"""
graph/flow.py
════════════════════════════════════════════════════════════════
Insighto — LangGraph Conversational Flow

Graph Architecture:
─────────────────────────────────────────────────────────────
                    ┌─────────────┐
                    │ START (user │
                    │   input)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   ROUTER    │  ← Intent detection node
                    └──────┬──────┘
                           │
              ┌────────────┼──────────────┐
              │            │              │
       ┌──────▼──┐   ┌─────▼──────┐  ┌───▼───────┐
       │ WEATHER │   │   GENERAL  │  │ CLARIFY   │
       │  NODE   │   │   LLM NODE │  │   NODE    │
       └──────┬──┘   └─────┬──────┘  └───┬───────┘
              │            │              │
              └────────────▼──────────────┘
                    ┌──────┴──────┐
                    │   MEMORY    │  ← Learn from exchange
                    │  UPDATE NODE│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    END      │
                    └─────────────┘

Nodes:
  • router_node    — classifies intent using helpers.detect_intent()
  • weather_node   — fetches live weather and enriches the LLM call
  • general_node   — standard LLM conversation (most messages go here)
  • clarify_node   — handles empty/unclear inputs politely
  • memory_node    — learns facts from the exchange, updates long-term memory
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict, Literal, Optional

from langgraph.graph import StateGraph, END

from utils.helpers import detect_intent, Intent, is_meaningful, get_logger
from memory.memory import InsightoMemory
from tools.weather_tool import get_weather_for_message
from chains.chat_chain import InsightoChatChain

logger = get_logger("insighto.graph")


# ─── Graph State ──────────────────────────────────────────────────────────────

class InsightoState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.

    Fields
    ------
    user_message    : Original user input.
    intent          : Detected intent label (set by router_node).
    special_context : Extra context for LLM (e.g., weather data).
    response        : Final assistant response text.
    error           : Error message if something went wrong.
    skip_memory     : True for greeting/clarify nodes where learning is skipped.
    """
    user_message:    str
    intent:          str
    special_context: str
    response:        str
    error:           str
    skip_memory:     bool


# ─── Node implementations ─────────────────────────────────────────────────────

def router_node(state: InsightoState) -> InsightoState:
    """
    Intent detection node.

    Analyses the user message and writes the detected Intent
    into state["intent"]. All routing decisions are made by
    the conditional edge function below, not here.

    Args:
        state: Current graph state.

    Returns:
        Updated state with intent field set.
    """
    text   = state["user_message"]
    intent = detect_intent(text)
    logger.info("Router: intent=%s for %r", intent, text[:50])
    return {**state, "intent": intent.value, "special_context": "", "error": ""}


def weather_node(
    chain:  InsightoChatChain,
    memory: InsightoMemory,
) -> callable:
    """
    Factory that returns a weather-handling node function.

    The node:
      1. Fetches live weather data for the detected city.
      2. Injects the data as special_context into the LLM call.
      3. Asks the LLM to produce a natural weather narrative.

    Args:
        chain:  Active chat chain.
        memory: Conversation memory.

    Returns:
        LangGraph node function.
    """
    def _node(state: InsightoState) -> InsightoState:
        user_msg     = state["user_message"]
        weather_ctx  = get_weather_for_message(user_msg)

        if weather_ctx:
            special_ctx = (
                "Use the following LIVE weather data to answer the user's question naturally. "
                "Include practical advice (clothing, activities) based on the conditions.\n\n"
                + weather_ctx
            )
        else:
            special_ctx = (
                "The user asked about weather but live data could not be fetched. "
                "Acknowledge this gracefully and offer general advice."
            )

        try:
            response = chain.invoke(user_msg, memory, special_ctx)
            return {**state, "response": response, "special_context": weather_ctx}
        except Exception as e:
            logger.error("weather_node error: %s", e)
            return {**state, "response": chain._error_reply(e), "error": str(e)}

    return _node


def general_node(
    chain:  InsightoChatChain,
    memory: InsightoMemory,
) -> callable:
    """
    Factory for the general-purpose LLM conversation node.

    Handles: general questions, memory/preference references,
             identity queries, calculations, greetings, farewells.

    Args:
        chain:  Active chat chain.
        memory: Conversation memory.

    Returns:
        LangGraph node function.
    """
    def _node(state: InsightoState) -> InsightoState:
        user_msg = state["user_message"]
        intent   = state["intent"]

        # Build intent-specific guidance for the LLM
        hint = _intent_hint(intent)

        try:
            response = chain.invoke(user_msg, memory, hint)
            return {**state, "response": response}
        except Exception as e:
            logger.error("general_node error: %s", e)
            return {**state, "response": chain._error_reply(e), "error": str(e)}

    return _node


def clarify_node(
    chain:  InsightoChatChain,
    memory: InsightoMemory,
) -> callable:
    """
    Factory for the clarification node.

    Activated when user input is empty, too short, or otherwise unclear.
    Uses the LLM to ask a polite clarifying question.

    Args:
        chain:  Active chat chain.
        memory: Conversation memory.

    Returns:
        LangGraph node function.
    """
    def _node(state: InsightoState) -> InsightoState:
        user_msg = state["user_message"]

        if not is_meaningful(user_msg):
            # Static fallback for completely empty/meaningless input
            profile = memory.long_term.get_profile()
            name    = f", {profile.name}" if profile.name else ""
            response = (
                f"Hi{name}! It looks like your message was empty or unclear. "
                "What can I help you with today? Feel free to ask me anything — "
                "I'm here to help! 😊"
            )
        else:
            # Let the LLM ask for clarification naturally
            hint = (
                "The user's input is ambiguous or very short. "
                "Ask a concise, friendly clarifying question to understand what they need. "
                "Don't make assumptions — just ask."
            )
            try:
                response = chain.invoke(user_msg, memory, hint)
            except Exception as e:
                response = "I didn't quite catch that. Could you rephrase?"

        return {**state, "response": response, "skip_memory": True}

    return _node


def memory_node(memory: InsightoMemory) -> callable:
    """
    Factory for the memory update node.

    Runs after every successful LLM response to:
      1. Add the full exchange to short-term memory.
      2. Extract and store learnable facts in long-term memory.

    Args:
        memory: Conversation memory instance.

    Returns:
        LangGraph node function.
    """
    def _node(state: InsightoState) -> InsightoState:
        if state.get("skip_memory"):
            return state

        user_msg = state["user_message"]
        response = state["response"]

        if response and not response.startswith("⚠️"):
            memory.add_exchange(user_msg, response)
            memory.learn_from_message(user_msg)
            logger.debug("Memory updated: +1 exchange")

        return state

    return _node


# ─── Intent hint builder ──────────────────────────────────────────────────────

def _intent_hint(intent: str) -> str:
    """
    Return a short instruction hint for the LLM based on detected intent.

    Args:
        intent: String value from Intent enum.

    Returns:
        Hint string (injected as special_context into the chain).
    """
    hints = {
        "greeting":    "The user is greeting you. Respond warmly and ask how you can help.",
        "farewell":    "The user is saying goodbye. Respond warmly with a brief, personalised farewell.",
        "memory":      "The user is referencing something from earlier in the conversation. Use the conversation history to answer accurately.",
        "preference":  "The user is sharing a preference or personal fact. Acknowledge it warmly and confirm you've noted it.",
        "calculation": "The user wants a calculation or mathematical reasoning. Show your work clearly and verify the result.",
        "identity":    "The user is asking about who you are. Introduce yourself as Insighto — intelligent, analytical, helpful — without mentioning the underlying model.",
    }
    return hints.get(intent, "")


# ─── Routing logic ────────────────────────────────────────────────────────────

def _route(state: InsightoState) -> Literal["weather", "general", "clarify"]:
    """
    Conditional edge function — decides which node to activate next.

    Args:
        state: Current graph state (must have 'intent' and 'user_message').

    Returns:
        Node name string.
    """
    user_msg = state["user_message"]
    intent   = state["intent"]

    if not is_meaningful(user_msg) or intent == Intent.UNCLEAR.value:
        return "clarify"
    if intent == Intent.WEATHER.value:
        return "weather"
    return "general"


# ─── Graph builder ────────────────────────────────────────────────────────────

class InsightoFlow:
    """
    LangGraph-based conversational flow for Insighto.

    Builds and compiles the state graph, then exposes a single
    `run(user_message)` method for external callers.

    Usage
    -----
    chain  = InsightoChatChain()
    memory = InsightoMemory()
    flow   = InsightoFlow(chain, memory)
    reply  = flow.run("What's the weather in Cairo?")
    """

    def __init__(self, chain: InsightoChatChain, memory: InsightoMemory):
        """
        Initialise and compile the LangGraph flow.

        Args:
            chain:  Configured InsightoChatChain instance.
            memory: InsightoMemory instance.
        """
        self._memory  = memory
        self._chain   = chain
        self._graph   = self._build()
        logger.info("InsightoFlow compiled successfully.")

    def _build(self):
        """Build and compile the StateGraph."""
        builder = StateGraph(InsightoState)

        # ── Add nodes ─────────────────────────────────────────────────────────
        builder.add_node("router",  router_node)
        builder.add_node("weather", weather_node(self._chain, self._memory))
        builder.add_node("general", general_node(self._chain, self._memory))
        builder.add_node("clarify", clarify_node(self._chain, self._memory))
        builder.add_node("memory",  memory_node(self._memory))

        # ── Entry point ───────────────────────────────────────────────────────
        builder.set_entry_point("router")

        # ── Conditional routing from router → processing nodes ────────────────
        builder.add_conditional_edges(
            source    = "router",
            path      = _route,
            path_map  = {
                "weather": "weather",
                "general": "general",
                "clarify": "clarify",
            },
        )

        # ── All processing nodes → memory update → END ────────────────────────
        for node in ("weather", "general", "clarify"):
            builder.add_edge(node, "memory")

        builder.add_edge("memory", END)

        return builder.compile()

    def run(self, user_message: str) -> str:
        """
        Process one user message through the full graph.

        Args:
            user_message: Raw user input string.

        Returns:
            Assistant response string.
        """
        initial_state: InsightoState = {
            "user_message":    user_message,
            "intent":          "",
            "special_context": "",
            "response":        "",
            "error":           "",
            "skip_memory":     False,
        }

        try:
            final_state = self._graph.invoke(initial_state)
            response    = final_state.get("response", "")
            if not response:
                response = "I couldn't generate a response. Please try again."
            return response
        except Exception as e:
            logger.error("Graph execution error: %s", e)
            return f"⚠️ An unexpected error occurred: {e}"

    @property
    def memory(self) -> InsightoMemory:
        """Access the memory instance."""
        return self._memory
