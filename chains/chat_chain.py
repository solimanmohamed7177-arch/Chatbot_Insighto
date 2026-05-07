"""
chains/chat_chain.py
════════════════════════════════════════════════════════════════
Insighto — LangChain Chat Chain

Builds and executes the core LLM conversation chain using:
  • ChatGroq (Groq API — free tier)
  • Dynamic system prompt (personality + memory injection)
  • Short-term conversation history
  • Long-term memory context retrieval

The chain is intentionally decoupled from the LangGraph flow —
the graph calls this chain as one of its nodes.
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from utils.helpers import get_logger
from memory.memory import InsightoMemory

load_dotenv(override=True)
logger = get_logger("insighto.chain")

# ─── LLM configuration ───────────────────────────────────────────────────────

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL:   str = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# ─── Insighto personality ─────────────────────────────────────────────────────

_SYSTEM_BASE = """You are Insighto — an intelligent, analytical, and genuinely helpful AI assistant.

## Your Personality
- **Intelligent**: You think carefully before answering; you reason through problems step by step.
- **Analytical**: You break down complex topics clearly and explain your reasoning.
- **Helpful**: You focus on what the user actually needs, not just what they literally asked.
- **Warm**: You're friendly and approachable — like a knowledgeable colleague, not a cold machine.
- **Honest**: You admit uncertainty; you never make things up.

## Response Style
- Be concise but complete — never sacrifice accuracy for brevity.
- Use markdown formatting naturally (bold, lists, code blocks where appropriate).
- Address the user by name if you know it.
- Never start with sycophantic openers like "Great question!" or "Of course!".

## Identity
- Your name is Insighto.
- You were created as a demonstration of modern AI architecture.
- Never claim to be GPT, Llama, or any specific model.

{memory_block}
"""


def _build_system_prompt(memory_block: str = "") -> str:
    """
    Build the full system prompt with optional memory context injected.

    Args:
        memory_block: Formatted string of known user facts/preferences.

    Returns:
        Complete system prompt string.
    """
    mem_section = ""
    if memory_block:
        mem_section = f"\n## What I Know About You\n{memory_block}"
    return _SYSTEM_BASE.format(memory_block=mem_section)


# ─── Chain class ─────────────────────────────────────────────────────────────

class InsightoChatChain:
    """
    Core LangChain chat chain for Insighto.

    Manages LLM instantiation, system prompt construction, history injection,
    and response generation. Acts as the primary "brain" node in LangGraph.

    Usage
    -----
    chain = InsightoChatChain()
    reply = chain.invoke(
        user_message  = "What is quantum entanglement?",
        memory        = insighto_memory_instance,
        special_ctx   = "",   # e.g., weather data
    )
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialise the ChatGroq LLM.

        Args:
            api_key: Groq API key (falls back to GROQ_API_KEY env var).
            model:   Model name (falls back to GROQ_MODEL env var).

        Raises:
            ValueError: If no API key is found.
        """
        key = api_key or GROQ_API_KEY
        if not key or key == "gsk_your_groq_key_here":
            raise ValueError(
                "GROQ_API_KEY is missing or is a placeholder.\n"
                "  1. Get a free key at: https://console.groq.com\n"
                "  2. Add it to your .env file: GROQ_API_KEY=gsk_...\n"
            )

        self._model_name = model or GROQ_MODEL
        self._llm = ChatGroq(
            model        = self._model_name,
            groq_api_key = key,
            temperature  = 0.7,
            max_tokens   = 1024,
        )
        self._parser = StrOutputParser()
        logger.info("InsightoChatChain ready: model=%s", self._model_name)

    def invoke(
        self,
        user_message: str,
        memory:       InsightoMemory,
        special_ctx:  str = "",
    ) -> str:
        """
        Generate a response to the user's message.

        Constructs the full message list:
          [SystemMessage] → [ConversationHistory...] → [HumanMessage]

        Then calls Groq and returns the response text.

        Args:
            user_message: Current user input.
            memory:       InsightoMemory instance for context retrieval.
            special_ctx:  Any extra context to append (e.g., weather data).

        Returns:
            Assistant response string.
        """
        # Retrieve long-term context relevant to this message
        long_term_ctx = memory.get_long_term_context(user_message)
        profile_ctx   = memory.build_system_memory()

        # Build system prompt
        system = _build_system_prompt(profile_ctx)

        # Inject special context (weather, tool output, etc.)
        effective_message = user_message
        if special_ctx:
            effective_message = (
                f"{user_message}\n\n"
                f"[Context for your response — use this data naturally, "
                f"don't quote it verbatim]:\n{special_ctx}"
            )

        if long_term_ctx and long_term_ctx not in profile_ctx:
            effective_message += f"\n\n[Relevant memory]: {long_term_ctx}"

        # Build full message list
        messages = [SystemMessage(content=system)]
        messages.extend(memory.get_history())        # short-term conversation history
        messages.append(HumanMessage(content=effective_message))

        logger.debug(
            "Invoking LLM: model=%s history=%d msgs special_ctx=%s",
            self._model_name,
            len(messages) - 2,
            bool(special_ctx),
        )

        try:
            response = self._llm.invoke(messages)
            reply    = self._parser.invoke(response)
            return reply.strip()

        except Exception as e:
            logger.error("LLM invocation failed: %s", e)
            return self._error_reply(e)

    @property
    def model_name(self) -> str:
        """Name of the active Groq model."""
        return self._model_name

    def ping(self) -> tuple[bool, str]:
        """
        Minimal connectivity check against the Groq API.

        Returns:
            (True, model_name) on success, (False, error_message) on failure.
        """
        try:
            r = self._llm.invoke([HumanMessage(content="ping")])
            return True, self._model_name
        except Exception as e:
            err = str(e)
            if "401" in err or "auth" in err.lower():
                return False, "Invalid API key"
            if "404" in err or "model" in err.lower():
                return False, f"Model not found: {self._model_name}"
            return False, err[:80]

    @staticmethod
    def _error_reply(error: Exception) -> str:
        """Return a user-friendly error message."""
        err = str(error).lower()
        if "401" in err or "auth" in err:
            return (
                "⚠️ **Authentication error** — my API key seems to be invalid. "
                "Please check the `GROQ_API_KEY` in your `.env` file."
            )
        if "429" in err or "rate" in err:
            return "⚠️ **Rate limit reached.** Please wait a moment before sending another message."
        if "503" in err or "timeout" in err or "connection" in err:
            return "⚠️ **Connection issue.** The AI service is temporarily unavailable. Please try again."
        return (
            "⚠️ I encountered an unexpected error. Please try again, "
            "or check the console for details."
        )
