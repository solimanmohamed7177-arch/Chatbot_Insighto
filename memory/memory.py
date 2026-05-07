"""
memory/memory.py
════════════════════════════════════════════════════════════════
Insighto — Dual-Layer Memory System

Layer 1 — Short-term (ConversationBufferWindowMemory):
    Keeps the last N conversation turns in a LangChain-compatible
    messages list for direct injection into every LLM call.

Layer 2 — Long-term (FAISS vector store):
    Stores summarised facts about the user. Retrieved by semantic
    similarity when relevant context is needed.
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from utils.helpers import get_logger

logger = get_logger("insighto.memory")

# Max short-term turns (each turn = 1 user + 1 assistant message)
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

# FAISS persistence path
FAISS_DIR = Path(__file__).parent.parent / "data" / "faiss_store"


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Message:
    """Single conversation turn."""
    role:      str      # "user" | "assistant" | "system"
    content:   str
    timestamp: float = field(default_factory=time.time)


@dataclass
class UserProfile:
    """Persisted facts extracted from conversation."""
    name:        Optional[str]   = None
    preferences: list[str]       = field(default_factory=list)
    facts:       dict[str, str]  = field(default_factory=dict)


# ─── Short-term memory ────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Rolling window conversation buffer.
    Stores last MAX_HISTORY pairs of (user, assistant) messages.
    Compatible with LangChain message format.
    """

    def __init__(self, max_turns: int = MAX_HISTORY):
        self._max: int = max_turns
        self._messages: list[Message] = []

    def add_user(self, content: str) -> None:
        """Append a user message."""
        self._messages.append(Message(role="user", content=content.strip()))
        self._prune()

    def add_assistant(self, content: str) -> None:
        """Append an assistant message."""
        self._messages.append(Message(role="assistant", content=content.strip()))
        self._prune()

    def get_messages(self) -> list[dict[str, str]]:
        """
        Return conversation history as a list of dicts for LangChain.

        Returns:
            [{"role": "user"|"assistant", "content": "..."}]
        """
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def get_langchain_messages(self):
        """
        Return history as LangChain HumanMessage / AIMessage objects.
        Used for direct injection into ChatGroq calls.
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        result = []
        for m in self._messages:
            if m.role == "user":
                result.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                result.append(AIMessage(content=m.content))
            elif m.role == "system":
                result.append(SystemMessage(content=m.content))
        return result

    def clear(self) -> None:
        """Wipe all stored messages."""
        self._messages.clear()
        logger.info("Short-term memory cleared.")

    def __len__(self) -> int:
        return len(self._messages)

    def is_empty(self) -> bool:
        return len(self._messages) == 0

    def _prune(self) -> None:
        """Keep only the most recent max_turns × 2 messages."""
        limit = self._max * 2
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]


# ─── Long-term memory (FAISS) ─────────────────────────────────────────────────

class LongTermMemory:
    """
    FAISS-backed semantic memory for user facts and preferences.

    Stores short text snippets (e.g., "User's name is Sara").
    Retrieves the top-k most relevant snippets given a query.

    Falls back gracefully if FAISS or embeddings are unavailable.
    """

    def __init__(self):
        self._store = None          # FAISS VectorStore or None
        self._profile = UserProfile()
        self._docs: list[str] = []  # raw texts for inspection
        self._available = False
        self._try_init()

    def _try_init(self) -> None:
        """
        Attempt to initialise FAISS with HuggingFace embeddings.
        Silently disables itself if dependencies are missing.
        """
        try:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS as FAISSStore

            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
            self._FAISSStore = FAISSStore
            FAISS_DIR.mkdir(parents=True, exist_ok=True)
            self._available = True
            logger.info("Long-term FAISS memory initialised.")
        except Exception as e:
            logger.warning("FAISS unavailable (%s) — long-term memory disabled.", e)
            self._available = False

    def store(self, text: str) -> None:
        """
        Embed and store a fact/preference string.

        Args:
            text: Short descriptive string (e.g., "User prefers dark mode").
        """
        if not text.strip():
            return

        self._docs.append(text)

        if not self._available:
            return

        try:
            from langchain_core.documents import Document
            doc = Document(page_content=text, metadata={"ts": time.time()})

            if self._store is None:
                self._store = self._FAISSStore.from_documents(
                    [doc], self._embeddings
                )
            else:
                self._store.add_documents([doc])

            logger.debug("Stored to FAISS: %s", text[:60])
        except Exception as e:
            logger.error("FAISS store error: %s", e)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """
        Retrieve top-k semantically similar stored facts.

        Args:
            query: The search query (usually the current user message).
            k:     Number of results to return.

        Returns:
            List of relevant fact strings (may be empty).
        """
        if not self._available or self._store is None:
            # Fallback: return all stored docs (there aren't many)
            return self._docs[-k:] if self._docs else []

        try:
            results = self._store.similarity_search(query, k=k)
            return [r.page_content for r in results]
        except Exception as e:
            logger.error("FAISS retrieval error: %s", e)
            return []

    def update_profile(self, name: Optional[str] = None,
                       preference: Optional[str] = None,
                       fact_key: Optional[str] = None,
                       fact_val: Optional[str] = None) -> None:
        """
        Update the structured user profile.

        Args:
            name:       User's name if detected.
            preference: A preference string to add.
            fact_key:   A fact key (e.g., "location").
            fact_val:   The fact value (e.g., "Cairo").
        """
        if name:
            self._profile.name = name
            self.store(f"User's name is {name}.")
        if preference:
            if preference not in self._profile.preferences:
                self._profile.preferences.append(preference)
                self.store(f"User preference: {preference}")
        if fact_key and fact_val:
            self._profile.facts[fact_key] = fact_val
            self.store(f"User {fact_key}: {fact_val}")

    def get_profile(self) -> UserProfile:
        """Return the current user profile."""
        return self._profile

    def build_context_string(self) -> str:
        """
        Build a compact memory context block for system prompt injection.

        Returns:
            Multi-line string summarising known user facts.
        """
        p = self._profile
        parts: list[str] = []
        if p.name:
            parts.append(f"User's name: {p.name}")
        if p.preferences:
            parts.append(f"User preferences: {'; '.join(p.preferences[:5])}")
        if p.facts:
            facts_str = "; ".join(f"{k}={v}" for k, v in list(p.facts.items())[:5])
            parts.append(f"Known facts: {facts_str}")
        return "\n".join(parts) if parts else ""

    @property
    def is_available(self) -> bool:
        return self._available


# ─── Unified memory interface ─────────────────────────────────────────────────

class InsightoMemory:
    """
    Top-level memory manager combining short-term and long-term layers.

    Usage
    -----
    memory = InsightoMemory()
    memory.add_exchange("Hello!", "Hi there!")
    history = memory.get_history()          # for LangChain injection
    context = memory.get_long_term_context("What do I prefer?")  # FAISS
    """

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term  = LongTermMemory()

    def add_exchange(self, user_msg: str, assistant_msg: str) -> None:
        """Store one full conversation turn."""
        self.short_term.add_user(user_msg)
        self.short_term.add_assistant(assistant_msg)

    def add_user_turn(self, user_msg: str) -> None:
        """Store just the user side (before assistant responds)."""
        self.short_term.add_user(user_msg)

    def add_assistant_turn(self, assistant_msg: str) -> None:
        """Store just the assistant side."""
        self.short_term.add_assistant(assistant_msg)

    def get_history(self):
        """Return short-term history as LangChain message objects."""
        return self.short_term.get_langchain_messages()

    def get_history_dicts(self) -> list[dict]:
        """Return short-term history as plain dicts."""
        return self.short_term.get_messages()

    def get_long_term_context(self, query: str) -> str:
        """
        Retrieve semantically relevant long-term memory for a query.

        Returns:
            Newline-joined string of relevant facts, or empty string.
        """
        facts = self.long_term.retrieve(query, k=3)
        return "\n".join(facts) if facts else ""

    def learn_from_message(self, text: str) -> None:
        """
        Extract and store user facts from a message.

        Args:
            text: User message to analyse for learnable information.
        """
        import re

        # Name extraction — stop at first word to avoid capturing 'And' etc.
        name_match = re.search(
            r"(?:my name is|call me|i am|i'm)\s+([A-Za-z]+)",
            text, re.IGNORECASE
        )
        if name_match:
            candidate = name_match.group(1).strip().capitalize()
            stopwords = {'a', 'an', 'the', 'not', 'also', 'just', 'really', 'very', 'called', 'known'}
            if candidate.lower() not in stopwords:
                self.long_term.update_profile(name=candidate)

        # Preference extraction
        pref_match = re.search(
            r"i\s+(like|love|prefer|enjoy|hate|dislike)\s+(.{3,50}?)(?:\.|,|$)",
            text, re.IGNORECASE
        )
        if pref_match:
            verb, subject = pref_match.group(1), pref_match.group(2).strip()
            self.long_term.update_profile(preference=f"{verb} {subject}")

    def build_system_memory(self) -> str:
        """Return the long-term profile as a system prompt snippet."""
        return self.long_term.build_context_string()

    def clear(self) -> None:
        """Clear short-term memory (long-term is persistent)."""
        self.short_term.clear()

    def stats(self) -> dict:
        """Return memory statistics for UI display."""
        return {
            "short_term_turns": len(self.short_term) // 2,
            "long_term_docs":   len(self.long_term._docs),
            "faiss_active":     self.long_term.is_available,
            "user_name":        self.long_term.get_profile().name,
        }
