"""Shared configuration and constants for the RAG QA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with",
    "ai", "la", "cua", "co", "cho", "cac", "nhung", "va", "ve", "o", "tai",
    "nao", "khi", "gi", "duoc", "trong", "tu", "den",
}

UNKNOWN_ANSWER = "I don't know"
EXIT_COMMANDS = {"exit", "quit"}


@dataclass(frozen=True)
class RAGConfig:
    kb_path: str | Path
    embedder_kind: str = "tfidf"
    retriever_model: str | None = None
    generator_model: str | None = None
    top_k: int = 5
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    llm_top_k: int | None = None
