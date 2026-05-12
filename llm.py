"""Backward-compatible wrapper for the RAG QA CLI.

Prefer running `python main.py ...` for new usage. This file keeps the previous
`python llm.py ...` command working.
"""

from __future__ import annotations

from chatbot import RAGChatbot, build_chatbot, interactive_chat, run_batch
from config import EXIT_COMMANDS, RAGConfig, STOPWORDS, UNKNOWN_ANSWER
from evaluate import EvaluationMetrics, calculate_metrics, evaluate_answers
from generators import (
    AnswerGenerator,
    ExtractiveAnswerGenerator,
    HuggingFaceAnswerGenerator,
    build_prompt,
    make_generator,
)
from io_utils import load_reference_answers, read_questions, write_answers
from main import main


__all__ = [
    "AnswerGenerator",
    "EvaluationMetrics",
    "ExtractiveAnswerGenerator",
    "EXIT_COMMANDS",
    "HuggingFaceAnswerGenerator",
    "RAGChatbot",
    "RAGConfig",
    "STOPWORDS",
    "UNKNOWN_ANSWER",
    "build_chatbot",
    "build_prompt",
    "calculate_metrics",
    "evaluate_answers",
    "interactive_chat",
    "load_reference_answers",
    "main",
    "make_generator",
    "read_questions",
    "run_batch",
    "write_answers",
]


if __name__ == "__main__":
    main()
