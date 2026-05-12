"""Command-line entrypoint for the RAG QA pipeline."""

from __future__ import annotations

import argparse

from chatbot import build_chatbot, interactive_chat, run_batch
from config import RAGConfig
from evaluate import evaluate_answers, print_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight RAG QA pipeline.")
    parser.add_argument("--kb", required=True, help="Knowledge-base file or directory.")
    parser.add_argument("--questions", help="Input questions.txt. If omitted, starts chat mode.")
    parser.add_argument("--output", default="system_outputs/system_output_1.txt")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedder", choices=["tfidf", "sentence-transformer"], default="tfidf")
    parser.add_argument("--retriever-model", help="SentenceTransformer model name/path.")
    parser.add_argument("--generator-model", help="Open HuggingFace causal/chat model name/path.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum LLM tokens to generate.")
    parser.add_argument("--temperature", type=float, help="Enable sampling with this temperature.")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling value for LLM generation.")
    parser.add_argument("--llm-top-k", type=int, help="Top-k sampling value for LLM generation.")
    parser.add_argument("--references", help="Optional reference_answers.txt for EM/F1/recall.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RAGConfig:
    return RAGConfig(
        kb_path=args.kb,
        embedder_kind=args.embedder,
        retriever_model=args.retriever_model,
        generator_model=args.generator_model,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        llm_top_k=args.llm_top_k,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)

    if args.questions:
        answers = run_batch(
            kb_path=config.kb_path,
            questions_path=args.questions,
            output_path=args.output,
            embedder_kind=config.embedder_kind,
            retriever_model=config.retriever_model,
            generator_model=config.generator_model,
            top_k=config.top_k,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            llm_top_k=config.llm_top_k,
        )
        print(f"Wrote {len(answers)} answers to {args.output}")
        if args.references:
            print_metrics(evaluate_answers(answers, args.references))
        return

    interactive_chat(build_chatbot(config))


if __name__ == "__main__":
    main()
