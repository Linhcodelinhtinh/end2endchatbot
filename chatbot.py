"""RAG chatbot orchestration and batch inference."""

from __future__ import annotations

from pathlib import Path
from typing import List

from config import EXIT_COMMANDS, RAGConfig, UNKNOWN_ANSWER
from generators import AnswerGenerator, ExtractiveAnswerGenerator, make_generator
from io_utils import read_questions, write_answers
from retriever import RetrievalResult, VectorRetriever, build_retriever


class RAGChatbot:
    def __init__(
        self,
        retriever: VectorRetriever,
        generator: AnswerGenerator | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ):
        self.retriever = retriever
        self.generator = generator or ExtractiveAnswerGenerator()
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str) -> List[RetrievalResult]:
        results = self.retriever.search(question, self.top_k)
        return [result for result in results if result.score >= self.min_score]

    def answer(self, question: str, return_context: bool = False):
        results = self.retrieve(question)
        contexts = [result.document.text for result in results]
        answer = self.generator.generate(question, contexts) if contexts else UNKNOWN_ANSWER
        if return_context:
            return answer, results
        return answer


def build_chatbot(config: RAGConfig) -> RAGChatbot:
    retriever = build_retriever(
        config.kb_path,
        embedder_kind=config.embedder_kind,
        model_name=config.retriever_model,
    )
    return RAGChatbot(
        retriever=retriever,
        generator=make_generator(
            model_name=config.generator_model,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.llm_top_k,
        ),
        top_k=config.top_k,
    )


def run_batch(
    kb_path: str | Path,
    questions_path: str | Path,
    output_path: str | Path,
    embedder_kind: str = "tfidf",
    retriever_model: str | None = None,
    generator_model: str | None = None,
    top_k: int = 5,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    llm_top_k: int | None = None,
) -> List[str]:
    bot = build_chatbot(
        RAGConfig(
            kb_path=kb_path,
            embedder_kind=embedder_kind,
            retriever_model=retriever_model,
            generator_model=generator_model,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            llm_top_k=llm_top_k,
        )
    )
    answers = [str(bot.answer(question)).strip() for question in read_questions(questions_path)]
    write_answers(output_path, answers)
    return answers


def interactive_chat(bot: RAGChatbot) -> None:
    print("RAG chatbot ready. Type 'exit' to quit.")
    while True:
        question = input("Question: ").strip()
        if question.lower() in EXIT_COMMANDS:
            break
        answer, results = bot.answer(question, return_context=True)
        print(f"Answer: {answer}")
        if results:
            best = results[0]
            print(f"Source: {best.document.source} | score={best.score:.3f}")
