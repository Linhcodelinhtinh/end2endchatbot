"""Answer generators used by the RAG chatbot."""

from __future__ import annotations

import copy
import re
from typing import List, Protocol, Sequence

from config import STOPWORDS, UNKNOWN_ANSWER
from embedder import tokenize


class AnswerGenerator(Protocol):
    def generate(self, question: str, contexts: Sequence[str]) -> str:
        """Generate a short answer from retrieved contexts."""


class ExtractiveAnswerGenerator:
    """Fallback answerer that selects the most relevant context sentence."""

    SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        candidates = self._candidate_sentences(contexts)
        if not candidates:
            return UNKNOWN_ANSWER

        return self._best_sentence(question, candidates)

    def _candidate_sentences(self, contexts: Sequence[str]) -> List[str]:
        candidates: List[str] = []
        for context in contexts:
            candidates.extend(
                sentence.strip()
                for sentence in self.SENTENCE_RE.split(context)
                if sentence.strip()
            )
        return candidates

    def _best_sentence(self, question: str, candidates: Sequence[str]) -> str:
        question_terms = {
            token for token in tokenize(question)
            if token not in STOPWORDS and len(token) > 1
        }

        scored = []
        for sentence in candidates:
            terms = set(tokenize(sentence))
            overlap = len(question_terms & terms)
            density = overlap / max(len(terms), 1)
            scored.append((overlap, density, -len(sentence), sentence))

        best = max(scored)
        return re.sub(r"\s+", " ", best[3]).strip()


class HuggingFaceAnswerGenerator:
    """Generative QA using an open HuggingFace causal/chat model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        max_new_tokens: int = 256,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers and torch to use HuggingFace generation.") from exc

        self.torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = temperature is not None and temperature > 0
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def generate(self, question: str, contexts: Sequence[str]) -> str:
        prompt = build_prompt(question, contexts)
        model_inputs = self._tokenize_prompt(prompt)

        with self.torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=self._generation_config(),
            )

        input_length = model_inputs["input_ids"].shape[-1]
        new_tokens = generated_ids[0][input_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = self._strip_thinking(text)
        return text or UNKNOWN_ANSWER

    def _generation_config(self):
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = self.max_new_tokens
        generation_config.pad_token_id = self.tokenizer.eos_token_id

        if self.do_sample:
            generation_config.do_sample = True
            generation_config.temperature = self.temperature
            if self.top_p is not None:
                generation_config.top_p = self.top_p
            if self.top_k is not None:
                generation_config.top_k = self.top_k
            return generation_config

        generation_config.do_sample = False
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50
        return generation_config

    def _tokenize_prompt(self, prompt: str):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a factual QA assistant. Answer using only the provided "
                    "context. If the context is insufficient, say you do not know."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            rendered = prompt

        return self.tokenizer(rendered, return_tensors="pt").to(self.device)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_prompt(question: str, contexts: Sequence[str]) -> str:
    context_block = "\n\n".join(f"[{idx + 1}] {text}" for idx, text in enumerate(contexts))
    return (
        "Answer the question using only the context.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def make_generator(
    model_name: str | None = None,
    max_new_tokens: int = 256,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> AnswerGenerator:
    if model_name:
        return HuggingFaceAnswerGenerator(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    return ExtractiveAnswerGenerator()
