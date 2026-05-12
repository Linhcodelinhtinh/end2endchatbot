"""Embedding backends for a lightweight RAG system.

The default backend is a dependency-light TF-IDF embedder so the project can run
offline. If sentence-transformers is installed and a local/open model is
available, use SentenceTransformerEmbedder for stronger semantic retrieval.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Normalize text into simple word tokens."""
    return TOKEN_RE.findall(text.lower())


class SimpleTfidfEmbedder:
    """Small TF-IDF embedder implemented with stdlib + numpy."""

    def __init__(self, max_features: int = 50000, min_df: int = 1):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, texts: Sequence[str]) -> "SimpleTfidfEmbedder":
        doc_freq: Counter[str] = Counter()
        term_freq: Counter[str] = Counter()

        for text in texts:
            tokens = tokenize(text)
            term_freq.update(tokens)
            doc_freq.update(set(tokens))

        terms = [
            term
            for term, df in doc_freq.items()
            if df >= self.min_df
        ]
        terms.sort(key=lambda term: (-term_freq[term], term))
        terms = terms[: self.max_features]

        self.vocabulary_ = {term: idx for idx, term in enumerate(terms)}
        n_docs = max(len(texts), 1)
        self.idf_ = np.array(
            [math.log((1 + n_docs) / (1 + doc_freq[term])) + 1 for term in terms],
            dtype=np.float32,
        )
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.idf_ is None:
            raise ValueError("Embedder must be fitted before transform().")

        matrix = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            counts = Counter(tokenize(text))
            if not counts:
                continue
            total = sum(counts.values())
            for term, count in counts.items():
                col_idx = self.vocabulary_.get(term)
                if col_idx is None:
                    continue
                matrix[row_idx, col_idx] = (count / total) * self.idf_[col_idx]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTfidfEmbedder":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj


class SentenceTransformerEmbedder:
    """Wrapper for local/open HuggingFace sentence-transformers models."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers to use SentenceTransformerEmbedder."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.transform(texts)


def cosine_similarity(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    """Return cosine scores for one query vector against many document vectors."""
    if query_vector.ndim == 2:
        query_vector = query_vector[0]
    if doc_vectors.size == 0:
        return np.array([], dtype=np.float32)
    return doc_vectors @ query_vector


def make_embedder(kind: str = "tfidf", model_name: str | None = None):
    """Factory for embedders used by the retriever."""
    if kind == "sentence-transformer":
        return SentenceTransformerEmbedder(model_name or "BAAI/bge-m3")
    if kind == "tfidf":
        return SimpleTfidfEmbedder()
    raise ValueError(f"Unknown embedder kind: {kind}")
