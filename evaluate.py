"""Evaluation helpers for exact match, token F1, and answer recall."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from config import STOPWORDS
from io_utils import load_reference_answers


@dataclass(frozen=True)
class EvaluationMetrics:
    exact_match: float
    f1: float
    answer_recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "exact_match": self.exact_match,
            "f1": self.f1,
            "answer_recall": self.answer_recall,
        }


def normalize_answer(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return [token for token in text.split() if token not in STOPWORDS]


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction)
    ref_tokens = normalize_answer(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    overlap = sum(min(pred_tokens.count(tok), ref_tokens.count(tok)) for tok in common)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def calculate_metrics(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> EvaluationMetrics:
    if len(predictions) != len(references):
        raise ValueError(
            f"Prediction/reference length mismatch: {len(predictions)} vs {len(references)}"
        )

    exact_matches = []
    f1_scores = []
    recalls = []
    for prediction, reference_group in zip(predictions, references):
        normalized_prediction = normalize_answer(prediction)
        normalized_refs = [normalize_answer(ref) for ref in reference_group]
        exact_matches.append(float(any(normalized_prediction == ref for ref in normalized_refs)))
        f1_scores.append(max(f1_score(prediction, ref) for ref in reference_group))
        recalls.append(
            float(any(set(ref).issubset(set(normalized_prediction)) for ref in normalized_refs if ref))
        )

    n = max(len(predictions), 1)
    return EvaluationMetrics(
        exact_match=sum(exact_matches) / n,
        f1=sum(f1_scores) / n,
        answer_recall=sum(recalls) / n,
    )


def evaluate_answers(predictions: Sequence[str], reference_path: str | Path) -> dict[str, float]:
    metrics = calculate_metrics(predictions, load_reference_answers(reference_path))
    return metrics.as_dict()


def print_metrics(metrics: dict[str, float]) -> None:
    print(
        "Metrics: "
        f"EM={metrics['exact_match']:.3f} "
        f"F1={metrics['f1']:.3f} "
        f"Recall={metrics['answer_recall']:.3f}"
    )
