"""File IO helpers for assignment-style QA data."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence


def read_questions(path: str | Path) -> List[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def write_answers(path: str | Path, answers: Sequence[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(answers) + "\n", encoding="utf-8")


def load_reference_answers(path: str | Path) -> List[List[str]]:
    reference_lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    return [
        [candidate.strip() for candidate in line.split(";") if candidate.strip()]
        for line in reference_lines
        if line.strip()
    ]
