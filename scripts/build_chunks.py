"""Build semantically-aware chunks from docs.jsonl.

Usage:
  python scripts/build_chunks.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

HEADING_RE = re.compile(r"^(chương|chapter|mục|section|điều|article)\b", flags=re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk docs.jsonl into chunks.jsonl.")
    parser.add_argument("--input", default="data/knowledge_base/processed/docs.jsonl")
    parser.add_argument("--output", default="data/knowledge_base/processed/chunks.jsonl")
    parser.add_argument("--chunk-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--min-chunk-words", type=int, default=30)
    parser.add_argument("--max-sentence-words", type=int, default=90)
    parser.add_argument("--dedupe-chunks", action="store_true", default=True)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def split_sentences(text: str, max_sentence_words: int) -> list[str]:
    raw_parts = SENTENCE_SPLIT_RE.split(text)
    sentences: list[str] = []
    for part in raw_parts:
        normalized = re.sub(r"\s+", " ", part).strip()
        if not normalized:
            continue
        # Hard split very long sentences for better retrieval granularity.
        words = normalized.split()
        if len(words) <= max_sentence_words:
            sentences.append(normalized)
            continue
        for i in range(0, len(words), max_sentence_words):
            sentences.append(" ".join(words[i : i + max_sentence_words]))
    return sentences


def chunk_sentences(
    sentences: list[str],
    max_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> list[tuple[int, int, str, str]]:
    if not sentences:
        return []

    chunks: list[tuple[int, int, str, str]] = []
    i = 0
    word_cursor = 0
    while i < len(sentences):
        chunk_sentences_acc: list[str] = []
        chunk_words = 0
        start_word = word_cursor
        section_label = "body"

        j = i
        while j < len(sentences):
            sent = sentences[j]
            sent_words = len(sent.split())
            if chunk_sentences_acc and (chunk_words + sent_words > max_words):
                break
            if not chunk_sentences_acc and HEADING_RE.match(sent):
                section_label = "heading"
            chunk_sentences_acc.append(sent)
            chunk_words += sent_words
            j += 1
            if chunk_words >= max_words:
                break

        if chunk_words >= min_chunk_words or not chunks:
            text = " ".join(chunk_sentences_acc).strip()
            end_word = start_word + chunk_words
            chunks.append((start_word, end_word, text, section_label))

        if j >= len(sentences):
            break

        # Sentence-aware overlap based on approximate overlap words.
        back_words = 0
        back_idx = j - 1
        while back_idx >= i and back_words < overlap_words:
            back_words += len(sentences[back_idx].split())
            back_idx -= 1
        i = max(back_idx + 1, i + 1)
        word_cursor = max(start_word + max(chunk_words - overlap_words, 0), 0)

    return chunks


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc_count = 0
    chunk_count = 0
    filtered_short = 0
    deduped_chunks = 0
    seen_chunk_hashes: set[str] = set()

    with output_path.open("w", encoding="utf-8") as out:
        for doc in iter_jsonl(input_path):
            doc_count += 1
            sentences = split_sentences(doc.get("text", ""), args.max_sentence_words)
            chunks = chunk_sentences(
                sentences=sentences,
                max_words=args.chunk_words,
                overlap_words=args.overlap_words,
                min_chunk_words=args.min_chunk_words,
            )
            for idx, (start_word, end_word, chunk, section_label) in enumerate(chunks):
                words = len(chunk.split())
                if words < args.min_chunk_words:
                    filtered_short += 1
                    continue

                chunk_digest = hashlib.sha1(chunk.lower().encode("utf-8")).hexdigest()
                if args.dedupe_chunks and chunk_digest in seen_chunk_hashes:
                    deduped_chunks += 1
                    continue
                seen_chunk_hashes.add(chunk_digest)

                row = {
                    "id": f"{doc['id']}::chunk-{idx}",
                    "doc_id": doc["id"],
                    "chunk_id": idx,
                    "source_id": doc.get("source_id"),
                    "source_url": doc.get("source_url"),
                    "title": doc.get("title"),
                    "category": doc.get("category"),
                    "language": doc.get("language"),
                    "collected_at": doc.get("collected_at"),
                    "text": chunk,
                    "metadata": {
                        **(doc.get("metadata") or {}),
                        "start_word": start_word,
                        "end_word": end_word,
                        "chunk_words": words,
                        "section_label": section_label,
                        "sentence_count": len(split_sentences(chunk, args.max_sentence_words)),
                    },
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                chunk_count += 1

    print(
        json.dumps(
            {
                "docs_in": doc_count,
                "chunks_out": chunk_count,
                "filtered_short_chunks": filtered_short,
                "deduped_chunks": deduped_chunks,
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
