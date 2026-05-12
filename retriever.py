"""Document loading, chunking, and retrieval for RAG QA."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from embedder import cosine_similarity, make_embedder


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".csv", ".html", ".htm", ".pdf"}


@dataclass
class Document:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    document: Document
    score: float


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError("Install pypdf to load PDF knowledge-base files.") from exc

    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_html(path: Path) -> str:
    html = _read_text_file(path)
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text("\n")


def _stringify_json_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(_stringify_json_value(item) for item in value if item is not None)
    if isinstance(value, dict):
        parts = []
        for key, nested_value in value.items():
            text = _stringify_json_value(nested_value)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    return str(value)


def _text_from_json_item(item) -> tuple[str, dict]:
    if isinstance(item, str):
        return item, {}
    if not isinstance(item, dict):
        return _stringify_json_value(item), {}

    text = item.get("text") or item.get("content") or item.get("body")
    metadata = {k: v for k, v in item.items() if k not in {"text", "content", "body"}}
    if text:
        return str(text), metadata

    flattened = []
    for key, value in item.items():
        value_text = _stringify_json_value(value)
        if value_text:
            flattened.append(f"{key}: {value_text}")
    return "\n".join(flattened), metadata


def _documents_from_json(path: Path) -> List[Document]:
    raw = json.loads(_read_text_file(path))
    items = raw if isinstance(raw, list) else [raw]
    docs: List[Document] = []
    for idx, item in enumerate(items):
        text, metadata = _text_from_json_item(item)
        if text.strip():
            docs.append(Document(f"{path.stem}-{idx}", text.strip(), str(path), metadata))
    return docs


def _documents_from_jsonl(path: Path) -> List[Document]:
    docs: List[Document] = []
    for idx, line in enumerate(_read_text_file(path).splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        text, metadata = _text_from_json_item(item)
        if text.strip():
            docs.append(Document(f"{path.stem}-{idx}", text.strip(), str(path), metadata))
    return docs


def _documents_from_csv(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text = row.get("text") or row.get("content") or row.get("body")
            if not text:
                text = " ".join(value for value in row.values() if value)
            if text.strip():
                docs.append(Document(f"{path.stem}-{idx}", text.strip(), str(path), dict(row)))
    return docs


def load_documents(path: str | Path) -> List[Document]:
    """Load raw knowledge-base documents from one file or a directory."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Knowledge-base path does not exist: {root}")

    files = [root] if root.is_file() else [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    documents: List[Document] = []

    for file_path in sorted(files):
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            documents.extend(_documents_from_json(file_path))
            continue
        if suffix == ".jsonl":
            documents.extend(_documents_from_jsonl(file_path))
            continue
        if suffix == ".csv":
            documents.extend(_documents_from_csv(file_path))
            continue
        if suffix == ".pdf":
            text = _read_pdf(file_path)
        elif suffix in {".html", ".htm"}:
            text = _read_html(file_path)
        else:
            text = _read_text_file(file_path)
        if text.strip():
            documents.append(Document(file_path.stem, text.strip(), str(file_path)))

    if not documents:
        raise ValueError(f"No supported documents found in {root}")
    return documents


def chunk_text(text: str, max_words: int = 220, overlap_words: int = 40) -> List[str]:
    """Split text into word chunks with overlap."""
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: List[str] = []
    step = max(max_words - overlap_words, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks


def chunk_documents(
    documents: Sequence[Document],
    max_words: int = 220,
    overlap_words: int = 40,
) -> List[Document]:
    chunks: List[Document] = []
    for doc in documents:
        for idx, text in enumerate(chunk_text(doc.text, max_words, overlap_words)):
            chunks.append(
                Document(
                    id=f"{doc.id}::chunk-{idx}",
                    text=text,
                    source=doc.source,
                    metadata={**doc.metadata, "parent_id": doc.id, "chunk_id": idx},
                )
            )
    return chunks


class VectorRetriever:
    """In-memory vector retriever."""

    def __init__(self, documents: Sequence[Document], embedder):
        if not documents:
            raise ValueError("VectorRetriever requires at least one document.")
        self.documents = list(documents)
        self.embedder = embedder
        self.document_vectors = self.embedder.fit_transform([doc.text for doc in self.documents])

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        query_vector = self.embedder.transform([query])
        scores = cosine_similarity(query_vector, self.document_vectors)
        if scores.size == 0:
            return []

        k = min(top_k, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievalResult(self.documents[idx], float(scores[idx]))
            for idx in top_indices
            if float(scores[idx]) > 0
        ]


def build_retriever(
    kb_path: str | Path,
    embedder_kind: str = "tfidf",
    model_name: str | None = None,
    chunk_words: int = 220,
    overlap_words: int = 40,
) -> VectorRetriever:
    raw_docs = load_documents(kb_path)
    chunks = chunk_documents(raw_docs, chunk_words, overlap_words)
    return VectorRetriever(chunks, make_embedder(embedder_kind, model_name))
