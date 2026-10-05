"""Small, local RAG layer for SQL explanation generation and judging.

This is deliberately lightweight: a checked-in JSONL knowledge base is indexed
with TF-IDF and retrieved with cosine similarity. No API, database, or remote
embedding service is required. Retrieval is deterministic and independent of
Model A/Model B output, so both models receive the exact same context.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RagChunk:
    id: str
    title: str
    text: str
    tags: tuple[str, ...] = ()
    source: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: RagChunk
    score: float


class LocalRAG:
    def __init__(self, path: Path):
        self.path = path
        self.chunks = self._load(path)
        if not self.chunks:
            raise ValueError(f"RAG knowledge base is empty: {path}")

        documents = [
            f"{chunk.title}\n{' '.join(chunk.tags)}\n{chunk.text}"
            for chunk in self.chunks
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r"(?u)\b[\w#][\w#.-]*\b",
        )
        self.matrix = self.vectorizer.fit_transform(documents)
        self.file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _load(path: Path) -> list[RagChunk]:
        chunks: list[RagChunk] = []
        for line_no, line in enumerate(path.open(encoding="utf-8"), start=1):
            if not line.strip():
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid RAG JSON on line {line_no}: {exc}") from exc

            for key in ("id", "title", "text"):
                if not str(row.get(key, "")).strip():
                    raise ValueError(f"RAG line {line_no} is missing '{key}'")

            chunks.append(
                RagChunk(
                    id=str(row["id"]),
                    title=str(row["title"]),
                    text=str(row["text"]).strip(),
                    tags=tuple(str(x) for x in row.get("tags", [])),
                    source=str(row.get("source", "")),
                )
            )
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]

        ranked = sorted(
            range(len(self.chunks)),
            key=lambda i: (-float(scores[i]), self.chunks[i].id),
        )
        results: list[RetrievedChunk] = []
        for i in ranked[:top_k]:
            score = float(scores[i])
            if score <= 0:
                continue
            results.append(RetrievedChunk(self.chunks[i], round(score, 4)))
        return results

    @staticmethod
    def format_context(results: list[RetrievedChunk]) -> str:
        if not results:
            return (
                "No relevant local RAG guidance was retrieved. "
                "Use the supplied SQL and schema only."
            )

        parts = [
            "Retrieved local SQL guidance. This is generic supporting knowledge, "
            "not query-specific evidence. The exact SQL and schema always take priority."
        ]
        for rank, result in enumerate(results, start=1):
            source = f" | source: {result.chunk.source}" if result.chunk.source else ""
            parts.append(
                f"\n[{rank}] {result.chunk.title} "
                f"(retrieval_score={result.score:.4f}{source})\n{result.chunk.text}"
            )
        return "\n".join(parts)

    def retrieve_for_case(
        self,
        *,
        schema: str,
        sql: str,
        categories: list[str],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], str]:
        query = (
            f"SQL categories: {', '.join(categories)}\n"
            f"Schema:\n{schema}\n"
            f"SQL:\n{sql}"
        )
        results = self.retrieve(query, top_k=top_k)
        return results, self.format_context(results)
