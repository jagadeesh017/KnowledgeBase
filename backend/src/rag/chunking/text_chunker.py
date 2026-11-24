import math
import re
from collections.abc import Iterator, Sequence
from itertools import pairwise

from rag.embeddings.base import EmbeddingProvider
from rag.schemas.document import DocumentChunk, LoadedDocument

_Unit = tuple[str, int]

_PARAGRAPH_PATTERN = re.compile(r"\n\s*\n+")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_CLAUSE_PATTERN = re.compile(r"(?<=[,;:])\s+|\s+[—–-]\s+")


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        embeddings: EmbeddingProvider | None = None,
        semantic_break_quantile: float = 0.75,
        embedding_batch_size: int = 64,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1")
        if not 0 < semantic_break_quantile < 1:
            raise ValueError("semantic_break_quantile must be between 0 and 1")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embeddings = embeddings
        self.semantic_break_quantile = semantic_break_quantile
        self.embedding_batch_size = embedding_batch_size
        self._min_chunk_size = max(1, chunk_size // 5)

    def chunk(self, document: LoadedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        counter = 0

        for page in document.pages:
            text = page.text.strip()
            if not text:
                continue

            for piece in self._split(text):
                counter += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.document_id}:chunk_{counter:03d}",
                        document_id=document.document_id,
                        content=piece,
                        page_number=page.page_number,
                        metadata={
                            "source": document.source,
                            **document.metadata,
                            **page.metadata,
                        },
                    )
                )

        return chunks

    def _split(self, text: str) -> Iterator[str]:
        units = self._units(text)
        if not units:
            return

        semantic_breaks = (
            self._semantic_breaks(units) if self.embeddings is not None else set()
        )
        current: list[_Unit] = []
        has_new_unit = False

        for index, unit in enumerate(units):
            too_long = has_new_unit and len(self._join(current + [unit])) > self.chunk_size
            paragraph_break = has_new_unit and current[-1][1] != unit[1]
            semantic_break = (
                has_new_unit
                and index in semantic_breaks
                and len(self._join(current)) >= self._min_chunk_size
            )

            if current and (too_long or paragraph_break or semantic_break):
                yield self._join(current)
                current = self._overlap(current)
                if current and current[-1][1] != unit[1]:
                    current = []
                has_new_unit = False
                while current and len(self._join(current + [unit])) > self.chunk_size:
                    current.pop(0)

            current.append(unit)
            has_new_unit = True

        if current:
            yield self._join(current)

    def _units(self, text: str) -> list[_Unit]:
        units: list[_Unit] = []
        for paragraph_index, paragraph in enumerate(_PARAGRAPH_PATTERN.split(text)):
            cleaned = paragraph.strip()
            if not cleaned:
                continue
            for sentence in _SENTENCE_PATTERN.split(cleaned):
                stripped = sentence.strip()
                if not stripped:
                    continue
                units.extend((piece, paragraph_index) for piece in self._split_long_unit(stripped))
        return units

    def _split_long_unit(self, text: str) -> Iterator[str]:
        if len(text) <= self.chunk_size:
            yield text
            return

        clauses = [clause.strip() for clause in _CLAUSE_PATTERN.split(text) if clause.strip()]
        if len(clauses) > 1:
            for clause in clauses:
                yield from self._split_long_unit(clause)
            return

        start = 0
        step = self.chunk_size - self.overlap
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            yield text[start:end]
            if end == len(text):
                break
            start += step

    def _semantic_breaks(self, units: list[_Unit]) -> set[int]:
        texts = [unit[0] for unit in units]
        vectors = self._embed_texts(texts)
        if len(vectors) != len(texts):
            return set()

        distances = [
            self._cosine_distance(vectors[index], vectors[index + 1])
            for index in range(len(vectors) - 1)
        ]
        if not distances:
            return set()

        threshold = self._quantile(distances, self.semantic_break_quantile)
        return {
            index + 1 for index, distance in enumerate(distances) if distance > threshold
        }

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.embeddings is None:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.embedding_batch_size):
            batch = texts[start : start + self.embedding_batch_size]
            vectors.extend([list(vector) for vector in self.embeddings.embed(batch)])
        return vectors

    @staticmethod
    def _quantile(values: list[float], quantile: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    @staticmethod
    def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 1.0

        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for left_value, right_value in zip(left, right):
            dot += left_value * right_value
            left_norm += left_value * left_value
            right_norm += right_value * right_value

        if left_norm == 0 or right_norm == 0:
            return 1.0

        distance = 1.0 - dot / math.sqrt(left_norm * right_norm)
        return max(0.0, min(2.0, distance))

    @staticmethod
    def _join(parts: list[_Unit]) -> str:
        if not parts:
            return ""

        output = [parts[0][0]]
        for previous, current in pairwise(parts):
            output.append("\n\n" if current[1] != previous[1] else " ")
            output.append(current[0])
        return "".join(output)

    def _overlap(self, parts: list[_Unit]) -> list[_Unit]:
        if self.overlap == 0:
            return []

        selected: list[_Unit] = []
        total = 0
        for part in reversed(parts):
            separator_length = 2 if selected else 0
            if total + len(part[0]) + separator_length > self.overlap:
                break
            selected.insert(0, part)
            total += len(part[0]) + separator_length
        return selected
