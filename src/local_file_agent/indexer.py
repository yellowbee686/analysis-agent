from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class DocChunk:
    path: str
    heading: str
    text: str
    start_line: int


def find_markdown_files(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    return sorted([p for p in root_dir.rglob("*.md") if p.is_file()])


def _tokenize_query(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", query)
    cleaned = [t.strip() for t in tokens if t.strip()]
    return cleaned or [query.strip()]


def _split_markdown(text: str) -> Iterable[tuple[str, str, int]]:
    lines = text.splitlines()
    current_heading = "(no heading)"
    current_lines: List[str] = []
    start_line = 1

    def flush():
        if current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                yield (current_heading, chunk_text, start_line)

    for idx, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            yield from flush()
            current_heading = stripped.strip()
            current_lines = [line]
            start_line = idx
        else:
            current_lines.append(line)

    yield from flush()


class LocalIndex:
    def __init__(
        self,
        root_dir: Path,
        chunk_max_chars: int = 1200,
        snippet_chars: int = 400,
        index_dir: Path | None = None,
    ):
        self.root_dir = root_dir.resolve()
        self.chunk_max_chars = chunk_max_chars
        self.snippet_chars = snippet_chars
        self.index_dir = index_dir
        self.chunks: List[DocChunk] = []
        self.files: List[Path] = []
        self.total_chars = 0
        self._bm25 = None
        self._bm25_tokens: List[List[str]] = []

    def build(self, force_rebuild: bool = False) -> None:
        if not force_rebuild and self._load_cache():
            return
        self.files = find_markdown_files(self.root_dir)
        chunks: List[DocChunk] = []
        total_chars = 0
        for path in self.files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            total_chars += len(text)
            for heading, chunk_text, start_line in _split_markdown(text):
                if not chunk_text:
                    continue
                if len(chunk_text) <= self.chunk_max_chars:
                    chunks.append(
                        DocChunk(
                            path=str(path),
                            heading=heading,
                            text=chunk_text,
                            start_line=start_line,
                        )
                    )
                else:
                    chunks.extend(
                        self._split_large_chunk(
                            path=str(path),
                            heading=heading,
                            text=chunk_text,
                            start_line=start_line,
                        )
                    )
        self.chunks = chunks
        self.total_chars = total_chars
        self._build_bm25_index()
        self._save_cache()

    def _split_large_chunk(
        self, path: str, heading: str, text: str, start_line: int
    ) -> List[DocChunk]:
        parts: List[DocChunk] = []
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        buffer: List[str] = []
        buffer_len = 0
        current_start = start_line
        for para in paragraphs:
            para_len = len(para)
            if buffer_len + para_len + 2 <= self.chunk_max_chars:
                buffer.append(para)
                buffer_len += para_len + 2
                continue
            if buffer:
                parts.append(
                    DocChunk(
                        path=path,
                        heading=heading,
                        text="\n\n".join(buffer).strip(),
                        start_line=current_start,
                    )
                )
                current_start += len("\n".join(buffer).splitlines())
                buffer = []
                buffer_len = 0
            buffer.append(para)
            buffer_len = para_len
        if buffer:
            parts.append(
                DocChunk(
                    path=path,
                    heading=heading,
                    text="\n\n".join(buffer).strip(),
                    start_line=current_start,
                )
            )
        return parts

    def search(
        self, query: str, top_k: int = 5, min_score: int = 1
    ) -> List[dict]:
        if self._bm25 is not None:
            return self._bm25_search(query=query, top_k=top_k)
        terms = _tokenize_query(query)
        if not terms:
            return []
        scored = []
        for chunk in self.chunks:
            haystack = chunk.text
            score = 0
            for term in terms:
                if not term:
                    continue
                score += haystack.count(term) + haystack.lower().count(
                    term.lower()
                )
            if score >= min_score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            snippet = chunk.text.strip().replace("\n", " ")
            if len(snippet) > self.snippet_chars:
                snippet = snippet[: self.snippet_chars] + "..."
            results.append(
                {
                    "path": chunk.path,
                    "heading": chunk.heading,
                    "start_line": chunk.start_line,
                    "score": score,
                    "snippet": snippet,
                }
            )
        return results

    def _tokenize_text(self, text: str) -> List[str]:
        return _tokenize_query(text)

    def _build_bm25_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self._bm25 = None
            self._bm25_tokens = []
            return

        tokens = [self._tokenize_text(chunk.text) for chunk in self.chunks]
        if not tokens:
            self._bm25 = None
            self._bm25_tokens = []
            return
        self._bm25_tokens = tokens
        self._bm25 = BM25Okapi(tokens)

    def _bm25_search(self, query: str, top_k: int = 5) -> List[dict]:
        if self._bm25 is None:
            return []
        tokens = self._tokenize_text(query)
        scores = self._bm25.get_scores(tokens)
        if not len(scores):
            return []
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda item: item[1], reverse=True)
        results = []
        for idx, score in indexed_scores[:top_k]:
            chunk = self.chunks[idx]
            snippet = chunk.text.strip().replace("\n", " ")
            if len(snippet) > self.snippet_chars:
                snippet = snippet[: self.snippet_chars] + "..."
            results.append(
                {
                    "path": chunk.path,
                    "heading": chunk.heading,
                    "start_line": chunk.start_line,
                    "score": float(score),
                    "snippet": snippet,
                }
            )
        return results

    def stats(self) -> dict:
        return {
            "root_dir": str(self.root_dir),
            "file_count": len(self.files),
            "chunk_count": len(self.chunks),
            "total_chars": self.total_chars,
        }

    def _cache_path(self) -> Path | None:
        if not self.index_dir:
            return None
        key_material = "|".join(
            [
                str(self.root_dir),
                str(self.chunk_max_chars),
                str(self.snippet_chars),
            ]
        )
        digest = hashlib.sha256(
            key_material.encode("utf-8")
        ).hexdigest()[:12]
        return self.index_dir / f"index_{digest}.json"

    def _load_cache(self) -> bool:
        cache_path = self._cache_path()
        if not cache_path or not cache_path.exists():
            return False
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("version") != 1:
            return False
        if payload.get("root_dir") != str(self.root_dir):
            return False
        if payload.get("chunk_max_chars") != self.chunk_max_chars:
            return False
        if payload.get("snippet_chars") != self.snippet_chars:
            return False
        self.files = [Path(p) for p in payload.get("files", [])]
        self.total_chars = int(payload.get("total_chars", 0))
        self.chunks = [
            DocChunk(
                path=chunk["path"],
                heading=chunk["heading"],
                text=chunk["text"],
                start_line=int(chunk["start_line"]),
            )
            for chunk in payload.get("chunks", [])
        ]
        self._build_bm25_index()
        return True

    def _save_cache(self) -> None:
        cache_path = self._cache_path()
        if not cache_path:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "root_dir": str(self.root_dir),
                "chunk_max_chars": self.chunk_max_chars,
                "snippet_chars": self.snippet_chars,
                "files": [str(path) for path in self.files],
                "total_chars": self.total_chars,
                "chunks": [
                    {
                        "path": chunk.path,
                        "heading": chunk.heading,
                        "text": chunk.text,
                        "start_line": chunk.start_line,
                    }
                    for chunk in self.chunks
                ],
            }
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=True),
                encoding="utf-8",
            )
        except OSError:
            return
