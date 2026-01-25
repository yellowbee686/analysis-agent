from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


logger = logging.getLogger(__name__)


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
        cache_path = self._cache_path()
        if not force_rebuild and self._load_cache():
            if cache_path:
                logger.info("Using cached index: %s", cache_path)
            return
        logger.info("Building index for %s", self.root_dir)
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
        logger.info(
            "Index build complete: %d files, %d chunks",
            len(self.files),
            len(self.chunks),
        )

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

    def search_exact(
        self, pattern: str, max_results: int = 20, case_sensitive: bool = False
    ) -> List[dict]:
        """Exact substring search in indexed chunks.

        Unlike BM25 semantic search, this performs exact string matching.
        Useful for finding specific keywords, error codes, function names,
        or phrases that BM25 might not rank highly.

        Args:
            pattern: The exact string to search for.
            max_results: Maximum number of results to return.
            case_sensitive: Whether to perform case-sensitive matching.

        Returns:
            List of matching chunks with path, heading, start_line, and snippet.
        """
        if not pattern:
            return []

        results = []
        pattern_match = pattern if case_sensitive else pattern.lower()

        for chunk in self.chunks:
            text_match = chunk.text if case_sensitive else chunk.text.lower()
            if pattern_match in text_match:
                # Find the position of the match for context
                pos = text_match.find(pattern_match)
                # Extract context around the match
                context_start = max(0, pos - 100)
                context_end = min(len(chunk.text), pos + len(pattern) + 100)
                context = chunk.text[context_start:context_end]
                if context_start > 0:
                    context = "..." + context
                if context_end < len(chunk.text):
                    context = context + "..."

                results.append({
                    "path": chunk.path,
                    "heading": chunk.heading,
                    "start_line": chunk.start_line,
                    "match_position": pos,
                    "context": context.replace("\n", " "),
                })

                if len(results) >= max_results:
                    break

        return results

    def get_chunk_content(
        self, path: str, heading: str | None = None, start_line: int | None = None
    ) -> dict | None:
        """Get full content of a specific chunk.

        Can match by path + heading, or by path + start_line for more precision.
        Useful after retrieve_local_docs returns snippets and user wants
        to read the full chunk content.

        Args:
            path: The file path of the chunk.
            heading: The heading of the chunk (optional if start_line provided).
            start_line: The start line of the chunk (optional, for precise matching).

        Returns:
            Dict with full chunk content, or None if not found.
        """
        for chunk in self.chunks:
            # Match by path first
            if chunk.path != path and not chunk.path.endswith(path):
                continue

            # If start_line is provided, use it for precise matching
            if start_line is not None:
                if chunk.start_line == start_line:
                    return {
                        "path": chunk.path,
                        "heading": chunk.heading,
                        "start_line": chunk.start_line,
                        "content": chunk.text,
                        "char_count": len(chunk.text),
                    }
            # Otherwise match by heading
            elif heading is not None:
                if chunk.heading == heading:
                    return {
                        "path": chunk.path,
                        "heading": chunk.heading,
                        "start_line": chunk.start_line,
                        "content": chunk.text,
                        "char_count": len(chunk.text),
                    }

        return None

    def list_chunks_in_file(self, path: str) -> List[dict]:
        """List all chunks in a specific file.

        Useful for browsing the structure of a document after finding it
        through search.

        Args:
            path: The file path to list chunks for.

        Returns:
            List of chunk metadata (heading, start_line, char_count).
        """
        results = []
        for chunk in self.chunks:
            if chunk.path == path or chunk.path.endswith(path):
                results.append({
                    "heading": chunk.heading,
                    "start_line": chunk.start_line,
                    "char_count": len(chunk.text),
                    "preview": chunk.text[:100].replace("\n", " ") + "..."
                    if len(chunk.text) > 100 else chunk.text.replace("\n", " "),
                })
        return results

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
            logger.info("Saved index cache: %s", cache_path)
            for path in cache_path.parent.glob("index_*.json"):
                if path == cache_path:
                    continue
                try:
                    path.unlink()
                    logger.info("Removed old index cache: %s", path)
                except OSError as exc:
                    logger.warning(
                        "Failed to remove old index cache %s: %s",
                        path,
                        exc,
                    )
        except OSError as exc:
            logger.warning(
                "Failed to write index cache %s: %s",
                cache_path,
                exc,
            )
            return
