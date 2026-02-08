from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

BM25_CACHE_VERSION = 2
TANTIVY_META_VERSION = 2
DEFAULT_CHUNK_MAX_WORDS = 400
DEFAULT_CHUNK_OVERLAP_WORDS = 100
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _default_backend() -> str:
    return os.environ.get("LOCAL_AGENT_INDEX_BACKEND", "tantivy").lower()


@dataclass(frozen=True)
class DocChunk:
    path: str
    heading: str
    text: str
    start_line: int


@dataclass(frozen=True)
class _TokenSpan:
    start: int
    end: int
    line: int


def find_markdown_files(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    return sorted([path for path in root_dir.rglob("*.md") if path.is_file()])


def _tokenize_query(query: str) -> list[str]:
    cleaned = [token.strip() for token in _QUERY_TOKEN_PATTERN.findall(query)]
    tokens = [token for token in cleaned if token]
    if tokens:
        return tokens
    stripped = query.strip()
    return [stripped] if stripped else []


def _split_markdown(text: str) -> Iterable[tuple[str, str, int]]:
    lines = text.splitlines()
    current_heading = "(no heading)"
    current_lines: list[str] = []
    start_line = 1

    def flush() -> Iterable[tuple[str, str, int]]:
        if not current_lines:
            return
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


def _make_snippet(text: str, terms: list[str], max_len: int) -> str:
    normalized = text.replace("\n", " ")
    if len(normalized) <= max_len:
        return normalized

    lower_text = normalized.lower()
    best_idx: int | None = None
    for term in terms:
        if not term:
            continue
        idx = lower_text.find(term.lower())
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx

    if best_idx is None:
        snippet = normalized[:max_len]
        return snippet + "..." if len(normalized) > max_len else snippet

    half = max_len // 2
    start = max(0, best_idx - half)
    end = min(len(normalized), start + max_len)
    snippet = normalized[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(normalized):
        snippet = snippet + "..."
    return snippet


def _escape_query_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _normalize_chunking(
    chunk_max_words: int,
    chunk_overlap_words: int,
) -> tuple[int, int]:
    if chunk_max_words <= 0:
        logger.error(
            "Invalid chunk_max_words=%s; falling back to %s",
            chunk_max_words,
            DEFAULT_CHUNK_MAX_WORDS,
        )
        return DEFAULT_CHUNK_MAX_WORDS, DEFAULT_CHUNK_OVERLAP_WORDS

    if chunk_overlap_words < 0 or chunk_overlap_words >= chunk_max_words:
        logger.error(
            "Invalid chunk overlap config max=%s overlap=%s; falling back to %s/%s",
            chunk_max_words,
            chunk_overlap_words,
            DEFAULT_CHUNK_MAX_WORDS,
            DEFAULT_CHUNK_OVERLAP_WORDS,
        )
        return DEFAULT_CHUNK_MAX_WORDS, DEFAULT_CHUNK_OVERLAP_WORDS

    return chunk_max_words, chunk_overlap_words


def _token_spans(text: str, base_line: int) -> list[_TokenSpan]:
    newline_offsets = [idx for idx, char in enumerate(text) if char == "\n"]
    spans: list[_TokenSpan] = []
    for match in _WORD_PATTERN.finditer(text):
        line = base_line + bisect_right(newline_offsets, match.start())
        spans.append(
            _TokenSpan(
                start=match.start(),
                end=match.end(),
                line=line,
            )
        )
    return spans


class LocalIndex:
    def __init__(
        self,
        root_dir: Path,
        chunk_max_words: int = DEFAULT_CHUNK_MAX_WORDS,
        chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
        snippet_chars: int = 400,
        index_dir: Path | None = None,
        backend: str | None = None,
    ):
        self.root_dir = root_dir.resolve()
        (
            self.chunk_max_words,
            self.chunk_overlap_words,
        ) = _normalize_chunking(chunk_max_words, chunk_overlap_words)

        if snippet_chars <= 0:
            logger.error("Invalid snippet_chars=%s; falling back to 400", snippet_chars)
            self.snippet_chars = 400
        else:
            self.snippet_chars = snippet_chars

        self.index_dir = index_dir
        self.backend = (backend or _default_backend()).lower()

        self.chunks: list[DocChunk] = []
        self.files: list[Path] = []
        self.total_chars = 0
        self.chunk_count = 0

        self._bm25 = None
        self._bm25_tokens: list[list[str]] = []

        self._tantivy = None
        self._tantivy_index = None
        self._tantivy_fields: dict[str, object] = {}

    def build(self, force_rebuild: bool = False) -> None:
        if self._should_use_tantivy() and self._build_tantivy(force_rebuild):
            return
        self._build_in_memory(force_rebuild)

    def _build_in_memory(self, force_rebuild: bool = False) -> None:
        cache_path = self._cache_path()
        if not force_rebuild and self._load_cache():
            if cache_path:
                logger.info("Using cached index: %s", cache_path)
            return

        logger.info("Building in-memory BM25 index for %s", self.root_dir)
        self.files = find_markdown_files(self.root_dir)
        chunks: list[DocChunk] = []
        total_chars = 0

        for path in self.files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            total_chars += len(text)
            chunks.extend(self._build_chunks(path=str(path), text=text))

        self.chunks = chunks
        self.total_chars = total_chars
        self.chunk_count = len(self.chunks)
        self._build_bm25_index()
        self._save_cache()
        logger.info(
            "Index build complete: %d files, %d chunks",
            len(self.files),
            self.chunk_count,
        )

    def _build_chunks(self, path: str, text: str) -> list[DocChunk]:
        chunks: list[DocChunk] = []
        for heading, section_text, section_line in _split_markdown(text):
            for chunk_text, start_line in self._split_section_by_words(
                text=section_text,
                section_start_line=section_line,
            ):
                chunks.append(
                    DocChunk(
                        path=path,
                        heading=heading,
                        text=chunk_text,
                        start_line=start_line,
                    )
                )
        return chunks

    def _split_section_by_words(
        self,
        text: str,
        section_start_line: int,
    ) -> list[tuple[str, int]]:
        spans = _token_spans(text=text, base_line=section_start_line)
        if not spans:
            stripped = text.strip()
            if not stripped:
                return []
            return [(stripped, section_start_line)]

        stride = self.chunk_max_words - self.chunk_overlap_words
        results: list[tuple[str, int]] = []
        start_idx = 0
        while start_idx < len(spans):
            end_idx = min(start_idx + self.chunk_max_words, len(spans))
            chunk_start = spans[start_idx].start
            chunk_end = spans[end_idx - 1].end
            chunk_text = text[chunk_start:chunk_end].strip()
            if chunk_text:
                results.append((chunk_text, spans[start_idx].line))
            if end_idx >= len(spans):
                break
            start_idx += stride

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: int = 1,
    ) -> list[dict]:
        if self._tantivy_index is not None:
            return self._tantivy_search(query=query, top_k=top_k)
        if self._bm25 is not None:
            return self._bm25_search(query=query, top_k=top_k)

        terms = _tokenize_query(query)
        if not terms:
            return []

        scored: list[tuple[int, DocChunk]] = []
        for chunk in self.chunks:
            haystack = chunk.text
            score = 0
            lower_haystack = haystack.lower()
            for term in terms:
                if not term:
                    continue
                score += haystack.count(term) + lower_haystack.count(term.lower())
            if score >= min_score:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[dict] = []
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

    def _tokenize_text(self, text: str) -> list[str]:
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

    def _bm25_search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._bm25 is None:
            return []

        tokens = self._tokenize_text(query)
        scores = self._bm25.get_scores(tokens)
        if not len(scores):
            return []

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda item: item[1], reverse=True)

        results: list[dict] = []
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
            "chunk_count": self.chunk_count,
            "total_chars": self.total_chars,
        }

    def search_exact(
        self,
        pattern: str,
        max_results: int = 20,
        case_sensitive: bool = False,
    ) -> list[dict]:
        if self._tantivy_index is not None:
            return self._tantivy_search_exact(
                pattern=pattern,
                max_results=max_results,
                case_sensitive=case_sensitive,
            )

        return self._search_exact_in_memory(
            pattern=pattern,
            max_results=max_results,
            case_sensitive=case_sensitive,
        )

    def _search_exact_in_memory(
        self,
        pattern: str,
        max_results: int,
        case_sensitive: bool,
    ) -> list[dict]:
        if not pattern:
            return []

        results: list[dict] = []
        pattern_match = pattern if case_sensitive else pattern.lower()

        for chunk in self.chunks:
            text_match = chunk.text if case_sensitive else chunk.text.lower()
            if pattern_match not in text_match:
                continue

            pos = text_match.find(pattern_match)
            context_start = max(0, pos - 100)
            context_end = min(len(chunk.text), pos + len(pattern) + 100)
            context = chunk.text[context_start:context_end]
            if context_start > 0:
                context = "..." + context
            if context_end < len(chunk.text):
                context = context + "..."

            results.append(
                {
                    "path": chunk.path,
                    "heading": chunk.heading,
                    "start_line": chunk.start_line,
                    "match_position": pos,
                    "context": context.replace("\n", " "),
                }
            )
            if len(results) >= max_results:
                break

        return results

    def get_chunk_content(
        self,
        path: str,
        heading: str | None = None,
        start_line: int | None = None,
    ) -> dict | None:
        if self._tantivy_index is not None:
            return self._tantivy_get_chunk_content(
                path=path,
                heading=heading,
                start_line=start_line,
            )

        return self._get_chunk_content_in_memory(
            path=path,
            heading=heading,
            start_line=start_line,
        )

    def _get_chunk_content_in_memory(
        self,
        path: str,
        heading: str | None,
        start_line: int | None,
    ) -> dict | None:
        for chunk in self.chunks:
            if chunk.path != path and not chunk.path.endswith(path):
                continue

            if start_line is not None and chunk.start_line != start_line:
                continue
            if start_line is None and heading is not None and chunk.heading != heading:
                continue

            return {
                "path": chunk.path,
                "heading": chunk.heading,
                "start_line": chunk.start_line,
                "content": chunk.text,
                "char_count": len(chunk.text),
            }

        return None

    def list_chunks_in_file(self, path: str) -> list[dict]:
        if self._tantivy_index is not None:
            return self._tantivy_list_chunks_in_file(path)
        return self._list_chunks_in_file_in_memory(path)

    def _list_chunks_in_file_in_memory(self, path: str) -> list[dict]:
        results: list[dict] = []
        for chunk in self.chunks:
            if chunk.path != path and not chunk.path.endswith(path):
                continue

            preview = chunk.text[:100].replace("\n", " ")
            if len(chunk.text) > 100:
                preview += "..."
            results.append(
                {
                    "heading": chunk.heading,
                    "start_line": chunk.start_line,
                    "char_count": len(chunk.text),
                    "preview": preview,
                }
            )

        results.sort(key=lambda item: item["start_line"])
        return results

    def _cache_path(self) -> Path | None:
        if not self.index_dir:
            return None

        key_material = "|".join(
            [
                str(self.root_dir),
                str(self.chunk_max_words),
                str(self.chunk_overlap_words),
                str(self.snippet_chars),
            ]
        )
        digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:12]
        return self.index_dir / f"index_{digest}.json"

    def _load_cache(self) -> bool:
        cache_path = self._cache_path()
        if not cache_path or not cache_path.exists():
            return False

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        if payload.get("version") != BM25_CACHE_VERSION:
            return False
        if payload.get("root_dir") != str(self.root_dir):
            return False
        if payload.get("chunk_max_words") != self.chunk_max_words:
            return False
        if payload.get("chunk_overlap_words") != self.chunk_overlap_words:
            return False
        if payload.get("snippet_chars") != self.snippet_chars:
            return False

        self.files = [Path(path) for path in payload.get("files", [])]
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
        self.chunk_count = len(self.chunks)
        self._build_bm25_index()
        return True

    def _save_cache(self) -> None:
        cache_path = self._cache_path()
        if not cache_path:
            return

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": BM25_CACHE_VERSION,
                "root_dir": str(self.root_dir),
                "chunk_max_words": self.chunk_max_words,
                "chunk_overlap_words": self.chunk_overlap_words,
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
                    logger.warning("Failed to remove old index cache %s: %s", path, exc)
        except OSError as exc:
            logger.warning("Failed to write index cache %s: %s", cache_path, exc)

    def _should_use_tantivy(self) -> bool:
        if self.backend in {"bm25", "memory", "in_memory"}:
            return False
        return self.index_dir is not None

    def _ensure_tantivy(self) -> bool:
        if self._tantivy is False:
            return False
        if self._tantivy is not None:
            return True

        try:
            import tantivy
        except ImportError:
            self._tantivy = False
            logger.warning("Tantivy not available; falling back to BM25")
            return False

        self._tantivy = tantivy
        return True

    def _tantivy_index_dir(self) -> Path | None:
        if not self.index_dir:
            return None

        key_material = "|".join(
            [
                str(self.root_dir),
                str(self.chunk_max_words),
                str(self.chunk_overlap_words),
                str(self.snippet_chars),
            ]
        )
        digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:12]
        return self.index_dir / f"tantivy_{digest}"

    def _tantivy_meta_path(self) -> Path | None:
        index_dir = self._tantivy_index_dir()
        if not index_dir:
            return None
        return index_dir / "tantivy_meta.json"

    def _build_tantivy(self, force_rebuild: bool = False) -> bool:
        if not self._ensure_tantivy():
            return False

        index_dir = self._tantivy_index_dir()
        if not index_dir:
            return False

        meta = None if force_rebuild else self._load_tantivy_meta()
        index_exists = (index_dir / "meta.json").exists()
        needs_full_rebuild = force_rebuild or meta is None or not index_exists

        if needs_full_rebuild:
            if index_dir.exists():
                shutil.rmtree(index_dir, ignore_errors=True)
            index_dir.mkdir(parents=True, exist_ok=True)
            meta = {"files": {}}

        schema, fields = self._build_tantivy_schema()
        self._tantivy_fields = fields

        index = self._open_or_create_tantivy_index(index_dir=index_dir, schema=schema)
        if index is None:
            return False
        self._tantivy_index = index

        current_files = find_markdown_files(self.root_dir)
        current_info: dict[str, dict] = {}
        for path in current_files:
            try:
                stat = path.stat()
            except OSError:
                continue
            current_info[str(path)] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }

        prev_files: dict[str, dict] = meta.get("files", {})
        changed_paths: list[str] = []
        deleted_paths: list[str] = []

        for path_str, info in current_info.items():
            prev = prev_files.get(path_str)
            if not prev:
                changed_paths.append(path_str)
                continue
            if prev.get("mtime") != info["mtime"] or prev.get("size") != info["size"]:
                changed_paths.append(path_str)

        for path_str in prev_files:
            if path_str not in current_info:
                deleted_paths.append(path_str)

        has_changes = bool(changed_paths or deleted_paths or force_rebuild)
        if has_changes:
            try:
                writer = index.writer()
            except Exception as exc:
                logger.warning("Failed to open Tantivy writer: %s", exc)
                return False

            path_field = self._tantivy_fields.get("path")
            for path_str in deleted_paths + changed_paths:
                try:
                    writer.delete_documents(path_field, path_str)
                except Exception as exc:
                    try:
                        writer.delete_documents("path", path_str)
                    except Exception:
                        logger.warning("Failed to delete documents for %s: %s", path_str, exc)

            for path_str in changed_paths:
                path = Path(path_str)
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue

                chunks = self._build_chunks(path=path_str, text=text)
                chunk_count = 0
                for chunk in chunks:
                    chunk_count += self._tantivy_add_doc(
                        writer=writer,
                        path=path_str,
                        heading=chunk.heading,
                        text=chunk.text,
                        start_line=chunk.start_line,
                    )

                current_info[path_str]["char_count"] = len(text)
                current_info[path_str]["chunk_count"] = chunk_count

            try:
                writer.commit()
            except Exception as exc:
                logger.warning("Failed to commit Tantivy index: %s", exc)
                return False

            if hasattr(index, "reload"):
                try:
                    index.reload()
                except Exception:
                    pass

        updated_meta: dict[str, dict] = {"files": {}}
        total_chars = 0
        total_chunks = 0
        for path_str, info in current_info.items():
            prev = prev_files.get(path_str, {})
            updated_meta["files"][path_str] = {
                "mtime": info.get("mtime"),
                "size": info.get("size"),
                "char_count": info.get(
                    "char_count",
                    prev.get("char_count", info.get("size", 0)),
                ),
                "chunk_count": info.get(
                    "chunk_count",
                    prev.get("chunk_count", 0),
                ),
            }
            total_chars += int(updated_meta["files"][path_str]["char_count"] or 0)
            total_chunks += int(updated_meta["files"][path_str]["chunk_count"] or 0)

        self.files = [Path(path_str) for path_str in sorted(current_info.keys())]
        self.total_chars = total_chars
        self.chunk_count = total_chunks
        self.chunks = []
        self._bm25 = None
        self._bm25_tokens = []

        self._save_tantivy_meta(meta=updated_meta)
        logger.info(
            "Tantivy index ready: %d files, %d chunks",
            len(self.files),
            self.chunk_count,
        )
        return True

    def _build_tantivy_schema(self) -> tuple[object, dict[str, object]]:
        tv = self._tantivy
        builder = tv.SchemaBuilder()
        path_field = self._add_text_field(
            builder=builder,
            name="path",
            stored=True,
            tokenizer_name="raw",
        )
        heading_field = self._add_text_field(
            builder=builder,
            name="heading",
            stored=True,
            tokenizer_name="raw",
        )
        text_field = self._add_text_field(
            builder=builder,
            name="text",
            stored=True,
        )
        start_line_field = self._add_integer_field(
            builder=builder,
            name="start_line",
            stored=True,
        )
        schema = builder.build()
        return schema, {
            "path": path_field,
            "heading": heading_field,
            "text": text_field,
            "start_line": start_line_field,
        }

    def _add_text_field(
        self,
        builder: object,
        name: str,
        stored: bool,
        tokenizer_name: str | None = None,
    ) -> object:
        if tokenizer_name is None:
            return builder.add_text_field(name, stored=stored)
        return builder.add_text_field(
            name,
            stored=stored,
            tokenizer_name=tokenizer_name,
        )

    def _add_integer_field(self, builder: object, name: str, stored: bool) -> object:
        return builder.add_integer_field(name, stored=stored)

    def _doc_add_integer(self, doc: object, field: object | str, value: int) -> None:
        doc.add_integer(field, int(value))

    def _open_or_create_tantivy_index(self, index_dir: Path, schema: object) -> object | None:
        tv = self._tantivy
        try:
            if (index_dir / "meta.json").exists():
                return tv.Index.open(str(index_dir))
            return tv.Index(schema, path=str(index_dir))
        except Exception as exc:
            logger.warning("Failed to open/create Tantivy index: %s", exc)
            return None

    def _tantivy_add_doc(
        self,
        writer: object,
        path: str,
        heading: str,
        text: str,
        start_line: int,
    ) -> int:
        tv = self._tantivy
        try:
            doc = tv.Document(
                path=path,
                heading=heading,
                text=text,
                start_line=int(start_line),
            )
        except Exception:
            doc = tv.Document()
            try:
                doc.add_text(self._tantivy_fields["path"], path)
            except Exception:
                doc.add_text("path", path)
            try:
                doc.add_text(self._tantivy_fields["heading"], heading)
            except Exception:
                doc.add_text("heading", heading)
            try:
                doc.add_text(self._tantivy_fields["text"], text)
            except Exception:
                doc.add_text("text", text)
            try:
                self._doc_add_integer(doc, self._tantivy_fields["start_line"], int(start_line))
            except Exception:
                self._doc_add_integer(doc, "start_line", int(start_line))

        try:
            writer.add_document(doc)
            return 1
        except Exception as exc:
            logger.warning("Failed to add document for %s: %s", path, exc)
            return 0

    def _load_tantivy_meta(self) -> dict | None:
        meta_path = self._tantivy_meta_path()
        if not meta_path or not meta_path.exists():
            return None

        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if payload.get("version") != TANTIVY_META_VERSION:
            return None
        if payload.get("root_dir") != str(self.root_dir):
            return None
        if payload.get("chunk_max_words") != self.chunk_max_words:
            return None
        if payload.get("chunk_overlap_words") != self.chunk_overlap_words:
            return None
        if payload.get("snippet_chars") != self.snippet_chars:
            return None

        return payload

    def _save_tantivy_meta(self, meta: dict) -> None:
        meta_path = self._tantivy_meta_path()
        if not meta_path:
            return

        payload = {
            "version": TANTIVY_META_VERSION,
            "root_dir": str(self.root_dir),
            "chunk_max_words": self.chunk_max_words,
            "chunk_overlap_words": self.chunk_overlap_words,
            "snippet_chars": self.snippet_chars,
            "files": meta.get("files", {}),
        }

        try:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(
                json.dumps(payload, ensure_ascii=True),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to write Tantivy meta %s: %s", meta_path, exc)

    def _tantivy_search(self, query: str, top_k: int) -> list[dict]:
        if not query.strip():
            return []
        if self._tantivy_index is None:
            return []

        try:
            searcher = self._tantivy_index.searcher()
            query_obj = self._tantivy_index.parse_query(
                query,
                ["text", "heading", "path"],
            )
            results = searcher.search(query_obj, top_k)
        except Exception as exc:
            logger.warning("Tantivy search failed: %s", exc)
            return []

        hits = getattr(results, "hits", results)
        terms = _tokenize_query(query)
        output: list[dict] = []
        for score, doc_address in hits:
            doc = searcher.doc(doc_address)
            path = str(self._doc_get_value(doc, "path") or "")
            heading = str(self._doc_get_value(doc, "heading") or "")
            text = str(self._doc_get_value(doc, "text") or "")
            start_line = int(self._doc_get_value(doc, "start_line") or 0)
            output.append(
                {
                    "path": path,
                    "heading": heading,
                    "start_line": start_line,
                    "score": float(score),
                    "snippet": _make_snippet(text=text, terms=terms, max_len=self.snippet_chars),
                }
            )
        return output

    def _tantivy_search_exact(
        self,
        pattern: str,
        max_results: int,
        case_sensitive: bool,
    ) -> list[dict]:
        if not pattern:
            return []
        if self._tantivy_index is None:
            return []

        tokens = _tokenize_query(pattern)
        query_text = " ".join(tokens) if tokens else pattern

        try:
            searcher = self._tantivy_index.searcher()
            query_obj = self._tantivy_index.parse_query(query_text, ["text"])
            candidate_limit = min(max_results * 20, 2000)
            results = searcher.search(query_obj, candidate_limit)
        except Exception as exc:
            logger.warning("Tantivy exact search failed: %s", exc)
            return []

        hits = getattr(results, "hits", results)
        pattern_match = pattern if case_sensitive else pattern.lower()
        output: list[dict] = []
        for _, doc_address in hits:
            doc = searcher.doc(doc_address)
            text = str(self._doc_get_value(doc, "text") or "")
            text_match = text if case_sensitive else text.lower()
            if pattern_match not in text_match:
                continue

            pos = text_match.find(pattern_match)
            context_start = max(0, pos - 100)
            context_end = min(len(text), pos + len(pattern) + 100)
            context = text[context_start:context_end]
            if context_start > 0:
                context = "..." + context
            if context_end < len(text):
                context = context + "..."

            output.append(
                {
                    "path": str(self._doc_get_value(doc, "path") or ""),
                    "heading": str(self._doc_get_value(doc, "heading") or ""),
                    "start_line": int(self._doc_get_value(doc, "start_line") or 0),
                    "match_position": pos,
                    "context": context.replace("\n", " "),
                }
            )
            if len(output) >= max_results:
                break

        return output

    def _tantivy_get_chunk_content(
        self,
        path: str,
        heading: str | None,
        start_line: int | None,
    ) -> dict | None:
        if self._tantivy_index is None:
            return None

        searcher = self._tantivy_index.searcher()
        query_text = f'"{_escape_query_text(path)}"'
        try:
            query_obj = self._tantivy_index.parse_query(query_text, ["path"])
            results = searcher.search(query_obj, 2000)
        except Exception as exc:
            logger.warning("Tantivy chunk lookup failed: %s", exc)
            return None

        hits = getattr(results, "hits", results)
        for _, doc_address in hits:
            doc = searcher.doc(doc_address)
            doc_path = str(self._doc_get_value(doc, "path") or "")
            if doc_path != path and not doc_path.endswith(path):
                continue

            doc_heading = str(self._doc_get_value(doc, "heading") or "")
            doc_start = int(self._doc_get_value(doc, "start_line") or 0)
            if start_line is not None and doc_start != start_line:
                continue
            if start_line is None and heading is not None and doc_heading != heading:
                continue

            text = str(self._doc_get_value(doc, "text") or "")
            return {
                "path": doc_path,
                "heading": doc_heading,
                "start_line": doc_start,
                "content": text,
                "char_count": len(text),
            }

        return None

    def _tantivy_list_chunks_in_file(self, path: str) -> list[dict]:
        if self._tantivy_index is None:
            return []

        searcher = self._tantivy_index.searcher()
        query_text = f'"{_escape_query_text(path)}"'
        try:
            query_obj = self._tantivy_index.parse_query(query_text, ["path"])
            results = searcher.search(query_obj, 5000)
        except Exception as exc:
            logger.warning("Tantivy list chunks failed: %s", exc)
            return []

        hits = getattr(results, "hits", results)
        output: list[dict] = []
        for _, doc_address in hits:
            doc = searcher.doc(doc_address)
            doc_path = str(self._doc_get_value(doc, "path") or "")
            if doc_path != path and not doc_path.endswith(path):
                continue

            text = str(self._doc_get_value(doc, "text") or "")
            preview = text[:100].replace("\n", " ")
            if len(text) > 100:
                preview += "..."

            output.append(
                {
                    "heading": str(self._doc_get_value(doc, "heading") or ""),
                    "start_line": int(self._doc_get_value(doc, "start_line") or 0),
                    "char_count": len(text),
                    "preview": preview,
                }
            )

        output.sort(key=lambda item: item.get("start_line", 0))
        return output

    def _doc_get_value(self, doc: object, field: str) -> object | None:
        if doc is None:
            return None

        if isinstance(doc, dict):
            value = doc.get(field)
        elif hasattr(doc, "to_dict"):
            value = doc.to_dict().get(field)
        elif hasattr(doc, "get"):
            try:
                value = doc.get(field)
            except Exception:
                value = None
        else:
            try:
                value = doc[field]
            except Exception:
                value = None

        if isinstance(value, list):
            return value[0] if value else None
        return value
