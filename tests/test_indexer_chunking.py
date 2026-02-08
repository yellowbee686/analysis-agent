from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from local_file_agent.indexer import LocalIndex
from local_file_agent.tools import LocalDocTools

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def test_english_chunking_with_overlap(tmp_path: Path) -> None:
    doc_path = tmp_path / "docs" / "english.md"
    text = " ".join(f"w{i}" for i in range(950))
    _write_markdown(doc_path, text)

    index = LocalIndex(
        root_dir=tmp_path,
        chunk_max_words=400,
        chunk_overlap_words=100,
        backend="bm25",
    )
    index.build()

    assert index.chunk_count == 3
    chunk_tokens = [_tokens(chunk.text) for chunk in index.chunks]
    assert [len(tokens) for tokens in chunk_tokens] == [400, 400, 350]
    assert chunk_tokens[0][-100:] == chunk_tokens[1][:100]
    assert chunk_tokens[1][-100:] == chunk_tokens[2][:100]


def test_chinese_chunking_treats_each_char_as_word(tmp_path: Path) -> None:
    doc_path = tmp_path / "docs" / "chinese.md"
    text = "你好世界" * 250
    _write_markdown(doc_path, text)

    index = LocalIndex(
        root_dir=tmp_path,
        chunk_max_words=400,
        chunk_overlap_words=100,
        backend="bm25",
    )
    index.build()

    assert index.chunk_count == 3
    chunk_tokens = [_tokens(chunk.text) for chunk in index.chunks]
    assert [len(tokens) for tokens in chunk_tokens] == [400, 400, 400]
    assert chunk_tokens[0][-100:] == chunk_tokens[1][:100]
    assert chunk_tokens[1][-100:] == chunk_tokens[2][:100]


def test_mixed_chunking_has_stable_window_stride(tmp_path: Path) -> None:
    doc_path = tmp_path / "docs" / "mixed.md"
    segments: list[str] = []
    for idx in range(520):
        segments.append(f"w{idx}")
        if idx % 10 == 0:
            segments.append("汉")
        if idx % 15 == 0:
            segments.append("字")
    text = " ".join(segments)
    _write_markdown(doc_path, text)

    index = LocalIndex(
        root_dir=tmp_path,
        chunk_max_words=120,
        chunk_overlap_words=20,
        backend="bm25",
    )
    index.build()

    full_tokens = _tokens(text)
    stride = 100
    start_idx = 0
    for chunk in index.chunks:
        chunk_tokens = _tokens(chunk.text)
        expected_tokens = full_tokens[start_idx : start_idx + len(chunk_tokens)]
        assert chunk_tokens == expected_tokens
        if start_idx + len(chunk_tokens) >= len(full_tokens):
            break
        start_idx += stride


def test_invalid_overlap_falls_back_to_default(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR)

    index = LocalIndex(
        root_dir=tmp_path,
        chunk_max_words=100,
        chunk_overlap_words=100,
        backend="bm25",
    )

    assert index.chunk_max_words == 400
    assert index.chunk_overlap_words == 100
    assert any("Invalid chunk overlap config" in rec.message for rec in caplog.records)


def test_bm25_and_tantivy_chunk_counts_match(tmp_path: Path) -> None:
    pytest.importorskip("tantivy")

    doc_path = tmp_path / "docs" / "shared.md"
    text = "\n".join(
        [
            "# Header",
            " ".join(f"w{i}" for i in range(730)),
        ]
    )
    _write_markdown(doc_path, text)

    bm25_index = LocalIndex(
        root_dir=tmp_path,
        chunk_max_words=400,
        chunk_overlap_words=100,
        backend="bm25",
    )
    bm25_index.build()

    tantivy_index = LocalIndex(
        root_dir=tmp_path,
        chunk_max_words=400,
        chunk_overlap_words=100,
        index_dir=tmp_path / "indices",
        backend="tantivy",
    )
    tantivy_index.build(force_rebuild=True)

    assert tantivy_index.chunk_count == bm25_index.chunk_count

    bm_chunks = bm25_index.list_chunks_in_file(str(doc_path))
    ta_chunks = tantivy_index.list_chunks_in_file(str(doc_path))
    assert len(ta_chunks) == len(bm_chunks)

    first_chunk = ta_chunks[0]
    content = tantivy_index.get_chunk_content(
        path=str(doc_path),
        start_line=int(first_chunk["start_line"]),
    )
    assert content is not None
    assert content["content"]


def test_local_doc_tools_schema_unchanged(tmp_path: Path) -> None:
    doc_path = tmp_path / "docs" / "schema.md"
    _write_markdown(doc_path, "hello world")

    index = LocalIndex(root_dir=tmp_path, backend="bm25")
    index.build()
    tools = LocalDocTools(index)

    retrieve = tools.retrieve_local_docs("hello", top_k=3, min_score=1)
    assert set(retrieve.keys()) == {"query", "matches", "total_chunks"}

    exact = tools.search_exact("world", max_results=5, case_sensitive=False)
    assert set(exact.keys()) == {"pattern", "case_sensitive", "matches", "total_matches"}

    listed = tools.list_chunks_in_file(str(doc_path))
    assert set(listed.keys()) == {"path", "chunks", "total_chunks"}
