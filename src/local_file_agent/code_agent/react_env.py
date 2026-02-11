"""Execution environment interface for React Code Agent.

`LocalEnv` is injected into LLM context. Keep this file as an interface-focused
API surface with concise, accurate usage guidance. Runtime implementations live
in `env_tools.py` and `local_file_agent.tools`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_file_agent.code_agent.env_tools import (
    CbetaMcpTools,
    FilePatternSearchTools,
    LocalDocSearchTools,
    SemanticScholarTools,
)
from local_file_agent.indexer import LocalIndex
from local_file_agent.tools import ChunkedFileTools, LocalDocTools

if TYPE_CHECKING:
    from camel.toolkits.mcp_toolkit import MCPToolkit


DEFAULT_RETRIEVE_TIMEOUT_S = float(
    os.environ.get("LOCAL_AGENT_RETRIEVE_TIMEOUT", "60")
)


class LocalEnv:
    """Public environment API for LLM-generated code."""

    def __init__(
        self,
        index: LocalIndex,
        working_directory: str | Path | None = None,
        mcp_toolkit: MCPToolkit | None = None,
    ) -> None:
        """Initialize LocalEnv.

        Args:
            index: Local index used by document retrieval tools.
            working_directory: Base directory for file operations.
            mcp_toolkit: Optional connected MCP toolkit.

        Returns:
            None.

        Example:
            env = LocalEnv(index, working_directory=".", mcp_toolkit=toolkit)
        """
        self._doc_tools = LocalDocTools(index)
        self._chunked_tools = ChunkedFileTools(working_directory)
        self._doc_search_tools = LocalDocSearchTools(
            doc_tools=self._doc_tools,
            default_timeout_s=DEFAULT_RETRIEVE_TIMEOUT_S,
        )
        self._file_pattern_tools = FilePatternSearchTools(working_directory)
        self._scholar_tools = SemanticScholarTools()
        self._cbeta_tools = CbetaMcpTools(mcp_toolkit)

    # --- Document Retrieval ---

    def retrieve_docs(
        self,
        query: str,
        top_k: int = 5,
        min_score: int = 1,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        """Retrieve relevant chunks from local documents.

        Args:
            query: Search query text.
            top_k: Maximum number of chunks.
            min_score: Minimum BM25 score.
            timeout_s: Optional timeout seconds (None uses env default).

        Returns:
            dict with retrieval result or timeout/error info.

        Example:
            env.retrieve_docs("MCP timeout fix", top_k=3)
        """
        return self._doc_search_tools.retrieve_docs(
            query=query,
            top_k=top_k,
            min_score=min_score,
            timeout_s=timeout_s,
        )

    def search_exact(
        self,
        pattern: str,
        max_results: int = 20,
        case_sensitive: bool = False,
    ) -> dict[str, object]:
        """Run exact substring search on indexed chunks.

        Args:
            pattern: Exact string to match.
            max_results: Maximum number of matches.
            case_sensitive: Whether matching is case-sensitive.

        Returns:
            dict with exact-match results.

        Example:
            env.search_exact("LOCAL_AGENT_MCP_ENABLED")
        """
        return self._doc_tools.search_exact(pattern, max_results, case_sensitive)

    def get_chunk_content(
        self,
        path: str,
        heading: str | None = None,
        start_line: int | None = None,
    ) -> dict[str, object]:
        """Get full content for one indexed chunk.

        Args:
            path: File path (full or partial).
            heading: Optional chunk heading filter.
            start_line: Optional chunk start line filter.

        Returns:
            dict with chunk content or error.

        Example:
            env.get_chunk_content("AGENTS.md", heading="MCP")
        """
        return self._doc_tools.get_chunk_content(path, heading, start_line)

    def list_chunks_in_file(self, path: str) -> dict[str, object]:
        """List chunk metadata for one file.

        Args:
            path: Target file path.

        Returns:
            dict with chunk list.

        Example:
            env.list_chunks_in_file("src/local_file_agent/code_agent/react_env.py")
        """
        return self._doc_tools.list_chunks_in_file(path)

    def corpus_stats(self) -> dict[str, object]:
        """Return corpus statistics for the local index.

        Returns:
            dict with file_count, chunk_count, and total_chars.

        Example:
            env.corpus_stats()
        """
        return self._doc_tools.corpus_stats()

    def list_docs(self, limit: int = 20) -> list[str]:
        """List indexed document paths.

        Args:
            limit: Maximum number of paths.

        Returns:
            list of file paths.

        Example:
            env.list_docs(limit=10)
        """
        return self._doc_tools.list_docs(limit)

    # --- File Operations ---

    def read_file_chunk(
        self,
        file_path: str,
        offset: int = 0,
        length: int = 8000,
    ) -> str:
        """Read part of a file by character range.

        Args:
            file_path: Path to file.
            offset: Start offset (0-based characters).
            length: Max characters to read.

        Returns:
            JSON string with chunk and navigation info.

        Example:
            env.read_file_chunk("cache/sessions/demo/mcp_responses/a.json", 0, 2000)
        """
        return self._chunked_tools.read_file_chunk(file_path, offset, length)

    def search_in_file(
        self,
        file_path: str,
        pattern: str,
        context_chars: int = 100,
        max_results: int = 20,
    ) -> str:
        """Search text pattern within one file.

        Args:
            file_path: Path to file.
            pattern: Pattern to find.
            context_chars: Context around each match.
            max_results: Maximum matches.

        Returns:
            JSON string with matches and contexts.

        Example:
            env.search_in_file("cache/sessions/demo/mcp_responses/a.json", "法鼓")
        """
        return self._chunked_tools.search_in_file(
            file_path,
            pattern,
            context_chars,
            max_results,
        )

    def get_file_info(self, file_path: str) -> str:
        """Get metadata for one file.

        Args:
            file_path: Path to file.

        Returns:
            JSON string with file metadata.

        Example:
            env.get_file_info("cache/sessions/demo/mcp_responses/a.json")
        """
        return self._chunked_tools.get_file_info(file_path)

    def search_files_pattern(
        self,
        pattern: str,
        file_types: list[str] | None = None,
        file_pattern: str | None = None,
        path: str | None = None,
    ) -> str:
        """Search pattern across files in a directory.

        Args:
            pattern: Case-insensitive text pattern.
            file_types: Optional extensions, e.g. ["md", "json"].
            file_pattern: Optional glob pattern, overrides file_types.
            path: Optional root directory.

        Returns:
            FileToolkit JSON string result.

        Example:
            env.search_files_pattern("MCP", file_types=["md"], path="src")
        """
        return self._file_pattern_tools.search_files_pattern(
            pattern=pattern,
            file_types=file_types,
            file_pattern=file_pattern,
            path=path,
        )

    # --- Academic Search ---

    def search_papers(
        self,
        query: str,
        limit: int = 5,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search papers with Semantic Scholar.

        Args:
            query: Search query.
            limit: Maximum number of returned papers.
            fields: Optional field list.

        Returns:
            Semantic Scholar response dict.

        Example:
            env.search_papers("retrieval augmented generation", limit=3)
        """
        return self._scholar_tools.search_papers(
            query=query,
            limit=limit,
            fields=fields,
        )

    def get_paper_details(
        self,
        paper_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get one paper by Semantic Scholar paper ID.

        Args:
            paper_id: Semantic Scholar paper ID.
            fields: Optional field list.

        Returns:
            paper detail dict.

        Example:
            env.get_paper_details("CorpusID:208324896")
        """
        return self._scholar_tools.get_paper_details(
            paper_id=paper_id,
            fields=fields,
        )

    def get_author_info(
        self,
        author_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get one author by Semantic Scholar author ID.

        Args:
            author_id: Semantic Scholar author ID.
            fields: Optional field list.

        Returns:
            author detail dict.

        Example:
            env.get_author_info("1741102")
        """
        return self._scholar_tools.get_author_info(
            author_id=author_id,
            fields=fields,
        )

    # --- CBETA MCP Tools ---

    def cbeta_search(
        self,
        query: str,
        rows: int = 20,
        start: int = 0,
        order: str | None = None,
    ) -> dict[str, Any]:
        """CBETA full-text search.

        Args:
            query: Query text (Traditional Chinese preferred).
            rows: Page size.
            start: Pagination offset.
            order: Optional sort expression, e.g. "time_from-".

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_search("四聖諦", rows=5)
        """
        params = {"q": query, "rows": rows, "start": start}
        if order:
            params["order"] = order
        return self._cbeta_tools.run("cbeta_fulltext_search", **params)

    def cbeta_search_all_in_one(
        self,
        query: str,
        rows: int = 20,
        start: int = 0,
        around: int = 10,
        facet: int = 0,
    ) -> dict[str, Any]:
        """CBETA all-in-one search with KWIC snippets.

        Args:
            query: Query text.
            rows: Page size.
            start: Pagination offset.
            around: KWIC context length.
            facet: Include facet summary (0/1).

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_search_all_in_one("法鼓", around=20, facet=1)
        """
        return self._cbeta_tools.run(
            "cbeta_all_in_one",
            q=query,
            rows=rows,
            start=start,
            around=around,
            facet=facet,
        )

    def cbeta_extended_search(
        self,
        query: str,
        rows: int = 20,
        start: int = 0,
    ) -> dict[str, Any]:
        """CBETA advanced full-text search with operators.

        Args:
            query: Query with AND/OR/NOT/NEAR syntax.
            rows: Page size.
            start: Pagination offset.

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_extended_search('"般若" NEAR/5 "波羅蜜"')
        """
        return self._cbeta_tools.run(
            "extended_search",
            q=query,
            rows=rows,
            start=start,
        )

    def cbeta_search_title(
        self,
        query: str,
        rows: int = 20,
        start: int = 0,
    ) -> dict[str, Any]:
        """Search CBETA scripture titles.

        Args:
            query: Title keyword (minimum 3 Chinese characters recommended).
            rows: Page size.
            start: Pagination offset.

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_search_title("觀無量壽經")
        """
        return self._cbeta_tools.run(
            "search_title",
            q=query,
            rows=rows,
            start=start,
        )

    def cbeta_kwic_search(
        self,
        work: str,
        juan: int,
        query: str,
        note: int = 1,
        mark: int = 1,
    ) -> dict[str, Any]:
        """Run KWIC search in one CBETA fascicle.

        Args:
            work: Work ID, e.g. "T0001".
            juan: Fascicle number.
            query: KWIC query text.
            note: Include annotation flag (0/1).
            mark: Highlight keyword flag (0/1).

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_kwic_search("T0001", 1, "老子")
        """
        return self._cbeta_tools.run(
            "cbeta_kwic_search",
            work=work,
            juan=juan,
            q=query,
            note=note,
            mark=mark,
        )

    def cbeta_search_catalog(self, query: str) -> dict[str, Any]:
        """Search CBETA text catalog by keyword or volume token.

        Args:
            query: Keyword or volume token, e.g. "阿含" or "T01".

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_search_catalog("阿含")
        """
        return self._cbeta_tools.run("search_cbeta_texts", q=query)

    def cbeta_search_by_translator(
        self,
        creator: str | None = None,
        creator_id: str | None = None,
    ) -> dict[str, Any]:
        """Search CBETA works by translator/author.

        Args:
            creator: Translator name fuzzy match, e.g. "玄奘".
            creator_id: Exact translator ID, e.g. "A000439".

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_search_by_translator(creator="玄奘")
        """
        params: dict[str, Any] = {}
        if creator_id:
            params["creator_id"] = creator_id
        elif creator:
            params["creator"] = creator
        else:
            return {
                "status": "error",
                "message": "Provide either creator or creator_id",
            }
        return self._cbeta_tools.run("search_works_by_translator", **params)

    def cbeta_search_by_dynasty(
        self,
        dynasty: str | None = None,
        time_start: int | None = None,
        time_end: int | None = None,
    ) -> dict[str, Any]:
        """Search CBETA works by dynasty or year range.

        Args:
            dynasty: Dynasty string, e.g. "唐" or "唐,宋".
            time_start: Start year CE.
            time_end: End year CE.

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_search_by_dynasty(dynasty="唐")
        """
        params: dict[str, Any] = {}
        if dynasty:
            params["dynasty"] = dynasty
        if time_start is not None:
            params["time_start"] = time_start
        if time_end is not None:
            params["time_end"] = time_end
        if not params:
            return {
                "status": "error",
                "message": "Provide dynasty or time_start/time_end",
            }
        return self._cbeta_tools.run("search_cbeta_by_dynasty", **params)

    def cbeta_get_work_info(self, work: str) -> dict[str, Any]:
        """Get CBETA metadata for one work.

        Args:
            work: Work ID, e.g. "T0001".

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_get_work_info("T0001")
        """
        return self._cbeta_tools.run("get_cbeta_work_info", work=work)

    def cbeta_get_toc(self, work: str) -> dict[str, Any]:
        """Get CBETA table of contents for one work.

        Args:
            work: Work ID, e.g. "T0001".

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_get_toc("T0001")
        """
        return self._cbeta_tools.run("get_cbeta_toc", work=work)

    def cbeta_get_juan_html(
        self,
        work: str,
        juan: int,
        work_info: int = 0,
        toc: int = 0,
    ) -> dict[str, Any]:
        """Get HTML body for one CBETA fascicle.

        Args:
            work: Work ID.
            juan: Fascicle number.
            work_info: Include work metadata (0/1).
            toc: Include table of contents (0/1).

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_get_juan_html("T0001", 1)
        """
        return self._cbeta_tools.run(
            "get_juan_html",
            work=work,
            juan=juan,
            work_info=work_info,
            toc=toc,
        )

    def cbeta_get_lines(
        self,
        linehead: str | None = None,
        linehead_start: str | None = None,
        linehead_end: str | None = None,
        before: int | None = None,
        after: int | None = None,
    ) -> dict[str, Any]:
        """Get CBETA text by one linehead or line range.

        Args:
            linehead: Single linehead, e.g. "T01n0001_p0001a04".
            linehead_start: Start linehead for range mode.
            linehead_end: End linehead for range mode.
            before: Optional context lines before linehead.
            after: Optional context lines after linehead.

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_get_lines(linehead="T01n0001_p0001a04", before=2, after=3)
        """
        params: dict[str, Any] = {}
        if linehead:
            params["linehead"] = linehead
        if linehead_start:
            params["linehead_start"] = linehead_start
        if linehead_end:
            params["linehead_end"] = linehead_end
        if before is not None:
            params["before"] = before
        if after is not None:
            params["after"] = after
        if not params:
            return {
                "status": "error",
                "message": "Provide linehead or linehead_start/linehead_end",
            }
        return self._cbeta_tools.run("get_cbeta_lines", **params)

    def cbeta_goto(
        self,
        linehead: str | None = None,
        canon: str | None = None,
        work: str | None = None,
        juan: int | None = None,
        vol: int | None = None,
        page: int | None = None,
        col: str | None = None,
        line: int | None = None,
    ) -> dict[str, Any]:
        """Build CBETA jump URL by linehead or structured coordinates.

        Args:
            linehead: Direct linehead, highest priority if provided.
            canon: Canon code, e.g. "T".
            work: Work number within canon, e.g. "1".
            juan: Fascicle number.
            vol: Volume number in canon.
            page: Page number.
            col: Column letter ("a"/"b"/"c").
            line: Line number.

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_goto(linehead="T01n0001_p0066c25")
        """
        params: dict[str, Any] = {}
        if linehead:
            params["linehead"] = linehead
        else:
            if canon:
                params["canon"] = canon
            if work:
                params["work"] = work
            if juan is not None:
                params["juan"] = juan
            if vol is not None:
                params["vol"] = vol
            if page is not None:
                params["page"] = page
            if col:
                params["col"] = col
            if line is not None:
                params["line"] = line

        if not params:
            return {
                "status": "error",
                "message": "Provide linehead or navigation parameters",
            }
        return self._cbeta_tools.run("cbeta_goto", **params)

    def cbeta_search_sc(
        self,
        query: str,
        fields: str | None = None,
        rows: int = 10,
        start: int = 0,
        order: str | None = None,
    ) -> dict[str, Any]:
        """Search CBETA with simplified/traditional auto-conversion.

        Args:
            query: Simplified or traditional Chinese query.
            fields: Optional fields filter string.
            rows: Page size.
            start: Pagination offset.
            order: Optional sort expression.

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_search_sc("四圣谛", rows=5)
        """
        params = {"q": query, "rows": rows, "start": start}
        if fields:
            params["fields"] = fields
        if order:
            params["order"] = order
        return self._cbeta_tools.run("cbeta_search_sc", **params)

    def cbeta_search_notes(
        self,
        query: str,
        around: int = 10,
        rows: int = 20,
        start: int = 0,
        facet: int = 0,
    ) -> dict[str, Any]:
        """Search CBETA notes/annotations.

        Args:
            query: Notes query text, supports boolean syntax.
            around: Highlight context length.
            rows: Page size.
            start: Pagination offset.
            facet: Include facet summary (0/1).

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_search_notes('"法鼓"', facet=1)
        """
        return self._cbeta_tools.run(
            "search_cbeta_notes",
            q=query,
            around=around,
            rows=rows,
            start=start,
            facet=facet,
        )

    def cbeta_facet_query(
        self,
        query: str,
        facet_type: str = "canon",
    ) -> dict[str, Any]:
        """Get CBETA facet aggregation by one dimension.

        Args:
            query: Query text for aggregation.
            facet_type: One of canon/category/dynasty/creator/work.

        Returns:
            MCP response dict with shape `{"status": "...", "result": ...}`.

        Example:
            env.cbeta_facet_query("法鼓", facet_type="dynasty")
        """
        return self._cbeta_tools.run("cbeta_facet_query", q=query, f=facet_type)


def run_env(env: LocalEnv, query: str) -> tuple[str, str]:
    """Example run function used by the React code loop.

    Args:
        env: LocalEnv instance.
        query: Current query string.

    Returns:
        tuple of (next_query, info).

    Example:
        return (query, str(env.corpus_stats()))
    """
    return query, ""
