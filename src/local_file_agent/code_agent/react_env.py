"""
Execution environment for React Code Agent.

This module provides the LocalEnv class that wraps local doc tools
for LLM-generated code to interact with during the agent loop.

Enhanced with:
- MCP tools (CBETA Buddhist scriptures)
- Academic search (Semantic Scholar)
- File pattern search (FileToolkit)
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_file_agent.indexer import LocalIndex
from local_file_agent.tools import LocalDocTools, ChunkedFileTools

if TYPE_CHECKING:
    from camel.toolkits.mcp_toolkit import MCPToolkit
    from camel.toolkits.semantic_scholar_toolkit import SemanticScholarToolkit
    from camel.toolkits.file_toolkit import FileToolkit

logger = logging.getLogger(__name__)


class LocalEnv:
    """Environment for React Code Agent execution.
    
    Wraps LocalDocTools and ChunkedFileTools methods as env methods
    that LLM-generated code can call.
    
    Additional capabilities:
    - MCP tools for CBETA Buddhist scripture search
    - Semantic Scholar for academic paper search
    - FileToolkit for pattern search in folders
    """
    
    def __init__(
        self,
        index: LocalIndex,
        working_directory: str | Path | None = None,
        mcp_toolkit: MCPToolkit | None = None,
    ):
        """Initialize LocalEnv.
        
        Args:
            index: LocalIndex instance for document search.
            working_directory: Working directory for file operations.
            mcp_toolkit: Optional MCPToolkit for CBETA and other MCP tools.
        """
        self._doc_tools = LocalDocTools(index)
        self._chunked_tools = ChunkedFileTools(working_directory)
        self._mcp_toolkit = mcp_toolkit
        self._working_directory = Path(working_directory) if working_directory else None
        
        # Initialize optional toolkits lazily
        self._scholar_toolkit: SemanticScholarToolkit | None = None
        self._file_toolkit: FileToolkit | None = None
    
    def _get_scholar_toolkit(self) -> SemanticScholarToolkit:
        """Get or create SemanticScholarToolkit instance."""
        if self._scholar_toolkit is None:
            try:
                from camel.toolkits.semantic_scholar_toolkit import SemanticScholarToolkit
                self._scholar_toolkit = SemanticScholarToolkit()
            except ImportError:
                logger.warning("SemanticScholarToolkit not available")
                return None
        return self._scholar_toolkit
    
    def _get_file_toolkit(self) -> FileToolkit:
        """Get or create FileToolkit instance."""
        if self._file_toolkit is None:
            try:
                from camel.toolkits.file_toolkit import FileToolkit
                working_dir = str(self._working_directory) if self._working_directory else None
                self._file_toolkit = FileToolkit(
                    working_directory=working_dir,
                    backup_enabled=False,
                )
            except ImportError:
                logger.warning("FileToolkit not available")
                return None
        return self._file_toolkit
    
    # --- Document Retrieval Methods ---
    
    def retrieve_docs(
        self,
        query: str,
        top_k: int = 5,
        min_score: int = 1,
    ) -> dict[str, object]:
        """Retrieve relevant passages from local documents using BM25.
        
        Uses BM25 semantic ranking to find the most relevant chunks.
        For exact string matching, use search_exact instead.
        
        Args:
            query: User question or keywords to search.
            top_k: Maximum number of results to return.
            min_score: Minimum match score to keep a result.
            
        Returns:
            dict with query, matches, and total_chunks.
        """
        return self._doc_tools.retrieve_local_docs(query, top_k, min_score)
    
    def search_exact(
        self,
        pattern: str,
        max_results: int = 20,
        case_sensitive: bool = False,
    ) -> dict[str, object]:
        """Exact substring search in indexed documents.
        
        Unlike retrieve_docs which uses BM25 semantic ranking, this
        performs exact string matching. Use this when you need to find:
        - Specific keywords or phrases
        - Error codes or technical identifiers
        - Function names, variable names, or code snippets
        
        Args:
            pattern: The exact string to search for.
            max_results: Maximum number of results to return.
            case_sensitive: Whether to match case exactly.
            
        Returns:
            dict with pattern, matches, and total_matches.
        """
        return self._doc_tools.search_exact(pattern, max_results, case_sensitive)
    
    def get_chunk_content(
        self,
        path: str,
        heading: str | None = None,
        start_line: int | None = None,
    ) -> dict[str, object]:
        """Get full content of a specific document chunk.
        
        After using retrieve_docs or search_exact which return snippets,
        use this to get the complete chunk content.
        
        Args:
            path: The file path of the chunk (can be partial path).
            heading: The heading of the chunk to retrieve.
            start_line: The start line for precise matching.
            
        Returns:
            dict with full chunk content or error message.
        """
        return self._doc_tools.get_chunk_content(path, heading, start_line)
    
    def list_chunks_in_file(self, path: str) -> dict[str, object]:
        """List all chunks in a specific file.
        
        Useful for browsing document structure after finding a file
        through search. Shows headings and their locations.
        
        Args:
            path: The file path to list chunks for.
            
        Returns:
            dict with path, chunks list, and total_chunks.
        """
        return self._doc_tools.list_chunks_in_file(path)
    
    def corpus_stats(self) -> dict[str, object]:
        """Return basic stats about the indexed corpus.
        
        Returns:
            dict with file_count, chunk_count, and total_chars.
        """
        return self._doc_tools.corpus_stats()
    
    def list_docs(self, limit: int = 20) -> list[str]:
        """List indexed document paths.
        
        Args:
            limit: Max number of paths to return.
            
        Returns:
            list of sorted file paths.
        """
        return self._doc_tools.list_docs(limit)
    
    # --- File Operations ---
    
    def read_file_chunk(
        self,
        file_path: str,
        offset: int = 0,
        length: int = 8000,
    ) -> str:
        """Read a chunk of content from a file by character offset.
        
        Use this to incrementally read through large files.
        
        Args:
            file_path: Path to the file to read.
            offset: Character offset to start reading from (0-based).
            length: Maximum number of characters to read.
            
        Returns:
            JSON string with chunk content and navigation info.
        """
        return self._chunked_tools.read_file_chunk(file_path, offset, length)
    
    def search_in_file(
        self,
        file_path: str,
        pattern: str,
        context_chars: int = 100,
        max_results: int = 20,
    ) -> str:
        """Search for a pattern within a single file.
        
        Args:
            file_path: Path to the file to search in.
            pattern: Text pattern to search for (case-insensitive).
            context_chars: Chars to show before/after each match.
            max_results: Maximum number of matches to return.
            
        Returns:
            JSON string with matching locations and context.
        """
        return self._chunked_tools.search_in_file(
            file_path, pattern, context_chars, max_results
        )
    
    def get_file_info(self, file_path: str) -> str:
        """Get information about a file without reading its content.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            JSON string with file metadata.
        """
        return self._chunked_tools.get_file_info(file_path)
    
    # --- File Pattern Search (FileToolkit) ---
    
    def search_files_pattern(
        self,
        pattern: str,
        file_types: list[str] | None = None,
        file_pattern: str | None = None,
        path: str | None = None,
    ) -> str:
        """Search for a text pattern in files within a directory.
        
        This searches for a text pattern (case-insensitive substring match)
        in files matching either the specified file types or a file pattern.
        
        Args:
            pattern: The text pattern to search for (case-insensitive).
            file_types: List of file extensions to search (e.g., ["md", "txt"]).
                If not provided, defaults to ["md"].
            file_pattern: Glob pattern for matching files (e.g., "*_workflow.md").
                If provided, this overrides file_types.
            path: Directory to search in. If not provided, uses working directory.
            
        Returns:
            JSON string with search results including file paths, line numbers,
            and matching content.
        """
        toolkit = self._get_file_toolkit()
        if toolkit is None:
            return json.dumps({"error": "FileToolkit not available"})
        
        try:
            return toolkit.search_files(
                pattern=pattern,
                file_types=file_types,
                file_pattern=file_pattern,
                path=path,
            )
        except Exception as e:
            logger.error("Error in search_files_pattern: %s", e)
            return json.dumps({"error": str(e)})
    
    # --- Academic Search (Semantic Scholar) ---
    
    def search_papers(
        self,
        query: str,
        limit: int = 5,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search for academic papers using Semantic Scholar.
        
        Args:
            query: Search query for papers.
            limit: Maximum number of papers to return.
            fields: Optional list of fields to retrieve.
            
        Returns:
            dict with search results or error message.
        """
        toolkit = self._get_scholar_toolkit()
        if toolkit is None:
            return {"error": "SemanticScholarToolkit not available"}
        
        try:
            return toolkit.fetch_bulk_paper_data(query=query, limit=limit)
        except Exception as e:
            logger.error("Error searching papers: %s", e)
            return {"error": str(e)}
    
    def get_paper_details(self, paper_id: str) -> dict[str, Any]:
        """Get detailed information about a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID.
            
        Returns:
            dict with paper details or error message.
        """
        toolkit = self._get_scholar_toolkit()
        if toolkit is None:
            return {"error": "SemanticScholarToolkit not available"}
        
        try:
            return toolkit.fetch_paper_data_title(paper_title=paper_id)
        except Exception as e:
            logger.error("Error fetching paper details: %s", e)
            return {"error": str(e)}
    
    def get_author_info(self, author_id: str) -> dict[str, Any]:
        """Get information about an author.
        
        Args:
            author_id: Semantic Scholar author ID.
            
        Returns:
            dict with author information or error message.
        """
        toolkit = self._get_scholar_toolkit()
        if toolkit is None:
            return {"error": "SemanticScholarToolkit not available"}
        
        try:
            return toolkit.fetch_author_data(author_ids=[author_id])
        except Exception as e:
            logger.error("Error fetching author info: %s", e)
            return {"error": str(e)}
    
    # --- CBETA MCP Tools ---
    # 
    # CBETA (Chinese Buddhist Electronic Text Association) provides access to
    # the largest digital collection of Chinese Buddhist scriptures.
    # 
    # Work ID Format:
    #   - T0001: 大正藏 (Taishō Tripiṭaka) - the main collection
    #   - X0001: 卍續藏 (Xuzangjing/Wan Continuation)
    #   - J0001: 嘉興藏 (Jiaxing Canon)
    #   - N0001: 南傳大藏經 (Nandenchō Daizōkyō/Pāli Canon)
    #
    # Linehead Format (for precise navigation):
    #   T01n0001_p0001a04 = Volume T01, Work 0001, Page 1, Column a, Line 4
    
    def _run_mcp_tool(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Run an MCP tool synchronously.
        
        Args:
            tool_name: Name of the MCP tool to run.
            **kwargs: Arguments to pass to the tool.
            
        Returns:
            Tool result or error dict.
        """
        if self._mcp_toolkit is None:
            return {"error": "MCP not connected. Start MCP server first."}
        
        try:
            # Get all tools and find the matching one
            tools = self._mcp_toolkit.get_tools()
            target_tool = None
            
            for tool in tools:
                schema = tool.get_openai_tool_schema()
                if isinstance(schema, dict):
                    func_schema = schema.get("function", {})
                    if func_schema.get("name") == tool_name:
                        target_tool = tool
                        break
            
            if target_tool is None:
                return {"error": f"MCP tool '{tool_name}' not found"}
            
            # Call the tool (handle both sync and async)
            result = target_tool.func(**kwargs)
            
            # If result is a coroutine, run it
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    # Already in an async context
                    future = asyncio.ensure_future(result)
                    return future
                except RuntimeError:
                    # No running loop, create one
                    result = asyncio.run(result)
            
            return result if isinstance(result, dict) else {"result": result}
            
        except Exception as e:
            logger.error("Error running MCP tool '%s': %s", tool_name, e)
            return {"error": str(e)}
    
    # === CBETA Search Tools ===
    
    def cbeta_search(
        self,
        query: str,
        rows: int = 20,
        start: int = 0,
        order: str | None = None,
    ) -> dict[str, Any]:
        """Search Buddhist scriptures using CBETA full-text search.
        
        This is the primary search tool for finding content in CBETA.
        
        Args:
            query: Search query in Chinese (Traditional preferred).
                Examples: "法鼓", "般若波羅蜜", "四聖諦"
            rows: Number of results per page (default: 20).
            start: Starting offset for pagination.
            order: Sort order, e.g., "time_from-" for descending by time.
            
        Returns:
            dict with:
                - num_found: total number of matching juan (卷)
                - total_term_hits: total keyword occurrences
                - results: list of matches with work, title, juan, term_hits, etc.
                
        Example:
            >>> env.cbeta_search("法鼓", rows=5)
            {"num_found": 2628, "results": [{"work": "T0270", "title": "大法鼓經", ...}]}
        """
        params = {"q": query, "rows": rows, "start": start}
        if order:
            params["order"] = order
        return self._run_mcp_tool("cbeta_fulltext_search", **params)
    
    def cbeta_search_all_in_one(
        self,
        query: str,
        rows: int = 20,
        start: int = 0,
        around: int = 10,
        facet: int = 0,
    ) -> dict[str, Any]:
        """Full-text search with KWIC (Keyword In Context) results.
        
        Returns both search results and keyword context snippets.
        More informative than cbeta_search but slightly slower.
        
        Args:
            query: Search query in Chinese. Supports AND/OR/NOT/NEAR syntax.
            rows: Number of results per page.
            start: Starting offset for pagination.
            around: Number of characters around keyword in KWIC (default: 10).
            facet: Whether to return facet breakdown (0=no, 1=yes).
            
        Returns:
            dict with results including kwics (keyword context) for each match.
            If facet=1, includes category/dynasty/canon distribution.
            
        Example:
            >>> env.cbeta_search_all_in_one("法鼓", around=20, facet=1)
            {"results": [{"kwics": {"results": [{"kwic": "擊於大<mark>法鼓</mark>..."}]}}]}
        """
        return self._run_mcp_tool(
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
        """Advanced search with AND/OR/NOT/NEAR operators.
        
        Args:
            query: Query with operators. Each term in double quotes.
                - AND: "法鼓" "聖嚴" (both terms)
                - OR: "波羅蜜" | "波羅密" (either term)
                - NOT: "迦葉" !"迦葉佛" (exclude)
                - NEAR: "法鼓" NEAR/7 "迦葉" (within 7 chars)
            rows: Number of results.
            start: Pagination offset.
            
        Returns:
            dict with total count and matching results.
            
        Example:
            >>> env.cbeta_extended_search('"般若" NEAR/5 "波羅蜜"')
        """
        return self._run_mcp_tool(
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
        """Search scripture titles (經名).
        
        Use this to find scriptures by their title rather than content.
        Query must be at least 3 characters.
        
        Args:
            query: Title keyword (min 3 chars). E.g., "法華經", "般若波羅蜜".
            rows: Number of results.
            start: Pagination offset.
            
        Returns:
            dict with matching scripture titles, work IDs, and metadata.
            
        Example:
            >>> env.cbeta_search_title("觀無量壽經")
            {"num_found": 49, "results": [{"work": "X0411", "content": "觀無量壽經義疏正觀記"}]}
        """
        return self._run_mcp_tool(
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
        """KWIC (Keyword In Context) search within a specific juan (卷).
        
        Search for a keyword within a single fascicle and get context.
        
        Args:
            work: Work ID, e.g., "T0001".
            juan: Juan (fascicle) number, starting from 1.
            query: Keyword to search. Supports NEAR syntax.
            note: Include annotations (0=no, 1=yes).
            mark: Add <mark> tags around keyword (0=no, 1=yes).
            
        Returns:
            dict with num_found and results with kwic context.
            
        Example:
            >>> env.cbeta_kwic_search("T0001", 1, "老子")
            {"num_found": 4, "results": [{"lb": "0002b03", "kwic": "...<mark>老子</mark>..."}]}
        """
        return self._run_mcp_tool(
            "cbeta_kwic_search",
            work=work,
            juan=juan,
            q=query,
            note=note,
            mark=mark,
        )
    
    # === CBETA Catalog/Metadata Tools ===
    
    def cbeta_search_catalog(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Search CBETA scripture catalog by keyword or volume.
        
        Args:
            query: Keyword or volume code.
                - Keyword: "阿含", "般若" → scriptures with this term
                - Volume: "T01" → scriptures in Taishō vol.1
                
        Returns:
            dict with results of type: catalog, work, or toc entry.
            
        Example:
            >>> env.cbeta_search_catalog("阿含")
            {"num_found": 46, "results": [{"type": "work", "n": "T0001", "label": "長阿含經"}]}
        """
        return self._run_mcp_tool("search_cbeta_texts", q=query)
    
    def cbeta_search_by_translator(
        self,
        creator: str | None = None,
        creator_id: str | None = None,
    ) -> dict[str, Any]:
        """Search scriptures by translator/author.
        
        Args:
            creator: Translator name (fuzzy match). E.g., "玄奘", "鳩摩羅什".
            creator_id: Exact translator ID. E.g., "A000439" (玄奘).
            
        Returns:
            dict with matching works and translator info.
            
        Example:
            >>> env.cbeta_search_by_translator(creator="玄奘")
            {"num_found": 76, "results": [{"work": "T0220", "title": "大般若波羅蜜多經"}]}
        """
        params = {}
        if creator_id:
            params["creator_id"] = creator_id
        elif creator:
            params["creator"] = creator
        else:
            return {"error": "Provide either creator or creator_id"}
        return self._run_mcp_tool("search_works_by_translator", **params)
    
    def cbeta_search_by_dynasty(
        self,
        dynasty: str | None = None,
        time_start: int | None = None,
        time_end: int | None = None,
    ) -> dict[str, Any]:
        """Search scriptures by dynasty or time period.
        
        Args:
            dynasty: Dynasty name(s), comma-separated.
                E.g., "唐", "唐,宋", "後漢".
            time_start: Start year (CE). E.g., 600.
            time_end: End year (CE). E.g., 900.
            
        Returns:
            dict with num_found and sample results.
            
        Example:
            >>> env.cbeta_search_by_dynasty(dynasty="唐")
            >>> env.cbeta_search_by_dynasty(time_start=600, time_end=900)
        """
        params = {}
        if dynasty:
            params["dynasty"] = dynasty
        if time_start:
            params["time_start"] = time_start
        if time_end:
            params["time_end"] = time_end
        if not params:
            return {"error": "Provide dynasty or time_start/time_end"}
        return self._run_mcp_tool("search_cbeta_by_dynasty", **params)
    
    # === CBETA Work/Content Tools ===
    
    def cbeta_get_work_info(self, work: str) -> dict[str, Any]:
        """Get detailed information about a scripture.
        
        Args:
            work: Work ID. E.g., "T0001", "T1501", "X0600".
            
        Returns:
            dict with:
                - work, title, byline: basic info
                - creators: translator/author names
                - category: CBETA classification
                - time_dynasty: dynasty
                - time_from/to: year range (CE)
                - cjk_chars: character count
                - places: translation location with coordinates
                
        Example:
            >>> env.cbeta_get_work_info("T0001")
            {"work": "T0001", "title": "長阿含經", "byline": "後秦 佛陀耶舍共竺佛念譯"}
        """
        return self._run_mcp_tool("get_cbeta_work_info", work=work)
    
    def cbeta_get_toc(self, work: str) -> dict[str, Any]:
        """Get table of contents for a scripture.
        
        Args:
            work: Work ID. E.g., "T0001".
            
        Returns:
            dict with mulu (目錄) structure containing:
                - title: section title
                - juan: fascicle number
                - lb: line position (page-column-line)
                - type: entry type (序/分/品/經)
                - children: nested sub-entries
                
        Example:
            >>> env.cbeta_get_toc("T0001")
            {"results": [{"mulu": [{"title": "序", "juan": 1, "lb": "0001a02"}]}]}
        """
        return self._run_mcp_tool("get_cbeta_toc", work=work)
    
    def cbeta_get_juan_html(
        self,
        work: str,
        juan: int,
        work_info: int = 0,
        toc: int = 0,
    ) -> dict[str, Any]:
        """Get HTML content of a specific juan (fascicle).
        
        Use this to read the actual scripture text.
        
        Args:
            work: Work ID. E.g., "T0001".
            juan: Juan number (starting from 1).
            work_info: Include work metadata (0=no, 1=yes).
            toc: Include table of contents (0=no, 1=yes).
            
        Returns:
            dict with HTML content of the juan.
            HTML includes semantic markup and annotation anchors.
            
        Example:
            >>> env.cbeta_get_juan_html("T0001", 1)
            {"results": [{"juan": 1, "html": "<div id='body'>如是我聞。一時佛在..."}]}
        """
        return self._run_mcp_tool(
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
        """Get specific lines of text by line position.
        
        Three modes:
        1. Single line: linehead only
        2. Range: linehead_start + linehead_end
        3. Context: linehead + before/after
        
        Args:
            linehead: Line position. Format: T01n0001_p0001a04
                (Vol T01, Work 0001, Page 1, Column a, Line 4)
            linehead_start: Start of range.
            linehead_end: End of range.
            before: Lines before linehead to include.
            after: Lines after linehead to include.
            
        Returns:
            dict with line content and any annotations.
            
        Example:
            >>> env.cbeta_get_lines(linehead="T01n0001_p0001a04")
            >>> env.cbeta_get_lines(linehead="T01n0001_p0001a04", before=2, after=3)
        """
        params = {}
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
            return {"error": "Provide linehead or linehead_start/linehead_end"}
        return self._run_mcp_tool("get_cbeta_lines", **params)
    
    def cbeta_goto(
        self,
        linehead: str | None = None,
        canon: str | None = None,
        work: str | None = None,
        juan: int | None = None,
        page: int | None = None,
        col: str | None = None,
        line: int | None = None,
    ) -> dict[str, Any]:
        """Navigate to a specific position in scripture.
        
        Two modes:
        1. By linehead: Direct jump (highest priority)
        2. By structure: canon + work + position info
        
        Args:
            linehead: Direct position. E.g., "T01n0001_p0066c25".
            canon: Canon code. E.g., "T" (Taishō), "X" (Xuzang).
            work: Work number within canon. E.g., "1", "150A".
            juan: Fascicle number.
            page: Page number.
            col: Column. One of "a", "b", "c".
            line: Line number.
            
        Returns:
            dict with URL to the position.
            
        Example:
            >>> env.cbeta_goto(linehead="T01n0001_p0066c25")
            >>> env.cbeta_goto(canon="T", work="1", page=11, col="b", line=10)
        """
        params = {}
        if linehead:
            params["linehead"] = linehead
        else:
            if canon:
                params["canon"] = canon
            if work:
                params["work"] = work
            if juan is not None:
                params["juan"] = juan
            if page is not None:
                params["page"] = page
            if col:
                params["col"] = col
            if line is not None:
                params["line"] = line
        if not params:
            return {"error": "Provide linehead or navigation parameters"}
        return self._run_mcp_tool("cbeta_goto", **params)
    
    # === Legacy Aliases (for backward compatibility) ===
    
    def search_cbeta(
        self,
        query: str,
        start: int = 0,
        rows: int = 10,
    ) -> dict[str, Any]:
        """[DEPRECATED] Use cbeta_search instead."""
        return self.cbeta_search(query=query, rows=rows, start=start)
    
    def get_cbeta_work_info(self, work: str) -> dict[str, Any]:
        """[DEPRECATED] Use cbeta_get_work_info instead."""
        return self.cbeta_get_work_info(work=work)
    
    def get_cbeta_toc(self, work: str) -> dict[str, Any]:
        """[DEPRECATED] Use cbeta_get_toc instead."""
        return self.cbeta_get_toc(work=work)


def run_env(env: LocalEnv, query: str) -> tuple[str, str]:
    # ... your code ...
    return query, ""
