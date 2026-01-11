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
    
    def search_cbeta(
        self,
        query: str,
        start: int = 0,
        rows: int = 10,
    ) -> dict[str, Any]:
        """Search Buddhist scriptures using CBETA full-text search.
        
        Requires MCP server to be running.
        
        Args:
            query: Search query in Chinese (Traditional preferred).
            start: Starting offset for pagination.
            rows: Number of results to return.
            
        Returns:
            dict with search results or error message.
        """
        return self._run_mcp_tool(
            "cbeta_fulltext_search",
            q=query,
            start=start,
            rows=rows,
        )
    
    def get_cbeta_work_info(self, work: str) -> dict[str, Any]:
        """Get information about a CBETA work (scripture).
        
        Requires MCP server to be running.
        
        Args:
            work: Work ID (e.g., "T0001" for 長阿含經).
            
        Returns:
            dict with work metadata or error message.
        """
        return self._run_mcp_tool("get_cbeta_work_info", work=work)
    
    def get_cbeta_toc(self, work: str) -> dict[str, Any]:
        """Get table of contents for a CBETA work.
        
        Requires MCP server to be running.
        
        Args:
            work: Work ID (e.g., "T0001").
            
        Returns:
            dict with table of contents or error message.
        """
        return self._run_mcp_tool("get_cbeta_toc", work=work)


def run_env(env: LocalEnv, query: str) -> tuple[str, str]:
    # ... your code ...
    return query, ""
