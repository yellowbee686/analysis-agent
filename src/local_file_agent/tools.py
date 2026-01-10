from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from camel.toolkits import FunctionTool

from local_file_agent.indexer import LocalIndex
from local_file_agent.session_storage import (
    generate_response_filename,
    get_session_storage_dir,
)

logger = logging.getLogger(__name__)

# Default threshold for tool response size (in characters)
DEFAULT_TOOL_RESPONSE_THRESHOLD = 8000


def wrap_tool_with_size_limit(
    tool: FunctionTool,
    session_id: str | None = None,
    threshold: int = DEFAULT_TOOL_RESPONSE_THRESHOLD,
) -> FunctionTool:
    """Wrap a FunctionTool to limit response size.

    If the tool's response exceeds the threshold, saves it to a file
    and returns metadata instead.

    Args:
        tool: The FunctionTool to wrap.
        session_id: Session identifier for storage directory.
        threshold: Size threshold in characters.

    Returns:
        A new FunctionTool with size-limited responses.
    """
    original_func = tool.func

    @functools.wraps(original_func)
    def wrapped_func(*args, **kwargs) -> Any:
        result = original_func(*args, **kwargs)

        # Estimate result size
        if isinstance(result, str):
            result_size = len(result)
            result_str = result
        elif isinstance(result, dict):
            result_str = json.dumps(result, ensure_ascii=False)
            result_size = len(result_str)
        else:
            result_str = str(result)
            result_size = len(result_str)

        # If result is small enough, return as-is
        if result_size <= threshold:
            return result

        # Save large result to file
        logger.info(
            "Tool '%s' response size %d > threshold %d, saving to file",
            tool.func.__name__,
            result_size,
            threshold,
        )

        storage_dir = get_session_storage_dir(session_id)
        tool_name = tool.func.__name__

        # Build args dict for filename generation
        args_dict = {}
        if args:
            args_dict["_args"] = str(args)[:50]
        args_dict.update(kwargs)

        # Determine extension based on content
        if isinstance(result, dict):
            extension = "json"
            content_to_save = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            extension = "txt"
            content_to_save = result_str

        filename = generate_response_filename(tool_name, args_dict, extension)
        file_path = storage_dir / filename

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content_to_save)

            # Return metadata about the saved file
            preview_len = min(500, len(content_to_save))
            metadata = {
                "status": "saved_to_file",
                "message": (
                    f"Response too large ({result_size} chars), saved to file. "
                    f"Use read_file_chunk or search_in_file to access the content."
                ),
                "file_path": str(file_path.resolve()),
                "file_size_chars": result_size,
                "tool_name": tool_name,
                "content_preview": content_to_save[:preview_len]
                + ("..." if result_size > preview_len else ""),
            }

            logger.info("Saved large tool response to: %s", file_path)
            return json.dumps(metadata, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Failed to save tool response to file: %s", e)
            # Return original result if save fails
            return result

    # Create a new FunctionTool with the wrapped function
    # Preserve the original schema
    return FunctionTool(
        func=wrapped_func,
        openai_tool_schema=tool.openai_tool_schema,
    )


def wrap_file_toolkit_tools(
    tools: list[FunctionTool],
    session_id: str | None = None,
    threshold: int = DEFAULT_TOOL_RESPONSE_THRESHOLD,
    tool_names: list[str] | None = None,
) -> list[FunctionTool]:
    """Wrap FileToolkit tools with size limits.

    Args:
        tools: List of FunctionTools from FileToolkit.
        session_id: Session identifier for storage directory.
        threshold: Size threshold in characters.
        tool_names: Optional list of tool names to wrap.
            If None, wraps 'search_files' and 'read_file'.

    Returns:
        List of tools with specified tools wrapped.
    """
    if tool_names is None:
        tool_names = ["search_files", "read_file"]

    wrapped_tools = []
    for tool in tools:
        if tool.func.__name__ in tool_names:
            wrapped_tools.append(
                wrap_tool_with_size_limit(tool, session_id, threshold)
            )
            logger.info(
                "Wrapped tool '%s' with size limit %d",
                tool.func.__name__,
                threshold,
            )
        else:
            wrapped_tools.append(tool)

    return wrapped_tools


class ChunkedFileTools:
    """Tools for reading large files in chunks and searching within files.

    These tools are designed to work with large files that have been saved
    by the size-limited tool wrappers.
    """

    def __init__(self, working_directory: str | Path | None = None):
        """Initialize ChunkedFileTools.

        Args:
            working_directory: Default working directory for relative paths.
        """
        if working_directory:
            self.working_directory = Path(working_directory).resolve()
        else:
            self.working_directory = Path.cwd()

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a file path to an absolute path."""
        path = Path(file_path)
        if path.is_absolute():
            return path.resolve()
        return (self.working_directory / path).resolve()

    def read_file_chunk(
        self,
        file_path: str,
        offset: int = 0,
        length: int = 8000,
    ) -> str:
        """Read a chunk of content from a file by character offset.

        Use this tool to incrementally read through large files. Start with
        offset=0, then increase offset by the length you read to get the next
        chunk.

        Args:
            file_path (str): Path to the file to read.
            offset (int): Character offset to start reading from (0-based).
            length (int): Maximum number of characters to read (default 8000).

        Returns:
            str: JSON with chunk content, metadata, and navigation info.
        """
        try:
            resolved_path = self._resolve_path(file_path)

            if not resolved_path.exists():
                return json.dumps({
                    "error": f"File not found: {file_path}",
                })

            if not resolved_path.is_file():
                return json.dumps({
                    "error": f"Not a file: {file_path}",
                })

            # Read file content
            content = resolved_path.read_text(encoding="utf-8")
            total_size = len(content)

            # Validate offset
            if offset < 0:
                offset = 0
            if offset >= total_size:
                return json.dumps({
                    "file_path": str(resolved_path),
                    "total_size": total_size,
                    "offset": offset,
                    "length": 0,
                    "content": "",
                    "has_more": False,
                    "message": "Offset is beyond end of file",
                })

            # Extract chunk
            end = min(offset + length, total_size)
            chunk = content[offset:end]

            return json.dumps({
                "file_path": str(resolved_path),
                "total_size": total_size,
                "offset": offset,
                "length": len(chunk),
                "content": chunk,
                "has_more": end < total_size,
                "next_offset": end if end < total_size else None,
                "remaining_chars": max(0, total_size - end),
            }, ensure_ascii=False)

        except Exception as e:
            logger.error("Error reading file chunk: %s", e)
            return json.dumps({"error": str(e)})

    def search_in_file(
        self,
        file_path: str,
        pattern: str,
        context_chars: int = 100,
        max_results: int = 20,
    ) -> str:
        """Search for a pattern within a single file.

        Use this tool to find specific content in a large file without
        reading the entire file.

        Args:
            file_path (str): Path to the file to search in.
            pattern (str): Text pattern to search for (case-insensitive).
            context_chars (int): Number of characters to show before/after
                each match (default 100).
            max_results (int): Maximum number of matches to return
                (default 20).

        Returns:
            str: JSON with matching locations and context.
        """
        try:
            resolved_path = self._resolve_path(file_path)

            if not resolved_path.exists():
                return json.dumps({
                    "error": f"File not found: {file_path}",
                })

            if not resolved_path.is_file():
                return json.dumps({
                    "error": f"Not a file: {file_path}",
                })

            # Read file content
            content = resolved_path.read_text(encoding="utf-8")
            content_lower = content.lower()
            pattern_lower = pattern.lower()
            total_size = len(content)

            # Find all matches
            matches = []
            start_pos = 0

            while len(matches) < max_results:
                pos = content_lower.find(pattern_lower, start_pos)
                if pos == -1:
                    break

                # Extract context around match
                ctx_start = max(0, pos - context_chars)
                ctx_end = min(total_size, pos + len(pattern) + context_chars)

                # Get line number
                line_num = content[:pos].count("\n") + 1

                matches.append({
                    "position": pos,
                    "line": line_num,
                    "context": content[ctx_start:ctx_end],
                    "context_start": ctx_start,
                    "context_end": ctx_end,
                })

                start_pos = pos + 1

            return json.dumps({
                "file_path": str(resolved_path),
                "pattern": pattern,
                "total_size": total_size,
                "total_matches": len(matches),
                "matches": matches,
                "truncated": start_pos < total_size
                and content_lower.find(pattern_lower, start_pos) != -1,
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Error searching in file: %s", e)
            return json.dumps({"error": str(e)})

    def get_file_info(self, file_path: str) -> str:
        """Get information about a file without reading its content.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: JSON with file metadata.
        """
        try:
            resolved_path = self._resolve_path(file_path)

            if not resolved_path.exists():
                return json.dumps({
                    "error": f"File not found: {file_path}",
                })

            stat = resolved_path.stat()

            # Try to get character count
            char_count = None
            line_count = None
            try:
                content = resolved_path.read_text(encoding="utf-8")
                char_count = len(content)
                line_count = content.count("\n") + 1
            except Exception:
                pass

            return json.dumps({
                "file_path": str(resolved_path),
                "exists": True,
                "is_file": resolved_path.is_file(),
                "size_bytes": stat.st_size,
                "size_chars": char_count,
                "line_count": line_count,
            })

        except Exception as e:
            logger.error("Error getting file info: %s", e)
            return json.dumps({"error": str(e)})

    def get_tools(self) -> list[FunctionTool]:
        """Return a list of FunctionTools for chunked file operations.

        Returns:
            list[FunctionTool]: Tools for reading file chunks and searching.
        """
        return [
            FunctionTool(self.read_file_chunk),
            FunctionTool(self.search_in_file),
            FunctionTool(self.get_file_info),
        ]


class LocalDocTools:
    def __init__(self, index: LocalIndex):
        self.index = index

    def retrieve_local_docs(
        self, query: str, top_k: int = 5, min_score: int = 1
    ) -> Dict[str, object]:
        """Retrieve relevant passages from local Markdown files using BM25.

        This uses BM25 semantic ranking to find the most relevant chunks.
        For exact string matching, use search_exact instead.

        Args:
            query (str): User question or keywords to search.
            top_k (int): Maximum number of results to return.
            min_score (int): Minimum match score to keep a result.

        Returns:
            Dict[str, object]: Retrieval results with basic metadata.
        """
        matches = self.index.search(
            query=query, top_k=top_k, min_score=min_score
        )
        return {
            "query": query,
            "matches": matches,
            "total_chunks": len(self.index.chunks),
        }

    def search_exact(
        self, pattern: str, max_results: int = 20, case_sensitive: bool = False
    ) -> Dict[str, object]:
        """Exact substring search in indexed documents.

        Unlike retrieve_local_docs which uses BM25 semantic ranking, this
        performs exact string matching. Use this when you need to find:
        - Specific keywords or phrases
        - Error codes or technical identifiers
        - Function names, variable names, or code snippets
        - Any content that requires precise matching

        Args:
            pattern (str): The exact string to search for.
            max_results (int): Maximum number of results to return.
            case_sensitive (bool): Whether to match case exactly.

        Returns:
            Dict[str, object]: Matching chunks with context around the match.
        """
        matches = self.index.search_exact(
            pattern=pattern,
            max_results=max_results,
            case_sensitive=case_sensitive,
        )
        return {
            "pattern": pattern,
            "case_sensitive": case_sensitive,
            "matches": matches,
            "total_matches": len(matches),
        }

    def get_chunk_content(
        self,
        path: str,
        heading: str | None = None,
        start_line: int | None = None,
    ) -> Dict[str, object]:
        """Get full content of a specific document chunk.

        After using retrieve_local_docs or search_exact which return snippets,
        use this to get the complete chunk content.

        Args:
            path (str): The file path of the chunk (can be partial path).
            heading (str | None): The heading of the chunk to retrieve.
            start_line (int | None): The start line for precise matching.

        Returns:
            Dict[str, object]: Full chunk content or error message.
        """
        result = self.index.get_chunk_content(
            path=path, heading=heading, start_line=start_line
        )
        if result is None:
            return {
                "error": "Chunk not found",
                "path": path,
                "heading": heading,
                "start_line": start_line,
            }
        return result

    def list_chunks_in_file(self, path: str) -> Dict[str, object]:
        """List all chunks in a specific file.

        Useful for browsing document structure after finding a file
        through search. Shows headings and their locations.

        Args:
            path (str): The file path to list chunks for.

        Returns:
            Dict[str, object]: List of chunk metadata.
        """
        chunks = self.index.list_chunks_in_file(path)
        return {
            "path": path,
            "chunks": chunks,
            "total_chunks": len(chunks),
        }

    def corpus_stats(self) -> Dict[str, object]:
        """Return basic stats about the indexed corpus.

        Returns:
            Dict[str, object]: File count, chunk count, and total character size.
        """
        return self.index.stats()

    def list_docs(self, limit: int = 20) -> List[str]:
        """List indexed document paths.

        Args:
            limit (int): Max number of paths to return.

        Returns:
            List[str]: Sorted file paths.
        """
        return [str(p) for p in self.index.files[:limit]]
