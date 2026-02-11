"""Implementation helpers for LocalEnv.

This module keeps runtime logic out of ``react_env.py`` so that ``LocalEnv``
can stay as a concise interface layer for LLM prompt injection.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_file_agent.tools import LocalDocTools

if TYPE_CHECKING:
    from camel.toolkits.file_toolkit import FileToolkit
    from camel.toolkits.mcp_toolkit import MCPToolkit
    from camel.toolkits.semantic_scholar_toolkit import SemanticScholarToolkit

logger = logging.getLogger(__name__)


class LocalDocSearchTools:
    """Local document retrieval helpers with timeout control."""

    def __init__(
        self,
        doc_tools: LocalDocTools,
        default_timeout_s: float | None,
    ) -> None:
        self._doc_tools = doc_tools
        self._default_timeout_s = default_timeout_s

    def retrieve_docs(
        self,
        query: str,
        top_k: int = 5,
        min_score: int = 1,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        """Retrieve documents with optional timeout protection."""
        timeout = self._default_timeout_s if timeout_s is None else timeout_s
        if timeout is not None and timeout <= 0:
            timeout = None

        if timeout is None:
            return self._doc_tools.retrieve_local_docs(query, top_k, min_score)

        result_holder: dict[str, object] = {}
        error_holder: dict[str, Exception] = {}

        def _run() -> None:
            try:
                result_holder["result"] = self._doc_tools.retrieve_local_docs(
                    query,
                    top_k,
                    min_score,
                )
            except Exception as exc:
                error_holder["error"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            logger.warning(
                "retrieve_docs timed out after %.1fs for query=%s",
                timeout,
                query,
            )
            return {
                "error": "retrieve_docs timed out",
                "timed_out": True,
                "timeout_s": timeout,
                "query": query,
                "top_k": top_k,
                "min_score": min_score,
            }

        if "error" in error_holder:
            err = error_holder["error"]
            logger.error("retrieve_docs failed: %s", err)
            return {
                "error": str(err),
                "query": query,
                "top_k": top_k,
                "min_score": min_score,
            }

        return result_holder.get("result", {})


class FilePatternSearchTools:
    """Wrapper around CAMEL FileToolkit pattern search."""

    def __init__(self, working_directory: str | Path | None = None) -> None:
        self._working_directory = (
            Path(working_directory) if working_directory is not None else None
        )
        self._file_toolkit: FileToolkit | None = None

    def _get_file_toolkit(self) -> FileToolkit | None:
        if self._file_toolkit is None:
            try:
                from camel.toolkits.file_toolkit import FileToolkit

                working_dir = (
                    str(self._working_directory)
                    if self._working_directory is not None
                    else None
                )
                self._file_toolkit = FileToolkit(
                    working_directory=working_dir,
                    backup_enabled=False,
                )
            except ImportError:
                logger.warning("FileToolkit not available")
                return None
        return self._file_toolkit

    def search_files_pattern(
        self,
        pattern: str,
        file_types: list[str] | None = None,
        file_pattern: str | None = None,
        path: str | None = None,
    ) -> str:
        """Search pattern in files and return FileToolkit raw JSON string."""
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
        except Exception as exc:
            logger.error("search_files_pattern failed: %s", exc)
            return json.dumps({"error": str(exc)})


class SemanticScholarTools:
    """Wrapper around CAMEL SemanticScholarToolkit."""

    def __init__(self) -> None:
        self._scholar_toolkit: SemanticScholarToolkit | None = None

    def _get_toolkit(self) -> SemanticScholarToolkit | None:
        if self._scholar_toolkit is None:
            try:
                from camel.toolkits.semantic_scholar_toolkit import (
                    SemanticScholarToolkit,
                )

                self._scholar_toolkit = SemanticScholarToolkit()
            except ImportError:
                logger.warning("SemanticScholarToolkit not available")
                return None
        return self._scholar_toolkit

    @staticmethod
    def _trim_paper_results(
        payload: dict[str, Any],
        limit: int,
    ) -> dict[str, Any]:
        if limit <= 0:
            return payload

        for key in ("data", "results", "papers"):
            value = payload.get(key)
            if isinstance(value, list):
                copied = dict(payload)
                copied[key] = value[:limit]
                return copied
        return payload

    def search_papers(
        self,
        query: str,
        limit: int = 5,
        fields: list[str] | None = None,
        year: str = "2023-",
    ) -> dict[str, Any]:
        """Search papers using bulk endpoint and trim locally by ``limit``."""
        toolkit = self._get_toolkit()
        if toolkit is None:
            return {"error": "SemanticScholarToolkit not available"}

        try:
            result = toolkit.fetch_bulk_paper_data(
                query=query,
                year=year,
                fields=fields,
            )
            if isinstance(result, dict):
                return self._trim_paper_results(result, limit)
            return {"result": result}
        except Exception as exc:
            logger.error("search_papers failed: %s", exc)
            return {"error": str(exc)}

    def get_paper_details(
        self,
        paper_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch one paper by Semantic Scholar paper ID."""
        toolkit = self._get_toolkit()
        if toolkit is None:
            return {"error": "SemanticScholarToolkit not available"}

        try:
            result = toolkit.fetch_paper_data_id(
                paper_id=paper_id,
                fields=fields,
            )
            if isinstance(result, dict):
                return result
            return {"result": result}
        except Exception as exc:
            logger.error("get_paper_details failed: %s", exc)
            return {"error": str(exc)}

    def get_author_info(
        self,
        author_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch one author by Semantic Scholar author ID."""
        toolkit = self._get_toolkit()
        if toolkit is None:
            return {"error": "SemanticScholarToolkit not available"}

        try:
            result = toolkit.fetch_author_data(
                ids=[author_id],
                fields=fields,
            )
            if isinstance(result, dict):
                return result
            return {"result": result}
        except Exception as exc:
            logger.error("get_author_info failed: %s", exc)
            return {"error": str(exc)}


class CbetaMcpTools:
    """CBETA MCP tool caller using ``MCPToolkit.call_tool_sync``."""

    def __init__(self, mcp_toolkit: MCPToolkit | None = None) -> None:
        self._mcp_toolkit = mcp_toolkit

    def run(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Call one MCP tool and return raw MCP response shape."""
        if self._mcp_toolkit is None:
            return {
                "status": "error",
                "message": "MCP not connected. Start MCP server first.",
            }

        try:
            result = self._mcp_toolkit.call_tool_sync(tool_name, kwargs)
        except Exception as exc:
            logger.error("CBETA MCP tool call failed (%s): %s", tool_name, exc)
            return {
                "status": "error",
                "message": str(exc),
                "tool": tool_name,
            }

        if isinstance(result, dict):
            return result
        return {
            "status": "success",
            "result": result,
        }
