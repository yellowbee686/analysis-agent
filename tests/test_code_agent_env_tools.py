from __future__ import annotations

from typing import Any

from local_file_agent.code_agent.env_tools import CbetaMcpTools, SemanticScholarTools
from local_file_agent.code_agent.prompt_engine import PromptEngine
from local_file_agent.code_agent import react_env as react_env_module


class _FakeScholarToolkit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def fetch_bulk_paper_data(
        self,
        query: str,
        year: str = "2023-",
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "fetch_bulk_paper_data",
                {"query": query, "year": year, "fields": fields},
            )
        )
        return {
            "data": [
                {"paperId": "p1"},
                {"paperId": "p2"},
                {"paperId": "p3"},
            ]
        }

    def fetch_paper_data_id(
        self,
        paper_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "fetch_paper_data_id",
                {"paper_id": paper_id, "fields": fields},
            )
        )
        return {"paperId": paper_id, "title": "Test Paper"}

    def fetch_author_data(
        self,
        ids: list[str],
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "fetch_author_data",
                {"ids": ids, "fields": fields},
            )
        )
        return {"data": [{"authorId": ids[0]}]}


class _FakeMcpToolkit:
    def __init__(
        self,
        *,
        result: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool_sync(self, tool_name: str, args: dict[str, Any]) -> Any:
        self.calls.append((tool_name, args))
        if self.error is not None:
            raise self.error
        return self.result


def test_semantic_scholar_tools_call_signatures() -> None:
    toolkit = _FakeScholarToolkit()
    tools = SemanticScholarTools()
    tools._scholar_toolkit = toolkit

    papers = tools.search_papers(
        query="retrieval augmented generation",
        limit=2,
        fields=["title", "year"],
    )
    assert papers["data"] == [{"paperId": "p1"}, {"paperId": "p2"}]

    details = tools.get_paper_details("CorpusID:123")
    assert details["paperId"] == "CorpusID:123"

    author = tools.get_author_info("1741102")
    assert author["data"][0]["authorId"] == "1741102"

    assert toolkit.calls[0] == (
        "fetch_bulk_paper_data",
        {
            "query": "retrieval augmented generation",
            "year": "2023-",
            "fields": ["title", "year"],
        },
    )
    assert toolkit.calls[1] == (
        "fetch_paper_data_id",
        {"paper_id": "CorpusID:123", "fields": None},
    )
    assert toolkit.calls[2] == (
        "fetch_author_data",
        {"ids": ["1741102"], "fields": None},
    )


def test_cbeta_mcp_tools_success_error_paths() -> None:
    disconnected = CbetaMcpTools(None)
    result = disconnected.run("cbeta_fulltext_search", q="法鼓")
    assert result["status"] == "error"

    connected = _FakeMcpToolkit(result={"status": "success", "result": {"ok": True}})
    success_tools = CbetaMcpTools(connected)
    success = success_tools.run("cbeta_fulltext_search", q="法鼓", rows=5)
    assert success == {"status": "success", "result": {"ok": True}}
    assert connected.calls[0] == (
        "cbeta_fulltext_search",
        {"q": "法鼓", "rows": 5},
    )

    broken = _FakeMcpToolkit(error=RuntimeError("tool not found"))
    broken_tools = CbetaMcpTools(broken)
    failed = broken_tools.run("missing_tool", q="法鼓")
    assert failed["status"] == "error"
    assert failed["tool"] == "missing_tool"


def test_local_env_surface_and_goto_vol_passthrough(monkeypatch) -> None:
    class _FakeLocalDocTools:
        def __init__(self, index: Any) -> None:
            self.index = index

        def retrieve_local_docs(
            self,
            query: str,
            top_k: int,
            min_score: int,
        ) -> dict[str, Any]:
            return {
                "query": query,
                "top_k": top_k,
                "min_score": min_score,
            }

        def search_exact(
            self,
            pattern: str,
            max_results: int,
            case_sensitive: bool,
        ) -> dict[str, Any]:
            return {"pattern": pattern, "max_results": max_results, "case_sensitive": case_sensitive}

        def get_chunk_content(
            self,
            path: str,
            heading: str | None,
            start_line: int | None,
        ) -> dict[str, Any]:
            return {"path": path, "heading": heading, "start_line": start_line}

        def list_chunks_in_file(self, path: str) -> dict[str, Any]:
            return {"path": path, "chunks": []}

        def corpus_stats(self) -> dict[str, Any]:
            return {"file_count": 1, "chunk_count": 1, "total_chars": 10}

        def list_docs(self, limit: int) -> list[str]:
            return ["a.md"][:limit]

    class _FakeChunkedFileTools:
        def __init__(self, working_directory: str | None = None) -> None:
            self.working_directory = working_directory

        def read_file_chunk(self, file_path: str, offset: int, length: int) -> str:
            return f"{file_path}:{offset}:{length}"

        def search_in_file(
            self,
            file_path: str,
            pattern: str,
            context_chars: int,
            max_results: int,
        ) -> str:
            return f"{file_path}:{pattern}:{context_chars}:{max_results}"

        def get_file_info(self, file_path: str) -> str:
            return file_path

    class _FakeFilePatternSearchTools:
        def __init__(self, working_directory: str | None = None) -> None:
            self.working_directory = working_directory

        def search_files_pattern(
            self,
            pattern: str,
            file_types: list[str] | None,
            file_pattern: str | None,
            path: str | None,
        ) -> str:
            return f"{pattern}:{file_types}:{file_pattern}:{path}"

    class _FakeSemanticScholarTools:
        def search_papers(
            self,
            query: str,
            limit: int,
            fields: list[str] | None,
        ) -> dict[str, Any]:
            return {"query": query, "limit": limit, "fields": fields}

        def get_paper_details(
            self,
            paper_id: str,
            fields: list[str] | None,
        ) -> dict[str, Any]:
            return {"paper_id": paper_id, "fields": fields}

        def get_author_info(
            self,
            author_id: str,
            fields: list[str] | None,
        ) -> dict[str, Any]:
            return {"author_id": author_id, "fields": fields}

    cbeta_calls: list[tuple[str, dict[str, Any]]] = []

    class _FakeCbetaMcpTools:
        def __init__(self, mcp_toolkit: Any = None) -> None:
            self.mcp_toolkit = mcp_toolkit

        def run(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
            cbeta_calls.append((tool_name, kwargs))
            return {"status": "success", "result": kwargs}

    monkeypatch.setattr(react_env_module, "LocalDocTools", _FakeLocalDocTools)
    monkeypatch.setattr(react_env_module, "ChunkedFileTools", _FakeChunkedFileTools)
    monkeypatch.setattr(
        react_env_module,
        "FilePatternSearchTools",
        _FakeFilePatternSearchTools,
    )
    monkeypatch.setattr(
        react_env_module,
        "SemanticScholarTools",
        _FakeSemanticScholarTools,
    )
    monkeypatch.setattr(react_env_module, "CbetaMcpTools", _FakeCbetaMcpTools)

    env = react_env_module.LocalEnv(index=object())

    assert hasattr(env, "cbeta_search_sc")
    assert hasattr(env, "cbeta_search_notes")
    assert hasattr(env, "cbeta_facet_query")
    assert not hasattr(env, "search_cbeta")
    assert not hasattr(env, "get_cbeta_work_info")
    assert not hasattr(env, "get_cbeta_toc")

    env.cbeta_goto(canon="T", work="1", vol=1, page=11, col="b", line=10)
    assert cbeta_calls[-1][0] == "cbeta_goto"
    assert cbeta_calls[-1][1]["vol"] == 1


def test_prompt_engine_env_api_is_synced() -> None:
    env_api = PromptEngine("gpt-4o")._get_env_api_string()

    assert "def cbeta_search_sc(" in env_api
    assert "def cbeta_search_notes(" in env_api
    assert "def cbeta_facet_query(" in env_api
    assert "def search_cbeta(" not in env_api
    assert "def get_cbeta_work_info(" not in env_api
    assert "def get_cbeta_toc(" not in env_api
