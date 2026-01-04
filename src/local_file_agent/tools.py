from __future__ import annotations

from typing import Dict, List

from local_file_agent.indexer import LocalIndex


class LocalDocTools:
    def __init__(self, index: LocalIndex):
        self.index = index

    def retrieve_local_docs(
        self, query: str, top_k: int = 5, min_score: int = 1
    ) -> Dict[str, object]:
        """Retrieve relevant passages from local Markdown files.

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
