"""arXiv API tool for the agent's USE_TOOL path."""

import logging

import arxiv
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers matching the query.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        Formatted string with paper titles, authors, abstracts, and URLs.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        entries = list(client.results(search))
    except Exception as exc:
        logger.warning("arXiv search failed: %s", exc)
        return f"[ERROR] arXiv search failed: {exc}"

    if not entries:
        return f"No results found for query: {query}"

    parts = []
    for i, paper in enumerate(entries, start=1):
        abstract = paper.summary[:300] + ("..." if len(paper.summary) > 300 else "")
        parts.append(
            f"[{i}] Title: {paper.title}\n"
            f"    Authors: {', '.join(a.name for a in paper.authors)}\n"
            f"    Published: {paper.published.date()}\n"
            f"    Abstract: {abstract}\n"
            f"    URL: {paper.entry_id}"
        )
    return "\n\n".join(parts)
