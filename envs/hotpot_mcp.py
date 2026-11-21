"""HotpotQA MCP Server running over stdio transport.

Provides tools for multi-hop QA agents to interact with HotpotQA article contexts:
- load_episode: Load Wikipedia context articles for a given episode.
- search: Look up and select an article by title (exact or substring).
- lookup: Search for keywords sentence-by-sentence in the currently selected article.
- finish: Submit the final answer.
"""

from __future__ import annotations

import json
from typing import Dict, List

from mcp.server.mcpserver import MCPServer

# Initialize MCP 2.x server
mcp = MCPServer("HotpotQA-Env")

# Module-level episode state
CONTEXT_PAGES: Dict[str, List[str]] = {}
CURRENT_PAGE_TITLE: str = ""
LOOKUP_INDEX: int = 0


@mcp.tool()
def load_episode(context_json: str) -> str:
    """Load episode context pages from a JSON string.

    Args:
        context_json: A JSON dict mapping article title to a list of sentence strings.

    Returns:
        Confirmation message with the count of loaded articles.
    """
    global CONTEXT_PAGES, CURRENT_PAGE_TITLE, LOOKUP_INDEX

    parsed_context = json.loads(context_json)
    CONTEXT_PAGES = {
        str(title): [str(sentence) for sentence in sentences]
        for title, sentences in parsed_context.items()
    }
    CURRENT_PAGE_TITLE = ""
    LOOKUP_INDEX = 0
    return f"Loaded {len(CONTEXT_PAGES)} articles."


@mcp.tool()
def search(entity: str) -> str:
    """Search for an article by entity/title.

    Performs a case-insensitive exact title match in CONTEXT_PAGES.
    If no exact match is found, falls back to a case-insensitive substring match.

    Args:
        entity: The entity or title name to search for.

    Returns:
        The title and concatenated text of the article if found, or an error listing available titles.
    """
    global CURRENT_PAGE_TITLE, LOOKUP_INDEX

    entity_query = entity.strip().lower()
    matched_title = None

    # 1. Exact match (case-insensitive)
    for title in CONTEXT_PAGES:
        if title.lower() == entity_query:
            matched_title = title
            break

    # 2. Substring match fallback (case-insensitive)
    if matched_title is None:
        for title in CONTEXT_PAGES:
            if entity_query in title.lower():
                matched_title = title
                break

    if matched_title is not None:
        CURRENT_PAGE_TITLE = matched_title
        LOOKUP_INDEX = 0
        sentences = CONTEXT_PAGES[matched_title]
        return f"[{matched_title}] " + " ".join(sentences)
    else:
        sample_titles = list(CONTEXT_PAGES.keys())[:5]
        return f"Error: Could not find '{entity}'. Available titles: " + ", ".join(sample_titles)


@mcp.tool()
def lookup(keyword: str) -> str:
    """Scan the current article for the next sentence containing keyword.

    Scans starting from LOOKUP_INDEX and updates the pointer to the next sentence.

    Args:
        keyword: The keyword to search for within the current article.

    Returns:
        The matched sentence with its 1-indexed number, or a message indicating no more occurrences.
    """
    global LOOKUP_INDEX

    if not CURRENT_PAGE_TITLE:
        return "Error: No article loaded. Call search(entity) first."

    sentences = CONTEXT_PAGES.get(CURRENT_PAGE_TITLE, [])
    keyword_query = keyword.strip().lower()

    for idx in range(LOOKUP_INDEX, len(sentences)):
        sentence = sentences[idx]
        if keyword_query in sentence.lower():
            LOOKUP_INDEX = idx + 1
            return f"(Sentence {idx + 1}) {sentence}"

    return f"No more occurrences of '{keyword}' in '{CURRENT_PAGE_TITLE}'."


@mcp.tool()
def finish(answer: str) -> str:
    """Submit the final answer for the current episode.

    Args:
        answer: The predicted answer string.

    Returns:
        Formatted final answer string.
    """
    return f"FINAL_ANSWER: {answer}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
