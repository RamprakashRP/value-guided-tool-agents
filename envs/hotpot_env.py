"""In-process HotpotQA environment with snapshot/restore for tree search.

Mirrors the tool semantics and output strings of envs/hotpot_mcp.py exactly,
but keeps state on an instance so search branches can fork it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

EnvState = Tuple[str, int]


class HotpotEnv:
    """HotpotQA search/lookup/finish tools over one episode's context pages."""

    def __init__(self, context: Dict[str, List[str]]):
        self.pages: Dict[str, List[str]] = {
            str(title): [str(s) for s in sentences] for title, sentences in context.items()
        }
        self.current_title: str = ""
        self.lookup_index: int = 0

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> HotpotEnv:
        """Build from a Hugging Face HotpotQA distractor row."""
        ctx = row.get("context", {})
        return cls(dict(zip(ctx.get("title", []), ctx.get("sentences", []))))

    def snapshot(self) -> EnvState:
        return (self.current_title, self.lookup_index)

    def restore(self, state: EnvState) -> None:
        self.current_title, self.lookup_index = state

    def search(self, entity: str) -> str:
        entity_query = entity.strip().lower()
        matched_title = None
        for title in self.pages:
            if title.lower() == entity_query:
                matched_title = title
                break
        if matched_title is None:
            for title in self.pages:
                if entity_query in title.lower():
                    matched_title = title
                    break

        if matched_title is not None:
            self.current_title = matched_title
            self.lookup_index = 0
            return f"[{matched_title}] " + " ".join(self.pages[matched_title])
        sample_titles = list(self.pages.keys())[:5]
        return f"Error: Could not find '{entity}'. Available titles: " + ", ".join(sample_titles)

    def lookup(self, keyword: str) -> str:
        if not self.current_title:
            return "Error: No article loaded. Call search(entity) first."
        sentences = self.pages.get(self.current_title, [])
        keyword_query = keyword.strip().lower()
        for idx in range(self.lookup_index, len(sentences)):
            if keyword_query in sentences[idx].lower():
                self.lookup_index = idx + 1
                return f"(Sentence {idx + 1}) {sentences[idx]}"
        return f"No more occurrences of '{keyword}' in '{self.current_title}'."

    def finish(self, answer: str) -> str:
        return f"FINAL_ANSWER: {answer}"

    def step(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call; unknown tools return the MCP client's error text."""
        if tool_name == "search":
            return self.search(str(arguments.get("entity", "")))
        if tool_name == "lookup":
            return self.lookup(str(arguments.get("keyword", "")))
        if tool_name == "finish":
            return self.finish(str(arguments.get("answer", "")))
        return f"Unknown tool: {tool_name}"
