"""Smoke-test script for HotpotQA MCP server over stdio transport.

Tests end-to-end functionality:
1. Loads sample rows from Hugging Face HotpotQA distractor validation split.
2. Formats context into {title: sentences} JSON.
3. Spawns envs/hotpot_mcp.py via stdio_client and ClientSession.
4. Lists tools, calls load_episode, search, and lookup.
5. Verifies observations.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import datasets
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def extract_result_text(call_result) -> str:
    """Extract plain text from MCP tool call response."""
    if hasattr(call_result, "content"):
        texts = [item.text for item in call_result.content if hasattr(item, "text")]
        if texts:
            return "\n".join(texts)
    return str(call_result)


async def main():
    print("=== 1. Loading HotpotQA validation sample rows ===")
    ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation[:2]")
    print(f"Loaded {len(ds)} sample rows.")

    server_script = Path(__file__).parent / "hotpot_mcp.py"

    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", str(server_script.resolve())],
        env=os.environ.copy(),
    )

    print("\n=== 2. Connecting to HotpotQA MCP Server via stdio ===")
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("ClientSession initialized successfully.")

            # List tools
            tools_response = await session.list_tools()
            tool_names = [t.name for t in tools_response.tools]
            print(f"Available tools ({len(tool_names)}): {tool_names}")
            assert "load_episode" in tool_names
            assert "search" in tool_names
            assert "lookup" in tool_names
            assert "finish" in tool_names

            # Run tests across both sample rows
            for i, row in enumerate(ds):
                question = row["question"]
                gold_answer = row["answer"]
                supporting_titles = list(set(row["supporting_facts"]["title"]))
                gold_title = supporting_titles[0] if supporting_titles else row["context"]["title"][0]

                print(f"\n--- Testing Episode {i + 1} ---")
                print(f"Question: {question}")
                print(f"Gold Answer: {gold_answer}")
                print(f"Gold Target Title: {gold_title}")

                # Format context into {title: sentences} JSON
                titles = row["context"]["title"]
                sentences_list = row["context"]["sentences"]
                context_dict = {
                    title: sentences for title, sentences in zip(titles, sentences_list)
                }
                context_json = json.dumps(context_dict)

                # Call load_episode
                load_res = await session.call_tool(
                    "load_episode", arguments={"context_json": context_json}
                )
                load_text = extract_result_text(load_res)
                print(f"load_episode -> {load_text}")
                assert f"Loaded {len(context_dict)} articles." in load_text

                # Call search on gold title
                search_res = await session.call_tool(
                    "search", arguments={"entity": gold_title}
                )
                search_text = extract_result_text(search_res)
                print(f"search('{gold_title}') -> {search_text[:120]}...")
                assert f"[{gold_title}]" in search_text

                # Pick a keyword from the gold title or answer to test lookup
                lookup_keyword = gold_title.split()[0]
                lookup_res = await session.call_tool(
                    "lookup", arguments={"keyword": lookup_keyword}
                )
                lookup_text = extract_result_text(lookup_res)
                print(f"lookup('{lookup_keyword}') -> {lookup_text}")
                assert "Sentence" in lookup_text or "No more occurrences" in lookup_text

                # Test finish tool
                finish_res = await session.call_tool(
                    "finish", arguments={"answer": gold_answer}
                )
                finish_text = extract_result_text(finish_res)
                print(f"finish('{gold_answer}') -> {finish_text}")
                assert f"FINAL_ANSWER: {gold_answer}" in finish_text

    print("\n[SUCCESS] All HotpotQA MCP Server smoke tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
