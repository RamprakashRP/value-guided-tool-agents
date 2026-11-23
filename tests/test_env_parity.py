"""HotpotEnv must behave exactly like the MCP server the greedy baselines used."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.rollout_runner import extract_result_text
from agent.search.common import LLM_TOOLS, SYSTEM_PROMPT
from envs.hotpot_env import HotpotEnv

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTEXT = {
    "Scott Derrickson": [
        "Scott Derrickson (born July 16, 1966) is an American director.",
        "He lives in Los Angeles, California.",
        "He is best known for directing horror films.",
    ],
    "Ed Wood": ["Edward Davis Wood Jr. was an American filmmaker.", "Wood was born in New York."],
    "Ed Wood (film)": ["Ed Wood is a 1994 American biographical film."],
}

CALLS = [
    ("lookup", {"keyword": "born"}),
    ("search", {"entity": "scott derrickson"}),
    ("lookup", {"keyword": "born"}),
    ("lookup", {"keyword": "born"}),
    ("lookup", {"keyword": "he"}),
    ("lookup", {"keyword": "he"}),
    ("search", {"entity": "Ed Wood"}),
    ("search", {"entity": "film"}),
    ("lookup", {"keyword": "biographical"}),
    ("search", {"entity": "Tim Burton"}),
    ("lookup", {"keyword": "1994"}),
    ("finish", {"answer": "yes"}),
    ("not_a_tool", {}),
]


async def _mcp_outputs():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-u", str(REPO_ROOT / "envs" / "hotpot_mcp.py")],
        env=os.environ.copy(),
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            await session.call_tool("load_episode", arguments={"context_json": json.dumps(CONTEXT)})
            outputs = []
            for name, args in CALLS:
                outputs.append(extract_result_text(await session.call_tool(name, arguments=args)))
            return tools, outputs


def test_env_matches_mcp_server():
    tools, mcp_outputs = asyncio.run(_mcp_outputs())

    env = HotpotEnv(CONTEXT)
    local_outputs = [env.step(name, args) for name, args in CALLS]
    assert local_outputs == mcp_outputs

    mcp_llm_tools = [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description or "", "parameters": t.input_schema},
        }
        for t in tools.tools
        if t.name in ["search", "lookup", "finish"]
    ]
    assert LLM_TOOLS == mcp_llm_tools


def test_snapshot_restore_forks_state():
    env = HotpotEnv(CONTEXT)
    env.search("Scott Derrickson")
    state = env.snapshot()
    first = env.lookup("he")
    env.search("Ed Wood")
    env.restore(state)
    assert env.lookup("he") == first


def test_system_prompt_matches_rollout_runner():
    tree = ast.parse((REPO_ROOT / "agent" / "rollout_runner.py").read_text())
    prompts = [
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "system_prompt" for t in node.targets)
    ]
    assert prompts == [SYSTEM_PROMPT]
