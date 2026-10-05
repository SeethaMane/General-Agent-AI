"""A minimal end-to-end agent: LLM + tools + loop.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python agent.py "What is 17 * 23, and what time is it in UTC?"
    python agent.py            # interactive mode
"""
import ast
import json
import operator
import os
import sys
from datetime import datetime, timezone

import anthropic

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 10  # safety cap on loop iterations
SYSTEM = (
    "You are a helpful agent. Use the provided tools when they help. "
    "When you have the final answer, reply with it plainly."
)

# ---------------------------------------------------------------- tools ----

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.FloorDiv: operator.floordiv,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression (no eval())."""
    return str(_eval(ast.parse(expression, mode="eval").body))


def current_time(timezone_name: str = "UTC") -> str:
    """Current time; only UTC is supported to keep this dependency-free."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_file(path: str) -> str:
    """Read a text file from the current directory tree (first 5000 chars)."""
    full = os.path.realpath(path)
    if not full.startswith(os.path.realpath(os.getcwd())):
        raise PermissionError("path outside working directory")
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        return f.read(5000)


TOOLS = {"calculator": calculator, "current_time": current_time, "read_file": read_file}

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '17 * 23 + 4'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "current_time",
        "description": "Get the current date and time in UTC.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "Read a text file under the current working directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]

# ----------------------------------------------------------- agent loop ----


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Execute a tool; return (output, is_error). Errors go back to the model."""
    try:
        return str(TOOLS[name](**args)), False
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}", True


def run_agent(client: anthropic.Anthropic, messages: list) -> str:
    for step in range(MAX_STEPS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out, is_err = run_tool(block.name, block.input)
                print(f"  [tool] {block.name}({json.dumps(block.input)}) -> {out[:80]}")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": out,
                    "is_error": is_err,
                })
        messages.append({"role": "user", "content": results})
    return "Stopped: reached max steps without a final answer."


def main():
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    messages: list = []
    if len(sys.argv) > 1:
        messages.append({"role": "user", "content": " ".join(sys.argv[1:])})
        print(run_agent(client, messages))
        return
    print("Agent ready. Ctrl-C or empty line to quit.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        messages.append({"role": "user", "content": user})
        print("agent>", run_agent(client, messages))


if __name__ == "__main__":
    main()
