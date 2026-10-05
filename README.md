# Simple Agent

A minimal end-to-end LLM agent: model + tools + a loop.

## How it works

1. Send the conversation and tool schemas to the model.
2. If `stop_reason == "tool_use"`, run each requested tool and append the results as a `tool_result` message.
3. Repeat until the model replies without a tool call (or `MAX_STEPS` is hit).

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python agent.py "What is 17 * 23, and what time is it in UTC?"   # one-shot
python agent.py                                                   # interactive chat
```

Set `AGENT_MODEL` to use a different model (default: `claude-sonnet-5-5`).

## Built-in tools

| Tool | Purpose |
|---|---|
| `calculator` | Safe arithmetic (AST-based, no `eval`) |
| `current_time` | Current UTC time |
| `read_file` | Read a text file under the working directory |

## Adding a tool

1. Write a Python function that takes keyword args and returns a string.
2. Add it to the `TOOLS` dict.
3. Add a matching JSON schema to `TOOL_SCHEMAS`.

## Notes

- Tool errors are returned to the model with `is_error: true` so it can recover.
- `MAX_STEPS` prevents runaway loops.
- `read_file` is restricted to the working directory.
