"""Tests for the tool registry: schema derivation, serialization, execution.

Pure/in-process — no DB, no LLM. Covers the tool lifecycle that the agent
loop and search tools depend on, including the run_tool error-handling we
added for LLM-driven (and therefore untrusted) tool args.
"""

from pydantic import BaseModel

from app.llm.registry import (
    _schema_from_signature,
    _serialize,
    make_run_tool,
    tool,
)

# --- _schema_from_signature -------------------------------------------------


def test_schema_derives_params_and_skips_session() -> None:
    async def search(session: object, query: str, k: int = 5) -> None: ...

    schema = _schema_from_signature(search)
    props = schema["properties"]
    assert isinstance(props, dict)
    assert "query" in props
    assert "k" in props
    assert "session" not in props  # plumbing the model never supplies
    assert schema["required"] == ["query"]  # k has a default


# --- _serialize -------------------------------------------------------------


class _Model(BaseModel):
    x: int


def test_serialize_basemodel() -> None:
    assert _serialize(_Model(x=1)) == '{"x":1}'


def test_serialize_list_of_models() -> None:
    out = _serialize([_Model(x=1), _Model(x=2)])
    assert '"x": 1' in out  # list path goes through json.dumps (spaced)
    assert '"x": 2' in out


def test_serialize_plain_value() -> None:
    assert _serialize("hello") == "hello"


# --- make_run_tool ----------------------------------------------------------


@tool("Echo the text back")
async def _echo(text: str) -> str:
    return f"echoed: {text}"


@tool("Always raises")
async def _boom(text: str) -> str:
    raise ValueError("kaboom")


async def test_run_tool_executes_and_serializes() -> None:
    run = make_run_tool()
    assert await run("_echo", {"text": "hi"}) == "echoed: hi"


async def test_run_tool_unknown_tool() -> None:
    run = make_run_tool()
    assert "Unknown tool" in await run("does_not_exist", {})


async def test_run_tool_catches_tool_errors() -> None:
    # an LLM can send bad args / a tool can throw — must not crash the loop
    run = make_run_tool()
    out = await run("_boom", {"text": "x"})
    assert "failed" in out
    assert "kaboom" in out


async def test_run_tool_injects_context() -> None:
    @tool("Needs a session injected by the caller")
    async def _needs_ctx(session: str, text: str) -> str:
        return f"{session}:{text}"

    run = make_run_tool(session="SESSION")
    assert await run("_needs_ctx", {"text": "hi"}) == "SESSION:hi"
