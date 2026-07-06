import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from pydantic import BaseModel, create_model


@dataclass
class ToolDefinition:
    name: str
    description: str
    func: Callable[..., object]
    parameters: dict[str, object] = field(default_factory=dict[str, object])


Tools = dict[str, ToolDefinition]

_TOOLS_REGISTRY: Tools = {}


F = TypeVar("F", bound=Callable[..., object])

_SKIP_PARAMS = {"self", "cls", "session"}


def _schema_from_signature(func: Callable[..., object]) -> dict[str, object]:
    empty = inspect.Parameter.empty
    fields: dict[str, Any] = {}
    for pname, p in inspect.signature(func).parameters.items():
        if pname in _SKIP_PARAMS:
            continue
        annotation = p.annotation if p.annotation is not empty else str
        default = ... if p.default is empty else p.default
        fields[pname] = (annotation, default)
    model = create_model(f"{func.__name__}_args", **fields)
    schema: dict[str, object] = model.model_json_schema()
    return schema


def tool(description: str, *, name: str | None = None) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        tool_name = name or func.__name__
        _TOOLS_REGISTRY[tool_name] = ToolDefinition(
            name=tool_name,
            description=description,
            func=func,
            parameters=_schema_from_signature(func),
        )
        return func

    return decorator


def get_tools() -> Tools:
    return dict(_TOOLS_REGISTRY)


def _serialize(result: object) -> str:
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    if isinstance(result, list):
        seq = cast(list[object], result)
        items = [
            r.model_dump(mode="json") if isinstance(r, BaseModel) else r for r in seq
        ]
        return json.dumps(items, default=str)
    return str(result)


def make_run_tool(
    *,
    on_result: Callable[[str, object], None] | None = None,
    audit: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    **context: object,
) -> Callable[[str, Mapping[str, Any]], Awaitable[str]]:
    async def run_tool(name: str, args: Mapping[str, Any]) -> str:
        td = _TOOLS_REGISTRY.get(name)
        if td is None:
            return f"Unknown tool: {name}"
        try:
            result = await cast(Awaitable[object], td.func(**context, **args))
        except Exception as exc:
            if audit is not None:
                await audit(
                    f"tool {name!r} failed",
                    {"tool": name, "args": dict(args), "error": str(exc)},
                )
            return f"Tool {name!r} failed: {exc}"
        # let the caller observe raw results (e.g. to collect citations)
        if on_result is not None:
            on_result(name, result)
        serialized = _serialize(result)
        if audit is not None:
            # NOTE: result may hold document text (PHI) — redact/truncate for real prod
            await audit(
                f"tool {name!r}",
                {"tool": name, "args": dict(args), "result": serialized},
            )
        return serialized

    return run_tool
