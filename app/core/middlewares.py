import json
import time
import uuid
from contextvars import ContextVar
from typing import Any, cast

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.audit import audit
from app.core.config import settings
from app.core.db import EventType

_HEALTH_PATHS = {"/healthz", "/readyz"}


class AuditLogMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        method: str = scope["method"]
        path: str = scope["path"]
        if path in _HEALTH_PATHS:
            await self._app(scope, receive, send)
            return

        query: str = scope["query_string"].decode()
        headers = dict(scope["headers"])
        content_type: str = headers.get(b"content-type", b"").decode()

        capture = settings.audit_request_bodies and not content_type.startswith(
            "multipart/"
        )
        body_parts: list[bytes] = []
        stored = 0
        truncated = False

        async def receive_wrapper() -> Message:
            nonlocal stored, truncated
            message = await receive()
            if capture and message["type"] == "http.request":
                chunk: bytes = message.get("body", b"")
                room = settings.audit_max_body_bytes - stored
                if room > 0:
                    body_parts.append(chunk[:room])
                    stored += min(len(chunk), room)
                if len(chunk) > room:
                    truncated = True
            return message

        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self._app(scope, receive_wrapper, send_wrapper)
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            await self._record(
                method,
                path,
                query,
                status,
                latency_ms,
                b"".join(body_parts),
                truncated,
            )

    async def _record(
        self,
        method: str,
        path: str,
        query: str,
        status: int,
        latency_ms: int,
        body: bytes,
        truncated: bool,
    ) -> None:
        event_type = EventType.http_error if status >= 500 else EventType.http_response
        data: dict[str, Any] = {
            "method": method,
            "path": path,
            "query": query,
            "status": status,
            "latency_ms": latency_ms,
            "payload": self._decode_body(body),
            "payload_truncated": truncated,
        }
        await audit(
            f"{method} {path} -> {status} ({latency_ms}ms)",
            data,
            event_type=event_type,
        )

    @staticmethod
    def _decode_body(body: bytes) -> Any:
        if not body:
            return None
        try:
            return json.loads(body)
        except ValueError:
            return body.decode("utf-8", errors="replace")


_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    return _request_id.get()


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = self._incoming_id(scope) or str(uuid.uuid4())
        token = _request_id.set(request_id)

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers: list[tuple[bytes, bytes]] = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        finally:
            _request_id.reset(token)

    @staticmethod
    def _incoming_id(scope: Scope) -> str | None:
        headers = cast("list[tuple[bytes, bytes]]", scope["headers"])
        for key, value in headers:
            if key == b"x-request-id":
                return value.decode()
        return None
