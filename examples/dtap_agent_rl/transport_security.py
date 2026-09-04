"""Bounded ASGI transport wrapper for the M4 FastMCP application."""

from __future__ import annotations

import json
from typing import Any

from .security_policy import M4SecurityPolicy


class M4RequestLimitMiddleware:
    def __init__(self, app: Any, *, max_body_bytes: int, max_header_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.max_header_bytes = max_header_bytes

    async def _reject(self, send: Any) -> None:
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "request unavailable"},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers") or []
        if sum(len(name) + len(value) for name, value in headers) > self.max_header_bytes:
            await self._reject(send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            chunk = message.get("body", b"")
            body.extend(chunk)
            if len(body) > self.max_body_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


def build_m4_http_app(server: Any, policy: M4SecurityPolicy) -> M4RequestLimitMiddleware:
    app = server.http_app(transport="http", stateless_http=True)
    return M4RequestLimitMiddleware(
        app,
        max_body_bytes=policy.max_http_body_bytes,
        max_header_bytes=policy.max_http_header_bytes,
    )
