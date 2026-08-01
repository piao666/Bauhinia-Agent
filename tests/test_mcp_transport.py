"""MCP SDK transport compatibility tests."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager

from bauhinia_agent.mcp import transport
from bauhinia_agent.mcp.models import McpRemoteServerConfig


class _FakeHttpClient:
    async def __aenter__(self) -> _FakeHttpClient:
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


def test_streamable_http_transport_forwards_headers_with_mcp_v2_api(monkeypatch) -> None:
    captured: dict[str, object] = {}
    read_stream = object()
    write_stream = object()

    def fake_create_mcp_http_client(*, headers: dict[str, str]) -> _FakeHttpClient:
        captured["headers"] = headers
        return _FakeHttpClient()

    @asynccontextmanager
    async def fake_streamable_http_client(url: str, *, http_client: _FakeHttpClient):
        captured["url"] = url
        captured["http_client"] = http_client
        yield read_stream, write_stream

    monkeypatch.setattr(transport, "_MCP_STREAMABLE_HTTP_V2", True)
    monkeypatch.setattr(transport, "create_mcp_http_client", fake_create_mcp_http_client)
    monkeypatch.setattr(transport, "streamable_http_client", fake_streamable_http_client)
    client = transport._StreamableHttpMcpTransport(
        McpRemoteServerConfig(
            name="remote",
            url="https://example.test/mcp",
            headers={"Authorization": "Bearer test-token"},
        )
    )

    async def open_streams() -> tuple[object, object]:
        stack = AsyncExitStack()
        try:
            return await client._open_streams(stack)
        finally:
            await stack.aclose()

    assert asyncio.run(open_streams()) == (read_stream, write_stream)
    assert captured["headers"] == {"Authorization": "Bearer test-token"}
    assert captured["url"] == "https://example.test/mcp"
    assert isinstance(captured["http_client"], _FakeHttpClient)
