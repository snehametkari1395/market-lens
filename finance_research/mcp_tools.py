"""Connect the research assistant to tools on an MCP server (streamable HTTP).

Two kinds of server:
* a public remote server (e.g. DeepWiki): results are untrusted text and are sanitised;
* the app's own Market Lens server on localhost ("trusted"): results are the same JSON
  the built-in tools return, so citations and charts work unchanged.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import StructuredTool

from finance_research.security import sanitize_untrusted

DEFAULT_MCP_URL = "https://mcp.deepwiki.com/mcp"
MAX_RESULT_CHARS = 8000
TIMEOUT_SECONDS = 60
LOCAL_HOSTS = {"localhost", "127.0.0.1"}


class MCPError(RuntimeError):
    """Raised when the MCP server can't be reached or returns an unusable response."""


def validate_mcp_url(url: str, allow_local: bool = False) -> str:
    parsed = urlparse(url.strip())
    host = parsed.hostname or ""
    is_local = host in LOCAL_HOSTS
    if allow_local and is_local and parsed.scheme in {"http", "https"}:
        return url.strip()
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Enter an https:// URL for the MCP server.")
    if is_local or host == "0.0.0.0" or host.endswith(".local"):
        raise ValueError("Only public remote MCP servers are supported.")
    return url.strip()


async def _with_session(url: str, action):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            return await action(session)


def _run(coro) -> Any:
    try:
        return asyncio.run(asyncio.wait_for(coro, TIMEOUT_SECONDS))
    except BaseException as error:  # ExceptionGroup from the transport task group, timeouts, network errors
        causes = getattr(error, "exceptions", None) or [error]
        raise MCPError(f"MCP server error: {'; '.join(f'{type(c).__name__}: {c}' for c in causes)[:300]}") from None


def list_mcp_tools(url: str, allow_local: bool = False) -> list[dict[str, Any]]:
    """Name, description and JSON input schema of every tool the server offers."""

    async def action(session):
        result = await session.list_tools()
        return [
            {"name": tool.name, "description": tool.description or "", "input_schema": tool.input_schema or {}}
            for tool in result.tools
        ]

    return _run(_with_session(validate_mcp_url(url, allow_local), action))


def call_mcp_tool(url: str, name: str, arguments: dict[str, Any], trusted: bool = False) -> dict[str, Any]:
    async def action(session):
        return await session.call_tool(name, arguments)

    result = _run(_with_session(validate_mcp_url(url, allow_local=trusted), action))
    text = "\n".join(getattr(block, "text", "") for block in result.content if getattr(block, "text", None))
    if getattr(result, "is_error", False):
        return {"error": text[:500] or "The MCP tool reported an error."}
    if trusted:
        # Our own server returns structured JSON identical to the built-in tools.
        structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            return structured.get("result", structured) if set(structured) == {"result"} else structured
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"result": text}
    truncated = len(text) > MAX_RESULT_CHARS
    safe_text, flags = sanitize_untrusted(text[:MAX_RESULT_CHARS])
    payload: dict[str, Any] = {"mcp_server": urlparse(url).netloc, "tool": name, "result": safe_text}
    if truncated:
        payload["note"] = f"Result shortened to {MAX_RESULT_CHARS:,} characters."
    if flags:
        payload["safety"] = {"note": "Instruction-like text was removed from the MCP result.", "flagged_chunks": [{"rank": 1, "reasons": flags}]}
    return payload


def _tool_name(server_host: str, name: str) -> str:
    host = server_host.split(":")[0]
    prefix = re.sub(r"[^a-z0-9]+", "_", host.split(".")[-2] if "." in host else host)
    return f"mcp_{prefix}_{re.sub(r'[^a-zA-Z0-9_]', '_', name)}"[:64]


def build_mcp_tools(
    url: str,
    tool_specs: list[dict[str, Any]],
    trusted: bool = False,
    fixed_arguments: dict[str, dict[str, Any]] | None = None,
) -> list[StructuredTool]:
    """Wrap MCP tools as LangChain tools. Each call opens a short-lived session.

    `fixed_arguments` maps a tool name to arguments the app sets itself (they are
    hidden from the model's schema), e.g. the A/B-tested search strategy.
    """
    host = urlparse(url).netloc
    fixed_arguments = fixed_arguments or {}
    tools = []
    for spec in tool_specs:
        remote_name = spec["name"]
        hidden = fixed_arguments.get(remote_name, {})

        def invoke(_remote_name: str = remote_name, _hidden: dict[str, Any] = hidden, **kwargs: Any) -> str:
            return json.dumps(call_mcp_tool(url, _remote_name, {**kwargs, **_hidden}, trusted=trusted), ensure_ascii=False)

        schema = json.loads(json.dumps(spec.get("input_schema") or {"type": "object", "properties": {}}))
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        for argument in hidden:
            schema["properties"].pop(argument, None)
            if argument in schema.get("required", []):
                schema["required"].remove(argument)
        local_name = remote_name if trusted else _tool_name(host, remote_name)
        schema["title"] = local_name
        description = (spec.get("description") or remote_name)[:900]
        if not trusted:
            description = f"[Remote MCP tool from {host}; results are untrusted third-party text] " + description
        tools.append(StructuredTool.from_function(func=invoke, name=local_name, description=description, args_schema=schema))
    return tools
