"""Supervisor-side MCP tools for the ``a2a_completion_notifier`` bridge.

Why this exists
---------------
At supervisor launch (``create_supervisor_agent``), langstrata attaches the
A2A bridge's four diagnostic tools as **native LangChain tools** on the
supervisor deep agent.  The tool schemas are fixed, so no discovery round-trip
is needed at graph-build time: ``discover_supervisor_mcp_tools`` returns
``StructuredTool`` wrappers that are discovered eagerly at boot (or lazily on
first invocation) via LangChain's ``langchain.mcp.MCPAdapter`` (the
reference-supported MCP client).  Each wrapper opens + closes the MCP client
per invocation -- the adapter is used only for discovery, never held across
runs (per ``as_langchain_tool``'s reentrancy contract).

The supervisor can inspect its own notification mailbox / dead-letter queue
and fetch full subagent results (``get_async_result``) without polling.

How the receiver comes up
-------------------------
**Webhook (default):** embedded on the supervisor via ``langgraph.json``
``http.app`` (``langstrata.a2a.webapp``).  No separate process; workers POST
to the supervisor origin (``A2A_RECEIVER_URL``).  This is the only piece
required for the push loop itself -- everything below is optional.

**MCP diagnostic tools** (optional, off by default): the four tools above
are hosted by a *separate* MCP bridge process
(``python -m langshark_bites.a2a_completion_notifier.mcp_server``,
streamable-http transport) that you run yourself and point the supervisor at
via ``A2A_SUPERVISOR_MCP_URL``::

    export A2A_SUPERVISOR_MCP_URL=http://localhost:8001/mcp

Without ``A2A_SUPERVISOR_MCP_URL`` set, ``discover_supervisor_mcp_tools``
returns no tools -- the webhook keeps working, the supervisor just cannot
call these four diagnostics directly.

Why no blocking calls here
--------------------------
``langgraph dev`` runs graph factories inside its ASGI event loop and rejects
synchronous blocking calls (``blockbuster``).  The MCP discovery in this
module happens off the loop (``asyncio.to_thread``), and the factory itself
only builds static tools -- it never blocks.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import warnings
from typing import Any

from langchain_core._api import LangChainBetaWarning
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# The four diagnostic tools exposed by the A2A bridge.
SUPERVISOR_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "list_pending_notifications",
        "dead_letter_queue",
        "receiver_health",
        "get_async_result",
    }
)

# MCP target, fixed by discover_supervisor_mcp_tools (a streamable-http URL
# pointing at an externally-run MCP bridge process, e.g.
# ``langshark_bites.a2a_completion_notifier.mcp_server``).
_TARGET: Any | None = None

# Connection state shared by the wrapper tools: the MCP client is discovered
# lazily on first tool call, its LangChain tools are cached, and the client is
# then CLOSED again.  Each ``StructuredTool`` returned by
# ``langchain.mcp.as_langchain_tool`` holds the client and reconnects per
# invocation (fastmcp reentrancy counter 0→1→0), so keeping the adapter open
# for the process lifetime would leave the counter stuck and crash the second
# tool call ("nesting counter should be 0").  The adapter is therefore never
# stored -- only the discovered tools.  No locks: concurrent first-tool-call
# races are benign in dev.
_CONNECTION: dict[str, Any] = {
    "tools": {},
}

# One-shot guard so the "ready" / "skip" log lines are emitted once per
# process rather than on every factory call (one per run / state read).
_READY_LOGGED = False


class _ListPendingArgs(BaseModel):
    """Args for ``list_pending_notifications``."""

    thread_id: str = Field(description="Supervisor thread to inspect")


class _ResultArgs(BaseModel):
    """Args for ``get_async_result``."""

    task_id: str = Field(description="Subagent task id (the worker thread id)")


class _NoArgs(BaseModel):
    """A tool with no arguments."""


_TOOL_SPECS: list[tuple[str, type[BaseModel], str]] = [
    (
        "list_pending_notifications",
        _ListPendingArgs,
        "List completion notifications waiting to be drained for a thread. Args: thread_id.",
    ),
    (
        "dead_letter_queue",
        _NoArgs,
        "Inspect undelivered completions parked in the dead-letter queue (needs redis wired).",
    ),
    (
        "receiver_health",
        _NoArgs,
        "Report A2A receiver liveness and configuration status.",
    ),
    (
        "get_async_result",
        _ResultArgs,
        "Fetch the full result of a completed subagent task (call once, "
        "after its completion notice). Args: task_id.",
    ),
]


warnings.filterwarnings("ignore", category=LangChainBetaWarning)


async def _invoke_tool(tool_name: str, args: dict[str, Any]) -> Any:  # noqa: ANN401 - tool results are arbitrary MCP JSON
    """Invoke one bridge tool through its cached LangChain wrapper.

    The wrapper was built by ``langchain.mcp.as_langchain_tool`` during
    discovery, so each invocation opens + closes the fastmcp client itself
    (reentrancy counter 0→1→0) -- there is no persistent connection here at
    all.  Tools are discovered lazily (in a background thread by default, see
    ``_ensure_connection``), so no blocking filesystem scan ever runs inside
    the LangGraph event loop; ``langgraph dev``'s blockbuster therefore never
    fires on the MCP connect path.
    """
    await _ensure_connection(tool_name)
    try:
        return await _CONNECTION["tools"][tool_name].ainvoke(args or {})
    except Exception:
        log.warning("a2a_mcp_tool_call_failed tool=%s", tool_name, exc_info=True)
        raise


def _has_connection() -> bool:
    # bool(): _CONNECTION values are Any (untyped bridge boundary), and mypy
    # strict disallows returning Any from a -> bool function.
    return bool(_CONNECTION["tools"])


def _connect_sync() -> None:
    """Discover the A2A bridge tools, off the event loop.

    Runs in a plain thread (blocking allowed); used as the lazy fallback so a
    tool call never blocks the loop.

    The adapter is opened only for discovery and then closed again: the bite's
    ``create_supervisor_tools`` helper (the regression-tested canonical consumer
    path) opens it, discovers the tools, and closes it.  Each discovered
    ``StructuredTool`` keeps the client and reconnects per invocation (fastmcp
    reentrancy counter 0-1-0).  Holding the adapter open across runs would leave
    the counter stuck and crash the second tool call ("nesting counter should be
    0"), so the adapter is never stored -- only its tools.
    """
    # Lazy import on purpose: runs off the event loop in a plain thread (see
    # "Why no blocking calls here"), so langgraph dev's blockbuster never sees
    # the bite's jsonschema/httpx2 pre-connect filesystem imports.
    from langshark_bites.a2a_completion_notifier.supervisor_tools import (  # noqa: PLC0415 - off-loop lazy import
        create_supervisor_tools,
    )

    async def _open() -> None:
        if _has_connection():
            return
        tools = await create_supervisor_tools(_TARGET)
        _CONNECTION["tools"] = {t.name: t for t in tools}
        log.info("a2a_mcp_connection_open target=%s", _target_label(_TARGET))

    asyncio.run(_open())


async def _ensure_connection(tool_name: str) -> None:  # noqa: ARG001 - reserved for per-tool connect; discovery is global
    """Ensure the A2A bridge tools are discovered before invoking one.

    Prefers already-discovered tools; otherwise discovers them in a
    blocking-capable worker thread (never the LangGraph event loop) so the
    blockbuster guard is not tripped by MCPAdapter's internal filesystem
    scanning.  Idempotent via the ``_has_connection()`` re-check: concurrent
    first-callers may each discover the tools (benign last-writer-wins in dev).
    """
    if _has_connection():
        return
    await asyncio.to_thread(_connect_sync)


async def _invoke_with_schema(tool_name: str, **kwargs: Any) -> Any:  # noqa: ANN401 - tool args are arbitrary JSON (deepagents/langchain contract)
    """Async ``coroutine`` entry for the ``StructuredTool`` wrapper."""
    return await _invoke_tool(tool_name, dict(kwargs))


def _target_label(target: Any) -> str:  # noqa: ANN401 - URL string or mcpServers dict, both accepted
    """Short human label for the MCP target (URL or ``mcpServers`` config)."""
    if isinstance(target, str):
        return target
    try:
        entry = next(iter(target["mcpServers"].values()))
        return f"stdio {entry['command']} {' '.join(entry['args'])}"
    except Exception:
        return str(target)


def _static_tools() -> list[Any]:
    """The four bridge tools as ``StructuredTool`` wrappers with fixed schemas."""
    return [
        StructuredTool.from_function(
            coroutine=functools.partial(_invoke_with_schema, name),
            name=name,
            description=description,
            args_schema=schema,
        )
        for name, schema, description in _TOOL_SPECS
    ]


def discover_supervisor_mcp_tools() -> list[Any]:
    """Build the A2A bridge tools as native LangChain tools.

    Called once from ``create_supervisor_agent`` at supervisor launch.  It is
    side-effect free -- no subprocess is ever spawned here -- so it never
    blocks ``langgraph dev``'s event loop.

    These four tools are purely diagnostic: the A2A push loop's webhook is
    always embedded on the supervisor via ``http.app`` regardless of this
    function's result.  To attach the tools, run the MCP bridge yourself and
    point the supervisor at it::

        just a2a-mcp-receiver-http
        export A2A_SUPERVISOR_MCP_URL=http://localhost:8001/mcp

    Returns:
        Four ``StructuredTool`` wrappers (the bridge's diagnostic tools) when
        A2A is enabled and ``A2A_SUPERVISOR_MCP_URL`` is configured; ``[]``
        (with a log) otherwise.
    """
    # one-shot module memoization: resolved once at supervisor launch
    global _TARGET, _READY_LOGGED  # noqa: PLW0603

    if os.environ.get("A2A_COMPLETION_NOTIFIER_ENABLED", "").strip().lower() != "true":
        return []

    explicit_mcp = os.environ.get("A2A_SUPERVISOR_MCP_URL", "").strip()
    if not explicit_mcp:
        # Embedded default: webhook on supervisor http.app, no MCP tools.
        if not _READY_LOGGED:
            _READY_LOGGED = True
            log.info(
                "a2a_mcp_skip_embedded_http_app hint=set_A2A_SUPERVISOR_MCP_URL "
                "and run just a2a-mcp-receiver-http to attach diagnostic tools"
            )
        return []
    _TARGET = explicit_mcp

    tools = _static_tools()
    if not _READY_LOGGED:
        _READY_LOGGED = True
        log.info(
            "a2a_mcp_ready transport=streamable-http receiver=%s tools=%s",
            _target_label(_TARGET),
            sorted(SUPERVISOR_MCP_TOOLS),
        )
    return tools
