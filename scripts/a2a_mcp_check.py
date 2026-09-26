#!/usr/bin/env python
"""Verify the A2A completion receiver (embedded or external).

Checks, in order:
1. ``GET /health`` on ``A2A_RECEIVER_URL`` (default: supervisor origin
   ``http://localhost:2024`` when the receiver is embedded via ``http.app``).
2. If ``A2A_SUPERVISOR_MCP_URL`` is set (or a sidecar MCP bridge is up), the
   four diagnostic tools via the bite's ``create_supervisor_tools``.
3. ``receiver_health`` twice (reconnect-per-call guard) when MCP is configured.

Run:      just a2a-mcp-check
Requires: supervisor running with A2A enabled (embedded webhook).  MCP tool
          checks are skipped when no bridge URL is configured.  Exits 0 when
          the webhook is healthy (and MCP checks pass if requested).
"""

from __future__ import annotations

import asyncio
import os
import sys
import warnings

from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

EXPECTED_TOOLS = {
    "list_pending_notifications",
    "dead_letter_queue",
    "receiver_health",
    "get_async_result",
}

# HTTP status a live ``/health`` endpoint answers with.
HTTP_OK = 200


def _fail(message: str) -> int:
    """Print a FAIL line and return the script's non-zero exit code."""
    print(f"[a2a-mcp-check] FAIL {message}")
    return 1


async def main() -> int:  # noqa: PLR0911 - sequential fail-fast checks
    """Run the live checks; return 0 (healthy) or 1 (any check failed)."""
    import httpx  # noqa: PLC0415 - lazy: health probe works even without the MCP stack
    from langshark_bites.a2a_completion_notifier.supervisor_tools import (  # noqa: PLC0415 - lazy: only needed for steps 2-3
        create_supervisor_tools,
    )

    receiver = os.environ.get("A2A_RECEIVER_URL", "http://localhost:2024").rstrip("/")
    mcp_url = os.environ.get("A2A_SUPERVISOR_MCP_URL", "").strip()

    # 1. HTTP liveness probe (embedded http.app or external sidecar).
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{receiver}/health")
        if resp.status_code != HTTP_OK:
            return _fail(f"/health -> HTTP {resp.status_code}")
        print(f"[a2a-mcp-check] /health OK -> {resp.json()}")
    except Exception as exc:
        return _fail(f"/health unreachable: {exc}")

    if not mcp_url:
        print(
            "[a2a-mcp-check] MCP tools skipped "
            "(set A2A_SUPERVISOR_MCP_URL to exercise the bridge)"
        )
        print("[a2a-mcp-check] OK (webhook healthy; embedded http.app path)")
        return 0

    # 2 + 3. Tools over MCP, and live receiver_health calls.  The helper uses
    # a short discovery scope; the returned tools reconnect per invocation, so
    # the second call here is the exact "nesting counter should be 0" guard.
    try:
        tools = await create_supervisor_tools(mcp_url)
        names = {tool.name for tool in tools}
        if not names >= EXPECTED_TOOLS:
            return _fail(f"unexpected tools: {sorted(names)}")
        health_tool = next(t for t in tools if t.name == "receiver_health")
        report = str(await health_tool.ainvoke({}))
        report_again = str(await health_tool.ainvoke({}))  # reconnect proof
    except Exception as exc:
        return _fail(f"MCP connect/call: {exc}")

    for label, probe in (("first", report), ("second", report_again)):
        if "status" not in probe or "ok" not in probe:
            return _fail(f"{label} receiver_health: {probe[:160]}")

    print(f"[a2a-mcp-check] MCP OK -> {len(tools)} tools: {', '.join(sorted(names))}")
    print(f"[a2a-mcp-check] receiver_health -> {report[:120]}...")
    print("[a2a-mcp-check] second receiver_health OK (tools reconnect per call)")
    print("A2A MCP receiver is functioning correctly.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(0)
