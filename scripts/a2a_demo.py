#!/usr/bin/env python
"""Live A2A completion-notifier demo — langstrata x langshark-bites.

Drives the real split-server stack (supervisor + worker + receiver) to prove
the full A2A push loop end-to-end:

    supervisor (deepagents + dispatch-config shim)
        -> dispatches a worker with an a2a_push_config
        -> the worker graph factory attaches the emitter middleware
        -> on terminal state the emitter POSTs a signed A2A completion webhook
        -> the receiver verifies + deduplicates + writes the mailbox
        -> the supervisor's MailboxDrainMiddleware injects the notice into
           the next model call

This script REQUIRES the langgraph dev servers already running:

    just supervisor    # terminal 1 (supervisor on :2024)
    just worker        # terminal 2 (worker on :2025)

The A2A completion webhook is embedded on the supervisor itself (see
``langstrata.a2a.webapp``). This script never starts a server; it is
a *client* that talks to the
running stack via the LangGraph SDK and streams the supervisor's turns
until the completion notice is drained.

Run:
    uv run python scripts/a2a_demo.pys

Exit codes: 0 when the a2a loop drained at least one completion notice into
the supervisor; non-zero otherwise (with a diagnostic).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from langchain_core._api import LangChainBetaWarning
from langgraph_sdk import get_client
from langgraph_sdk._async.client import LangGraphClient
from langgraph_sdk.errors import LangGraphError
from langshark_bites.a2a_completion_notifier.push_config import build_push_config
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree

from langstrata.a2a.supervisor_mcp import SUPERVISOR_MCP_TOOLS

# langchain.mcp (MCPAdapter) is in beta; silence its one-per-process warning.
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

CALLBACK_SECRET = "langstrata-demo-secret-0123456789abcdef"

# HTTP status a live ``/health`` probe answers with.
HTTP_OK = 200
_TERMINAL_NODE_IDS = frozenset({"__start__", "__end__"})
_WORKER_GRAPH_IDS = frozenset({"researcher", "coder", "analyst"})

# Seconds between "still waiting" heartbeats, and the overall wake-up budget.
_HEARTBEAT_SECONDS = 15.0
_WAKE_TIMEOUT_SECONDS = 90.0

log = logging.getLogger(__name__)

console = Console()


def _demo(msg: str) -> None:
    """Demo-generated status (not from any model): dim, prefixed ``demo``."""
    console.print(f"[dim]· demo[/dim] {msg}")


def _a2a_ok(msg: str) -> None:
    """A2A loop success marker (the demo's verification signal)."""
    console.print(f"[bold green]✓ a2a[/bold green] {msg}")


def _warn(msg: str) -> None:
    """Warning / diagnostic (demo-generated)."""
    console.print(f"[bold yellow]⚠[/bold yellow] {msg}")


async def _diagnose_wake(client: LangGraphClient, thread_id: str) -> None:
    """Explain why the a2a wake-up turn never surfaced (delivery vs drain)."""
    mailbox_items: list = []
    with contextlib.suppress(LangGraphError, httpx.HTTPError):
        _resp = await client.store.search_items(("notifications", thread_id))
        mailbox_items = _resp.get("items", []) if isinstance(_resp, dict) else list(_resp or [])

    runs: list = []
    with contextlib.suppress(LangGraphError, httpx.HTTPError):
        runs = await client.runs.list(thread_id, limit=10)

    console.print()
    if mailbox_items:
        _warn(
            f"the completion notice IS in the supervisor Store mailbox "
            f"({len(mailbox_items)} item(s)) but no wake-up turn surfaced it — "
            f"the MailboxDrainMiddleware needs Store access on the supervisor graph"
        )
    else:
        _warn(
            "the supervisor Store mailbox is EMPTY — the completion "
            "notification never reached the receiver; check for `a2a_emitted` "
            "on the worker server and `a2a_mailbox_write` on the receiver"
        )
    if runs:
        console.print(
            f"  [dim]runs on thread   : {[(r['run_id'][:8], r['status']) for r in runs]}[/dim]"
        )


def _worker_url_from_supervisor_url(supervisor_url: str) -> str:
    """Derive the worker server URL from the supervisor URL (port + 1)."""
    parsed = urlparse(supervisor_url)
    try:
        port = int(parsed.port) if parsed.port else 2024
    except ValueError:
        port = 2024
    return f"{parsed.scheme}://{parsed.hostname}:{port + 1}"


def _classify_agent_type(nodes: list[str]) -> str:
    """Classify an agent type from its node names (mirrors ``scripts/demo.py``)."""
    for n in nodes:
        if "PatchToolCallsMiddleware" in n or "before_agent" in n:
            return "DeepAgent"
    return "Standard agent"


async def _detect_mode(supervisor_url: str) -> str:
    """Detect whether workers are co-deployed (ASGI) or on a separate server (HTTP).

    Derives the worker server port from the supervisor port by adding 1
    (e.g. 2024→2025, 8123→8124).  If the worker server is reachable and
    returns assistants, returns ``"http"``; otherwise ``"asgi"``.
    """
    worker_url = _worker_url_from_supervisor_url(supervisor_url)
    try:
        worker = get_client(url=worker_url)
        wa = await worker.assistants.search()
        if wa:
            return "http"
    except (LangGraphError, httpx.HTTPError) as exc:
        log.debug("worker server unreachable for mode detection: %s", exc)
    return "asgi"


async def _add_graph_type_line(
    branch: Tree,
    client: LangGraphClient,
    assistant_id: str,
    *,
    xray: bool = False,
) -> None:
    """Add a ``type:`` child to ``branch`` from the assistant's graph topology."""
    try:
        detail = await client.assistants.get_graph(assistant_id, xray=xray)
        nodes = [
            n["id"] for n in detail.get("nodes", []) if n["id"] not in ("__start__", "__end__")
        ]
        branch.add(f"[dim]type:[/dim] {_classify_agent_type(nodes)}")
    except Exception as exc:
        log.debug("graph topology unavailable for %s: %s", assistant_id, exc)


def _partition_assistants(
    assistants: list,
) -> tuple[dict | None, dict[str, dict], list[dict]]:
    """Split assistants into supervisor, named workers, and everything else."""
    supervisor = None
    workers: dict[str, dict] = {}
    others: list[dict] = []
    for a in assistants:
        gid = a.get("graph_id", "?")
        if gid == "supervisor":
            supervisor = a
        elif gid in _WORKER_GRAPH_IDS:
            workers[gid] = a
        else:
            others.append(a)
    return supervisor, workers, others


def _deploy_label(mode: str, supervisor_url: str, worker_count: int) -> str:
    """Rich label describing how the supervisor reaches its workers."""
    if mode == "http":
        worker_url = _worker_url_from_supervisor_url(supervisor_url)
        return f"[yellow]⚡ delegated to workers (separate server, {worker_url})[/yellow]"
    return f"[yellow]⚡ delegates to ({worker_count} workers, co-deployed)[/yellow]"


async def _add_worker_branches(
    parent: Tree,
    client: LangGraphClient,
    workers: dict[str, dict],
    mode: str,
    supervisor_url: str,
) -> None:
    """Attach worker sub-branches under the supervisor branch."""
    sub_branch = parent.add(_deploy_label(mode, supervisor_url, len(workers)))
    for gid, a in workers.items():
        wid = a.get("assistant_id", "")
        w_branch = sub_branch.add(f"[cyan]{gid}[/cyan]  (id: {wid[:12]})")
        await _add_graph_type_line(w_branch, client, wid)


async def _print_graph_structure(client: LangGraphClient, supervisor_url: str) -> None:
    """Fetch registered assistants and display the graph topology (like run-demo)."""
    console.print(Rule("[bold]Graph Structure[/bold]"))
    mode = await _detect_mode(supervisor_url)

    try:
        assistants = await client.assistants.search()
    except Exception as e:
        console.print(f"  [dim]Could not fetch structure: {e}[/dim]")
        console.print()
        return

    if not assistants:
        console.print("  [dim]No assistants registered on this server.[/dim]")
        console.print()
        return

    supervisor, workers, others = _partition_assistants(assistants)
    tree = Tree("[bold]Graph topology[/bold]")

    if supervisor:
        sup_id = supervisor.get("assistant_id", "")
        model = os.environ.get("AGENT_SERVER_MODEL", "default")
        sup_branch = tree.add(f"[cyan]⭐ supervisor[/cyan]  (model: [green]{model}[/green])")
        await _add_graph_type_line(sup_branch, client, sup_id, xray=True)
        if workers:
            await _add_worker_branches(sup_branch, client, workers, mode, supervisor_url)
        for a in others:
            gid = a.get("graph_id", "?")
            aid = a.get("assistant_id", "")[:12]
            others_branch = tree.add(f"[dim]{gid}[/dim]  (id: {aid})")
            await _add_graph_type_line(others_branch, client, a.get("assistant_id", ""))
    else:
        for a in assistants:
            tree.add(f"[dim]{a.get('graph_id', '?')}[/dim]")

    console.print(tree)
    console.print()


def _extract_text(content: object) -> str:
    """Extract plain text from a message content block or string."""
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return content.strip() if isinstance(content, str) else ""


_NOTICE_HEADER = "[SUBAGENT COMPLETION NOTICE]"


async def _notice_blocks_in_history(client: LangGraphClient, thread_id: str) -> list[dict]:
    """Find the notifier's static completion markers in the prompt history.

    ``MailboxDrainMiddleware`` injects a ``HumanMessage`` whose content is the
    ``render_notice`` block (``[SUBAGENT COMPLETION NOTICE]`` + ``task_id:`` /
    ``status:`` / ``summary:``) -- the bite's defined completion marker.  Read
    the thread's checkpoint history via the SDK and return every message whose
    content carries that static header.
    """
    history = await client.threads.get_history(thread_id, limit=50)
    latest = history[0] if history else {}
    values = latest.get("values", {})
    messages = values.get("messages", []) if isinstance(values, dict) else []
    return [
        m
        for m in messages
        if isinstance(m, dict)
        and m.get("type") == "human"
        and str(m.get("content", "")).startswith(_NOTICE_HEADER)
    ]


@dataclass
class _RunStats:
    """Live counters gathered while streaming the supervisor's runs."""

    launched: set[str] = field(default_factory=set)
    launched_names: set[str] = field(default_factory=set)
    checks: int = 0
    supervisor_count: int = 0
    last_ai_msg_id: str | None = None


def _print_supervisor_turn(message: dict, stats: _RunStats) -> str:
    """Print one supervisor model response; return its text (``""`` if empty)."""
    content = _extract_text(message.get("content", ""))
    if not content:
        return ""
    stats.supervisor_count += 1
    console.print(
        Panel(
            content,
            title=f"[green]Supervisor · turn {stats.supervisor_count}[/green]",
            border_style="green",
            width=72,
        )
    )
    return content


async def _preflight_receiver(receiver_url: str) -> bool:
    """Wait for the receiver's ``/health``; return True once it answers HTTP_OK."""
    for _attempt in range(6):
        try:
            async with httpx.AsyncClient(timeout=2.0) as probe:
                resp = await probe.get(f"{receiver_url.rstrip('/')}/health")
            if resp.status_code == HTTP_OK:
                _demo(f"receiver /health OK -> {resp.json()}")
                return True
        except Exception as exc:
            log.debug("receiver health probe failed: %s", exc)
        await asyncio.sleep(2)  # the supervisor's eager spawn may still be booting
    _warn(
        f"receiver NOT reachable at {receiver_url}/health — "
        "a2a is not live.  Add the A2A_* block to .env "
        "(A2A_COMPLETION_NOTIFIER_ENABLED=true, A2A_RECEIVER_URL, "
        "A2A_SUPERVISOR_URL, A2A_CALLBACK_TOKEN_SECRET), restart both "
        "servers, then re-run."
    )
    return False


def _note_tool_calls(last: dict, stats: _RunStats) -> None:
    """Record compact tool-call events from the latest AI message."""
    for tc in last.get("tool_calls", []):
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}
        if name == "start_async_task":
            stats.launched_names.add(args.get("subagent_type", "worker"))
            _demo(f"→ launching {args.get('subagent_type', 'worker')}")
        elif name == "check_async_task":
            stats.checks += 1
            _demo(f"… poll check_async_task (#{stats.checks})")
        elif name == "get_async_result":
            _a2a_ok(f"get_async_result({(args.get('task_id') or '?')[:8]}…)")


def _note_launch_results(messages: list, stats: _RunStats) -> None:
    """Detect "Launched async subagent" tool messages and record task ids."""
    for msg in messages[-3:]:
        if not (isinstance(msg, dict) and msg.get("type") == "tool"):
            continue
        text_content = _extract_text(msg.get("content", ""))
        if not text_content.startswith("Launched async subagent."):
            continue
        tid = text_content.partition("task_id: ")[2].strip()
        if tid and tid not in stats.launched:
            stats.launched.add(tid)
            _a2a_ok(f"running: task_id {tid[:8]}…")


async def _stream_dispatch_run(
    client: LangGraphClient,
    thread_id: str,
    prompt: str,
    push_config: dict,
    stats: _RunStats,
) -> None:
    """Stream the dispatch run, printing supervisor turns and compact tool events."""
    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id="supervisor",
        input={"messages": [{"role": "user", "content": prompt}]},
        config=push_config,
        stream_mode="values",
    ):
        if chunk.event != "values":
            continue
        messages = chunk.data.get("messages", [])
        ai_msgs = [m for m in messages if isinstance(m, dict) and m.get("type") == "ai"]
        if not ai_msgs:
            continue
        last = ai_msgs[-1]
        _note_tool_calls(last, stats)
        _note_launch_results(messages, stats)
        msg_id = last.get("id")
        if msg_id and msg_id == stats.last_ai_msg_id:
            continue
        stats.last_ai_msg_id = msg_id
        _print_supervisor_turn(last, stats)


async def _poll_wake_state(
    client: LangGraphClient,
    thread_id: str,
    stats: _RunStats,
    notice_count: int,
    notify_printed: bool,
) -> tuple[int, bool, bool]:
    """One wake-up poll iteration.

    Returns ``(notice_count, notify_printed, done)``.
    """
    try:
        notices = await _notice_blocks_in_history(client, thread_id)
        if notices:
            notice_count = len(notices)
            if not notify_printed:
                notify_printed = True
                _a2a_ok(f"completion notice found in prompt history (count={notice_count})")
        messages = (await client.threads.get_state(thread_id)).get("values", {}).get("messages", [])
    except Exception:
        return notice_count, notify_printed, False

    ai_msgs = [m for m in messages if isinstance(m, dict) and m.get("type") == "ai"]
    if not ai_msgs:
        return notice_count, notify_printed, bool(notice_count)

    last = ai_msgs[-1]
    if last.get("id") == stats.last_ai_msg_id:
        return notice_count, notify_printed, bool(notice_count)

    stats.last_ai_msg_id = last.get("id")
    for tc in last.get("tool_calls", []):
        args = tc.get("args", {}) or {}
        if tc.get("name") == "get_async_result":
            _a2a_ok(f"get_async_result({(args.get('task_id') or '?')[:8]}…)")
    content = _print_supervisor_turn(last, stats)
    done = notice_count and any(k in content.lower() for k in ("summary", "langgraph", "learned"))
    return notice_count, notify_printed, bool(done)


async def _wait_for_wake_run(
    client: LangGraphClient,
    thread_id: str,
    stats: _RunStats,
) -> int:
    """Poll the thread until the a2a wake-up turns surface; return the notice count."""
    console.print()
    _demo("(dispatch run ended — waiting for the a2a wake-up run…)")
    started_wait = time.monotonic()
    last_beat = started_wait
    deadline = started_wait + _WAKE_TIMEOUT_SECONDS
    notice_count = 0
    notify_printed = False
    while time.monotonic() < deadline:
        await asyncio.sleep(1.5)
        if time.monotonic() - last_beat > _HEARTBEAT_SECONDS:
            last_beat = time.monotonic()
            _demo(f"… still waiting ({(last_beat - started_wait):.0f}s)")
        notice_count, notify_printed, done = await _poll_wake_state(
            client, thread_id, stats, notice_count, notify_printed
        )
        if done:
            break
    return notice_count


async def _print_worker_results(supervisor_url: str, launched: set[str]) -> None:
    """Print each launched worker's final answer, read from the worker server."""
    for tid in sorted(launched):
        try:
            worker_client = get_client(url=_worker_url_from_supervisor_url(supervisor_url))
            worker_state = await worker_client.threads.get_state(tid)
            worker_msgs = worker_state.get("values", {}).get("messages", [])
        except Exception as exc:
            log.debug("worker state unavailable for %s: %s", tid, exc)
            worker_msgs = []
        if not worker_msgs:
            console.print(
                "  [dim]worker result is on the worker server (see `just langshark-worker`)[/dim]"
            )
            continue
        worker_ai = [m for m in worker_msgs if isinstance(m, dict) and m.get("type") == "ai"]
        text = _extract_text(worker_ai[-1].get("content", "")) if worker_ai else ""
        if text:
            console.print(
                Panel(
                    text[:400],
                    title="[yellow]Worker · researcher[/yellow]",
                    border_style="yellow",
                    width=72,
                )
            )


async def _print_summary(
    client: LangGraphClient,
    thread_id: str,
    stats: _RunStats,
    notice_count: int,
) -> None:
    """Print the end-of-run summary table (mirrors ``scripts/demo.py``)."""
    msg_count: int | str = "?"
    try:
        final_state = await client.threads.get_state(thread_id)
        messages = final_state["values"].get("messages", [])
        msg_count = sum(
            1 for m in messages if isinstance(m, dict) and m.get("type") in ("human", "ai")
        )
    except Exception as exc:
        log.debug("could not read the final state for the summary: %s", exc)

    table = Table(show_header=False, box=None)
    table.add_column("Label", style="bold", width=24)
    table.add_column("Value")
    if stats.launched_names:
        table.add_row("Workers launched", ", ".join(sorted(stats.launched_names)))
    if stats.launched:
        table.add_row("Worker tasks", ", ".join(t[:8] for t in sorted(stats.launched)))
    table.add_row("check_async_task", f"{stats.checks} poll(s)")
    if notice_count:
        table.add_row("A2A push", f"✓ {notice_count} completion notice(s) drained (no polling)")
    table.add_row("Supervisor turns", f"{stats.supervisor_count} (model responses)")
    table.add_row("Total messages", str(msg_count))
    table.add_row("Thread", thread_id)

    console.print()
    console.print(Rule("[bold]Live demo summary[/bold]"))
    console.print(table)
    console.print(
        "  [dim]supervisor logs confirm the loop: "
        "a2a_emitted (worker) · a2a_mailbox_write (receiver) · "
        "a2a_wake_triggered (wake) · a2a_mcp_ready (parent MCP tools)[/dim]"
    )


async def main() -> int:
    """Drive the real split-server stack (supervisor + worker + receiver).

    Renders supervisor turns only (``values`` stream, deduped by message id),
    compact tool events, an end-of-run summary, and a non-zero exit if the
    notifier's static completion marker never appears in the supervisor's
    prompt history (read from the thread checkpoints).  The receiver is
    embedded on the supervisor via ``http.app``in the langgraph.json configuration
    file.
    """
    supervisor_url = os.environ.get("SUPERVISOR_URL", "http://localhost:2024")
    receiver_url = os.environ.get("A2A_RECEIVER_URL", "http://localhost:2024")
    secret = os.environ.get("A2A_CALLBACK_TOKEN_SECRET", CALLBACK_SECRET)

    client = get_client(url=supervisor_url)
    try:
        thread = await client.threads.create()
        thread_id = thread["thread_id"]
    except Exception as exc:
        _warn(
            f"Could not reach the supervisor at {supervisor_url}.\n"
            "This demo is LIVE-ONLY: it never starts a server.  Start the "
            "split-server stack first, then re-run:\n\n"
            "    just supervisor    # terminal 1\n"
            "    just worker             # terminal 2\n"
            f"\n  ({type(exc).__name__}: {exc})\n"
        )
        return 2

    push_config = build_push_config(
        thread_id=thread_id,
        assistant_id="supervisor",
        dispatch_id=f"dispatch-{thread_id[:8]}",
        receiver_url=receiver_url,
        callback_token_secret=secret,
    )

    # Same research objective as scripts/demo.py (so the demos are comparable),
    # plus the a2a no-polling guidance: the worker's emitter pushes a
    # completion notice, the receiver writes it to the mailbox, and the
    # supervisor's next model call drains it.
    prompt = (
        "You MUST delegate the research task to the researcher subagent: "
        "research and write a one-sentence summary of LangGraph's purpose. "
        "Wait for the COMPLETION NOTICE (the worker will notify you when it "
        "finishes -- do NOT poll check_async_task in a loop). When the notice "
        "arrives, write the one-sentence summary; if you need the full "
        "result, call get_async_result once with the exact task_id from the "
        "notice."
    )
    await _print_graph_structure(client, supervisor_url)
    console.print(Rule("[bold]Live A2A push demo[/bold]"))
    _demo(f"thread={thread_id}")
    _demo(f"dispatching with a2a_push_config → {receiver_url}/a2a/notifications")
    _demo(
        "native MCP tools attached at launch (see supervisor logs): "
        + ", ".join(sorted(SUPERVISOR_MCP_TOOLS))
    )

    # Preflight: the a2a loop needs the receiver up before the worker
    # completes.  It's embedded on the supervisor via http.app when A2A is
    # enabled; missing here means the servers did not pick up the A2A_* env.
    await _preflight_receiver(receiver_url)
    _demo("stream:")

    stats = _RunStats()
    await _stream_dispatch_run(client, thread_id, prompt, push_config, stats)

    # The dispatch run is done.  The a2a wake-up is a SECOND supervisor run
    # that the receiver starts once the worker's completion has been written
    # to the mailbox; its first model call drains the notice and writes the
    # summary.  Poll the thread state until those turns appear.
    notice_count = await _wait_for_wake_run(client, thread_id, stats)
    if notice_count == 0:
        await _diagnose_wake(client, thread_id)
    # ── Worker results (mirrors scripts/demo.py) ─────────────────
    await _print_worker_results(supervisor_url, stats.launched)

    # ── Summary (mirrors scripts/demo.py's table) ────────────────
    await _print_summary(client, thread_id, stats, notice_count)
    return 0 if notice_count else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
