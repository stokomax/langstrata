#!/usr/bin/env python
"""Demo: send a message to the supervisor and watch it delegate to workers.

Usage
-----
1. Start the single-server (recommended):

        just standalone

2. Run this demo:

        uv run python scripts/demo.py

3. Watch subagent delegation in the output, then inspect threads in Langshark:

        just langshark-supervisor

4. (Optional) Graduate to dual-server mode (separate terminals):

        Terminal 1:  just worker
        Terminal 2:  just supervisor
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from langgraph_sdk import get_client
from langgraph_sdk._async.client import LangGraphClient
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree

SUPERVISOR_URL = os.environ.get("SUPERVISOR_URL", "http://localhost:2024")
console = Console()

# HTTP status codes at or above this mean the server is not considered healthy.
_HTTP_SERVER_ERROR = 500
_TERMINAL_NODE_IDS = frozenset({"__start__", "__end__"})
_WORKER_GRAPH_IDS = frozenset({"researcher", "coder", "analyst"})

# ── Provider API key map ─────────────────────────────────────────
_PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _resolve_provider() -> str | None:
    """Return the provider prefix from AGENT_SERVER_MODEL, or None."""
    model = os.environ.get("AGENT_SERVER_MODEL", "anthropic:claude-sonnet-5")
    if ":" in model:
        return model.split(":", 1)[0]
    return None


def _check_api_key() -> None:
    """Print a warning if the API key for the resolved provider is not set."""
    provider = _resolve_provider()
    if provider is None:
        return
    env_var = _PROVIDER_KEY_MAP.get(provider)
    if env_var is None:
        return
    if not os.environ.get(env_var, "").strip():
        console.print(
            Panel(
                f"[yellow]The model provider [bold]{provider}[/bold] is configured "
                f"but [bold]{env_var}[/bold] is not set in your environment or "
                f".env file.  The LangGraph server will fail at runtime.[/yellow]",
                title="[bold]API Key Warning[/bold]",
                border_style="yellow",
                width=72,
            )
        )


# ── Built-in prompts ─────────────────────────────────────────────
DEMO_PROMPT_ABRIDGED = (
    "You MUST delegate the research task to the researcher subagent. "
    "Wait for it to complete, then write a one-sentence summary of LangGraph's purpose."
)

DEMO_PROMPT_ORIGINAL = (
    "Research the latest trends in LangGraph agent architectures "
    "and then write a Python script demonstrating a simple state machine."
)


def _resolve_prompt() -> str:
    """Return the prompt to use.

    Set DEMO_PROMPT_MODE env var to ``"original"`` for the full research+code
    prompt, or ``"abridged"`` (default) for a shorter single-task prompt.
    """
    choice = os.environ.get("DEMO_PROMPT_MODE", "abridged")
    if choice == "original":
        return DEMO_PROMPT_ORIGINAL
    return DEMO_PROMPT_ABRIDGED


def _classify_agent_type(nodes: list[str]) -> str:
    """Classify an agent type from its node names.

    DeepAgents include ``PatchToolCallsMiddleware`` nodes; standard
    LangGraph agents do not.
    """
    for n in nodes:
        if "PatchToolCallsMiddleware" in n or "before_agent" in n:
            return "DeepAgent"
    return "Standard agent"


def _extract_text(content: object) -> str:
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return content.strip() if isinstance(content, str) else ""


async def _check_server(url: str, timeout: float = 3) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url.rstrip("/") + "/")
            return resp.status_code < _HTTP_SERVER_ERROR
    except httpx.ConnectError, httpx.TimeoutException:
        return False


def _print_guidance() -> None:
    console.print()
    console.print("[red]LangGraph server is not available.[/red]")
    console.print()
    console.print("  Start from this project root with:")
    console.print()
    console.print("    [cyan]just standalone[/cyan]  (single-server, recommended)")
    console.print()
    console.print("  For dual-server mode use two terminals:")
    console.print("    terminal 1:  [cyan]just worker[/cyan]")
    console.print("    terminal 2:  [cyan]just supervisor[/cyan]")
    console.print()
    console.print("  See [bold]just --list[/bold] for all available recipes.")


def _worker_url_from_supervisor_url() -> str:
    """Derive the worker URL from SUPERVISOR_URL by adding 1 to its port."""
    sup_url = os.environ.get("SUPERVISOR_URL", "http://localhost:2024")
    parsed = urlparse(sup_url)
    try:
        sup_port = int(parsed.port) if parsed.port else 2024
    except ValueError, TypeError:
        sup_port = 2024
    return f"{parsed.scheme}://{parsed.hostname}:{sup_port + 1}"


async def _detect_mode() -> str:
    """Detect whether workers are co-deployed (ASGI) or on a separate server (HTTP).

    Derives the worker server port from SUPERVISOR_URL by adding 1 to the
    supervisor port (e.g. 2024→2025, 8123→8124).  If the worker server is
    reachable and returns assistants, returns "http"; otherwise "asgi".
    """
    worker_url = _worker_url_from_supervisor_url()
    try:
        worker = get_client(url=worker_url)
        wa = await worker.assistants.search()
        if wa:
            return "http"
    except Exception:
        pass
    return "asgi"


def _node_ids(detail: dict) -> list[str]:
    """Return non-terminal node ids from an assistants.get_graph payload."""
    return [n["id"] for n in detail.get("nodes", []) if n["id"] not in _TERMINAL_NODE_IDS]


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
        branch.add(f"[dim]type:[/dim] {_classify_agent_type(_node_ids(detail))}")
    except Exception:
        pass


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


def _deploy_label(mode: str, worker_count: int) -> str:
    """Rich label describing how the supervisor reaches its workers."""
    if mode == "http":
        worker_url = _worker_url_from_supervisor_url()
        return f"[yellow]⚡ delegated to workers (separate server, {worker_url})[/yellow]"
    return f"[yellow]⚡ delegates to ({worker_count} workers, co-deployed)[/yellow]"


async def _add_worker_branches(
    parent: Tree,
    client: LangGraphClient,
    workers: dict[str, dict],
    mode: str,
) -> None:
    """Attach worker sub-branches under the supervisor branch."""
    sub_branch = parent.add(_deploy_label(mode, len(workers)))
    for gid, a in workers.items():
        wid = a.get("assistant_id", "")
        w_branch = sub_branch.add(f"[cyan]{gid}[/cyan]  (id: {wid[:12]})")
        await _add_graph_type_line(w_branch, client, wid)


async def _print_graph_structure(client: LangGraphClient) -> None:
    """Fetch registered assistants from the server and display their graph topology."""
    console.print(Rule("[bold]Graph Structure[/bold]"))
    mode = await _detect_mode()

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
            await _add_worker_branches(sup_branch, client, workers, mode)
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


@dataclass
class _RunStats:
    """Counters gathered while streaming the supervisor run."""

    launched_workers: set[str] = field(default_factory=set)
    worker_tasks: dict[str, str] = field(default_factory=dict)
    launch_order: list[str] = field(default_factory=list)
    last_ai_msg_id: str | None = None
    supervisor_count: int = 0
    had_error: bool = False


def _track_tool_calls(last: dict, stats: _RunStats) -> None:
    """Update launch tracking from tool calls on the latest AI message."""
    for tc in last.get("tool_calls", []):
        name = tc.get("name", "")
        if name == "check_async_task":
            tid = tc["args"].get("taskId", "") or tc["args"].get("task_id", "")
            if tid and tid not in stats.worker_tasks and stats.launch_order:
                stats.worker_tasks[tid] = stats.launch_order.pop(0)
            continue
        if name == "start_async_task":
            sub = tc["args"].get("subagent_type", "unknown")
            if sub not in stats.launched_workers:
                stats.launched_workers.add(sub)
                stats.launch_order.append(sub)


def _print_ai_turn(last: dict, stats: _RunStats) -> None:
    """Print a new supervisor AI message once (deduped by message id)."""
    msg_id = last.get("id", "")
    if msg_id and msg_id == stats.last_ai_msg_id:
        return
    stats.last_ai_msg_id = msg_id
    content = _extract_text(last.get("content", ""))
    if not content:
        return
    stats.supervisor_count += 1
    print(flush=True)
    console.print(
        Panel(
            content,
            title=f"[green]Supervisor · turn {stats.supervisor_count}[/green]",
            border_style="green",
            width=72,
        )
    )


def _handle_stream_event(chunk: object, stats: _RunStats) -> None:
    """Dispatch one stream chunk into launch tracking / UI updates."""
    event = getattr(chunk, "event", None)
    data = getattr(chunk, "data", None) or {}

    if event == "values":
        messages = data.get("messages", []) if isinstance(data, dict) else []
        ai_msgs = [m for m in messages if isinstance(m, dict) and m.get("type") == "ai"]
        if not ai_msgs:
            return
        last = ai_msgs[-1]
        _track_tool_calls(last, stats)
        _print_ai_turn(last, stats)
        return

    if event == "start_async_task":
        sub = data.get("subagentType", "") if isinstance(data, dict) else ""
        if sub and sub not in stats.launched_workers:
            stats.launched_workers.add(sub)
            stats.launch_order.append(sub)
            print(flush=True)
            console.print(f"  [cyan]→[/cyan] launched [bold]{sub}[/bold]")
        return

    if event == "check_async_task" and isinstance(data, dict):
        tid = data.get("taskId", "") or data.get("task_id", "")
        if tid and tid not in stats.worker_tasks and stats.launch_order:
            stats.worker_tasks[tid] = stats.launch_order.pop(0)
        return

    if event == "error":
        console.print(f"  [red][error][/red] {data}")
        stats.had_error = True


async def _stream_supervisor_run(
    client: LangGraphClient,
    thread_id: str,
    prompt: str,
) -> _RunStats:
    """Stream the supervisor run; return launch stats and error flag."""
    stats = _RunStats()
    stop_heartbeat = False

    async def _heartbeat() -> None:
        while not stop_heartbeat:
            await asyncio.sleep(3)
            print(".", end="", flush=True)
        print(flush=True)

    hb = asyncio.create_task(_heartbeat())
    console.print("\n[bold]Running supervisor agent...[/bold]")

    try:
        async for chunk in client.runs.stream(
            thread_id,
            "supervisor",
            input={"messages": [{"role": "human", "content": prompt}]},
            stream_mode="values",
        ):
            _handle_stream_event(chunk, stats)
    finally:
        stop_heartbeat = True
        await hb

    return stats


async def _print_worker_results(
    client: LangGraphClient,
    worker_tasks: dict[str, str],
) -> None:
    """Print final AI answers from each launched worker thread."""
    if not worker_tasks:
        return
    console.print()
    console.print(Rule("[bold]Worker Results[/bold]"))
    dual_server_hint = "  [dim]Worker results live on a separate server in dual-server mode.[/dim]"
    for tid, name in worker_tasks.items():
        try:
            state = await client.threads.get_state(tid)
            msgs = state["values"].get("messages", [])
            ai_msgs = [m for m in msgs if isinstance(m, dict) and m.get("type") == "ai"]
            if not ai_msgs:
                continue
            body = _extract_text(ai_msgs[-1].get("content", ""))
            if body:
                console.print(
                    Panel(
                        body[:400],
                        title=f"[yellow]{name}[/yellow]",
                        border_style="yellow",
                        width=72,
                    )
                )
        except Exception:
            console.print(dual_server_hint)
            console.print("  [dim]View them in Langshark:[/dim]")
            console.print("  [dim]  just langshark-worker[/dim]")


async def _print_summary(
    client: LangGraphClient,
    thread_id: str,
    stats: _RunStats,
) -> None:
    """Print the end-of-demo summary table and next-step hints."""
    msg_count = 0
    try:
        state = await client.threads.get_state(thread_id)
        messages = state["values"].get("messages", [])
        msg_count = sum(
            1 for m in messages if isinstance(m, dict) and m.get("type") in ("human", "ai")
        )
    except Exception:
        pass

    table = Table(show_header=False, box=None)
    table.add_column("Label", style="bold", width=22)
    table.add_column("Value")
    table.add_row("Supervisor responses", str(stats.supervisor_count))
    if stats.worker_tasks:
        worker_info = ", ".join(f"{name} ({tid})" for tid, name in stats.worker_tasks.items())
        table.add_row("Worker threads", worker_info)
    elif stats.launched_workers:
        table.add_row("Workers deployed", ", ".join(sorted(stats.launched_workers)))
    else:
        table.add_row(
            "[yellow]Workers deployed[/yellow]",
            "[yellow]none — supervisor answered directly[/yellow]",
        )
    table.add_row("Total messages", str(msg_count))
    table.add_row("Supervisor thread", thread_id)

    console.print()
    if not stats.had_error:
        console.print(Rule("[bold green]✅ Demo completed successfully[/bold green]"))
    else:
        console.print(Rule("[bold red]⚠ Demo finished with errors[/bold red]"))
    console.print(table)

    console.print()
    console.print("  Next step — [cyan]dual-server mode[/cyan] (separate terminals):")
    console.print("    Terminal 1:  [cyan]just worker[/cyan]")
    console.print("    Terminal 2:  [cyan]just supervisor[/cyan]")
    console.print()


async def main() -> None:
    """Run the supervisor demo against a live LangGraph server."""
    if not await _check_server(SUPERVISOR_URL):
        _print_guidance()
        sys.exit(0)

    _check_api_key()

    client = get_client(url=SUPERVISOR_URL)
    await _print_graph_structure(client)

    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    prompt = _resolve_prompt()

    console.print()
    console.print(Rule("[bold]LangStrata Demo[/bold]"))
    console.print()
    console.print(f"  Thread: [yellow]{thread_id}[/yellow]")
    console.print(f"  User:   {prompt}")
    console.print()

    stats = await _stream_supervisor_run(client, thread_id, prompt)
    await _print_worker_results(client, stats.worker_tasks)
    await _print_summary(client, thread_id, stats)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt, asyncio.CancelledError:
        print("\n\nInterrupted.", file=sys.stderr)
        sys.exit(0)
