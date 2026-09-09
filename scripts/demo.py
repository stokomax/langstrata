#!/usr/bin/env python
"""Demo: send a message to the supervisor and watch it delegate to workers.

Usage
-----
1. Start the single-server (recommended):

        just supervisor

2. Run this demo:

        uv run python scripts/demo.py

3. Watch subagent delegation in the output, then inspect threads in Langshark:

        just langshark-supervisor

4. (Optional) Graduate to dual-server mode (separate terminals):

        Terminal 1:  just worker
        Terminal 2:  just supervisor-http
"""

import asyncio
import os
import sys

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree

from langgraph_sdk import get_client

SUPERVISOR_URL = os.environ.get("SUPERVISOR_URL", "http://localhost:2024")
console = Console()

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
        console.print(Panel(
            f"[yellow]The model provider [bold]{provider}[/bold] is configured "
            f"but [bold]{env_var}[/bold] is not set in your environment or "
            f".env file.  The LangGraph server will fail at runtime.[/yellow]",
            title="[bold]API Key Warning[/bold]",
            border_style="yellow",
            width=72,
        ))

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
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return content.strip() if isinstance(content, str) else ""


async def _check_server(url: str, timeout: float = 3) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url.rstrip("/") + "/")
            return resp.status_code < 500
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _print_guidance() -> None:
    console.print()
    console.print("[red]LangGraph server is not available.[/red]")
    console.print()
    console.print("  Start from this project root with:")
    console.print()
    console.print("    [cyan]just supervisor[/cyan]  (single-server, recommended)")
    console.print()
    console.print("  For dual-server mode use two terminals:")
    console.print("    terminal 1:  [cyan]just worker[/cyan]")
    console.print("    terminal 2:  [cyan]just supervisor-http[/cyan]")
    console.print()
    console.print("  See [bold]just --list[/bold] for all available recipes.")

def _worker_url_from_supervisor_url() -> str:
    """Derive the worker URL from SUPERVISOR_URL by adding 1 to its port."""
    from urllib.parse import urlparse
    sup_url = os.environ.get("SUPERVISOR_URL", "http://localhost:2024")
    parsed = urlparse(sup_url)
    try:
        sup_port = int(parsed.port) if parsed.port else 2024
    except (ValueError, TypeError):
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


async def _print_graph_structure(client) -> None:
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

    # Find supervisor and workers
    supervisor = None
    workers: dict[str, dict] = {}
    others: list[dict] = []

    for a in assistants:
        gid = a.get("graph_id", "?")
        if gid == "supervisor":
            supervisor = a
        elif gid in ("researcher", "coder", "analyst"):
            workers[gid] = a
        else:
            others.append(a)

    tree = Tree("[bold]Graph topology[/bold]")

    # ── Supervisor (root) ──────────────────────────────────────
    if supervisor:
        sup_id = supervisor.get("assistant_id", "")
        model = os.environ.get("AGENT_SERVER_MODEL", "default")
        sup_branch = tree.add(
            f"[cyan]⭐ supervisor[/cyan]  (model: [green]{model}[/green])"
        )

        # Fetch supervisor's internal graph structure
        try:
            detail = await client.assistants.get_graph(sup_id, xray=True)
            nodes = [n["id"] for n in detail.get("nodes", []) if n["id"] not in ("__start__", "__end__")]
            agent_type = _classify_agent_type(nodes)
            sup_branch.add(f"[dim]type:[/dim] {agent_type}")
        except Exception:
            pass

        # ── Workers (children) ─────────────────────────────────
        if workers:
            deploy_label = (
                f"[yellow]⚡ delegated to workers (separate server, {_worker_url_from_supervisor_url()})[/yellow]"
                if mode == "http"
                else f"[yellow]⚡ delegates to ({len(workers)} workers, co-deployed)[/yellow]"
            )
            sub_branch = sup_branch.add(deploy_label)
            for gid, a in workers.items():
                wid = a.get("assistant_id", "")
                w_branch = sub_branch.add(f"[cyan]{gid}[/cyan]  (id: {wid[:12]})")

                try:
                    w_detail = await client.assistants.get_graph(wid)
                    w_nodes = [n["id"] for n in w_detail.get("nodes", []) if n["id"] not in ("__start__", "__end__")]
                    w_agent_type = _classify_agent_type(w_nodes)
                    w_branch.add(f"[dim]type:[/dim] {w_agent_type}")
                except Exception:
                    pass

        # ── Other assistants ───────────────────────────────────
        for a in others:
            gid = a.get("graph_id", "?")
            aid = a.get("assistant_id", "")[:12]
            others_branch = tree.add(f"[dim]{gid}[/dim]  (id: {aid})")
            try:
                o_detail = await client.assistants.get_graph(a.get("assistant_id", ""))
                o_nodes = [n["id"] for n in o_detail.get("nodes", []) if n["id"] not in ("__start__", "__end__")]
                o_agent_type = _classify_agent_type(o_nodes)
                others_branch.add(f"[dim]type:[/dim] {o_agent_type}")
            except Exception:
                pass

    else:
        # No supervisor found — flat list
        for a in assistants:
            gid = a.get("graph_id", "?")
            tree.add(f"[dim]{gid}[/dim]")

    console.print(tree)
    console.print()


async def main() -> None:
    # ── Pre-flight check ──────────────────────────────────────────
    if not await _check_server(SUPERVISOR_URL):
        _print_guidance()
        sys.exit(0)

    # ── API key check ────────────────────────────────────────────
    _check_api_key()

    client = get_client(url=SUPERVISOR_URL)

    # ── Live graph structure ─────────────────────────────────────
    await _print_graph_structure(client)

    # Create thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    prompt = _resolve_prompt()

    console.print()
    console.print(Rule("[bold]LangStrata Demo[/bold]"))
    console.print()
    console.print(f"  Thread: [yellow]{thread_id}[/yellow]")
    console.print(f"  User:   {prompt}")
    console.print()

    # ── Stream the supervisor run ────────────────────────────────
    launched_workers: set[str] = set()
    worker_tasks: dict[str, str] = {}  # taskId -> worker_name
    launch_order: list[str] = []  # ordered list of worker names
    last_ai_msg_id: str | None = None
    supervisor_count = 0
    had_error = False

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
            if chunk.event == "values":
                state = chunk.data
                messages = state.get("messages", [])
                ai_msgs = [m for m in messages if isinstance(m, dict) and m.get("type") == "ai"]
                if not ai_msgs:
                    continue
                last = ai_msgs[-1]

                for tc in last.get("tool_calls", []):
                    name = tc.get("name", "")
                    if name == "check_async_task":
                        # Capture taskId from the check args
                        tid = tc["args"].get("taskId", "") or tc["args"].get("task_id", "")
                        if tid and tid not in worker_tasks and launch_order:
                            worker_tasks[tid] = launch_order.pop(0)
                        continue
                    if name == "start_async_task":
                        sub = tc["args"].get("subagent_type", "unknown")
                        if sub not in launched_workers:
                            launched_workers.add(sub)
                            launch_order.append(sub)

                # Dedup: only print when the AI message actually changes
                msg_id = last.get("id", "")
                if msg_id and msg_id == last_ai_msg_id:
                    continue
                last_ai_msg_id = msg_id

                content = _extract_text(last.get("content", ""))
                if content:
                    supervisor_count += 1
                    print(flush=True)  # clear heartbeat line
                    console.print(Panel(
                        content,
                        title="[green]Supervisor[/green]",
                        border_style="green",
                        width=72,
                    ))

            elif chunk.event == "start_async_task":
                sub = chunk.data.get("subagentType", "")
                if sub not in launched_workers:
                        launched_workers.add(sub)
                        launch_order.append(sub)
                        print(flush=True)
                        console.print(f"  [cyan]→[/cyan] launched [bold]{sub}[/bold]")

            elif chunk.event == "check_async_task":
                tid = chunk.data.get("taskId", "") or chunk.data.get("task_id", "")
                if tid and tid not in worker_tasks and launch_order:
                    worker_tasks[tid] = launch_order.pop(0)

            elif chunk.event == "error":
                console.print(f"  [red][error][/red] {chunk.data}")
                had_error = True

    finally:
        stop_heartbeat = True
        await hb

    # ── Worker results ────────────────────────────────────────────
    if worker_tasks:
        console.print()
        console.print(Rule("[bold]Worker Results[/bold]"))
        for tid, name in worker_tasks.items():
            try:
                state = await client.threads.get_state(tid)
                msgs = state["values"].get("messages", [])
                ai_msgs = [m for m in msgs if isinstance(m, dict) and m.get("type") == "ai"]
                if ai_msgs:
                    text = _extract_text(ai_msgs[-1].get("content", ""))
                    if text:
                        console.print(Panel(
                            text[:400],
                            title=f"[yellow]{name}[/yellow]",
                            border_style="yellow",
                            width=72,
                        ))
            except Exception:
                console.print(f"  [dim]Worker results live on a separate server in dual-server mode.[/dim]")
                console.print(f"  [dim]View them in Langshark:[/dim]")
                console.print(f"  [dim]  just langshark-worker[/dim]")

    # ── Summary ───────────────────────────────────────────────────
    msg_count = 0
    status = "unknown"
    try:
        state = await client.threads.get_state(thread_id)
        messages = state["values"].get("messages", [])
        msg_count = sum(
            1 for m in messages
            if isinstance(m, dict) and m.get("type") in ("human", "ai")
        )
    except Exception:
        pass

    table = Table(show_header=False, box=None)
    table.add_column("Label", style="bold", width=22)
    table.add_column("Value")
    table.add_row("Supervisor responses", str(supervisor_count))
    if worker_tasks:
        worker_info = ", ".join(
            f"{name} ({tid})" for tid, name in worker_tasks.items()
        )
        table.add_row("Worker threads", worker_info)
    elif launched_workers:
        table.add_row("Workers deployed", ", ".join(sorted(launched_workers)))
    else:
        table.add_row("[yellow]Workers deployed[/yellow]", "[yellow]none — supervisor answered directly[/yellow]")
    table.add_row("Total messages", str(msg_count))
    table.add_row("Supervisor thread", thread_id)

    console.print()
    if not had_error:
        console.print(Rule("[bold green]✅ Demo completed successfully[/bold green]"))
    else:
        console.print(Rule("[bold red]⚠ Demo finished with errors[/bold red]"))
    console.print(table)

    console.print()
    console.print("  Next step — [cyan]dual-server mode[/cyan] (separate terminals):")
    console.print("    Terminal 1:  [cyan]just worker[/cyan]")
    console.print("    Terminal 2:  [cyan]just supervisor-http[/cyan]")
    console.print()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\nInterrupted.", file=sys.stderr)
        sys.exit(0)
