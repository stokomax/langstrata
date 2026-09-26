"""Supervisor agent factory — dynamically builds AsyncSubAgent list from config."""

import logging
from typing import Any

from deepagents import AsyncSubAgent, create_deep_agent
from langgraph_sdk.runtime import ServerRuntime
from langshark_bites.a2a_completion_notifier.supervisor_tools import (
    prewarm_supervisor_tools,
)

from langstrata.config import Settings

log = logging.getLogger(__name__)

# Warm the mcp SDK's lazy `jsonschema` import at graph-module import time.
# `langgraph dev` imports this module before its ASGI event loop (and
# blockbuster) exists, so this synchronous directory scan runs safely here --
# the same eager-import pattern langgraph_api.graph uses for ddtrace.  If it
# ran lazily on the first MCP tool call, the first live `get_async_result`
# would trip blockbuster ("Blocking call to ScandirIterator.__next__").
prewarm_supervisor_tools()

# ── Worker registry ────────────────────────────────────────────────
# (name, description) pairs that the supervisor knows about.
# The graph_id must match the assistant name on the worker server.
_WORKER_SPECS: list[tuple[str, str]] = [
    (
        "researcher",
        (
            "Conducts in-depth research using web search. "
            "Use for questions requiring multiple searches and synthesis."
        ),
    ),
    (
        "coder",
        (
            "Generates and reviews code. Use for implementation, "
            "debugging, code review, and refactoring."
        ),
    ),
    (
        "analyst",
        (
            "Performs data analysis and visualisation. "
            "Use for querying datasets, finding trends, and generating charts."
        ),
    ),
]


def _resolve_url(name: str, config: Settings) -> str | None:
    """Resolve the worker URL for a given worker name.

    Returns ``None`` in ASGI mode (co-deployed), or an HTTP URL in
    ``http`` mode.  Per-worker overrides take precedence over the
    default ``worker_api_url``.
    """
    if config.mode == "asgi":
        return None  # ASGI transport — co-deployed

    # Per-worker override (optional)
    override_attr = f"{name}_url"
    override = getattr(config, override_attr, None)
    return override or config.worker_api_url


def _build_async_subagents(config: Settings) -> list[AsyncSubAgent]:
    """Construct the list of ``AsyncSubAgent`` specs for the supervisor.

    Each spec's ``url`` is resolved from config, so the same code path
    yields co-deployed (ASGI) or remote (HTTP) subagents depending on
    ``config.mode``.
    """
    subagents: list[AsyncSubAgent] = []
    for name, description in _WORKER_SPECS:
        spec = AsyncSubAgent(name=name, description=description, graph_id=name)
        url = _resolve_url(name, config)
        if url is not None:
            # HTTP mode.  For ASGI transport the key must be OMITTED, not set
            # to None (deepagents declares AsyncSubAgent.url NotRequired[str]
            # and documents "Omit to use ASGI transport for local servers").
            spec["url"] = url
        subagents.append(spec)
    return subagents


def _build_mcp_tools(config: Settings) -> list[Any]:
    """Discover the A2A bridge's diagnostic tools as native supervisor tools.

    When ``A2A_COMPLETION_NOTIFIER_ENABLED=true`` the supervisor attaches the four
    MCP tools exposed by the ``a2a_completion_notifier`` bridge
    (``list_pending_notifications``, ``dead_letter_queue``,
    ``receiver_health``, ``get_async_result``) at launch, using LangChain's
    native ``langchain.mcp.MCPAdapter``.  The bridge is spawned over stdio by
    default (dev) or reached over streamable-http when run separately.
    """
    if not config.a2a_enabled:
        return []
    from langstrata.a2a.supervisor_mcp import (  # noqa: PLC0415 - conditional: only import the MCP stack when A2A is on
        discover_supervisor_mcp_tools,
    )

    return discover_supervisor_mcp_tools()


def _build_extra_middleware(config: Settings) -> list[Any]:
    """Middleware langstrata adds on top of the deepagents stack.

    The a2a_completion_notifier drain reads the supervisor's Store mailbox
    (``("notifications", thread_id)``, written by the receiver) at the start
    of each model call and injects pending completions into the model's
    context.  The bite's ``MailboxDrainMiddleware`` works on both the
    platform/Agent Server (``runtime.store``) and ``langgraph dev`` (HTTP
    fallback via ``A2A_SUPERVISOR_URL``) — the single source of truth.
    """
    if not config.a2a_enabled:
        return []
    from langshark_bites.a2a_completion_notifier.drain import (  # noqa: PLC0415 - conditional: drain only wires up with A2A on
        MailboxDrainMiddleware,
    )

    return [MailboxDrainMiddleware()]


def _build_dispatch_forwarding(config: Settings, async_subagents: list[AsyncSubAgent]) -> list[Any]:
    """The deepagents middleware that forwards the a2a push config to subagents.

    deepagents' built-in async-subagent middleware drops the parent run's
    ``config.configurable`` on dispatch, so the worker-side emitter would never
    see the ``a2a_push_config``.  ``ConfigForwardingAsyncSubAgentMiddleware``
    keeps the built-in middleware's ``name``, so deepagents replaces that entry
    in place (rather than appending a duplicate).
    """
    if not config.a2a_enabled:
        return []
    from langstrata.a2a.a2a_dispatch_forwarder import (  # noqa: PLC0415 - conditional: forwarding only with A2A on
        ConfigForwardingAsyncSubAgentMiddleware,
    )

    return [ConfigForwardingAsyncSubAgentMiddleware(async_subagents=async_subagents)]


def create_supervisor_agent(
    runtime: ServerRuntime | None = None,
) -> Any:  # noqa: ANN401 - returns the deepagents graph (dynamic CompiledStateGraph shape)
    """Build the supervisor Deep Agent.

    Reads runtime settings from environment variables (``AGENT_SERVER_*``)
    so it works without arguments in ``langgraph dev``.

    LangGraph Server calls this factory per run and, because the parameter is
    annotated ``ServerRuntime``, injects the server runtime -- whose
    ``store`` is the Agent Server's Store.  Compiling the deep agent with
    that store is what lets the ``MailboxDrainMiddleware`` read the a2a
    notification mailbox the receiver writes.

    When ``A2A_COMPLETION_NOTIFIER_ENABLED=true``, also attaches the deepagents
    config-forwarding middleware (so subagent runs receive the ``a2a_push_config``
    deepagents would otherwise drop), attaches the a2a mailbox drain, and
    discovers the A2A bridge's four diagnostic tools as native supervisor
    tools via ``langchain.mcp.MCPAdapter`` (see ``langstrata.a2a.supervisor_mcp``).
    """
    config = Settings()

    async_subagents = _build_async_subagents(config)

    tools = _build_mcp_tools(config)

    store = getattr(runtime, "store", None) if runtime is not None else None
    if store is not None:
        log.info("a2a_server_store=wired")
    else:
        log.info("a2a_server_store=missing")

    middleware = [
        *_build_extra_middleware(config),
        *_build_dispatch_forwarding(config, async_subagents),
    ]

    return create_deep_agent(
        model=config.model,
        system_prompt=config.system_prompt,
        subagents=async_subagents,  # ← the core wiring
        tools=tools or None,
        store=store,
        middleware=middleware,
    )
