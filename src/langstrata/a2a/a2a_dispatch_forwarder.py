"""deepagents dispatch-config forwarding for the ``a2a_completion_notifier`` bite.

Why this exists
---------------
langstrata's supervisor is built on deepagents ``create_deep_agent`` with
declarative ``AsyncSubAgent`` specs.  deepagents' built-in
``AsyncSubAgentMiddleware`` launches subagent runs with only
``thread_id``/``assistant_id``/``input`` -- it never forwards the parent run's
``config.configurable``.  The langshark-bites emitter, however, requires
``config.configurable["a2a_push_config"]`` on the *subagent* run.  Without
forwarding, the push config cannot cross the deepagents boundary, so no
completion webhook would ever be emitted.

How it plugs in -- public API only, no monkeypatching
----------------------------------------------------
``ConfigForwardingAsyncSubAgentMiddleware`` subclasses the **public**
``AsyncSubAgentMiddleware`` and replaces only its ``start_async_task`` tool
with one that forwards the push config.  Two documented extension points make
that safe:

- ``AgentMiddleware.name`` -- documented as overridable for custom naming.
  Keeping the built-in name lets deepagents' middleware merge recognise this
  instance as a *replacement* for the built-in entry and swap it in place,
  preserving stack position, the other four async tools and the state schema.
- ``create_deep_agent(middleware=...)`` -- the documented channel for
  caller-supplied middleware.

What the forwarding start tool does with the push config:

- verbatim, when the supervisor run's ``config.configurable`` already carries
  an ``a2a_push_config`` (the canonical bite contract: ``url`` + opaque
  ``token``);
- otherwise minted from ``A2A_*`` env via ``build_push_config``, using the
  supervisor's real ``thread_id`` from ``runtime.execution_info`` and a
  per-dispatch ``dispatch_id``.

Caveats
-------
- Only public deepagents/langgraph API is used here.  The earlier revision
  imported private ``_*`` builders and monkeypatched
  ``deepagents.graph.AsyncSubAgentMiddleware``; both are gone.
- ``name`` must stay equal to the built-in class name, otherwise deepagents
  would *append* this middleware instead of replacing the built-in entry,
  leaving two ``start_async_task`` tools in the stack.
- langstrata pins ``deepagents>=0.7.0,<0.8``.  If deepagents starts forwarding
  ``config.configurable`` natively, delete this module instead of extending it.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from deepagents.middleware.async_subagents import (
    ASYNC_TASK_TOOL_DESCRIPTION,
    AsyncSubAgent,
    AsyncSubAgentMiddleware,
    StartAsyncTaskSchema,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from langgraph_sdk import get_client, get_sync_client
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
from langshark_bites.a2a_completion_notifier.push_config import build_push_config
from langshark_bites.a2a_completion_notifier.settings import A2AVerifyMode

logger = logging.getLogger(__name__)

# The built-in middleware entry this subclass replaces (see module docstring).
_BUILTIN_MIDDLEWARE_NAME = "AsyncSubAgentMiddleware"

# The one tool rebuilt here; the other four stay exactly as deepagents built them.
_START_TOOL_NAME = "start_async_task"

# Cache key for the Agent Protocol clients: (url, resolved headers).
_CacheKey = tuple[str | None, frozenset[tuple[str, str]]]


# ── Dispatch-config resolution ───────────────────────────────────────


def _dispatch_config_for(runtime: ToolRuntime) -> Mapping[str, Any] | None:
    """Build the ``config`` to forward to a subagent run, if any.

    Verbatim forwarding wins: a caller that already placed a push config
    under ``config.configurable["a2a_push_config"]`` (the canonical bite
    contract) gets it passed through untouched.  Otherwise the shim mints
    one with ``build_push_config`` from ``A2A_*`` env when both the receiver
    URL and the callback-token secret are configured.
    """
    configurable = dict((runtime.config or {}).get("configurable") or {})
    existing = configurable.get("a2a_push_config")
    if isinstance(existing, Mapping) and existing.get("url") and "token" in existing:
        return {"configurable": {"a2a_push_config": existing}}

    receiver_url = os.environ.get("A2A_RECEIVER_URL", "").strip()
    secret = os.environ.get("A2A_CALLBACK_TOKEN_SECRET", "").strip()
    if not receiver_url or not secret:
        return None

    info = getattr(runtime, "execution_info", None)
    thread_id = getattr(info, "thread_id", None) or configurable.get("thread_id")
    if not thread_id:
        return None

    return build_push_config(
        thread_id=thread_id,
        assistant_id=str(configurable.get("assistant_id") or "supervisor"),
        dispatch_id=str(configurable.get("dispatch_id") or uuid.uuid4()),
        receiver_url=receiver_url,
        callback_token_secret=secret,
        mode=A2AVerifyMode(os.environ.get("A2A_VERIFY_MODE", "dev")),
    )


# ── Agent Protocol clients (public SDK only) ─────────────────────────


class _SubAgentClients:
    """Lazily-created Agent Protocol clients keyed by (url, resolved headers).

    A public-API stand-in for deepagents' private ``_ClientCache``, so this
    module imports nothing private.  Behavior mirrors it: ``x-auth-scheme:
    langsmith`` is added unless the spec overrides it, clients are cached per
    (url, headers), and the sync path rejects ``url=None`` because ASGI
    transport is async-only.
    """

    def __init__(self, agents: dict[str, AsyncSubAgent]) -> None:
        self._agents = agents
        self._sync: dict[_CacheKey, SyncLangGraphClient] = {}
        self._async: dict[_CacheKey, LangGraphClient] = {}

    @staticmethod
    def _headers(spec: AsyncSubAgent) -> dict[str, str]:
        """Resolved headers for one spec (langsmith auth scheme by default)."""
        headers = dict(spec.get("headers") or {})
        headers.setdefault("x-auth-scheme", "langsmith")
        return headers

    def _cache_key(self, spec: AsyncSubAgent) -> _CacheKey:
        return (spec.get("url"), frozenset(self._headers(spec).items()))

    def get_sync(self, name: str) -> SyncLangGraphClient:
        """Get or create the sync client for the named subagent (HTTP only)."""
        spec = self._agents[name]
        if spec.get("url") is None:
            msg = (
                f"Async subagent '{name}' has no url configured. "
                "ASGI transport (url=None) requires async invocation."
            )
            raise ValueError(msg)
        key = self._cache_key(spec)
        if key not in self._sync:
            self._sync[key] = get_sync_client(url=spec.get("url"), headers=self._headers(spec))
        return self._sync[key]

    def get_async(self, name: str) -> LangGraphClient:
        """Get or create the async client for the named subagent."""
        spec = self._agents[name]
        key = self._cache_key(spec)
        if key not in self._async:
            self._async[key] = get_client(url=spec.get("url"), headers=self._headers(spec))
        return self._async[key]


def _validate_agent_type(agent_map: dict[str, AsyncSubAgent], agent_type: str) -> str | None:
    """Return an error message when ``agent_type`` is unknown, else ``None``."""
    if agent_type not in agent_map:
        allowed = ", ".join(f"`{name}`" for name in agent_map)
        return f"Unknown async subagent type `{agent_type}`. Available types: {allowed}"
    return None


# ── Start tool with config forwarding ───────────────────────────────


def _launch_kwargs(
    spec: AsyncSubAgent,
    thread_id: str,
    description: str,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Run-create kwargs for one subagent launch, with optional config."""
    kwargs: dict[str, Any] = {
        "thread_id": thread_id,
        "assistant_id": spec["graph_id"],
        "input": {"messages": [{"role": "user", "content": description}]},
    }
    if config is not None:
        kwargs["config"] = config
    return kwargs


def _launched_command(
    run: Any,  # noqa: ANN401 - SDK run handle, shape not part of any public type
    thread_id: str,
    subagent_type: str,
    tool_call_id: str | None,
) -> Command[Any]:
    """Build the identical ``Command`` update deepagents returns."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    task: dict[str, Any] = {
        "task_id": thread_id,
        "agent_name": subagent_type,
        "thread_id": thread_id,
        "run_id": run["run_id"],
        "status": "running",
        "created_at": now,
        "last_checked_at": now,
        "last_updated_at": now,
    }
    msg = f"Launched async subagent. task_id: {thread_id}"
    return Command(
        update={
            "messages": [ToolMessage(msg, tool_call_id=tool_call_id)],
            "async_tasks": {thread_id: task},
        }
    )


def _build_forwarding_start_tool(
    agent_map: dict[str, AsyncSubAgent],
    clients: _SubAgentClients,
    tool_description: str,
) -> StructuredTool:
    """Build ``start_async_task`` exactly like deepagents, plus ``config``."""

    def start_async_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command[Any]:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        config = _dispatch_config_for(runtime)
        try:
            client = clients.get_sync(subagent_type)
            thread = client.threads.create()
            run = client.runs.create(
                **_launch_kwargs(spec, thread["thread_id"], description, config)
            )
        except Exception as exc:  # LangGraph SDK raises untyped errors
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, exc)
            return f"Failed to launch async subagent '{subagent_type}': {exc}"
        return _launched_command(run, thread["thread_id"], subagent_type, runtime.tool_call_id)

    async def astart_async_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command[Any]:
        error = _validate_agent_type(agent_map, subagent_type)
        if error:
            return error
        spec = agent_map[subagent_type]
        config = _dispatch_config_for(runtime)
        try:
            client = clients.get_async(subagent_type)
            thread = await client.threads.create()
            run = await client.runs.create(
                **_launch_kwargs(spec, thread["thread_id"], description, config)
            )
        except Exception as exc:  # LangGraph SDK raises untyped errors
            logger.warning("Failed to launch async subagent '%s': %s", subagent_type, exc)
            return f"Failed to launch async subagent '{subagent_type}': {exc}"
        return _launched_command(run, thread["thread_id"], subagent_type, runtime.tool_call_id)

    return StructuredTool.from_function(
        name=_START_TOOL_NAME,
        func=start_async_task,
        coroutine=astart_async_task,
        description=tool_description,
        infer_schema=False,
        args_schema=StartAsyncTaskSchema,
    )


def _swap_start_tool(
    tools: Sequence[BaseTool],
    agents: list[AsyncSubAgent],
) -> list[BaseTool]:
    """Replace deepagents' stock ``start_async_task`` with the forwarding one.

    Only that one tool is rebuilt; the four monitoring tools are reused
    verbatim, so they keep deepagents' own client cache and behavior.  The
    rebuilt tool gets its own cache (deepagents does not expose the stock one),
    which is harmless: clients are cached per (url, headers) either way.
    """
    agent_map: dict[str, AsyncSubAgent] = {a["name"]: a for a in agents}
    clients = _SubAgentClients(agent_map)
    agents_desc = "\n".join(f"- {a['name']}: {a['description']}" for a in agents)
    launch_desc = ASYNC_TASK_TOOL_DESCRIPTION.format(available_agents=agents_desc)
    forwarding = _build_forwarding_start_tool(agent_map, clients, launch_desc)

    swapped: list[BaseTool] = []
    replaced = False
    for tool in tools:
        if tool.name == _START_TOOL_NAME:
            swapped.append(forwarding)
            replaced = True
        else:
            swapped.append(tool)

    if not replaced:
        # deepagents renamed/removed its start tool: fail loudly rather than
        # ship a supervisor that silently never forwards the push config.
        msg = (
            f"deepagents' async subagent tools carry no {_START_TOOL_NAME!r} "
            f"(got {[t.name for t in tools]}); cannot forward the a2a push config"
        )
        raise RuntimeError(msg)
    return swapped


# ── The middleware deepagents swaps in ─────────────────────────────


class ConfigForwardingAsyncSubAgentMiddleware(AsyncSubAgentMiddleware):
    """``AsyncSubAgentMiddleware`` whose start tool forwards a push config.

    Pass an instance to ``create_deep_agent(..., middleware=[...])`` alongside
    the same ``subagents`` specs: deepagents matches middleware by ``name`` and
    replaces the built-in entry with this instance, so the stack position, the
    four monitoring tools and the async-task state schema are unchanged.
    """

    @property
    def name(self) -> str:
        """The built-in entry's name, so deepagents swaps this in place.

        ``AgentMiddleware.name`` defaults to the class name and is documented
        as overridable for custom naming.  Without this override deepagents
        would *append* this middleware, leaving two ``start_async_task`` tools.
        """
        return _BUILTIN_MIDDLEWARE_NAME

    def __init__(
        self,
        *,
        async_subagents: list[AsyncSubAgent],
        system_prompt: str | None = None,
    ) -> None:
        """Build the stock tool stack, then swap in the forwarding start tool."""
        super().__init__(async_subagents=async_subagents, system_prompt=system_prompt)
        self.tools = _swap_start_tool(self.tools, async_subagents)
