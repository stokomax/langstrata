"""Regression tests relocated from the former ``scripts/a2a_demo.py``.

The standalone demo was removed: it confused users into thinking it launched a
server.  Two pieces of durable coverage it proved -- which the bite's own suite
does not cover because they are langstrata glue -- live here instead:

1. ``build_worker_middleware`` attach-or-skip: langstrata's worker graph
   factory attaches the A2A emitter only when the run is dispatched with an
   ``a2a_push_config``, and builds the plain worker otherwise.
2. ``supervisor_mcp`` discovery through the bite's ``create_supervisor_tools``
   against a REAL ``mcpServers`` stdio subprocess + the reconnect-per-call
   regression (the "nesting counter should be 0" crash) and the run-off-the-
   event-loop discovery contract.
"""

from __future__ import annotations

import socket
import warnings
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)



RECEIVER_URL = "http://receiver.example"
CALLBACK_SECRET = "langstrata-test-secret-0123456789abcdef"


def _free_port() -> int:
    """Return a currently-free localhost TCP port for the spawned bridge."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _push_config(thread_id: str = "supervisor-thread-1") -> dict:
    from langshark_bites.a2a_completion_notifier.push_config import build_push_config

    return build_push_config(
        thread_id=thread_id,
        assistant_id="supervisor",
        dispatch_id=f"dispatch-{thread_id[:8]}",
        receiver_url=RECEIVER_URL,
        callback_token_secret=CALLBACK_SECRET,
    )


class TestBuildWorkerMiddlewareAttachOrSkip:
    """The worker graph factory's attach-or-skip step (langstrata glue)."""

    @staticmethod
    def _has_emitter_nodes(graph) -> bool:
        """The A2A emitter middleware compiles into graph nodes named
        ``A2APushNotifierMiddleware.before_agent`` / ``.after_agent``.
        """
        assert hasattr(graph, "nodes"), "expected a compiled graph"
        return any("A2APushNotifierMiddleware." in name for name in graph.nodes)

    def test_dispatched_run_attaches_emitter(self) -> None:
        from langstrata.workers.researcher import create_researcher_agent

        graph = create_researcher_agent(_push_config())
        assert self._has_emitter_nodes(graph)

    def test_plain_run_builds_without_emitter(self) -> None:
        from langstrata.workers.researcher import create_researcher_agent

        graph = create_researcher_agent({"configurable": {}})
        assert not self._has_emitter_nodes(graph)

    def test_none_config_is_plain(self) -> None:
        from langstrata.workers.researcher import create_researcher_agent

        graph = create_researcher_agent(None)
        assert not self._has_emitter_nodes(graph)


class TestCreateSupervisorToolsStdioSubprocess:
    """`create_supervisor_tools` against a real `mcpServers` stdio subprocess.

    This is the exact path the supervisor's eager discovery uses at launch:
    declare the bridge in a standard mcpServers block and let MCPAdapter
    spawn it.  The regression locks:
    - the four bridge tools are discovered;
    - a SECOND consecutive call to the same tool succeeds (the fastmcp
      "nesting counter should be 0" crash);
    - discovery is off the event loop (blockbuster / jsonschema prewarm).

    SKIPPED: The langshark-bites package no longer includes an `mcp_server`
    module (the optional MCP diagnostic bridge was removed in favor of the
    embedded HTTP receiver).  This test requires a real MCP server to exercise
    the reconnect-per-call behavior.  Re-enable if a suitable test server is
    added to langshark-bites or langstrata.
    """

    _SKIP_REASON = (
        "langshark-bites mcp_server module removed; "
        "no stdio MCP server available for testing"
    )

    @pytest.mark.skip(reason=_SKIP_REASON)
    def _mcp_config(self) -> dict:  # pragma: no cover
        ...

    @pytest.mark.skip(reason=_SKIP_REASON)
    @pytest.mark.asyncio
    async def test_discovers_tools_and_reconnects_per_call(self) -> None:  # pragma: no cover
        ...

    @pytest.mark.asyncio
    async def test_graph_module_import_prewarms_jsonschema(self) -> None:
        """The mcp SDK lazily imports jsonschema on the first call_tool; that
        import scans a directory (jsonschema_specifications' registry), which
        blockbuster rejects inside a running loop.  `supervisor/agent.py` (the
        graph `langgraph dev` imports at startup, before blockbuster exists)
        must have already warmed the subtree -- this is the fix for the
        first-live-`get_async_result` BlockingError.
        """
        import sys as _sys

        from langstrata.supervisor import agent as _agent  # noqa: F401 - runs prewarm at import

        assert "jsonschema.exceptions" in _sys.modules
        assert "jsonschema.validators" in _sys.modules
        assert "jsonschema_specifications" in _sys.modules


class _FakeRuntime:
    """Stand-in for `ToolRuntime`: only the attributes the tools read."""

    def __init__(self, config: dict, tool_call_id: str = "call-1") -> None:
        self.config = config
        self.tool_call_id = tool_call_id
        self.execution_info = SimpleNamespace(thread_id="supervisor-thread-1")


class _FakeRuns:
    """Captures `runs.create` kwargs instead of calling an Agent Protocol server."""

    def __init__(self, captured: dict) -> None:
        self._captured = captured

    async def create(self, **kwargs: object) -> dict:
        self._captured.update(kwargs)
        return {"run_id": "run-1"}


class _FakeThreads:
    async def create(self) -> dict:
        return {"thread_id": "subagent-thread-1"}


class _FakeClient:
    def __init__(self, captured: dict) -> None:
        self.threads = _FakeThreads()
        self.runs = _FakeRuns(captured)


def _fake_get_async(captured: dict) -> Callable[[object, str], _FakeClient]:
    """Return a `_SubAgentClients.get_async` replacement yielding a fake client."""

    def get_async(_self: object, _name: str) -> _FakeClient:
        return _FakeClient(captured)

    return get_async


_SUBAGENT_SPECS = [{"name": "researcher", "description": "research", "graph_id": "researcher"}]


class TestDispatchConfigForwarding:
    """The deepagents middleware that forwards `a2a_push_config` to subagents.

    deepagents' built-in async-subagent middleware drops the parent run's
    `config.configurable` on dispatch, so the worker-side emitter would never
    receive a push config.  These lock the replacement mechanism and the
    forwarding behavior.
    """

    _SPECS = _SUBAGENT_SPECS

    def _middleware(self) -> object:
        from langstrata.a2a.a2a_dispatch_forwarder import ConfigForwardingAsyncSubAgentMiddleware

        return ConfigForwardingAsyncSubAgentMiddleware(async_subagents=list(self._SPECS))

    @staticmethod
    def _start_tool(middleware) -> object:
        return next(t for t in middleware.tools if t.name == "start_async_task")

    def test_name_matches_builtin_entry(self) -> None:
        """Deepagents merges caller middleware by `name`, replacing in place.

        A different name would APPEND ours and leave two `start_async_task`
        tools in the stack -- hence this exact value is load-bearing.
        """
        assert self._middleware().name == "AsyncSubAgentMiddleware"

    def test_only_start_tool_is_rebuilt(self) -> None:
        """The four monitoring tools stay exactly as deepagents built them."""
        middleware = self._middleware()
        assert [t.name for t in middleware.tools] == [
            "start_async_task",
            "check_async_task",
            "update_async_task",
            "cancel_async_task",
            "list_async_tasks",
        ]
        assert self._start_tool(middleware).coroutine.__qualname__.startswith(
            "_build_forwarding_start_tool"
        )

    def test_deepagents_replaces_builtin_in_place(self) -> None:
        """Pin the merge contract: one entry afterwards, and it is ours.

        Uses deepagents' own merge helper deliberately -- if it ever stops
        replacing by `name`, this fails loudly instead of duplicating tools.
        """
        from deepagents.graph import _apply_custom_middleware
        from deepagents.middleware.async_subagents import AsyncSubAgentMiddleware

        stock = AsyncSubAgentMiddleware(async_subagents=list(self._SPECS))
        ours = self._middleware()
        merged = _apply_custom_middleware([stock], [ours])
        assert len(merged) == 1
        assert merged[0] is ours

    @pytest.mark.asyncio
    async def test_forwards_existing_push_config(self, monkeypatch) -> None:
        """A push config already on the supervisor run passes through verbatim."""
        from langstrata.a2a import a2a_dispatch_forwarder as shim

        captured: dict = {}
        monkeypatch.setattr(shim._SubAgentClients, "get_async", _fake_get_async(captured))

        middleware = self._middleware()
        push = _push_config("supervisor-thread-1")["configurable"]["a2a_push_config"]
        runtime = _FakeRuntime(config={"configurable": {"a2a_push_config": push}})
        result = await self._start_tool(middleware).coroutine(
            description="do it", subagent_type="researcher", runtime=runtime
        )

        forwarded = captured["config"]["configurable"]["a2a_push_config"]
        assert forwarded == push  # verbatim pass-through, not re-minted
        assert captured["thread_id"] == "subagent-thread-1"
        assert captured["assistant_id"] == "researcher"
        assert "Launched async subagent" in str(result)

    @pytest.mark.asyncio
    async def test_mints_push_config_from_env(self, monkeypatch) -> None:
        """With no push config on the run, one is minted from `A2A_*` env."""
        from langstrata.a2a import a2a_dispatch_forwarder as shim

        captured: dict = {}
        monkeypatch.setattr(shim._SubAgentClients, "get_async", _fake_get_async(captured))
        monkeypatch.setenv("A2A_RECEIVER_URL", RECEIVER_URL)
        monkeypatch.setenv("A2A_CALLBACK_TOKEN_SECRET", CALLBACK_SECRET)
        monkeypatch.setenv("A2A_VERIFY_MODE", "dev")

        middleware = self._middleware()
        runtime = _FakeRuntime(config={"configurable": {}})
        await self._start_tool(middleware).coroutine(
            description="do it", subagent_type="researcher", runtime=runtime
        )

        forwarded = captured["config"]["configurable"]["a2a_push_config"]
        # The bite appends its webhook path to the configured receiver base URL.
        assert forwarded["url"] == f"{RECEIVER_URL}/a2a/notifications"
        assert "token" in forwarded

    @pytest.mark.asyncio
    async def test_no_config_when_unconfigured(self, monkeypatch) -> None:
        """No push config on the run and no env -> nothing is forwarded."""
        from langstrata.a2a import a2a_dispatch_forwarder as shim

        captured: dict = {}
        monkeypatch.setattr(shim._SubAgentClients, "get_async", _fake_get_async(captured))
        monkeypatch.delenv("A2A_RECEIVER_URL", raising=False)
        monkeypatch.delenv("A2A_CALLBACK_TOKEN_SECRET", raising=False)

        middleware = self._middleware()
        runtime = _FakeRuntime(config={"configurable": {}})
        await self._start_tool(middleware).coroutine(
            description="do it", subagent_type="researcher", runtime=runtime
        )

        assert "config" not in captured

    def test_swap_fails_loudly_without_start_tool(self) -> None:
        """If deepagents renames its start tool, raise rather than go silent."""
        from langchain_core.tools import StructuredTool

        from langstrata.a2a import a2a_dispatch_forwarder as shim

        renamed = StructuredTool.from_function(
            func=lambda: "x", name="renamed_start", description="not the start tool"
        )
        with pytest.raises(RuntimeError, match="cannot forward the a2a push config"):
            shim._swap_start_tool([renamed], list(self._SPECS))
