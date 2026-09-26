"""langstrata's ``a2a_completion_notifier`` integration.

Brings the langshark-bites ``a2a_completion_notifier`` to langstrata's
split-server topology.  The pieces:

- ``ConfigForwardingAsyncSubAgentMiddleware`` -- the deepagents middleware that
  forwards ``a2a_push_config`` into each subagent run (deepagents' built-in
  ``AsyncSubAgentMiddleware`` drops it today).  Pass an instance to
  ``create_deep_agent(middleware=[...])``; it keeps the built-in ``name`` so
  deepagents swaps it in place instead of appending a duplicate.
- ``build_worker_middleware`` -- the attach-or-skip step worker graph
  factories call with the run's ``config``; attaches the emitter only when
  the run was dispatched with a push config.
- ``build_emitter_signer`` -- RS256 signer from ``A2A_*`` env, or ``None``
  in dev mode without a key.
- ``webapp`` -- **default** receiver packaging: bite ``create_receiver_app``
  mounted on the supervisor Agent Server via ``langgraph.json`` ``http.app``
  (same origin as graphs; set ``A2A_RECEIVER_URL`` to the supervisor URL).
- ``supervisor_mcp`` -- ``discover_supervisor_mcp_tools`` attaches the
  bridge's diagnostic tools when an external MCP bridge is configured
  (``A2A_SUPERVISOR_MCP_URL``).  Embedded ``http.app`` mode leaves tools
  empty unless that bridge is present.

The supervisor drain is the bite's single ``MailboxDrainMiddleware``, which
now covers both the platform/Agent Server ``runtime.store`` path and
``langgraph dev`` via an HTTP ``StoreClient`` fallback (``A2A_SUPERVISOR_URL``).

Public API
----------
- ``ConfigForwardingAsyncSubAgentMiddleware(async_subagents=...)`` -- pass to
  ``create_deep_agent(middleware=[...])`` when ``A2A_COMPLETION_NOTIFIER_ENABLED=true``.
- ``build_worker_middleware(config)`` -- returns the emitter middleware
  list for a worker graph factory.
- ``build_emitter_signer()`` -- the subagent deployment signer, or None.
"""

from __future__ import annotations

from langstrata.a2a.a2a_dispatch_forwarder import ConfigForwardingAsyncSubAgentMiddleware
from langstrata.a2a.worker_emitter import build_emitter_signer, build_worker_middleware

__all__ = [
    "ConfigForwardingAsyncSubAgentMiddleware",
    "build_emitter_signer",
    "build_worker_middleware",
]
