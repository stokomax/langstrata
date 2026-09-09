"""Supervisor agent factory — dynamically builds AsyncSubAgent list from config."""

from typing import Any

from deepagents import AsyncSubAgent, create_deep_agent

from langstrata.config import Settings

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
        url = _resolve_url(name, config)
        subagents.append(
            AsyncSubAgent(
                name=name,
                description=description,
                graph_id=name,
                url=url,  # None → ASGI; str → HTTP
            )
        )
    return subagents


def create_supervisor_agent() -> Any:
    """Build the supervisor Deep Agent.

    Reads runtime settings from environment variables (``AGENT_SERVER_*``)
    so it works without arguments in ``langgraph dev``.
    """
    config = Settings()

    async_subagents = _build_async_subagents(config)

    return create_deep_agent(
        model=config.model,
        system_prompt=config.system_prompt,
        subagents=async_subagents,  # ← the core wiring
    )
