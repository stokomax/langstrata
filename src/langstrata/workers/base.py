"""Worker agent factory — shared helper for building worker graphs."""

from typing import Any

from langchain.agents import create_agent


def create_worker_agent(
    *,
    model: str,
    tools: list[Any],
    system_prompt: str,
    middleware: list[Any] | None = None,
    **kwargs: Any,  # noqa: ANN401 - forwarded verbatim to create_agent (varied langchain options)
) -> Any:  # noqa: ANN401 - langchain's CompiledStateGraph is a dynamic generic we don't specialize
    """Build a standalone LangGraph agent suitable for LangGraph Server registration.

    Each worker graph is a compiled ``create_agent(...)`` graph that can
    be registered in ``langgraph.json`` and invoked over HTTP via the
    Agent Protocol.

    Args:
        model: LLM model or model spec for the worker.
        tools: Tool list for the worker.
        system_prompt: System prompt for the worker.
        middleware: Optional ``AgentMiddleware`` list.  langstrata uses this
            to attach the a2a_completion_notifier emitter for dispatched
            subagent runs (see ``langstrata.a2a.build_worker_middleware``).
        **kwargs: Forwarded unchanged to ``create_agent`` (e.g. ``store``).
    """
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware or [],
        **kwargs,
    )
