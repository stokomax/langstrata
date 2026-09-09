"""Worker agent factory — shared helper for building worker graphs."""

from typing import Any

from langchain.agents import create_agent


def create_worker_agent(
    *,
    model: str,
    tools: list[Any],
    system_prompt: str,
    **kwargs: Any,
) -> Any:
    """Build a standalone LangGraph agent suitable for registration as a
    LangGraph Server assistant.

    Each worker graph is a compiled ``create_agent(...)`` graph that can
    be registered in ``langgraph.json`` and invoked over HTTP via the
    Agent Protocol.
    """
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        **kwargs,
    )
