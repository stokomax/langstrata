"""Researcher worker — an agent specialised in information gathering."""

from collections.abc import Mapping
from typing import Any

from langchain.tools import tool

from langstrata.a2a.worker_emitter import build_worker_middleware
from langstrata.config import Settings
from langstrata.workers.base import create_worker_agent


@tool
def web_search(query: str) -> str:
    """Search the web for information.

    Use this for current events, facts, or any topic requiring up-to-date
    sources.
    """
    return (
        f"[web_search result for: {query}] (placeholder — implement with Tavily, DuckDuckGo, etc.)"
    )


@tool
def fetch_url(url: str) -> str:
    """Fetch and summarise the content of a URL."""
    return f"[fetch_url result for: {url}] (placeholder — implement with requests/html parser)"


def create_researcher_agent(
    config: Mapping[str, Any] | None = None,
) -> Any:  # noqa: ANN401 - returns the compiled worker graph (dynamic generic)
    """Dynamic graph factory for the researcher worker.

    Registered in ``langgraph.json``/``langgraph.workers.json`` so the server
    rebuilds the graph per run.  When a run is dispatched with an
    ``a2a_push_config`` (deepagents subagent dispatch with the a2a shim
    installed), the A2A emitter middleware is attached; any other invocation
    builds the plain worker.
    """
    return create_worker_agent(
        model=Settings().model,
        tools=[web_search, fetch_url],
        system_prompt=(
            "You are a research specialist. Gather information thoroughly, "
            "synthesise findings, and present a concise summary."
        ),
        middleware=build_worker_middleware(config),
    )


graph = create_researcher_agent()
