"""Analyst worker — an agent specialised in data analysis and visualisation."""

from collections.abc import Mapping
from typing import Any

from langchain.tools import tool

from langstrata.a2a.worker_emitter import build_worker_middleware
from langstrata.config import Settings
from langstrata.workers.base import create_worker_agent


@tool
def run_query(dataset: str, query: str) -> str:
    """Run a data query against the specified dataset (CSV, Parquet, etc.)."""
    return f"[run_query on {dataset}: {query}] (placeholder)"


@tool
def visualize(data: str, chart_type: str) -> str:  # noqa: ARG001 - data is a declared tool input (schema); the placeholder ignores it
    """Generate a chart (bar, line, scatter, etc.) from the provided data."""
    return f"[visualize: {chart_type}] (placeholder — would return image URL or base64)"


def create_analyst_agent(
    config: Mapping[str, Any] | None = None,
) -> Any:  # noqa: ANN401 - returns the compiled worker graph (dynamic generic)
    """Dynamic graph factory for the analyst worker.

    Registered in ``langgraph.json``/``langgraph.workers.json`` so the server
    rebuilds the graph per run.  When a run is dispatched with an
    ``a2a_push_config`` (deepagents subagent dispatch with the a2a shim
    installed), the A2A emitter middleware is attached; any other invocation
    builds the plain worker.
    """
    return create_worker_agent(
        model=Settings().model,
        tools=[run_query, visualize],
        system_prompt=(
            "You are a data analysis specialist. Interpret data, identify trends, "
            "create clear visualisations, and explain findings in plain language."
        ),
        middleware=build_worker_middleware(config),
    )


graph = create_analyst_agent()
