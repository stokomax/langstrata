"""Analyst worker — an agent specialised in data analysis and visualisation."""

from langchain.tools import tool

from langstrata.config import Settings

from langstrata.workers.base import create_worker_agent


@tool
def run_query(dataset: str, query: str) -> str:
    """Run a data query against the specified dataset (CSV, Parquet, etc.)."""
    return f"[run_query on {dataset}: {query}] (placeholder)"


@tool
def visualize(data: str, chart_type: str) -> str:
    """Generate a chart (bar, line, scatter, etc.) from the provided data."""
    return f"[visualize: {chart_type}] (placeholder — would return image URL or base64)"


graph = create_worker_agent(
    model=Settings().model,
    tools=[run_query, visualize],
    system_prompt=(
        "You are a data analysis specialist. Interpret data, identify trends, "
        "create clear visualisations, and explain findings in plain language."
    ),
)
