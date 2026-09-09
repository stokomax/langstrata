"""Researcher worker — an agent specialised in information gathering."""

from langchain.tools import tool

from langstrata.config import Settings

from langstrata.workers.base import create_worker_agent


@tool
def web_search(query: str) -> str:
    """Search the web for information. Use this for current events,
    facts, or any topic requiring up-to-date sources."""
    return (
        f"[web_search result for: {query}] (placeholder — implement with Tavily, DuckDuckGo, etc.)"
    )


@tool
def fetch_url(url: str) -> str:
    """Fetch and summarise the content of a URL."""
    return f"[fetch_url result for: {url}] (placeholder — implement with requests/html parser)"


graph = create_worker_agent(
    model=Settings().model,
    tools=[web_search, fetch_url],
    system_prompt=(
        "You are a research specialist. Gather information thoroughly, "
        "synthesise findings, and present a concise summary."
    ),
)
