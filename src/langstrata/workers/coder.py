"""Coder worker — an agent specialised in code generation and review."""

from collections.abc import Mapping
from typing import Any

from langchain.tools import tool

from langstrata.a2a.worker_emitter import build_worker_middleware
from langstrata.config import Settings
from langstrata.workers.base import create_worker_agent


@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    return f"[read_file: {path}] (placeholder)"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path."""
    return f"[write_file: {path}] (placeholder — {len(content)} chars)"


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its output."""
    return f"[run_command: {command}] (placeholder)"


def create_coder_agent(
    config: Mapping[str, Any] | None = None,
) -> Any:  # noqa: ANN401 - returns the compiled worker graph (dynamic generic)
    """Dynamic graph factory for the coder worker.

    Registered in ``langgraph.json``/``langgraph.workers.json`` so the server
    rebuilds the graph per run.  When a run is dispatched with an
    ``a2a_push_config`` (deepagents subagent dispatch with the a2a shim
    installed), the A2A emitter middleware is attached; any other invocation
    builds the plain worker.
    """
    return create_worker_agent(
        model=Settings().model,
        tools=[read_file, write_file, run_command],
        system_prompt=(
            "You are a coding specialist. Write clean, well-tested code. "
            "Review provided code for bugs, security issues, and style."
        ),
        middleware=build_worker_middleware(config),
    )


graph = create_coder_agent()
