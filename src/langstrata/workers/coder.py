"""Coder worker — an agent specialised in code generation and review."""

from langchain.tools import tool

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


graph = create_worker_agent(
    model=Settings().model,
    tools=[read_file, write_file, run_command],
    system_prompt=(
        "You are a coding specialist. Write clean, well-tested code. "
        "Review provided code for bugs, security issues, and style."
    ),
)
