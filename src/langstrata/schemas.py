"""Shared Pydantic models for langstrata supervisor and workers."""

from typing import Any

from pydantic import BaseModel


class WorkerTask(BaseModel):
    """A task description routed to a worker."""

    worker_name: str
    description: str
    input: dict[str, Any] = {}


class WorkerResult(BaseModel):
    """Structured result returned by a worker."""

    worker_name: str
    status: str  # "success" | "error"
    summary: str
    raw_output: dict[str, Any] | None = None
