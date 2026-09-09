"""Runtime configuration for langstrata.

The mode toggle controls whether workers run co-deployed (ASGI) or
on a separate server (HTTP).  Change by setting AGENT_SERVER_MODE env var.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Topology and model configuration for a langstrata deployment."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_SERVER_",
        env_file=".env",
        extra="ignore",
    )

    # ── Topology ──────────────────────────────────────────────────
    mode: Literal["asgi", "http"] = "http"
    """asgi → workers co-deployed on same server (ASGI transport).
       http → workers on a separate server via Agent Protocol."""

    # ── Supervisor model ──────────────────────────────────────────
    model: str = "anthropic:claude-sonnet-5"
    """LLM model for the supervisor agent (provider:model format)."""

    system_prompt: str = (
        "You are a coordinating supervisor. "
        "Delegate tasks to specialist workers and synthesize their results."
    )

    # ── Worker server endpoint ────────────────────────────────────
    worker_api_url: str = "http://worker:2024"
    """Base URL of the worker Agent Protocol server (used in http mode)."""

    # ── Per-worker URL overrides (optional, for hybrid setups) ────
    researcher_url: str | None = None
    """Override the worker URL for the researcher sub-agent only."""
    coder_url: str | None = None
    """Override the worker URL for the coder sub-agent only."""
    analyst_url: str | None = None
    """Override the worker URL for the analyst sub-agent only."""
