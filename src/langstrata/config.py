"""Runtime configuration for langstrata.

The mode toggle controls whether workers run co-deployed (ASGI) or
on a separate server (HTTP).  Change by setting AGENT_SERVER_MODE env var.
"""

from typing import Literal

from pydantic import Field
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
    model: str = "anthropic:claude-sonnet-4-6"
    """LLM model for the supervisor agent (provider:model format)."""

    system_prompt: str = (
        "You are a coordinating supervisor. "
        "Delegate tasks to specialist workers and synthesize their results."
    )

    # ── Worker server endpoint ────────────────────────────────────
    worker_api_url: str = "http://worker:2025"
    """Base URL of the worker Agent Protocol server (used in http mode)."""

    # ── Per-worker URL overrides (optional, for hybrid setups) ────
    researcher_url: str | None = None
    """Override the worker URL for the researcher sub-agent only."""
    coder_url: str | None = None
    """Override the worker URL for the coder sub-agent only."""
    analyst_url: str | None = None
    """Override the worker URL for the analyst sub-agent only."""

    # ── A2A completion notifier (langshark-bites) ─────────────────
    # No AGENT_SERVER_ prefix: this toggle is part of the A2A_* family
    # (A2A_VERIFY_MODE, A2A_CALLBACK_TOKEN_SECRET, ...), not an Agent Server setting.
    a2a_enabled: bool = Field(default=False, validation_alias="A2A_COMPLETION_NOTIFIER_ENABLED")
    """Opt in to the langshark-bites ``a2a_completion_notifier``.

    Set A2A_COMPLETION_NOTIFIER_ENABLED=true on both servers to: install the
    deepagents config-forwarding middleware on the supervisor, attach the
    ``MailboxDrainMiddleware`` to the supervisor graph, and let worker graph
    factories attach the A2A emitter when dispatched with a push config.

    The bite itself is configured through its own ``A2A_*`` environment
    variables (``A2A_VERIFY_MODE``, ``A2A_CALLBACK_TOKEN_SECRET``,
    ``A2A_RECEIVER_URL``, ``A2A_SUPERVISOR_URL``, ...) — see
    ``langshark_bites.a2a_completion_notifier.settings``.
    """
