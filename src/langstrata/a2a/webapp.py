"""Agent Server custom app: embed the A2A completion receiver via ``http.app``.

langstrata's default packaging mounts the langshark-bites completion-notifier
receiver on the **supervisor** Agent Server (see ``langgraph.json`` ``http.app``).
Workers POST completions to the supervisor origin; no separate :8001 process.

Paths (bite wire contract — origin only in ``A2A_RECEIVER_URL``):

- ``POST /a2a/notifications`` — completion webhook
- ``POST /a2a/result`` — full task result fetch
- ``GET /health`` — liveness

Reserved ``/a2a/`` leaves for the notifier: ``notifications``, ``result``.
Built-in Google A2A remains at ``/a2a/{assistant_id}``.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from langshark_bites.a2a_completion_notifier.receiver import create_receiver_app
from langshark_bites.a2a_completion_notifier.settings import ReceiverSettings

log = logging.getLogger(__name__)

_A2A_ENABLED_TRUTHY = frozenset({"1", "true", "yes", "on"})


def build_app() -> FastAPI:
    """Build the FastAPI app mounted by Agent Server ``http.app``.

    When ``A2A_COMPLETION_NOTIFIER_ENABLED`` is not true, returns an empty
    FastAPI app so the supervisor still boots without A2A.  When enabled,
    returns the bite's ``create_receiver_app`` so webhook routes share the
    supervisor origin.

    Returns:
        A FastAPI application instance suitable for ``langgraph.json`` ``http.app``.
    """
    enabled = os.environ.get("A2A_COMPLETION_NOTIFIER_ENABLED", "").strip().lower()
    if enabled not in _A2A_ENABLED_TRUTHY:
        log.info("a2a_http_app_disabled")
        return FastAPI(title="langstrata-a2a-disabled")

    settings = ReceiverSettings.from_env()
    log.info(
        "a2a_http_app_enabled supervisor_url=%s receiver_url=%s mode=%s",
        settings.supervisor_url,
        settings.receiver_url,
        settings.mode,
    )
    return create_receiver_app(settings=settings)


app = build_app()
