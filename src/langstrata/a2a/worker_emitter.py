"""Emitter-side wiring: attach the A2A notifier middleware to worker graphs.

Why this exists
---------------
langstrata workers are built by ``create_worker_agent`` (``create_agent``).
The langshark-bites emitter is an ``AgentMiddleware`` that must be attached
*per run*, because the webhook target + opaque callback token arrive per
dispatch in ``config.configurable["a2a_push_config"]``.  The middleware is
constructed by ``build_a2a_notifier_from_config`` and cannot be shared
across runs.

``build_worker_middleware`` is the attach-or-skip step worker graph
factories call with the run's ``config``:

- dispatched with ``a2a_push_config`` -> ``[A2APushNotifierMiddleware]``;
- any other run (plain invocation, no push config) -> ``[]`` so the worker
  behaves exactly as before the integration.

The signer comes from the subagent deployment's ``A2A_*`` env
(``EmitterSettings``); in dev mode there is intentionally no signing key, so
notifications go out unsigned — accepted by ``dev`` receivers, rejected by
``strict`` ones.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from langshark_bites.a2a_completion_notifier.middleware import (
    PushNotificationConfigError,
    build_a2a_notifier_from_config,
)
from langshark_bites.a2a_completion_notifier.push_client import PushClient
from langshark_bites.a2a_completion_notifier.settings import EmitterSettings
from langshark_bites.a2a_completion_notifier.signer import A2ASigner

log = logging.getLogger(__name__)


def build_emitter_signer() -> A2ASigner | None:
    """Build the subagent deployment's RS256 signer from ``A2A_*`` env.

    Returns:
        A configured ``A2ASigner`` when a private key is present, or ``None``
        when it is not (the dev-mode default) so notifications are sent
        unsigned.
    """
    settings = EmitterSettings.from_env()
    if not settings.private_key_pem:
        return None
    return A2ASigner(
        private_key_pem=settings.private_key_pem,
        kid=settings.kid,
        issuer=settings.issuer or "https://langstrata-worker",
        audience=settings.audience or os.environ.get("A2A_RECEIVER_URL", "http://localhost:2024"),
    )


def build_worker_middleware(config: Mapping[str, Any] | None) -> list[Any]:
    """Return the A2A emitter middleware for one dispatched worker run.

    Args:
        config: The runnable ``config`` (containing ``configurable``) for the
            worker graph being built.

    Returns:
        ``[A2APushNotifierMiddleware]`` when the run was dispatched with an
        ``a2a_push_config``, otherwise ``[]``.
    """
    if not config:
        return []
    configurable = config.get("configurable") or {}
    if not configurable.get("a2a_push_config"):
        return []
    try:
        notifier = build_a2a_notifier_from_config(
            config,
            signer=build_emitter_signer(),
            push_client=PushClient(),
        )
    except PushNotificationConfigError as exc:
        log.warning("a2a_emitter_skip reason=%s", exc)
        return []
    log.info("a2a_emitter_attached url=%s", configurable["a2a_push_config"].get("url"))
    return [notifier]
