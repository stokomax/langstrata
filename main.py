"""langstrata entrypoint — scaffold smoke test."""

import os

from langstrata.config import Settings

_PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _check_api_key(settings: Settings) -> None:
    """Warn if the API key for the configured model provider is missing."""
    model = settings.model
    provider = model.split(":", 1)[0] if ":" in model else None
    if provider is None:
        return
    env_var = _PROVIDER_KEY_MAP.get(provider)
    if env_var is None:
        return
    if not os.environ.get(env_var, "").strip():
        print(f"  [warning] {env_var} is not set — {provider} model will fail at runtime")


def _show_demo_prompt_mode() -> None:
    """Show which demo prompt mode is configured."""
    mode = os.environ.get("DEMO_PROMPT_MODE", "abridged")
    print(f"  demo_prompt_mode={mode}  # set DEMO_PROMPT_MODE=abridged|full")


def main() -> None:
    settings = Settings()
    print(f"Hello from langstrata!")
    print(f"  mode={settings.mode}              # set AGENT_SERVER_MODE=http|asgi")
    print(f"  model={settings.model}            # set AGENT_SERVER_MODEL=provider:model")
    print(f"  worker_api_url={settings.worker_api_url}  # set AGENT_SERVER_WORKER_API_URL=<url>")
    print(f"  env prefix: AGENT_SERVER_          # all config vars use this prefix")
    _check_api_key(settings)
    _show_demo_prompt_mode()


if __name__ == "__main__":
    main()
