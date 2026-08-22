from functools import lru_cache
from typing import Any, Callable

from .config import get_settings


@lru_cache
def is_enabled() -> bool:
    settings = get_settings()
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


@lru_cache
def _client() -> Any:
    """Constructs the Langfuse client once, explicitly from Settings (which reads .env
    via pydantic-settings) rather than letting the SDK fall back to its own os.environ
    read. Without this, get_callbacks()/observe() below would work by coincidence in
    Docker -- where `env_file:` also happens to populate the process environment -- but
    silently use the wrong host (or none) running the app straight from .venv, where
    only Settings sees .env. get_callbacks()/observe() call this before touching
    anything from the langfuse package, so CallbackHandler()/observe()'s own internal
    get_client() lookups resolve to this same instance instead of constructing a second,
    env-derived one."""
    from langfuse import Langfuse

    settings = get_settings()
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def get_callbacks() -> list[Any]:
    """LangChain callback handlers for demos built on LangChain; empty (no-op) if unconfigured."""
    if not is_enabled():
        return []
    _client()
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def observe(**kwargs: Any) -> Callable[[Callable], Callable]:
    """Decorator for plain-Python demos; a no-op passthrough if Langfuse isn't configured."""
    if not is_enabled():
        def _passthrough(fn: Callable) -> Callable:
            return fn

        return _passthrough

    _client()
    from langfuse import observe as _observe

    return _observe(**kwargs)
