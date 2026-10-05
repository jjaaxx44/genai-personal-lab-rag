import warnings
from functools import lru_cache

from langchain_core.runnables import Runnable
from langchain_typesafe import TypeSafeClassifier
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from .config import get_settings

# TypeSafeClassifier does no retrying of its own (the standalone typesafe-sdk does), so
# transient failures are retried here through LangChain's .with_retry(). Auth and
# bad-request errors are deliberately absent: retrying them can't help.
_RETRYABLE = (
    TypeSafeRateLimitError,
    TypeSafeInternalServerError,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
)
_MAX_ATTEMPTS = 3


def is_configured() -> bool:
    """True when a TypeSafe key is set. Check this before get_classifier(): the classifier
    raises ValueError on an empty key, which would otherwise surface as a stack trace."""
    return bool(get_settings().typesafe_api_key.strip())


@lru_cache
def get_classifier() -> Runnable:
    """The TypeSafe Jev classifier, pinned to settings.typesafe_model.

    Cached because the instance owns httpx2 connection pools and is meant to be long-lived.
    The key comes from Settings, not the environment variable the class would read itself:
    run straight from .venv, only Settings sees .env (same reason as core/tracing.py).
    """
    settings = get_settings()
    with warnings.catch_warnings():
        # The class is @beta(); without this every first use logs a LangChainBetaWarning.
        warnings.simplefilter("ignore")
        classifier = TypeSafeClassifier(
            model=settings.typesafe_model,
            api_key=settings.typesafe_api_key,
            timeout=settings.typesafe_timeout_s,
        )
    return classifier.with_retry(
        retry_if_exception_type=_RETRYABLE,
        stop_after_attempt=_MAX_ATTEMPTS,
    )
