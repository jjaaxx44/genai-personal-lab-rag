import streamlit as st
from langchain_core.language_models.chat_models import BaseChatModel

from .config import get_settings
from .tracing import get_callbacks

# Providers are tried in this fixed order, each falling back to the next. A provider
# joins the chain only when its .env config is filled in, so running on one provider
# and running on all four are the same code path.
PROVIDER_ORDER = ("gemini", "groq", "openai", "local_ollama")

_TEMPERATURE = 0.2
_MAX_RETRIES = 2
# No client defaults to a timeout, so a stalled connection would hang forever instead
# of raising -- which skips every demo's try/except and leaves the page spinning with
# no error shown. Ollama gets its own budget: a cold 7B load on CPU is slow, not stuck.
_TIMEOUT = 30
_OLLAMA_TIMEOUT = 300


def _pick(fast: bool, fast_model: str, model: str) -> str:
    """The fast model when one is asked for and configured, otherwise the normal one."""
    return fast_model if fast and fast_model else model


def _build(provider: str, fast: bool) -> BaseChatModel | None:
    """Builds the chat model for one provider, or returns None when it isn't configured.

    Skipping happens here rather than via with_fallbacks() because every cloud client
    validates its API key at construction time, and with_fallbacks() only covers errors
    raised during invoke() -- not ones raised by the model's own __init__.
    """
    settings = get_settings()

    if provider == "gemini":
        if not (settings.gemini_api_key and settings.gemini_llm_model):
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=_pick(fast, settings.gemini_llm_fast_model, settings.gemini_llm_model),
            google_api_key=settings.gemini_api_key,
            temperature=_TEMPERATURE,
            max_retries=_MAX_RETRIES,
            timeout=_TIMEOUT,
        )

    if provider == "groq":
        if not (settings.groq_api_key and settings.groq_llm_model):
            return None
        from langchain_groq import ChatGroq

        return ChatGroq(
            model_name=_pick(fast, settings.groq_llm_fast_model, settings.groq_llm_model),
            groq_api_key=settings.groq_api_key,
            temperature=_TEMPERATURE,
            max_retries=_MAX_RETRIES,
            request_timeout=_TIMEOUT,
        )

    if provider == "openai":
        if not (settings.openai_api_key and settings.openai_llm_model):
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=_pick(fast, settings.openai_llm_fast_model, settings.openai_llm_model),
            api_key=settings.openai_api_key,
            temperature=_TEMPERATURE,
            max_retries=_MAX_RETRIES,
            timeout=_TIMEOUT,
        )

    if provider == "local_ollama":
        if not (settings.local_ollama_base_url and settings.local_ollama_llm_model):
            return None
        from langchain_ollama import ChatOllama

        # One local model, so `fast` has nothing to switch to. ChatOllama has no timeout
        # field of its own -- it forwards client_kwargs to the underlying httpx client.
        # Model existence is deliberately not validated at construction: Ollama is last
        # in the chain, so an unreachable host should fail at invoke() like any other
        # provider outage, not at import time.
        return ChatOllama(
            model=settings.local_ollama_llm_model,
            base_url=settings.local_ollama_base_url,
            temperature=_TEMPERATURE,
            client_kwargs={"timeout": _OLLAMA_TIMEOUT},
        )

    raise ValueError(f"Unknown provider: {provider}")


def _configured_models(fast: bool) -> list[BaseChatModel]:
    models = [model for name in PROVIDER_ORDER if (model := _build(name, fast)) is not None]
    if not models:
        raise RuntimeError(
            "No LLM provider is configured. Set the API key and model for at least one of "
            "GEMINI_, GROQ_, OPENAI_ or LOCAL_OLLAMA_ in .env -- see .env.example."
        )
    return models


@st.cache_resource(show_spinner=False)
def get_chat_model(fast: bool = False) -> BaseChatModel:
    """The first configured provider, falling back through the rest in PROVIDER_ORDER."""
    primary, *fallbacks = _configured_models(fast)
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


@st.cache_resource(show_spinner=False)
def get_plain_chat_model(fast: bool = False) -> BaseChatModel:
    """The first configured provider with no fallback wrapper, for callers that need a
    plain BaseChatModel they can introspect or mutate (Ragas does exactly that)."""
    return _configured_models(fast)[0]


def complete(prompt: str, *, fast: bool = False) -> str:
    """Single-shot completion for the plain-Python demos."""
    model = get_chat_model(fast=fast)
    response = model.invoke(prompt, config={"callbacks": get_callbacks()})
    return response.text
