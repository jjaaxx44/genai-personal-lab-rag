import streamlit as st
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from .config import get_settings


@st.cache_resource(show_spinner="Loading embedding model...")
def _get_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


class LocalEmbeddings(Embeddings):
    """Wraps the shared sentence-transformers model so it's loaded once."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed([text])[0]
