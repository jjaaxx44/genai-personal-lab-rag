import time
from typing import Any, Iterable

import streamlit as st
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from .config import get_settings


@st.cache_resource(show_spinner=False)
def get_client() -> MongoClient:
    settings = get_settings()
    return MongoClient(settings.mongodb_uri)


def get_collection(name: str) -> Collection:
    settings = get_settings()
    return get_client()[settings.mongodb_db][name]


def ensure_indexes(
    collection: Collection,
    *,
    vector_dims: int = 384,
    filter_fields: Iterable[str] = ("doc_id",),
    with_text_index: bool = False,
    timeout_s: float = 60.0,
) -> None:
    """Create vector_index (and optionally text_index) if missing, and block until queryable."""
    existing = {ix["name"] for ix in collection.list_search_indexes()}

    if "vector_index" not in existing:
        definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": vector_dims,
                    "similarity": "cosine",
                },
                *({"type": "filter", "path": field} for field in filter_fields),
            ]
        }
        collection.create_search_index(
            SearchIndexModel(definition, name="vector_index", type="vectorSearch")
        )

    if with_text_index and "text_index" not in existing:
        definition = {"mappings": {"dynamic": False, "fields": {"text": {"type": "string"}}}}
        collection.create_search_index(
            SearchIndexModel(definition, name="text_index", type="search")
        )

    _wait_until_queryable(collection, "vector_index", timeout_s)
    if with_text_index:
        _wait_until_queryable(collection, "text_index", timeout_s)


def _wait_until_queryable(collection: Collection, name: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for ix in collection.list_search_indexes(name):
            if ix.get("queryable"):
                return
        time.sleep(1)
    raise TimeoutError(f"Search index '{name}' did not become queryable within {timeout_s}s.")


def insert_passages(collection: Collection, passages: list[dict[str, Any]]) -> None:
    if passages:
        collection.insert_many(passages)


def clear_doc(collection: Collection, doc_id: str) -> int:
    return collection.delete_many({"doc_id": doc_id}).deleted_count


def vector_search(
    collection: Collection,
    query_vector: list[float],
    *,
    doc_id: str,
    top_k: int = 5,
    extra_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    match_filter: dict[str, Any] = {"doc_id": {"$eq": doc_id}}
    if extra_filter:
        match_filter.update(extra_filter)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "filter": match_filter,
                "numCandidates": max(top_k * 10, 100),
                "limit": top_k,
            }
        },
        {"$project": {"embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(collection.aggregate(pipeline))


def text_search(
    collection: Collection,
    query: str,
    *,
    doc_id: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    pipeline = [
        {"$search": {"index": "text_index", "text": {"query": query, "path": "text"}}},
        {"$match": {"doc_id": doc_id}},
        {"$limit": top_k},
        {"$project": {"embedding": 0, "score": {"$meta": "searchScore"}}},
    ]
    return list(collection.aggregate(pipeline))
