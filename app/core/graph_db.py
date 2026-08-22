import streamlit as st

from .config import get_settings


class GraphUnavailable(Exception):
    """Raised when Neo4j isn't configured, or its Aura database is paused/unreachable."""


@st.cache_resource(show_spinner=False)
def get_graph():
    from langchain_neo4j import Neo4jGraph

    settings = get_settings()
    if not settings.neo4j_uri:
        raise GraphUnavailable("Neo4j is not configured (NEO4J_URI is empty).")

    try:
        return Neo4jGraph(
            url=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
        )
    except Exception as exc:
        raise GraphUnavailable(
            "Could not reach Neo4j. AuraDB Free instances pause after inactivity — "
            "resume it in the Aura console and try again."
        ) from exc


def clear_graph_doc(graph, demo: str, doc_id: str) -> int:
    """Deletes every node tagged with this demo + doc_id (and the relationships on them).

    Aura Free has a single shared database, so GraphRAG and KAG -- and different
    doc_ids within the same demo -- rely on every node carrying `demo` and `doc_id`
    properties rather than separate databases or labels.
    """
    rows = graph.query(
        "MATCH (n {demo: $demo, doc_id: $doc_id}) DETACH DELETE n RETURN count(n) AS deleted",
        {"demo": demo, "doc_id": doc_id},
    )
    return rows[0]["deleted"] if rows else 0
