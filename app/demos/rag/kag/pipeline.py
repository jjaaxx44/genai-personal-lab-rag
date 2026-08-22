import asyncio
import json
import re
import time

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from core.config import get_settings
from core.graph_db import clear_graph_doc, get_graph
from core.llm import get_chat_model
from core.pdf import extract_text
from core.tracing import get_callbacks, observe
from core.types import IngestStats, RagResult

DEMO_TAG = "kag"

DEFAULT_SCHEMA = {
    "entity_types": ["Person", "Organization", "Location", "Event", "Concept"],
    "relationship_types": ["WORKS_AT", "LOCATED_IN", "PARTICIPATED_IN", "FOUNDED", "RELATED_TO"],
}

_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|FOREACH|CALL)\b", re.IGNORECASE
)

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """You have a knowledge graph with these entity types (node labels): {entity_types}
and these relationship types: {relationship_types}

Break the question below into an ordered list of logical steps, each answerable by one \
read-only Cypher query against this graph. Use as few steps as the question genuinely \
needs, up to {max_steps}.

Question: {question}"""
)

CYPHER_PROMPT = ChatPromptTemplate.from_template(
    """Knowledge graph schema:
Entity types (node labels): {entity_types}
Relationship types: {relationship_types}

Every entity node's name is stored in its `id` property -- there is no `name` property. \
Every node also carries `doc_id` and `demo` properties. Always filter every node pattern by \
{{doc_id: $doc_id, demo: 'kag'}}, and match on `id` (not `name`) when you need a specific \
entity, for example (p:Person {{id: 'Marie Curie', doc_id: $doc_id, demo: 'kag'}}). \
$doc_id is a query parameter -- do not hardcode a document id. Return `id`, not `name`.

Original question: {question}
All planned steps: {all_steps}
Results from earlier steps (use concrete values from these directly in this step's query \
where needed):
{prior_results}

Write a single read-only Cypher query (MATCH ... RETURN, no writes, no CALL) for this step only.

Step {step_number}: {step_description}"""
)

FIX_PROMPT = ChatPromptTemplate.from_template(
    """This Cypher query was rejected:
{cypher}

Reason:
{error}

Knowledge graph schema:
Entity types (node labels): {entity_types}
Relationship types: {relationship_types}
Every entity node's name is stored in its `id` property -- there is no `name` property. \
Every node also carries `doc_id` and `demo` properties; filter every node pattern by \
{{doc_id: $doc_id, demo: 'kag'}} and match/return `id`, not `name`. $doc_id is a query parameter.

Write a corrected single read-only Cypher query (MATCH ... RETURN, no writes, no CALL) for \
this step: {step_description}"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Original question: {question}

Each step's description, Cypher query, and results:
{steps_text}

Answer the question using only these results. If a step returned nothing or was blocked, \
say what's missing instead of guessing.
Answer:"""
)


class Schema(BaseModel):
    entity_types: list[str]
    relationship_types: list[str]


def default_schema_json() -> str:
    return json.dumps(DEFAULT_SCHEMA, indent=2)


def parse_schema(raw: str) -> Schema:
    data = json.loads(raw)
    entity_types = [str(t) for t in data["entity_types"]]
    relationship_types = [str(t) for t in data["relationship_types"]]
    if not entity_types or not relationship_types:
        raise ValueError("Schema needs at least one entity type and one relationship type.")
    return Schema(entity_types=entity_types, relationship_types=relationship_types)


class PlannedSteps(BaseModel):
    steps: list[str] = Field(description="Ordered logical steps, each answerable by one Cypher query.")


class GeneratedCypher(BaseModel):
    cypher: str = Field(description="A single read-only Cypher MATCH ... RETURN query.")
    explanation: str = Field(description="One sentence on what this query looks up.")


class KagIngestStats(IngestStats):
    """IngestStats plus what the schema-constrained graph build cost."""

    extracted: int = 0
    over_cap: int = 0
    failed: int = 0
    nodes: int = 0
    relationships: int = 0
    llm_calls: int = 0
    tokens: int = 0


class StepResult(BaseModel):
    description: str
    cypher: str = ""
    columns: list[str] = []
    rows: list[dict] = []
    truncated: bool = False
    retried: bool = False
    blocked_reason: str | None = None


class KagResult(RagResult):
    """RagResult plus the schema used and each step's Cypher and results."""

    schema_used: Schema
    kag_steps: list[StepResult] = []


def _chunk_pdf(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int) -> list[dict]:
    parsed = extract_text(pdf_bytes)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        {"page": page_num, "text": chunk_text}
        for page_num, text in enumerate(parsed.pages, start=1)
        if text.strip()
        for chunk_text in splitter.split_text(text)
    ]


async def _extract_all(transformer: LLMGraphTransformer, documents: list[Document], max_concurrency: int) -> list:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _one(doc: Document):
        async with semaphore:
            return await transformer.aprocess_response(doc)

    return await asyncio.gather(*(_one(d) for d in documents), return_exceptions=True)


def ingest(pdf_bytes: bytes, doc_id: str, *, schema: Schema | None = None) -> KagIngestStats:
    start = time.monotonic()
    settings = get_settings()
    schema = schema or Schema(**DEFAULT_SCHEMA)
    graph = get_graph()
    clear_graph_doc(graph, DEMO_TAG, doc_id)

    chunks = _chunk_pdf(pdf_bytes, settings.kag_chunk_size, settings.kag_chunk_overlap)
    if not chunks:
        return KagIngestStats(doc_id=doc_id, chunks=0, latency_ms=(time.monotonic() - start) * 1000)

    extract_count = min(len(chunks), settings.kag_max_chunks)
    over_cap = len(chunks) - extract_count
    to_extract = chunks[:extract_count]
    documents = [Document(page_content=c["text"], metadata={"page": c["page"]}) for c in to_extract]

    # strict_mode (the default) drops any node or relationship outside the schema, which is
    # what keeps this demo's graph constrained instead of GraphRAG's open extraction.
    transformer = LLMGraphTransformer(
        llm=get_chat_model(fast=True),
        allowed_nodes=schema.entity_types,
        allowed_relationships=schema.relationship_types,
    )
    results = asyncio.run(_extract_all(transformer, documents, settings.kag_extract_concurrency))

    graph_documents = []
    failed = 0
    node_count = 0
    rel_count = 0
    tokens = sum(len(d.page_content) for d in documents) // 4
    for result in results:
        if isinstance(result, Exception):
            failed += 1
            continue
        for node in result.nodes:
            node.properties["doc_id"] = doc_id
            node.properties["demo"] = DEMO_TAG
        for rel in result.relationships:
            rel.properties["doc_id"] = doc_id
            rel.properties["demo"] = DEMO_TAG
        node_count += len(result.nodes)
        rel_count += len(result.relationships)
        graph_documents.append(result)

    if graph_documents:
        graph.add_graph_documents(graph_documents)
    graph.refresh_schema()

    return KagIngestStats(
        doc_id=doc_id,
        chunks=len(chunks),
        latency_ms=(time.monotonic() - start) * 1000,
        extracted=len(graph_documents),
        over_cap=over_cap,
        failed=failed,
        nodes=node_count,
        relationships=rel_count,
        llm_calls=len(documents),
        tokens=tokens,
    )


def _entity_type(labels: list[str]) -> str:
    return labels[0] if labels else "Entity"


def graph_summary(doc_id: str, limit: int = 200) -> list[dict]:
    """The whole schema-constrained graph for this document, for the 'extracted graph' view."""
    graph = get_graph()
    rows = graph.query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.doc_id = $doc_id AND a.demo = $demo AND b.doc_id = $doc_id AND b.demo = $demo
        RETURN a.id AS source, labels(a) AS source_labels, type(r) AS rel,
               b.id AS target, labels(b) AS target_labels
        LIMIT $limit
        """,
        {"doc_id": doc_id, "demo": DEMO_TAG, "limit": limit},
    )
    return [
        {
            "source": r["source"],
            "source_type": _entity_type(r["source_labels"]),
            "rel": r["rel"],
            "target": r["target"],
            "target_type": _entity_type(r["target_labels"]),
        }
        for r in rows
    ]


def _validate_read_only(cypher: str) -> str | None:
    """Returns an error message if the Cypher isn't a single read-only query, else None."""
    cleaned = cypher.strip().rstrip(";").strip()
    if not cleaned:
        return "The model returned an empty query."
    if ";" in cleaned:
        return "Only a single statement is allowed."
    if "RETURN" not in cleaned.upper():
        return "The query must end in a RETURN clause."
    forbidden = _WRITE_KEYWORDS.search(cleaned)
    if forbidden:
        return f"The keyword '{forbidden.group().upper()}' is not allowed -- only read queries are."
    return None


def _run_step(
    graph,
    schema: Schema,
    question: str,
    all_steps: list[str],
    step_number: int,
    step_description: str,
    prior_results: list[StepResult],
    doc_id: str,
    row_limit: int,
    max_retries: int,
) -> tuple[StepResult, int, int]:
    llm_calls = 0
    tokens = 0
    entity_types = ", ".join(schema.entity_types)
    relationship_types = ", ".join(schema.relationship_types)
    prior_text = (
        "\n".join(
            f"Step {i + 1} ({r.description}): {r.rows[:5]}" if not r.blocked_reason else f"Step {i + 1}: blocked -- {r.blocked_reason}"
            for i, r in enumerate(prior_results)
        )
        or "(none yet)"
    )

    chain = CYPHER_PROMPT | get_chat_model().with_structured_output(GeneratedCypher, method="json_schema")
    generated = chain.invoke(
        {
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "question": question,
            "all_steps": all_steps,
            "prior_results": prior_text,
            "step_number": step_number,
            "step_description": step_description,
        },
        config={"callbacks": get_callbacks()},
    )
    llm_calls += 1
    tokens += (len(entity_types) + len(relationship_types) + len(prior_text) + len(generated.cypher)) // 4
    cypher = generated.cypher.strip()

    retried = False
    for attempt in range(max_retries + 1):
        error = _validate_read_only(cypher)
        if error is None:
            try:
                rows = graph.query(cypher, {"doc_id": doc_id})
                truncated = len(rows) > row_limit
                rows = rows[:row_limit]
                return (
                    StepResult(
                        description=step_description,
                        cypher=cypher,
                        columns=list(rows[0].keys()) if rows else [],
                        rows=rows,
                        truncated=truncated,
                        retried=retried,
                        blocked_reason=None,
                    ),
                    llm_calls,
                    tokens,
                )
            except Exception as exc:
                error = str(exc)

        if attempt >= max_retries:
            return (
                StepResult(description=step_description, cypher=cypher, retried=retried, blocked_reason=error),
                llm_calls,
                tokens,
            )
        retried = True
        fix_chain = FIX_PROMPT | get_chat_model().with_structured_output(GeneratedCypher, method="json_schema")
        generated = fix_chain.invoke(
            {
                "cypher": cypher,
                "error": error,
                "entity_types": entity_types,
                "relationship_types": relationship_types,
                "step_description": step_description,
            },
            config={"callbacks": get_callbacks()},
        )
        llm_calls += 1
        tokens += len(generated.cypher) // 4
        cypher = generated.cypher.strip()

    # Unreachable, but keeps type-checkers happy.
    return StepResult(description=step_description, cypher=cypher, retried=retried, blocked_reason="Exhausted retries."), llm_calls, tokens


@observe(name="kag_ask")
def ask(
    question: str,
    doc_id: str,
    *,
    schema: Schema | None = None,
    max_steps: int | None = None,
    **_: object,
) -> KagResult:
    start = time.monotonic()
    settings = get_settings()
    schema = schema or Schema(**DEFAULT_SCHEMA)
    max_steps = max_steps if max_steps is not None else settings.kag_max_steps
    row_limit = settings.kag_row_limit
    max_retries = settings.kag_max_retries

    graph = get_graph()
    llm_calls = 0
    tokens = 0
    steps: list[str] = []

    entity_types = ", ".join(schema.entity_types)
    relationship_types = ", ".join(schema.relationship_types)
    decompose_chain = DECOMPOSE_PROMPT | get_chat_model(fast=True).with_structured_output(
        PlannedSteps, method="json_schema"
    )
    planned = decompose_chain.invoke(
        {"entity_types": entity_types, "relationship_types": relationship_types, "question": question, "max_steps": max_steps},
        config={"callbacks": get_callbacks()},
    )
    llm_calls += 1
    tokens += len(question) // 4
    plan = planned.steps[:max_steps] or [question]
    steps.append(f"Planned {len(plan)} step(s): {plan}")

    kag_steps: list[StepResult] = []
    for i, description in enumerate(plan, start=1):
        result, step_llm_calls, step_tokens = _run_step(
            graph, schema, question, plan, i, description, kag_steps, doc_id, row_limit, max_retries
        )
        kag_steps.append(result)
        llm_calls += step_llm_calls
        tokens += step_tokens
        if result.blocked_reason:
            steps.append(f"Step {i} ({description}): blocked -- {result.blocked_reason}")
        else:
            steps.append(f"Step {i} ({description}): {result.cypher} -> {len(result.rows)} row(s).")

    steps_text = "\n\n".join(
        f"Step {i + 1}: {r.description}\nCypher: {r.cypher or '(none)'}\n"
        + (f"Blocked: {r.blocked_reason}" if r.blocked_reason else f"Results: {r.rows}")
        for i, r in enumerate(kag_steps)
    )
    answer_chain = ANSWER_PROMPT | get_chat_model()
    answer_msg = answer_chain.invoke(
        {"question": question, "steps_text": steps_text}, config={"callbacks": get_callbacks()}
    )
    answer = answer_msg.text
    llm_calls += 1
    tokens += (len(steps_text) + len(answer)) // 4
    steps.append("Answered from the step results via the chat model.")

    return KagResult(
        answer=answer,
        contexts=[],
        steps=steps,
        llm_calls=llm_calls,
        tokens=tokens,
        latency_ms=(time.monotonic() - start) * 1000,
        schema_used=schema,
        kag_steps=kag_steps,
    )
