import re
import time
from pathlib import Path

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy import Engine, create_engine, inspect, text

from core.config import get_settings
from core.llm import get_chat_model
from core.tracing import get_callbacks, observe
from core.types import RagResult

# The bundled sample database -- no upload for this demo, see README.
DB_PATH = Path(__file__).resolve().parents[4] / "samples" / "chinook.db"

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|GRANT|REVOKE|BEGIN|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)

SQL_PROMPT = ChatPromptTemplate.from_template(
    """You are a SQLite expert. Given this database schema:

{schema}

Write a single SQL SELECT statement (SQLite dialect) that answers the question below. \
Only read data -- never write, and never use more than one statement. Include a LIMIT \
clause of at most {row_limit} rows unless the question calls for fewer.

Question: {question}"""
)

FIX_PROMPT = ChatPromptTemplate.from_template(
    """You are a SQLite expert. This SQL statement was rejected:

{sql}

Reason:
{error}

Database schema:
{schema}

Write a corrected single SQL SELECT statement (SQLite dialect) that answers the original \
question below. Only read data -- never write, and never use more than one statement. \
Include a LIMIT clause of at most {row_limit} rows unless the question calls for fewer.

Original question: {question}"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Question: {question}

This SQL query was run against the database:
{sql}

It returned these results (columns: {columns}):
{rows}

Answer the question in plain language, using only these results. If the results are \
empty, say so instead of guessing."""
)


class GeneratedSql(BaseModel):
    sql: str = Field(description="A single read-only SQLite SELECT statement.")
    explanation: str = Field(description="One sentence on what the query does.")


class SqlAskResult(RagResult):
    """RagResult plus the generated SQL and result table, so the page can show both."""

    sql: str = ""
    columns: list[str] = []
    rows: list[dict] = []
    truncated: bool = False
    retried: bool = False
    blocked_reason: str | None = None


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    # SQLite's own "mode=ro" URI flag blocks writes at the database layer, as a second
    # line of defense behind the SELECT-only check below.
    return create_engine(f"sqlite:///file:{DB_PATH}?mode=ro&uri=true")


def describe_schema(engine: Engine) -> str:
    inspector = inspect(engine)
    lines: list[str] = []
    for table in inspector.get_table_names():
        columns = ", ".join(f"{c['name']} {c['type']}" for c in inspector.get_columns(table))
        lines.append(f"{table}({columns})")
        for fk in inspector.get_foreign_keys(table):
            if fk["constrained_columns"] and fk["referred_table"]:
                lines.append(
                    f"  FK: {table}.{fk['constrained_columns'][0]} -> "
                    f"{fk['referred_table']}.{fk['referred_columns'][0]}"
                )
    return "\n".join(lines)


def _validate_select_only(sql: str) -> str | None:
    """Returns an error message if the SQL isn't a single, read-only SELECT, else None."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        return "The model returned an empty query."
    if ";" in cleaned:
        return "Only a single statement is allowed."
    first_word = re.match(r"[A-Za-z]+", cleaned)
    if not first_word or first_word.group().upper() not in {"SELECT", "WITH"}:
        return "Only SELECT statements are allowed."
    forbidden = _FORBIDDEN.search(cleaned)
    if forbidden:
        return f"The keyword '{forbidden.group().upper()}' is not allowed -- only read queries are."
    return None


def _run_select(engine: Engine, sql: str, row_limit: int) -> tuple[list[str], list[dict], bool]:
    with engine.connect() as conn:
        cursor = conn.execute(text(sql))
        columns = list(cursor.keys())
        fetched = cursor.fetchmany(row_limit + 1)
        truncated = len(fetched) > row_limit
        rows = [dict(zip(columns, row)) for row in fetched[:row_limit]]
    return columns, rows, truncated


@st.cache_resource(show_spinner=False)
def _get_schema_cached() -> str:
    return describe_schema(get_engine())


@observe(name="sql_rag_ask")
def ask(question: str, *, row_limit: int | None = None, max_retries: int | None = None, **_: object) -> SqlAskResult:
    start = time.monotonic()
    settings = get_settings()
    row_limit = row_limit if row_limit is not None else settings.sql_rag_row_limit
    max_retries = max_retries if max_retries is not None else settings.sql_rag_max_retries

    steps: list[str] = []
    llm_calls = 0
    tokens = 0

    engine = get_engine()
    schema = _get_schema_cached()
    steps.append(f"Read schema via SQLAlchemy inspector: {len(schema.splitlines())} line(s).")

    sql_chain = SQL_PROMPT | get_chat_model().with_structured_output(GeneratedSql, method="json_schema")
    generated = sql_chain.invoke(
        {"schema": schema, "question": question, "row_limit": row_limit},
        config={"callbacks": get_callbacks()},
    )
    llm_calls += 1
    tokens += (len(schema) + len(question) + len(generated.sql)) // 4
    sql = generated.sql.strip()
    steps.append(f"Generated SQL: {generated.explanation}")

    retried = False
    columns: list[str] = []
    rows: list[dict] = []
    truncated = False
    error: str | None = None

    for attempt in range(max_retries + 1):
        error = _validate_select_only(sql)
        if error is None:
            try:
                columns, rows, truncated = _run_select(engine, sql, row_limit)
                steps.append(f"Ran the query: {len(rows)} row(s) returned{' (truncated)' if truncated else ''}.")
                break
            except Exception as exc:
                error = str(exc)

        if attempt >= max_retries:
            break
        retried = True
        steps.append(f"Attempt {attempt + 1} rejected: {error}. Asking the model to fix it.")
        fix_chain = FIX_PROMPT | get_chat_model().with_structured_output(GeneratedSql, method="json_schema")
        generated = fix_chain.invoke(
            {"schema": schema, "question": question, "row_limit": row_limit, "sql": sql, "error": error},
            config={"callbacks": get_callbacks()},
        )
        llm_calls += 1
        tokens += (len(schema) + len(question) + len(generated.sql)) // 4
        sql = generated.sql.strip()

    if error is not None:
        # Either every attempt was blocked, or the last one still errored at execution time.
        answer = f"Couldn't run a safe query for this question after {max_retries + 1} attempt(s). {error}"
        return SqlAskResult(
            answer=answer,
            contexts=[],
            steps=steps,
            llm_calls=llm_calls,
            tokens=tokens,
            latency_ms=(time.monotonic() - start) * 1000,
            sql=sql,
            columns=[],
            rows=[],
            truncated=False,
            retried=retried,
            blocked_reason=error,
        )

    rows_text = "\n".join(str(row) for row in rows) or "(no rows)"
    answer_chain = ANSWER_PROMPT | get_chat_model()
    answer_msg = answer_chain.invoke(
        {"question": question, "sql": sql, "columns": ", ".join(columns), "rows": rows_text},
        config={"callbacks": get_callbacks()},
    )
    answer = answer_msg.text
    llm_calls += 1
    tokens += (len(rows_text) + len(question) + len(answer)) // 4
    steps.append("Answered from the query results via the chat model.")

    return SqlAskResult(
        answer=answer,
        contexts=[],
        steps=steps,
        llm_calls=llm_calls,
        tokens=tokens,
        latency_ms=(time.monotonic() - start) * 1000,
        sql=sql,
        columns=columns,
        rows=rows,
        truncated=truncated,
        retried=retried,
        blocked_reason=None,
    )
