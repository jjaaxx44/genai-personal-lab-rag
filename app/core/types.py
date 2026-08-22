from typing import Protocol

from pydantic import BaseModel


class Passage(BaseModel):
    text: str
    score: float
    page: int


class RagResult(BaseModel):
    answer: str
    contexts: list[Passage]
    steps: list[str]
    llm_calls: int
    tokens: int
    latency_ms: float


class IngestStats(BaseModel):
    doc_id: str
    chunks: int
    latency_ms: float


class RagDemo(Protocol):
    def ingest(self, pdf_bytes: bytes, doc_id: str) -> IngestStats: ...

    def ask(self, question: str, doc_id: str, **settings: object) -> RagResult: ...
