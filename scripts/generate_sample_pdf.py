"""One-off generator for samples/sample.pdf — a short, real, extractable PDF
so the debug page (and later, Naive RAG) has something to ingest without
requiring anyone to supply their own file.

Run: .venv/bin/python scripts/generate_sample_pdf.py
"""

import pymupdf

PAGES = [
    (
        "The GenAI Personal Lab — sample document",
        """This is a short sample document used to exercise the RAG demos in
this project. It has five pages, each covering one retrieval technique
in plain language, so a question about a specific technique should
retrieve the matching page.""",
    ),
    (
        "Naive RAG",
        """Naive RAG is the simplest retrieval pipeline: split a document into
chunks, embed each chunk, store the embeddings, and at question time
embed the question and run a vector search for the closest chunks.
Those chunks are placed in a prompt and a language model writes the
answer. It has no re-ranking, no query rewriting, and no agentic
control -- it is a straight line from question to answer.""",
    ),
    (
        "Chunking strategies",
        """How a document is split changes what gets retrieved. Fixed-size
chunking cuts text every N characters. Recursive chunking tries to
split on paragraph and sentence boundaries first. Sliding-window
chunking overlaps neighbouring chunks so a fact near a boundary is not
lost. Semantic chunking splits where the meaning changes, detected by
a drop in embedding similarity between neighbouring sentences.""",
    ),
    (
        "Hybrid search",
        """Hybrid search runs a vector search and a keyword (BM25) search
separately, then fuses the two ranked lists, commonly with reciprocal
rank fusion. This helps with queries that name an exact code,
identifier, or proper noun, which vector search alone can miss because
embeddings capture meaning rather than exact tokens.""",
    ),
    (
        "RAG evaluation",
        """Evaluating a RAG pipeline means scoring both retrieval and
generation. Faithfulness checks whether the answer is supported by the
retrieved context. Answer relevancy checks whether the answer
addresses the question. Context precision and context recall check
whether the retrieved passages are both correct and sufficient.""",
    ),
]


def build(path: str) -> None:
    doc = pymupdf.open()
    for title, body in PAGES:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_textbox((72, 72, 523, 130), title, fontsize=18, fontname="helv")
        page.insert_textbox((72, 150, 523, 770), body, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()


if __name__ == "__main__":
    build("samples/sample.pdf")
    print("Wrote samples/sample.pdf")
