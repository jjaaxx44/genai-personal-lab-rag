# GenAI Personal Lab

Sixteen RAG techniques, one Streamlit app, one page each. Every page runs the real pipeline on a PDF you upload and shows its working: the stages it went through, the passages it retrieved with their scores, and what the whole thing cost in latency and tokens.

It is a teaching repo. Each demo is written to make its technique legible, not to be reused — so the code is deliberately repetitive, there is no shared abstraction layer, and there are no tests. The point is that you can read one folder top to bottom and understand what the technique actually does.

Everything runs on a laptop: 4 CPUs, 8 GB of RAM, no GPU. Embeddings and re-ranking are local models; only the LLM calls (and Neo4j, for the two graph demos) leave the machine.

## The demos

The app's home page is organised around symptoms rather than names — you pick the way your retrieval is failing, and it points you at the techniques that address it. The full catalogue:

### Retrieval basics
The baseline pipeline, the first repairs anyone makes to it, and how to tell whether they worked.

| # | Technique | What it does |
|---|---|---|
| 01 | [Naive RAG](app/demos/rag/naive/README.md) | Fetch the passages whose meaning sits closest to the question, and trust that one of them holds the answer. |
| 02 | [Chunking](app/demos/rag/chunking/README.md) | Four splitters cut the same document four ways and answer the same question. |
| 03 | [Hybrid search](app/demos/rag/hybrid/README.md) | A keyword retriever runs alongside the vector one, and the two rankings are fused by position. |
| 04 | [Re-ranking](app/demos/rag/rerank/README.md) | A cheap search casts a wide net; a model that reads question and passage together reorders what it caught. |
| 05 | [RAG evaluation](app/demos/rag/evaluation/README.md) | Faithfulness, relevancy, precision and recall, computed the same way for every pipeline. |

### Enrichment
Change what goes into the index, rather than how the index is searched.

| # | Technique | What it does |
|---|---|---|
| 06 | [Contextual retrieval](app/demos/rag/contextual/README.md) | A model writes the document context back into each passage before it is indexed. |

### Agentic
Hand the model control over retrieval: whether to search at all, how often, and whether to believe the result.

| # | Technique | What it does |
|---|---|---|
| 07 | [Agentic RAG](app/demos/rag/agentic/README.md) | Retrieval becomes a tool the model calls with arguments it chooses, as often as it decides it needs to. |
| 08 | [CRAG](app/demos/rag/crag/README.md) | Passages are graded before they are trusted, and a web search takes over when none of them survive. |
| 09 | [Self-RAG](app/demos/rag/self_rag/README.md) | The draft is critiqued against its own evidence, and rewritten when the critique fails it. |
| 10 | [Adaptive RAG](app/demos/rag/adaptive/README.md) | A classifier reads the question first and routes it down a path sized for it. |
| 11 | [Multi-hop RAG](app/demos/rag/multi_hop/README.md) | The question is broken into a chain of simpler ones, each answer carried into the next search. |

### Beyond vectors
Retrieval with no embedding anywhere in the pipeline.

| # | Technique | What it does |
|---|---|---|
| 12 | [Vectorless RAG](app/demos/rag/vectorless/README.md) | The table of contents becomes the index: descend the headings, then read the pages you land on. |
| 13 | [SQL RAG](app/demos/rag/sql_rag/README.md) | The model is shown a schema and writes a query; rows come back as the context it answers from. |

### Graph
Store the relationships between things, not only the passages that mention them.

| # | Technique | What it does |
|---|---|---|
| 14 | [GraphRAG](app/demos/rag/graph_rag/README.md) | Entities and relationships are extracted at ingest, and the answer comes from a node's neighbourhood. |
| 15 | [KAG](app/demos/rag/kag/README.md) | A fixed schema constrains extraction, and the question is planned as a sequence of Cypher queries. |

### Multimodal
Retrieve from what the page shows, not only from what it says.

| # | Technique | What it does |
|---|---|---|
| 16 | [Multimodal RAG](app/demos/rag/multimodal/README.md) | Every image is captioned at ingest, and the captions compete with the paragraphs in one search. |

Each demo folder has a `README.md` covering the technique in general — how it works, what it is good at, how it fails, where to use it — followed by an `In this demo` section with every implementation detail. The diagrams are mermaid, so they render on GitHub and inside the app.

## Stack

| Layer | Choice |
|---|---|
| UI | Streamlit multipage app, Python 3.12 |
| LLM | LangChain chat models chained with `.with_fallbacks(...)`: Gemini → Groq → OpenAI → local Ollama. A provider joins the chain only when its key *and* its model are set. |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers — local, CPU |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers — local, CPU |
| PDF parsing | PyMuPDF — **AGPL-3.0**, see [Licence](#licence) before reusing this code |
| Vector + keyword store | MongoDB Atlas Local, in Docker — no cloud database account needed |
| Graph store | Neo4j AuraDB Free, via `langchain-neo4j` |
| SQL | SQLite (the bundled Chinook database) via SQLAlchemy |
| Orchestration | LangChain v1 and LangGraph, except in Naive RAG, Hybrid search and Vectorless RAG, which are plain Python so the mechanics stay visible |
| Web search | `ddgs` (no API key), used by CRAG's fallback |
| Evaluation | Ragas |
| Tracing | Langfuse — optional, and silently disabled when keys are missing |

Every direct dependency is pinned in [pyproject.toml](pyproject.toml), with a committed `uv.lock` for the transitive ones. Torch comes from the CPU-only index; the PyPI Linux wheels would pull in several GB of CUDA packages that this app has no use for.

## Running it

**Prerequisites:** Docker (with Compose). Optionally [Ollama](https://ollama.com) on the host if you want a local LLM as the last fallback.

1. Copy the environment template:

   ```sh
   cp .env.example .env
   ```

2. Fill in at least one LLM provider key. Gemini and Groq both have usable free tiers, and the defaults in the template point at their cheapest models. Change `MONGODB_PASSWORD` from `change-me`, and make `MONGODB_URI` match it.

   The graph demos (14, 15) also need a Neo4j AuraDB Free instance. The other fourteen run without one. Langfuse is optional everywhere.

   If you want the local fallback: `ollama pull qwen2.5:7b-instruct` on the host. It is about 4.7 GB resident, so on an 8 GB machine it competes with the app and the local embedding models.

3. Start it:

   ```sh
   docker compose up --build
   ```

   Then open <http://localhost:8501>. First boot downloads the embedding and re-ranker models into a cached volume, so it is slower than later ones.

A sample PDF ships in [samples/](samples/), so you can try any demo without uploading your own. Uploads are PDF-only and capped by `MAX_UPLOAD_MB` / `MAX_PDF_PAGES`.

Note that application code is **not** bind-mounted into the container — only `./data` and the model cache are. A code change needs `docker compose up --build`, not a restart. Streamlit's file watcher is switched off in [.streamlit/config.toml](.streamlit/config.toml) for the same reason.

## Layout

```
├── app/
│   ├── Home.py              # symptom grid + full catalogue
│   ├── core/                # config, LLM, embeddings, PDF, Mongo, Neo4j, tracing, shared UI
│   ├── static/fonts/        # the three typefaces the theme loads
│   └── demos/rag/<name>/    # pipeline.py, page.py, README.md — one folder per technique
├── samples/                 # sample PDFs + the Chinook SQLite database
├── scripts/                 # one-off helpers
├── .streamlit/config.toml   # theme tokens and server settings
├── docker-compose.yml       # app + mongodb
├── Dockerfile
├── DESIGN.md                # design tokens and page patterns
└── CLAUDE.md                # working rules for this repo
```

A demo imports from `app/core/` and never from another demo, and touches only its own MongoDB collection. Something moves into `core/` when a second demo needs it, not in anticipation — duplication between demos is the intended trade-off, because it keeps each one readable on its own.

## PDF extraction in production

Extraction here is one line — `page.get_text()` per page, in [app/core/pdf.py](app/core/pdf.py) — and that is a teaching decision, not a recommendation. It keeps the retrieval technique the visible part of every demo. What it gives you is a flat string per page: no OCR, no table structure, no reading order for multi-column layouts, no figure or header/footer handling.

That matters more than it looks. In a real system extraction quality usually dominates retrieval quality: a table flattened into prose, or a two-column page read straight across, produces chunks that **no** technique in this repo can rescue. Fix the parser before reaching for a cleverer retriever.

If you are building something real, extract with one of these instead:

| Tool | Licence | Why you'd pick it |
|---|---|---|
| [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) | Apache-2.0 / BSD-3-Clause | Bindings to PDFium, the PDF engine in Chrome. The closest thing to a straight swap for PyMuPDF — comparable speed, permissive licence. |
| [pypdf](https://github.com/py-pdf/pypdf) | BSD-3-Clause | Pure Python, no native dependency. Fine for clean digital PDFs; weakest on layout fidelity. |
| [pdfplumber](https://github.com/jsvine/pdfplumber) | MIT | Word-level coordinates and genuine table extraction. Slower, and worth it when the tables carry the meaning. |
| [pdfminer.six](https://github.com/pdfminer/pdfminer.six) | MIT | Low-level layout analysis — what pdfplumber is built on. Reach for it when you need to control the layout algorithm yourself. |
| [Docling](https://github.com/docling-project/docling) | MIT | Model-based conversion to a structured document: headings, tables, reading order. Heavier, and the best fit when documents are messy. |
| [Apache Tika](https://tika.apache.org/) | Apache-2.0 | A JVM service covering far more than PDF. Sensible when you ingest a dozen formats, not one. |

For **scanned** documents none of the above is enough on its own, because there is no text layer to extract. Pair one with [Tesseract](https://github.com/tesseract-ocr/tesseract) (Apache-2.0), or use a managed service — AWS Textract, Azure AI Document Intelligence, Google Document AI — which do OCR plus table and form structure, billed per page. Feeding page images to a vision model works too and handles anything, at meaningfully higher cost per page and with no guarantee of a stable output shape.

Whatever you choose, check its licence rather than assuming: PyMuPDF is not unusual in the PDF ecosystem, and several widely-used document tools are GPL or AGPL.

## Licence

The code in this repository is MIT — see [LICENSE](LICENSE).

> [!IMPORTANT]
> **PyMuPDF is AGPL-3.0, and that governs anything you build out of this.**
>
> Every demo here reads PDFs with [PyMuPDF](https://github.com/pymupdf/pymupdf), which is dual-licensed: GNU AGPL-3.0, or a paid commercial licence from [Artifex](https://artifex.com/licensing/).
>
> The AGPL does not stop you charging money for software. What it stops is keeping the source closed — and its section 13 network clause is aimed squarely at web apps like this one: **put it somewhere other people can use it over a network, and you owe those users the complete source of your entire application under the AGPL.** You never have to hand out a copy for that obligation to bite; letting someone use it remotely is enough.
>
> Reading this repo, running it locally, forking it and learning from it are all fine. Lifting the code into a closed-source product — including one you only ever host for your own customers or staff — is not, unless you buy Artifex's commercial licence or swap PyMuPDF for one of the permissively licensed parsers listed under [PDF extraction in production](#pdf-extraction-in-production) above.
>
> This describes the licence; it isn't legal advice. Check with your own counsel before relying on it.

Bundled assets carry their own licences: the three typefaces in [app/static/fonts/](app/static/fonts/) are SIL Open Font License 1.1 (see [the notice there](app/static/fonts/LICENSES.md)), and `samples/chinook.db` is the [Chinook sample database](https://github.com/lerocha/chinook-database), MIT.
