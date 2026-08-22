import json
import re
import time
from pathlib import Path
from typing import Any

import pymupdf

from core.config import get_settings
from core.llm import complete
from core.tracing import observe
from core.types import IngestStats, Passage, RagResult

# data/ is the compose volume -- the tree (and the page text it points into) is saved
# here per doc_id, since this demo has no MongoDB collection to keep it in.
TREE_DIR = Path(__file__).resolve().parents[4] / "data" / "vectorless"

ANSWER_PROMPT = """Answer the question using only the pages below, taken from the section \
"{section}" of the document. Cite the page number(s) you relied on in square brackets, \
e.g. [p.3]. If these pages don't contain the answer, say so instead of guessing.

{context}

Question: {question}
Answer:"""

SUMMARY_PROMPT = """Write one short sentence (max 15 words) summarizing what pages {start}-{end} \
of a document are about, so a reader could decide whether to open them from a table of contents.

Pages:
{text}

One-sentence summary:"""

NAVIGATE_PROMPT = """You are navigating a document's table of contents to find the answer to a \
question. You may descend into one child section, or stop and read the current section's pages \
directly.

Question: {question}

Current section: "{title}" (pages {start}-{end})

Child sections:
{children}
0. Stop here and read "{title}" (pages {start}-{end}) directly.

Respond with only the number of your choice, nothing else."""


def _tree_path(doc_id: str) -> Path:
    return TREE_DIR / f"{doc_id}.json"


def has_tree(doc_id: str) -> bool:
    return _tree_path(doc_id).exists()


def clear_doc(doc_id: str) -> int:
    path = _tree_path(doc_id)
    if path.exists():
        path.unlink()
        return 1
    return 0


def _load_tree(doc_id: str) -> dict[str, Any] | None:
    path = _tree_path(doc_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _save_tree(doc_id: str, data: dict[str, Any]) -> None:
    TREE_DIR.mkdir(parents=True, exist_ok=True)
    _tree_path(doc_id).write_text(json.dumps(data))


def _build_toc_tree(toc: list[list], page_count: int) -> dict:
    """A stack-based build: each entry's page range runs to just before the next entry's
    page (in document order, not level order), regardless of how TOC levels are numbered."""
    root: dict[str, Any] = {"title": "Document", "page_start": 1, "page_end": page_count, "children": []}
    stack: list[tuple[int, dict]] = [(0, root)]

    for i, (level, title, page) in enumerate(toc):
        end = page_count
        for _, _, next_page in toc[i + 1 :]:
            if next_page > page:
                end = next_page - 1
                break
        node = {"title": title, "page_start": page, "page_end": max(page, end), "children": []}
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        stack[-1][1]["children"].append(node)
        stack.append((level, node))

    return root


def _build_page_group_tree(pages: list[str], group_size: int) -> dict:
    """No TOC to fall back on: one flat level of page-group nodes, each summarized by the LLM."""
    page_count = len(pages)
    children = []
    for start in range(1, page_count + 1, group_size):
        end = min(start + group_size - 1, page_count)
        text = "\n\n".join(pages[start - 1 : end])
        try:
            summary = complete(SUMMARY_PROMPT.format(start=start, end=end, text=text[:4000]), fast=True).strip()
        except Exception:
            summary = f"Pages {start}-{end}"
        children.append({"title": summary or f"Pages {start}-{end}", "page_start": start, "page_end": end, "children": []})
    return {"title": "Document", "page_start": 1, "page_end": page_count, "children": children}


@observe(name="vectorless_ingest")
def ingest(pdf_bytes: bytes, doc_id: str) -> IngestStats:
    start = time.monotonic()

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]
        toc = doc.get_toc(simple=True)

    settings = get_settings()
    if toc:
        tree = _build_toc_tree(toc, len(pages))
        source = "toc"
    else:
        tree = _build_page_group_tree(pages, settings.vectorless_page_group_size)
        source = "page_groups"

    _save_tree(doc_id, {"doc_id": doc_id, "source": source, "pages": pages, "tree": tree})

    def _count(node: dict) -> int:
        return 1 + sum(_count(c) for c in node["children"])

    return IngestStats(doc_id=doc_id, chunks=_count(tree), latency_ms=(time.monotonic() - start) * 1000)


def _parse_choice(raw: str, max_choice: int) -> int:
    match = re.search(r"-?\d+", raw)
    if not match:
        return 0
    choice = int(match.group())
    return choice if 0 <= choice <= max_choice else 0


@observe(name="vectorless_ask")
def ask(question: str, doc_id: str, *, max_depth: int | None = None, **_: object) -> RagResult:
    start = time.monotonic()
    settings = get_settings()
    max_depth = max_depth if max_depth is not None else settings.vectorless_max_depth

    data = _load_tree(doc_id)
    if data is None:
        raise RuntimeError(f"No tree found for doc_id={doc_id!r} -- ingest the document first.")
    pages: list[str] = data["pages"]
    steps: list[str] = [f"Tree built from {'the table of contents' if data['source'] == 'toc' else 'page-group summaries'}."]
    llm_calls = 0
    tokens = 0

    node = data["tree"]
    depth = 0
    while node["children"] and depth < max_depth:
        children = node["children"]
        listing = "\n".join(
            f"{i}. {c['title']} (pages {c['page_start']}-{c['page_end']})" for i, c in enumerate(children, start=1)
        )
        prompt = NAVIGATE_PROMPT.format(
            question=question,
            title=node["title"],
            start=node["page_start"],
            end=node["page_end"],
            children=listing,
        )
        raw = complete(prompt, fast=True)
        llm_calls += 1
        tokens += (len(prompt) + len(raw)) // 4
        choice = _parse_choice(raw, len(children))

        if choice == 0:
            steps.append(f"At \"{node['title']}\": stopped to read this section directly.")
            break
        chosen = children[choice - 1]
        steps.append(f"At \"{node['title']}\": descended into \"{chosen['title']}\" (pages {chosen['page_start']}-{chosen['page_end']}).")
        node = chosen
        depth += 1
    else:
        if node["children"]:
            steps.append(f"Hit the {max_depth}-step navigation cap at \"{node['title']}\" -- reading this section.")

    read_start, read_end = node["page_start"], node["page_end"]
    read_pages = list(range(read_start, read_end + 1))
    context = "\n\n".join(f"[p.{p}] {pages[p - 1]}" for p in read_pages)
    if len(context) > settings.vectorless_max_read_chars:
        context = context[: settings.vectorless_max_read_chars]
        steps.append(f"Read pages {read_start}-{read_end}, truncated to {settings.vectorless_max_read_chars} chars.")
    else:
        steps.append(f"Read pages {read_start}-{read_end} ({len(context)} chars) from \"{node['title']}\".")

    prompt = ANSWER_PROMPT.format(section=node["title"], context=context, question=question)
    answer = complete(prompt)
    llm_calls += 1
    tokens += (len(prompt) + len(answer)) // 4
    steps.append("Answered from the read pages via core.llm.complete().")

    contexts = [Passage(text=pages[p - 1], score=1.0, page=p) for p in read_pages]

    return RagResult(
        answer=answer,
        contexts=contexts,
        steps=steps,
        llm_calls=llm_calls,
        tokens=tokens,
        latency_ms=(time.monotonic() - start) * 1000,
    )
