import base64
import time
from pathlib import Path
from typing import Callable

import pymupdf
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo.collection import Collection

from core.config import get_settings
from core.embeddings import LocalEmbeddings
from core.llm import get_chat_model
from core.mongo import clear_doc, ensure_indexes, get_collection
from core.tracing import get_callbacks
from core.types import IngestStats, Passage, RagResult

COLLECTION_NAME = "rag_multimodal"

# data/ is the compose volume -- saved images live here, one subfolder per doc_id, so
# "Clear my data" can drop a document's images without touching another's.
IMAGES_DIR = Path(__file__).resolve().parents[4] / "data" / "multimodal"

CAPTION_PROMPT = (
    "Write one or two sentences describing this image from a document, for someone who "
    "cannot see it. Name the chart type, axis labels, key numbers or diagram structure "
    "visible, so the description alone could match a question about it. Do not start with "
    '"This image shows" or similar.'
)

ANSWER_PROMPT = """Answer the question using only the numbered context below, which mixes \
document text and captions describing document images. Cite the page number(s) you relied \
on in square brackets, e.g. [p.3].

{context}

Question: {question}
Answer:"""


class MultimodalIngestStats(IngestStats):
    """IngestStats plus what the image side of ingest found and captioned."""

    text_chunks: int = 0
    images_found: int = 0
    images_captioned: int = 0
    images_failed: int = 0
    images_over_cap: int = 0
    llm_calls: int = 0
    tokens: int = 0


def _get_vector_store(collection: Collection) -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        collection,
        LocalEmbeddings(),
        index_name="vector_index",
        auto_create_index=False,
    )


def image_full_path(rel_path: str) -> Path:
    return IMAGES_DIR / rel_path


def clear_images(doc_id: str) -> None:
    doc_dir = IMAGES_DIR / doc_id
    if not doc_dir.exists():
        return
    for f in doc_dir.iterdir():
        f.unlink(missing_ok=True)
    doc_dir.rmdir()


def _save_image(doc_id: str, xref: int, png_bytes: bytes) -> str:
    doc_dir = IMAGES_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    rel_path = f"{doc_id}/{xref}.png"
    (IMAGES_DIR / rel_path).write_bytes(png_bytes)
    return rel_path


def _extract_images(doc: pymupdf.Document, min_px: int) -> list[dict]:
    """One entry per distinct embedded image (deduplicated by xref, since a logo repeated
    on every page would otherwise be extracted, captioned and stored once per page),
    normalized to RGB PNG so an unusual embedded encoding (CMYK, indexed, masked) doesn't
    break either the vision call or `st.image`."""
    seen_xrefs: set[int] = set()
    images: list[dict] = []
    for page_num, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                pixmap = pymupdf.Pixmap(doc, xref)
                if pixmap.n - pixmap.alpha >= 4:  # CMYK or other non-RGB colorspace
                    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
                if pixmap.width < min_px or pixmap.height < min_px:
                    continue
                png_bytes = pixmap.tobytes("png")
            except Exception:
                continue  # Some embedded encodings (e.g. JPX, JBIG2 masks) can't be decoded this way.
            images.append({"xref": xref, "page": page_num, "png_bytes": png_bytes})
    return images


def _caption_messages(png_bytes: bytes) -> list[HumanMessage]:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return [
        HumanMessage(
            content=[
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image", "base64": b64, "mime_type": "image/png"},
            ]
        )
    ]


def ingest(
    pdf_bytes: bytes,
    doc_id: str,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> MultimodalIngestStats:
    start = time.monotonic()
    settings = get_settings()
    collection = get_collection(COLLECTION_NAME)
    clear_doc(collection, doc_id)
    clear_images(doc_id)

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages_text = [page.get_text() for page in doc]
        raw_images = _extract_images(doc, settings.multimodal_min_image_px)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.multimodal_chunk_size, chunk_overlap=settings.multimodal_chunk_overlap
    )
    text_chunks = [
        {"page": page_num, "text": chunk_text}
        for page_num, text in enumerate(pages_text, start=1)
        if text.strip()
        for chunk_text in splitter.split_text(text)
    ]

    to_caption = raw_images[: settings.multimodal_max_images]
    images_over_cap = len(raw_images) - len(to_caption)

    # Captioning goes through the same provider fallback chain as every other LangChain
    # demo (core.llm.get_chat_model()), not a hardcoded vision model. A provider that can't
    # handle the image content block fails over to the next one, same as a rate limit would;
    # if none of the configured providers support vision, every caption fails and the image
    # is dropped -- the text side of the document still ingests normally.
    captioned: list[dict] = []
    llm_calls = 0
    tokens = sum(len(c["text"]) for c in text_chunks) // 4
    if to_caption:
        model = get_chat_model(fast=True)
        concurrency = max(1, settings.multimodal_caption_concurrency)
        for wave_start in range(0, len(to_caption), concurrency):
            wave = to_caption[wave_start : wave_start + concurrency]
            outputs = model.batch(
                [_caption_messages(img["png_bytes"]) for img in wave],
                config={"callbacks": get_callbacks(), "max_concurrency": concurrency},
                return_exceptions=True,
            )
            for img, output in zip(wave, outputs):
                llm_calls += 1
                caption = "" if isinstance(output, Exception) else (output.text or "").strip()
                if caption:
                    # Vision calls don't cost tokens proportional to caption length alone;
                    # this is a rough flat add-on, same spirit as the char/4 text estimate.
                    tokens += len(caption) // 4 + 300
                    captioned.append({**img, "caption": caption})
            if on_progress:
                on_progress(min(wave_start + len(wave), len(to_caption)), len(to_caption))

    saved_images = [{**img, "image_path": _save_image(doc_id, img["xref"], img["png_bytes"])} for img in captioned]

    texts = [c["text"] for c in text_chunks] + [i["caption"] for i in saved_images]
    metadatas = [
        {"doc_id": doc_id, "modality": "text", "page": c["page"], "image_path": ""} for c in text_chunks
    ] + [
        {"doc_id": doc_id, "modality": "image", "page": i["page"], "image_path": i["image_path"]}
        for i in saved_images
    ]
    if texts:
        _get_vector_store(collection).add_texts(texts, metadatas=metadatas)
        ensure_indexes(collection, filter_fields=("doc_id", "modality"))

    return MultimodalIngestStats(
        doc_id=doc_id,
        chunks=len(texts),
        latency_ms=(time.monotonic() - start) * 1000,
        text_chunks=len(text_chunks),
        images_found=len(raw_images),
        images_captioned=len(saved_images),
        images_failed=len(to_caption) - len(saved_images),
        images_over_cap=images_over_cap,
        llm_calls=llm_calls,
        tokens=tokens,
    )


def ask_detailed(
    question: str,
    doc_id: str,
    *,
    top_k: int = 5,
    **_: object,
) -> tuple[RagResult, list[dict]]:
    """Runs the full pipeline and also returns per-passage rows (with modality and image
    path) for the page to render images inline next to text evidence."""
    start = time.monotonic()
    steps: list[str] = []
    collection = get_collection(COLLECTION_NAME)
    vector_store = _get_vector_store(collection)

    hits = vector_store.similarity_search_with_score(question, k=top_k, pre_filter={"doc_id": doc_id})
    rows = [
        {
            "text": doc.page_content,
            "page": doc.metadata["page"],
            "modality": doc.metadata.get("modality", "text"),
            "image_path": doc.metadata.get("image_path", ""),
            "score": float(score),
        }
        for doc, score in hits
    ]
    steps.append(
        f"MongoDBAtlasVectorSearch (k={top_k}) on rag_multimodal returned {len(rows)} passage(s), "
        f"searched over both text chunks and image captions in one index."
    )

    context = "\n\n".join(
        f"[{i + 1}] (p.{r['page']}{', image caption' if r['modality'] == 'image' else ''}) {r['text']}"
        for i, r in enumerate(rows)
    )
    chain = ChatPromptTemplate.from_template(ANSWER_PROMPT) | get_chat_model() | StrOutputParser()
    answer = chain.invoke(
        {"context": context, "question": question},
        config={"callbacks": get_callbacks(), "run_name": "multimodal_ask"},
    )
    steps.append("Answered from the retrieved text passages and image captions via an LCEL chain.")

    result = RagResult(
        answer=answer,
        contexts=[Passage(text=r["text"], score=r["score"], page=r["page"]) for r in rows],
        steps=steps,
        llm_calls=1,
        tokens=(len(context) + len(question) + len(answer)) // 4,
        latency_ms=(time.monotonic() - start) * 1000,
    )
    return result, rows


def ask(question: str, doc_id: str, *, top_k: int = 5, **settings: object) -> RagResult:
    result, _rows = ask_detailed(question, doc_id, top_k=top_k, **settings)
    return result
