import streamlit as st
from pymongo.collection import Collection

from .mongo import clear_doc
from .pdf import PdfValidationError, compute_doc_id, validate_pdf
from .types import RagResult


def upload_widget(demo_key: str) -> tuple[bytes, str] | None:
    """Sidebar PDF upload + validation. Returns (pdf_bytes, doc_id), or None if nothing valid yet."""
    # Keyed on a generation counter so clear_data_button can force a fresh
    # (empty) uploader by bumping it -- clearing session_state[key] directly
    # does not reliably reset a file_uploader widget.
    gen = st.session_state.get(f"{demo_key}_upload_gen", 0)
    uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], key=f"{demo_key}_upload_{gen}")
    if uploaded is None:
        return None

    data = uploaded.getvalue()
    try:
        validate_pdf(uploaded.name, data)
    except PdfValidationError as exc:
        st.sidebar.error(str(exc))
        return None

    return data, compute_doc_id(data)


def pipeline_ribbon(stages: list[str], *, active: int = -1, timings: dict[str, float] | None = None) -> None:
    """Stage chips in execution order: filled = done, highlighted = running, muted = not reached."""
    timings = timings or {}
    for col, stage in zip(st.columns(len(stages)), stages):
        with col:
            label = stage
            timing = timings.get(stage)
            if timing is not None:
                label += f"  `{timing:.2f}s`"
            if active < 0:
                st.caption(label)
            elif stages.index(stage) < active:
                st.success(label, icon=":material/check:")
            elif stages.index(stage) == active:
                st.info(label, icon=":material/play_arrow:")
            else:
                st.caption(label)


PREVIEW_CHARS = 100


def evidence_view(passages: list[dict]) -> None:
    """Ranked passages as bordered blocks: mono score, relevance bar, page number, text.

    Text over PREVIEW_CHARS is truncated with an expander for the full chunk,
    since some chunking strategies (sliding window especially) produce chunks
    long enough to dominate the page.
    """
    max_score = max((p["score"] for p in passages), default=1.0) or 1.0
    for p in passages:
        with st.container(border=True):
            bar_len = round(20 * (p["score"] / max_score))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            st.markdown(f"`{p['score']:.3f}` {bar}  p.{p['page']}")
            text = p["text"]
            if len(text) <= PREVIEW_CHARS:
                st.write(text)
            else:
                st.write(text[:PREVIEW_CHARS].rstrip() + "…")
                with st.expander(f"View complete chunk ({len(text):,} chars)"):
                    st.write(text)


def metrics_row(result: RagResult) -> None:
    cols = st.columns(4)
    cols[0].metric("Latency", f"{result.latency_ms:.0f} ms")
    cols[1].metric("Tokens", f"{result.tokens:,}")
    cols[2].metric("LLM calls", result.llm_calls)
    cols[3].metric("Passages", len(result.contexts))


def graph_triples_dot(triples: list[dict], highlight: set[str] = frozenset()) -> str:
    """Renders entity-relationship triples (source/source_type/rel/target/target_type
    dicts) as a DOT digraph, for the two Neo4j-backed demos (GraphRAG, KAG)."""
    node_ids: dict[str, str] = {}

    def _id(name: str) -> str:
        if name not in node_ids:
            node_ids[name] = f"n{len(node_ids)}"
        return node_ids[name]

    node_types: dict[str, str] = {}
    for t in triples:
        node_types[t["source"]] = t["source_type"]
        node_types[t["target"]] = t["target_type"]

    lines = ["digraph G {", 'rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];']
    for name, node_type in node_types.items():
        label = name.replace('"', "'")
        fill = "#FFD54F" if name in highlight else "#E8EAF6"
        lines.append(f'{_id(name)} [label="{label}\\n({node_type})", fillcolor="{fill}"];')

    seen_edges = set()
    for t in triples:
        edge_key = (t["source"], t["rel"], t["target"])
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        lines.append(f'{_id(t["source"])} -> {_id(t["target"])} [label="{t["rel"]}", fontsize=9];')
    lines.append("}")
    return "\n".join(lines)


def graph_diagram(mermaid_source: str, path: list[str]) -> None:
    """Renders a compiled LangGraph's mermaid diagram, outlining the nodes actually
    visited on the last run -- so the page can show which branch a question took.

    LangGraph's own classDefs hardcode a light node fill with no text color, so
    labels inherit the page's theme text color -- near-white on near-white in
    dark mode, unreadable. Mermaid can't consume the app's CSS theme variables,
    so (as with graph_triples_dot below) fill/text are pinned to this app's own
    light/dark tokens (DESIGN.md) and appended last, since a later classDef for
    the same class wins in mermaid.
    """
    dark = st.context.theme.type != "light"
    surface = "#221F1A" if dark else "#F1ECE2"
    text = "#EFEAE0" if dark else "#1B1A17"
    border = "#3A352D" if dark else "#DCD5C7"
    accent = "#2FB39A" if dark else "#00806B"

    theme_override = "\n".join(
        [
            f"classDef default fill:{surface},stroke:{border},color:{text},line-height:1.2",
            f"classDef first fill-opacity:0,stroke:{border},color:{text}",
            f"classDef last fill:{border},stroke:{border},color:{text}",
        ]
    )
    visited = dict.fromkeys(n for n in path if n)
    highlight = "\n".join(f"style {node} stroke:{accent},stroke-width:3px,color:{text}" for node in visited)
    diagram = f"{mermaid_source}\n{theme_override}" + (f"\n{highlight}" if highlight else "")
    st.mermaid_chart(diagram)


def clear_data_button(demo_key: str, collection: Collection, doc_id: str | None) -> bool:
    """Renders the button. Returns True the run it's clicked, so the page can drop its
    own doc_id/result state -- the rerun this triggers is left to the caller."""
    flash_key = f"{demo_key}_cleared_count"
    if flash_key in st.session_state:
        st.sidebar.success(f"Cleared {st.session_state.pop(flash_key)} record(s).")

    if st.sidebar.button("Clear my data", key=f"{demo_key}_clear", disabled=doc_id is None):
        deleted = clear_doc(collection, doc_id)
        st.session_state[flash_key] = deleted
        st.session_state[f"{demo_key}_upload_gen"] = st.session_state.get(f"{demo_key}_upload_gen", 0) + 1
        return True
    return False
