import streamlit as st

st.set_page_config(page_title="GenAI Personal Lab", layout="wide")

# index -> (name, what the technique does, its pipeline stages).
#
# `stages` mirrors the STAGES list each demo page declares for its own ribbon
# (the evaluation demo has no ribbon, so its chain names that page's own flow).
# It is duplicated here rather than imported because a demo page is a script --
# importing one would run it.
DEMOS: dict[str, tuple[str, str, list[str]]] = {
    "01": (
        "Naive RAG",
        "Fetch the passages whose meaning sits closest to the question, and trust that one of them holds the answer.",
        ["parse", "chunk", "embed", "retrieve", "answer"],
    ),
    "02": (
        "Chunking",
        "Four splitters cut the same document four ways and answer the same question.",
        ["parse", "split ×4", "embed", "index", "answer ×4"],
    ),
    "03": (
        "Hybrid search",
        "A keyword retriever runs alongside the vector one, and the two rankings are fused by position.",
        ["parse", "chunk", "embed", "index", "fuse", "answer ×3"],
    ),
    "04": (
        "Re-ranking",
        "A cheap search casts a wide net; a model that reads question and passage together reorders what it caught.",
        ["parse", "chunk", "embed", "index", "retrieve", "rerank", "answer"],
    ),
    "05": (
        "RAG evaluation",
        "Faithfulness, relevancy, precision and recall, computed the same way for every pipeline.",
        ["generate test set", "run pipelines", "score", "compare"],
    ),
    "06": (
        "Contextual retrieval",
        "A model writes the document context back into each passage before it is indexed.",
        ["parse", "chunk", "contextualize", "embed", "index", "retrieve ×2", "answer ×2"],
    ),
    "07": (
        "Agentic RAG",
        "Retrieval becomes a tool the model calls with arguments it chooses, as often as it decides it needs to.",
        ["parse", "chunk", "embed", "index", "agent loop", "answer"],
    ),
    "08": (
        "CRAG",
        "Passages are graded before they are trusted, and a web search takes over when none of them survive.",
        ["parse", "chunk", "embed", "index", "retrieve", "grade", "web fallback", "answer"],
    ),
    "09": (
        "Self-RAG",
        "The draft is critiqued against its own evidence, and rewritten when the critique fails it.",
        ["parse", "chunk", "embed", "index", "route", "retrieve", "generate", "critique", "answer"],
    ),
    "10": (
        "Adaptive RAG",
        "A classifier reads the question first and routes it down a path sized for it.",
        ["parse", "chunk", "embed", "index", "route", "retrieve", "generate", "answer"],
    ),
    "11": (
        "Multi-hop RAG",
        "The question is broken into a chain of simpler ones, each answer carried into the next search.",
        ["parse", "chunk", "embed", "index", "decompose", "hop", "combine", "answer"],
    ),
    "12": (
        "Vectorless RAG",
        "The table of contents becomes the index: descend the headings, then read the pages you land on.",
        ["parse", "build tree", "navigate", "read", "answer"],
    ),
    "13": (
        "SQL RAG",
        "The model is shown a schema and writes a query; rows come back as the context it answers from.",
        ["schema", "generate SQL", "run SQL", "answer"],
    ),
    "14": (
        "GraphRAG",
        "Entities and relationships are extracted at ingest, and the answer comes from a node's neighbourhood.",
        ["parse", "chunk", "extract graph", "embed", "vector search", "expand", "answer"],
    ),
    "15": (
        "KAG",
        "A fixed schema constrains extraction, and the question is planned as a sequence of Cypher queries.",
        ["parse", "chunk", "extract (schema-constrained)", "plan steps", "run Cypher", "answer"],
    ),
    "16": (
        "Multimodal RAG",
        "Every image is captioned at ingest, and the captions compete with the paragraphs in one search.",
        ["parse", "caption", "embed", "index", "retrieve", "answer"],
    ),
}

# The entry surface: each technique exists because plain vector search broke in
# a particular way, so the reader picks the break rather than the technique.
# The last entry is the way in for someone who has no symptom yet.
SYMPTOMS: list[tuple[str, list[str]]] = [
    ("It cannot find the exact part number, error code or surname.", ["03", "04"]),
    ("It answers confidently when the document says nothing on the subject.", ["08", "09"]),
    ("The answer needs facts from three pages, and no single passage holds them all.", ["11", "14", "15"]),
    ("The passage lost the facts that identified it when it was cut out of the document.", ["02", "06"]),
    ("It is blind to every chart, diagram and scanned table on the page.", ["16"]),
    ("The answer has to be computed — summed, joined, ranked — not found.", ["13"]),
    ('Every question gets the same search, from "hello" to a three-part comparison.', ["07", "10"]),
    ("Nothing tells me whether any of this helped, or whether embeddings were overkill.", ["05", "12"]),
]

BASELINE_PROMPT = "Nothing yet — show me the baseline that everything else is a repair of."

FAMILIES: dict[str, tuple[str, list[str]]] = {
    "Retrieval basics": (
        "The baseline pipeline, the first three repairs anyone makes to it, and how to tell whether they worked.",
        ["01", "02", "03", "04", "05"],
    ),
    "Enrichment": (
        "Change what goes into the index, rather than how the index is searched.",
        ["06"],
    ),
    "Agentic": (
        "Hand the model control over retrieval: whether to search at all, how often, and whether to believe the result.",
        ["07", "08", "09", "10", "11"],
    ),
    "Beyond vectors": (
        "Retrieval with no embedding anywhere in the pipeline.",
        ["12", "13"],
    ),
    "Graph": (
        "Store the relationships between things, not only the passages that mention them.",
        ["14", "15"],
    ),
    "Multimodal": (
        "Retrieve from what the page shows, not only from what it says.",
        ["16"],
    ),
}

BUILT_PAGES: dict[str, st.Page] = {
    "01": st.Page("demos/rag/naive/page.py", title="Naive RAG", url_path="naive"),
    "02": st.Page("demos/rag/chunking/page.py", title="Chunking", url_path="chunking"),
    "03": st.Page("demos/rag/hybrid/page.py", title="Hybrid search", url_path="hybrid"),
    "04": st.Page("demos/rag/rerank/page.py", title="Re-ranking", url_path="rerank"),
    "05": st.Page("demos/rag/evaluation/page.py", title="RAG evaluation", url_path="evaluation"),
    "06": st.Page("demos/rag/contextual/page.py", title="Contextual retrieval", url_path="contextual"),
    "07": st.Page("demos/rag/agentic/page.py", title="Agentic RAG", url_path="agentic"),
    "08": st.Page("demos/rag/crag/page.py", title="CRAG", url_path="crag"),
    "09": st.Page("demos/rag/self_rag/page.py", title="Self-RAG", url_path="self_rag"),
    "10": st.Page("demos/rag/adaptive/page.py", title="Adaptive RAG", url_path="adaptive"),
    "11": st.Page("demos/rag/multi_hop/page.py", title="Multi-hop RAG", url_path="multi_hop"),
    "12": st.Page("demos/rag/vectorless/page.py", title="Vectorless RAG", url_path="vectorless"),
    "13": st.Page("demos/rag/sql_rag/page.py", title="SQL RAG", url_path="sql_rag"),
    "14": st.Page("demos/rag/graph_rag/page.py", title="GraphRAG", url_path="graph_rag"),
    "15": st.Page("demos/rag/kag/page.py", title="KAG", url_path="kag"),
    "16": st.Page("demos/rag/multimodal/page.py", title="Multimodal RAG", url_path="multimodal"),
}

# Constrains the reading measure. The app is laid out wide for the demo pages,
# where evidence and trace genuinely use the width; a page of sentences does
# not, so home gets margins instead of 120-character lines.
HOME_WIDTH = 1040
SHOW_ALL = "home_show_all"


def open_demo(index: str) -> None:
    """Navigates to a demo, or says so quietly if its page is missing."""
    page = BUILT_PAGES.get(index)
    if page is None:
        st.warning(f"Demo {index} has no page yet.")
        return
    st.switch_page(page)


def symptom_tile(symptom: str, indices: list[str], key: str) -> None:
    # No `height`: the tile takes exactly the height of its text. Any height at
    # all -- a pixel count or "stretch" -- turns the container into a scroll
    # surface, which hides the end of the longer symptoms.
    with st.container(border=True):
        st.markdown(symptom)
        with st.container(horizontal=True, gap="small"):
            for index in indices:
                name = DEMOS[index][0]
                if st.button(f"`{index}`  {name}", key=f"{key}_{index}", type="tertiary"):
                    open_demo(index)


def baseline_tile() -> None:
    with st.container(border=True):
        st.markdown(BASELINE_PROMPT)
        if st.button("Open Naive RAG", key="symptom_baseline", type="primary"):
            open_demo("01")


def catalog_card(index: str) -> None:
    """One demo as a card: index and name, what it does, the chain it runs on."""
    name, hook, stages = DEMOS[index]
    with st.container(border=True):
        if st.button(f"`{index}`  {name}", key=f"catalog_{index}", type="tertiary"):
            open_demo(index)
        st.markdown(hook)
        # The same signal-chain strip the demo page runs on, at rest. Mono, and
        # every card the same width, so the chains can be compared card to card.
        st.caption("`" + " → ".join(stages) + "`")


def full_catalog() -> None:
    """Every demo as a card, grouped by family, three to a row."""
    for family, (note, indices) in FAMILIES.items():
        st.subheader(family, divider="gray")
        st.caption(note)
        st.space("small")
        for row_start in range(0, len(indices), 3):
            cols = st.columns(3, gap="medium")
            for col, index in zip(cols, indices[row_start : row_start + 3]):
                with col:
                    catalog_card(index)
        st.space("small")


def home() -> None:
    with st.container(horizontal_alignment="center"):
        with st.container(width=HOME_WIDTH):
            st.title("GenAI Personal Lab")
            st.markdown(
                "Sixteen retrieval techniques, one runnable demo each. Every one of them exists "
                "because plain vector search broke in a particular way — so start from the way "
                "yours is breaking."
            )
            st.markdown(
                ":gray[Each page loads a PDF, runs the pipeline stage by stage, and shows the "
                "passages it retrieved with their scores, the decisions it took, and what the run "
                "cost in latency and tokens.]"
            )
            st.divider()

            st.header("What is your retrieval getting wrong?")
            st.space("small")

            # The eight symptoms, then the way in for a reader who has no
            # symptom yet, three to a row.
            tiles = [
                (lambda s=symptom, i=indices, n=position: symptom_tile(s, i, key=f"symptom_{n}"))
                for position, (symptom, indices) in enumerate(SYMPTOMS)
            ]
            tiles.append(baseline_tile)
            for row_start in range(0, len(tiles), 3):
                cols = st.columns(3, gap="medium")
                for col, render_tile in zip(cols, tiles[row_start : row_start + 3]):
                    with col:
                        render_tile()

            st.space("medium")

            show_all = st.session_state.get(SHOW_ALL, False)
            label = "Hide the full catalog" if show_all else "Show all sixteen techniques"
            if st.button(label, key="toggle_catalog", width="stretch", icon=":material/list:"):
                st.session_state[SHOW_ALL] = not show_all
                st.rerun()
            if not show_all:
                st.caption(
                    "Every technique, grouped by family, with the pipeline each one runs — "
                    "useful once you know what you are looking for."
                )
            else:
                st.space("small")
                full_catalog()


HOME_PAGE = st.Page(home, title="Home", icon=":material/home:", default=True)


def sidebar_nav(current: st.Page) -> None:
    """Home link, then the sixteen demos inside a collapsed expander.

    Streamlit's own sidebar navigation is hidden (`position="hidden"`) and
    rebuilt here because its `expanded=` argument cannot be relied on: the first
    time a reader clicks "View N more", the frontend writes
    `sidebarNavState=expanded` into localStorage and force-expands the list on
    every page from then on, whatever `expanded=` says. There is no way to clear
    that key from Python, and an expanded list of sixteen pushes the document
    picker, the demo's settings and "Clear my data" below the fold.

    The expander is keyed on the current page, so every navigation gives it a
    fresh identity: arriving at a demo always finds the list closed, and
    opening it on one demo does not leave it open on the next.
    """
    st.sidebar.page_link(HOME_PAGE)
    with st.sidebar.expander("Demos", expanded=False, key=f"nav_demos_{current.title}"):
        for page in BUILT_PAGES.values():
            st.page_link(page)


nav = st.navigation(
    {
        "": [HOME_PAGE],
        "Demos": list(BUILT_PAGES.values()),
    },
    position="hidden",
)
# Rendered before the page runs, so the page's own sidebar content sits below it.
sidebar_nav(nav)
nav.run()
