# Design decisions — GenAI Personal Lab

The values and patterns this app is built from. Method lives in the `app-ux-design` skill; Streamlit mechanics live in the `developing-with-streamlit` skill that ships inside the Streamlit package. This file holds what is specific to this repo — and the reason for each choice, so later screens can be checked against intent rather than taste.

## Grounding

The subject is **retrieval**: text goes through a chain of stages, and an answer comes out carrying evidence that can be inspected and doubted. The references the design borrows from:

1. **Signal-chain strips** on audio and lab equipment — a row of stages, each with its own reading, one of them lit.
2. **Lab notebooks** — warm paper, ruled structure, handwritten-in-margin annotations, numbers recorded in a fixed format.
3. **Library catalog and index pages** — numbered entries, grouped by family, scanned by eye rather than browsed as tiles.

So: the pipeline is the hero, evidence sits beside every claim, and the chrome behaves like instrument housing — quiet, square, uniform.

**The first screen shows the catalog itself** — the sixteen techniques as an index — not a welcome banner.

## Vocabulary

One word per concept, everywhere: UI labels, READMEs, log lines, variable names.

| Use | Not |
|---|---|
| passage | chunk, snippet, context block |
| stage | step, phase, node |
| trace | log, history, debug output |
| evidence | sources, references, citations panel |
| document | file, PDF, corpus item |
| score | relevance, similarity, rank value |
| demo | example, module, app |

`chunk` survives in one place only: the chunking demo, where it is the subject.

## Identity, with reasons

- **Concept:** a lab instrument, not a chatbot — the app exists to show *how* an answer was produced, so the pipeline and its evidence get the visual weight.
- **Surface temperature:** warm paper (light) / warm charcoal (dark). Reason: the notebook reference, and distance from the cool grey-blue every framework ships with.
- **Accent with one job:** deep pine green marks *what is live or selected* — the active stage, retrieved passages, the primary button. Nowhere else, so its presence always means something.
- **Type pair:** serif headings (notebook), sans body (legibility), monospace for every number and identifier (instrument readings are fixed-width and comparable down a column).
- **Edges:** 1px borders, small radius, no shadows, no gradients. Reason: housing, not cards; borders encode grouping without implying elevation that isn't there.
- **Signature element:** the **pipeline ribbon** — the signal-chain strip, one per demo, different per technique because the pipelines genuinely differ.

**Considered and rejected:** slate-and-amber (reads more like hardware, but flattens the paper/notebook reference and makes long reading harder); the default Streamlit theme (interchangeable with every other demo app).

**Known adjacency:** warm paper sits near the "cream page + terracotta accent" cliché. What keeps it clear: a green accent rather than terracotta, serif + mono rather than geometric sans, no cards, no shadows, and an identity that rests on the ribbon rather than on the background colour. If a screen starts to feel like stationery, that's the drift to correct.

## What structure encodes

Structural devices carry meaning here; none are decorative.

| Device | Means |
|---|---|
| Order of ribbon chips | execution sequence |
| Filled vs outlined vs muted chip | running / completed / not reached |
| Rule under a family heading | grouping in the catalog |
| Bordered block | one retrieved passage, inspectable |
| Bar length beside a score | relative relevance, from the sequential ramp |
| Mono type | a value you can compare with another value |

**Motion:** exactly one moment — the ribbon advancing as stages complete. Everything else is still.

## Tokens

### Light

| Token | Value |
|---|---|
| `bg` | `#FAF8F3` |
| `surface` (sidebar, code, raised) | `#F1ECE2` |
| `border` | `#DCD5C7` |
| `text` | `#1B1A17` |
| `text-muted` | `#6B655B` |
| `accent` (fills, active) | `#00806B` |
| `accent-text` (links, accent text) | `#00705E` |
| `warning` | `#8A5A00` |
| `error` | `#A32F2A` |

### Dark

| Token | Value |
|---|---|
| `bg` | `#191713` |
| `surface` | `#221F1A` |
| `sidebar-bg` | `#141210` |
| `border` | `#3A352D` |
| `text` | `#EFEAE0` |
| `text-muted` | `#9C9486` |
| `accent` | `#2FB39A` |
| `warning` | `#E0A94A` |
| `error` | `#E4736B` |

### Data colours

Categorical, in fixed order (demo comparisons on the evaluation page):

- light: `#00806B`, `#C2571B`, `#2F5FD0`, `#A77A00`, `#B03E8C`
- dark: `#1FA58E`, `#D9773A`, `#5C82D8`, `#B58A22`, `#C568A8`

Sequential (relevance scores), light → dark:

- light: `#68B9A8`, `#53A594`, `#3E9180`, `#277D6D`, `#066A5B`, `#005649`, `#004238`, `#003028`, `#001E18`, `#000E0A`
- dark: `#005448`, `#026959`, `#257C6C`, `#3C8F7F`, `#51A392`, `#66B8A6`, `#7BCDBA`, `#90E2CF`, `#A5F8E4`, `#ECFFFA`

All values were computed and validated for contrast, lightness banding, chroma and colour-vision separation in both modes. Don't substitute by eye; re-validate if you change one.

### Type and space

- Headings: Instrument Serif (400/500); body: Inter at 15px; numbers, IDs, code: JetBrains Mono.
- Self-hosted from `app/static/fonts/` with `[[theme.fontFaces]]` and `server.enableStaticServing = true`, so there's no font-CDN dependency at runtime.
- Spacing scale: 4 / 8 / 16 / 24 / 48. Nothing in between.
- Reading measure 60–80 characters; wide screens get margins, not longer lines.
- Numbers: fixed precision with units — `1.24 s`, `0.837`, `1,204 tokens`, `doc a3f91c…`.

Everything above is set in `.streamlit/config.toml` as theme tokens — never CSS injection, never colour literals in page code.

## Page patterns

### Home — symptom first, catalog behind one click

**Superseded:** home used to open on the numbered index itself. Sixteen equally-weighted rows gave a reader no way in, and every row ended in the same word (`built`), so the one varying column carried no information. The index is still here; it is no longer the first thing.

The entry surface is the **symptom grid**: eight bordered tiles, three to a row, each stating a way retrieval breaks in the reader's own words — never a technique name.

```
It cannot find the exact part number,     It answers confidently when the
error code or surname.                    document says nothing on the subject.

`03` Hybrid search  `04` Re-ranking       `08` CRAG  `09` Self-RAG
```

The symptom is body text; the demos it routes to are tertiary buttons carrying a mono index. The grid's ninth tile is the way in for a reader with no symptom yet — `Nothing yet — show me the baseline` — and holds the page's **one** accent button, `Open Naive RAG`.

The reason this beats the index: every technique here exists because plain vector search broke in a particular way, so the break is the thing a reader actually recognises. It also cuts the first choice from sixteen to nine.

Below the grid, one full-width bordered button, `Show all sixteen techniques`, toggles the **full catalog** inline: the same card as the symptom tile, three to a row, grouped under serif family headings with a thin rule. Each card carries a tertiary button holding mono index and name, one line on what the technique does, and the **stage chain it runs on** (`parse → chunk → embed → retrieve → answer`), taken from that demo's own `STAGES`.

The chains are the reason the catalog is worth showing, and why the cards share a width: the chains line up, so the pipelines visibly diverge card to card — SQL RAG never embeds, Self-RAG carries a `critique`, KAG runs Cypher. A card grid is the default that makes apps interchangeable, so it is used here on purpose and only here: the grid is a **comparison**, not a menu of tiles.

Both surfaces read from one `DEMOS` dict keyed by index, so a name or chain is written once.

**No card or tile ever scrolls its own text**, so neither carries a `height` at all — `st.container(border=True)` and nothing more. Each one is exactly as tall as its own content. Two attempts failed before this: a pixel height (`176`/`208`) clipped the longest stage chains behind an inner scrollbar, and `height="stretch"` scrolled as well, so in this version *any* height makes a container a scroll surface. The cost is that cards in a row end at different depths; that is the accepted trade, because a hidden stage chain defeats the reason the catalog exists.

Home is width-constrained (`HOME_WIDTH`) and centred. The app is `layout="wide"` for the demo pages, where evidence and trace genuinely use the width; a page of sentences gets margins instead of 120-character lines.

### Sidebar navigation

Built by hand in `sidebar_nav()`: a `Home` page link, then a **collapsed expander** labelled `Demos` holding all sixteen `st.page_link`s. Streamlit's own navigation is turned off with `st.navigation(..., position="hidden")`.

The reason is the demo page's sidebar, not the navigation: the document picker, the demo's own settings and "Clear my data" live there, and an open list of sixteen pushes all three below the fold, so arriving on a demo means scrolling the sidebar before the page can be used at all. The controls matter more than the list, so the list yields.

**`expanded=` on `st.navigation` cannot do this**, which is why the navigation is rebuilt rather than configured. The first time anyone clicks "View N more", Streamlit's frontend writes `sidebarNavState=expanded` into `localStorage` and force-expands the list on every page from then on, ignoring whatever `expanded=` says. Nothing in Python can clear that key, and clearing it would mean injecting JS, which the house rules forbid. Note the flag is per origin *including port*, so the same build can look correct on one port and wrong on another — that difference is the symptom, not two different builds.

The expander is keyed on the current page (`key=f"nav_demos_{current.title}"`), so every navigation gives it a fresh identity. Arriving at a demo always finds the list closed, and opening it on one demo does not leave it open on the next.

`sidebar_nav()` is called between `st.navigation()` and `nav.run()`, so it sits above whatever sidebar content the page itself adds.

### Demo page

1. **Header** — demo name (serif) + one plain sentence on what the technique does.
2. **Pipeline ribbon** — stage chips in execution order (`parse → chunk → embed → retrieve → rerank → answer`), each showing its timing in mono once run. Visible before a question is asked, muted, so it also serves as the empty state and the explanation.
3. **Question row** — input + Ask, wrapped in a form so Enter submits it, not just a click on Ask.
4. **Answer** — streamed where the chain allows, with page citations.
5. **Evidence** — ranked passages as bordered blocks: mono score, bar from the sequential ramp, page number, then text. Never a dataframe dump.
6. **Detail tabs** — `How it works` (renders the demo's README.md) first, then `Trace` (stages, timings, decisions). The pair is stateful: `st.tabs([...], key=f"{DEMO}_tabs", on_change="rerun")`, so the selected tab lives in session state and the page can move it.

   The rule is **explain first, then get out of the way**. On arrival nothing has run, so the reader gets the explanation: `How it works` is first and Streamlit selects it by default. The moment a run produces a result, the page sets `st.session_state[f"{DEMO}_tabs"] = "Trace"` — set immediately after the result is stored, which is always above the `st.tabs` call, so the widget picks it up the same run. The reader asked a question; what they now want is what the pipeline did, not the prose they already had the chance to read. The tab is a normal control throughout, so `How it works` is one click away and stays available.

   Both tabs render from page load. `Trace` shows a placeholder until a run has happened.

   READMEs carry two mermaid flow diagrams (ingestion, then retrieval and generation). `st.markdown` renders ` ```mermaid ` fences natively, so the README stays one plain file that renders in the app and on GitHub — no splitting, no `st.mermaid_chart` call, no custom component. Diagrams are unstyled: no `classDef`, no colour literals, so they inherit the app theme in both modes. The one exception is `core.ui.graph_diagram`, which has to override LangGraph's own hardcoded classDefs — see the comment there.
7. **Metrics row** — latency, tokens, LLM calls, passages retrieved.

Sidebar: document picker + upload, demo settings, "Clear my data".

### Shared behaviour

- Session-state keys are namespaced per demo (`f"{DEMO}_question"`), since state is shared across pages.
- `@st.cache_resource` for models and clients; `@st.cache_data` for per-`doc_id` derived data.
- Long work drives the ribbon plus a `st.status` with named sub-steps — never an anonymous spinner.
- Errors follow the three-kind split: expected/actionable → inline warning, app keeps working; input rejected → name the limit and the actual value; unexpected → short message, traceback to the log.
- Charts: evaluation page only, themed colours in fixed order, one measure per chart.
- Accessibility floor: validated contrast, real labels on every control, focus visible, meaning never carried by colour alone (chips and scores carry text too).

## House rules

No emoji (Material Symbols if an icon is genuinely needed). Sentence case. Buttons are verbs naming their object. Active voice, plain language, the vocabulary above. No custom CSS, no third-party components needing a Node build, no browser storage.
