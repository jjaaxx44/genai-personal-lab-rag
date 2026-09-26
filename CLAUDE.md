# GenAI Personal Lab

A Streamlit app where each page demonstrates one RAG technique, 16 in total. Demo/POC quality — the point is to show the technique and the reasoning, not to ship production software.

**Read `taxonomy/plans/RAG_IMPLEMENTATION_PLAN.md` before starting a step.** It holds the stack, exact versions, repo layout, the shared `ingest()` / `ask()` contract, the step order and each step's "done when" list. Don't restate it here; if this file and the plan disagree on technical detail, the plan wins.

Capability checklist: `taxonomy/genai-capability-taxonomy.md` — tick a demo off when it lands.

> `taxonomy/` is gitignored on purpose: the plan and the checklist are working notes and stay in the local copy. If you cloned this repo, those two paths won't exist for you, and nothing in `app/` depends on them.

UI work: `DESIGN.md` holds this app's design tokens and page patterns. Method comes from the `app-ux-design` skill; Streamlit mechanics from the `developing-with-streamlit` skill that ships inside the Streamlit package.

## Rules

1. **Standalone demos.** Each demo is `app/demos/rag/<name>/` (`pipeline.py`, `page.py`, `README.md`, `flow.json`, `flow.html`), imports from `app/core/` only, never from another demo, and touches only its own MongoDB collection. Duplication between demos beats shared abstractions, and demos are never refactored together "for consistency".
2. **Thin core.** Something moves into `core/` when a *second* demo needs it, not in anticipation.
3. **No automated tests.** Verification is manual: run the app, walk the step's "done when" list.
4. **Fixed stack.** Never add, remove or bump a dependency, service or local model, or change a pin, while implementing a step — raise it as a decision first. Everything has to keep running on 4 CPUs / 8 GB, CPU only, no GPU. Archify (see *Flow diagrams*) is an authoring tool run on the host with Node, not an app dependency: the app only serves the pre-rendered `flow.html`.
5. **Import discipline.** Never import `langchain_community` or `langchain_classic` — both are archived/legacy and are present only as transitive dependencies, so the import will work and nothing will warn you. Naive RAG, hybrid search and vectorless RAG have no `langchain` imports at all; their mechanics stay visible.
6. **Teaching value first.** Each page shows pipeline steps, retrieved context with scores, and latency/tokens. Code that hides the technique is a defect.
   Every README follows one fixed structure, in this order: `What it is` (the technique in general — how it works and why, never this repo's implementation), `Ingestion flow` and `Retrieval and generation flow` (one mermaid diagram each; `st.markdown` renders mermaid fences natively, and so does GitHub), `Strengths`, `Limitations` (inherent trade-offs *and* the ways it fails, merged — not a separate failure-modes section), `Where to use it`, and a closing `In this demo` holding every implementation specific: libraries, collection names, env vars, caps, caveats.
   Each page also shows an interactive flow diagram in an `Explore the flow` expander at the top of the `How it works` tab. The mermaid in the README stays, because GitHub can't render the HTML. The two must describe the same flow: change one, change the other.
7. **Soft failure.** Rate limits, a paused database and missing optional keys give a clear UI message — never a stack trace, never a crash on startup.

## Flow diagrams

Each demo's `flow.json` is the source, written for the `archify` skill. `flow.html` is its render and is never edited by hand. Install the skill into this repo from its root with `npx -y skills add tt-a1i/archify --skill archify --agent claude-code --copy --yes` (v2.17 was used for the first diagrams). It lands in `.claude/skills/archify/`, which is gitignored. Draw one `dataflow` diagram per demo, with ingestion and question time as two rows that meet at the store. Labels name the real models, collections and calls from `pipeline.py`. Regenerate from the skill folder:

```bash
node bin/archify.mjs validate dataflow <demo>/flow.json --quality showcase --json
node bin/archify.mjs deliver dataflow <demo>/flow.json <demo>/flow.html --quality showcase --json
```

A render counts only when both commands exit 0 with all 9 checks passing and no errors or warnings. Delete the `flow.visual-check.*` files that `visual-check` writes; they don't belong in the repo.

Rolling this out to all 16 demos in one change was approved on 2026-09-26. It's the one sanctioned exception to "never refactored together" in rule 1.

## How to work

**Nothing is committed or pushed without the user's approval.** Make the changes and stage them if you like, then stop, show what would be committed and the message you propose, and wait for an explicit yes. The repo is public (`github.com/jjaaxx44/genai-personal-lab-rag`), so a push is visible immediately and undoing one means rewriting published history. An approval covers the commit in front of you, not the next one — ask again each time, even within a session.

One step from the plan per session, in order. Implement it, then stop so a human can check it against that step's "done when" list. If a step turns out to be wrong or impossible, say so and propose the change instead of working around it — and once agreed, update `RAG_IMPLEMENTATION_PLAN.md` in the same session, since every later step reads from it.
