# TechDoc-Helper

A technical documentation assistant over the LangChain / LangGraph **docs and source**,
built as a LangGraph supervisor on AWS Bedrock. Ask "how do interrupts work and how are
they implemented?" and it routes to the prose corpus, the code corpus, or both in
parallel, then streams one answer with inline `[n]` citations you can expand and check.

The point of the project is not the chat box — it is that **every design decision has a
number behind it**. Retrieval, chunking, and generation each have their own eval harness,
and the interesting parts are the places where a measurement contradicted the obvious
choice. Full engineering log with rationale and results: **[DECISIONS.md](DECISIONS.md)**.

---

## Architecture

```
                    ┌── search_docs ──┐
question ── route ──┼── search_code ──┼── synthesize ── answer + sources
   (Haiku 4.5)      └── (both, parallel)   (Sonnet 5)

retrieval per agent:  dense (Qdrant/Cohere) ─┐
                      BM25 (in-memory)      ─┴─ RRF ── Cohere rerank-v3.5 ── top-5
                                                       └ post-filter by corpus type
```

| Layer | Choice | Why |
| --- | --- | --- |
| Orchestration | LangGraph `StateGraph`, conditional-edge fan-out | the two corpora fail differently, so they get separate candidate pools |
| Concurrency | `Annotated[list[Document], operator.add]` reducer on `docs` | both agents write the same key in one superstep; without the reducer that is an `InvalidUpdateError` |
| Retrieval | hand-rolled hybrid + RRF (k=60) + cross-encoder rerank | RRF is scale-invariant, so no normalising unbounded BM25 scores against cosine similarity |
| Models | Haiku routes, Sonnet 5 writes, Sonnet 4.6 judges | routing runs on every query and needs the least capability; the judge needs `temperature=0`, which Sonnet 5 no longer honours |
| Observability | LangSmith via env vars only | one `load_dotenv()` covers eval, CLI and UI with no tracing code in the app |

---

## Results

**Retrieval** — 38-question golden set, zero LLM calls, ~1 min to re-run. The three configs
are literal prefixes of one `HybridRetriever.retrieve(mode=)` code path, so the eval
measures the real pipeline rather than a reimplementation of it.

| config | hit@1 | hit@3 | hit@5 | MRR | doc hit@5 | code hit@5 |
| --- | --- | --- | --- | --- | --- | --- |
| dense only | 0.711 | 0.737 | 0.789 | 0.736 | 1.000 | 0.579 |
| hybrid (RRF) | 0.684 | 0.816 | 0.895 | 0.767 | 1.000 | 0.789 |
| **hybrid + rerank** | **0.763** | **0.921** | **0.921** | **0.842** | **1.000** | **0.842** |

Rerank earns its ~0.55s/query in MRR and hit@3, not in hit@10 — a cross-encoder can only
reorder what fusion supplied, never add candidates.

No LLM calls, but **not bit-reproducible**: Bedrock's Cohere embedder returns vectors that
differ by ~1e-3 for byte-identical input, so cosine scores move in the 4th decimal and
near-tied ranks flip. Re-running reproduced dense and rerank exactly and moved hybrid hit@5
between 0.868 and 0.895 — one question (q019) whose gold chunk sits on the rank-5/6 boundary
with a 0.0008 score gap. Worth knowing before reading a ±0.03 move as a change.

**Generation** — 38 questions, Sonnet 4.6 as judge. Three of the five metrics use an LLM;
router accuracy and citation syntax are deterministic, because ground truth already exists
for both.

| slice | n | faithfulness | relevance | citation precision | % with citations |
| --- | --- | --- | --- | --- | --- |
| all | 38 | 4.789 | 4.632 | 0.876 | 0.974 |
| doc | 19 | 4.895 | 4.737 | 0.883 | 0.947 |
| code | 19 | 4.684 | 4.526 | 0.870 | 1.000 |

Pooled over the 126 distinct (claim, source) pairs actually judged, citation precision is
**0.857** — the honest figure for "how often is a citation trustworthy", where the 0.876
per-question mean answers "how good is a typical answer". Both are reported.

**Three findings worth more than the aggregates:**

- **BM25's default tokenizer is `str.split()`**, which cannot match query token
  `_dict_int_op` against source token `_dict_int_op(` — so exact-symbol matching, the entire
  reason BM25 is in this pipeline, was silently broken and hybrid scored *worse* than dense.
  A corpus-aware tokenizer took BM25 recall@20 from 0.474 to **0.921**. Nothing threw; only
  the eval caught it, and only the per-query dump explained it.
- **Faithfulness and relevance are independent, and the run proves it.** Three questions
  scored faithfulness 5 with relevance ≤ 3 — the generator correctly refused or hedged on
  context that did not contain the answer. That combination localises the failure to
  retrieval, which neither metric alone can do.
- **The router hedges rather than errs.** strict 0.316 / lenient 0.947 / both_rate 0.632:
  only 2 of 38 are true misses, and the other 24 "strict failures" searched the right corpus
  alongside a second one. Reporting one number without the other would either slander a
  cautious router or hide a useless one.

**Chunking** was swept (500 / 1000 / 1500) and the first-guess 1000/150 won. The sweep
needed its own harness because changing `chunk_size` renumbers every `chunk_id`: at 500,
33/38 gold ids still resolve to a real chunk but only 1/38 point at the text that was
labelled. Labels are re-anchored by content instead.

---

## Quickstart

**Prerequisites**

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- AWS credentials on the standard boto3 chain (`aws configure` / SSO / instance role).
  There are deliberately **no AWS keys in `.env`**.
- Bedrock model access in `us-east-1` for `anthropic.claude-sonnet-5`,
  `anthropic.claude-sonnet-4-6`, `anthropic.claude-haiku-4-5`, `cohere.embed-english-v3`
  and `cohere.rerank-v3-5`. The reranker ARN pins `us-east-1`.

**1. Install and configure**

```bash
uv sync
cat > .env <<'EOF'
AWS_REGION=us-east-1
# optional, for tracing:
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<your key>
LANGSMITH_PROJECT=techdoc-helper
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # must match your key's region
EOF
```

**2. Fetch the corpus.** Third-party source is not vendored;
[`data/corpus_manifest.yaml`](data/corpus_manifest.yaml) pins the commits so the index is
reproducible from a clone.

```bash
mkdir -p data/raw && cd data/raw
git clone https://github.com/langchain-ai/docs.git      && git -C docs      checkout 7c417783c90d5ee9fb03378eeddaf20082fddfc5
git clone https://github.com/langchain-ai/langgraph.git && git -C langgraph checkout 55ec2f21939ce7755e6398c11b541de8926245ee
git clone https://github.com/langchain-ai/langchain.git && git -C langchain checkout a8fd0da2b7c3409db9a16d0c7bcd55463967351b
```

**3. Build the index** — 9997 chunks from 432 files, ~13 min of embedding, once.
Re-running detects the existing collection and skips it; `chunk_id` → `uuid5` makes
re-ingestion an idempotent upsert rather than a duplicate insert.

```bash
uv run python -c "
from techdoc.ingest import load_manifest, iter_source_files, chunk_records
from techdoc.vectorstore import build_index
from techdoc.config import get_embeddings
build_index(chunk_records(list(iter_source_files(load_manifest()))), get_embeddings())"
```

**4. Ask it something**

```bash
uv run streamlit run app.py            # chat UI, streamed, with source expanders
uv run python -m techdoc.graph         # CLI: three questions, one per route
```

> Qdrant runs in local on-disk mode and holds an **exclusive file lock** — one process at a
> time. Stop the UI before running an eval.

---

## Eval harness

```bash
uv run python -m techdoc.eval.retrieval_metrics                 # ~1 min, no LLM calls
uv run python -m techdoc.eval.generation_metrics                # judge the cached answers
uv run python -m techdoc.eval.generation_metrics --regenerate   # + ~15 min of generation
uv run python -m techdoc.eval.generation_metrics --limit 5      # smoke test
uv run python -m techdoc.eval.sweep_chunking                    # re-embeds per config
```

Generation eval is **two stages on purpose** — generate (expensive, cached to
`answers_cache.json`) then judge (cheap, re-runnable). That makes rubric iteration a cache
read instead of a 15-minute regeneration, and it removes a confound: Sonnet 5 ignores
`temperature`, so regenerating produces different answers and any score movement would be
unattributable to either the rubric or generator noise. Judging fixed answers makes rubric
changes measurable.

`--limit` is a smoke test, not a small experiment. A `--limit 3` run once showed
`both_rate = 1.000`, which looked exactly like router collapse and was n=3 noise.

---

## Layout

```
app.py                          Streamlit chat UI (streaming + source expanders)
techdoc/
  config.py                     Bedrock clients + model tiering; load_dotenv() lives here
  ingest.py                     manifest -> files -> chunks, doc/code splitter fork
  vectorstore.py                Qdrant collection, idempotent embedding
  retrieval.py                  HybridRetriever: dense + BM25 -> RRF -> rerank
  graph.py                      LangGraph supervisor; answer() / astream_events()
  eval/
    gen_goldenset.py            bootstrap the golden set from sampled chunks
    goldenset.jsonl             38 hand-curated questions with gold chunk ids
    retrieval_metrics.py        hit@k / recall@k / MRR across the three modes
    generation_metrics.py       faithfulness, relevance, citations, router accuracy
    sweep_chunking.py           chunk-size sweep with content-re-anchored labels
    results_*.json              recorded runs
data/corpus_manifest.yaml       pinned commits + include globs (source of truth)
DECISIONS.md                    engineering log: rationale, measurements, bugs found
```

---

## Observability

With `LANGSMITH_TRACING=true`, a `both`-route request traces as its own architecture
diagram, with the latency and token cost of each node attached:

```
techdoc_answer [chain] 25.1s   3098 in / 1352 out
  ├ route        9.11s    904 tok   (Haiku)
  ├ fan_out      0.00s
  ├ search_code  2.34s              ┐ parallel
  ├ search_docs  3.42s              ┘
  └ synthesize  12.58s   3546 tok   (Sonnet 5)
```

Routing being 9s of a 25s request is the strongest argument for keeping a cheap model on
that node — visible here in a way no log line would make visible.

Runs are named so they stay filterable at volume: `techdoc_stream` (UI), `techdoc_answer`
(blocking), and `judge_faithfulness` / `judge_relevance` / `judge_citation` for the eval,
each carrying `qid`, `q_type`, `route` and — for citations — `marker` metadata. Before that,
one eval run produced 432 anonymous `RunnableSequence` roots with no path from a surprising
score back to the question that caused it.

---

## Known limitations

Honest list; per-area detail is in [DECISIONS.md](DECISIONS.md).

- **No multi-turn memory.** No checkpointer, so the graph sees only the current question —
  the UI transcript is display-only. `MemorySaver` + a `thread_id` is the fix.
- **Post-filter can under-fill, and it does.** Corpus filtering happens after retrieval
  (BM25 here is an in-memory index with no filter API), so if fewer than `k` of the `k*4`
  candidates match the requested type, the agent returns fewer than 5 chunks. One golden-set
  question got **0 sources** and correctly refused rather than hallucinating.
- **Filtered retrieval is unmeasured.** The retrieval eval measured the *unfiltered*
  retriever; nothing yet proves `retrieve_filtered` beats it on the matching half of the
  corpus, which is the experiment that would justify the doc/code fork.
- **n=38, one judge, no human-labelled subset.** Scores are self-consistent, not externally
  calibrated. Judge and generator are different models, which blunts self-preference bias
  without eliminating family bias.
- **Faithfulness is measured against retrieved context, not truth.** An answer that
  faithfully reports a wrong document scores 5. That is the correct definition of
  groundedness and a real ceiling on the metric.
- **Citation markers are deduplicated**, so an answer citing `[1]` five times is judged once
  on its first claim — ~250 Bedrock calls down to ~126, at the cost of missing a later
  mis-citation of the same source.
- **Code retrieval is the weak half** (hit@5 0.842 vs 1.000). The sweep showed it is *not*
  chunk-size-limited; the remaining failures are signature-vs-body misattribution, which
  needs a symbol-aware splitter rather than more size tuning.
- **Index build has no CLI entry point** — hence the `python -c` in step 3.
- **Tracing is unsampled and on by default.** Fine for development, wrong for production.

## Next steps

Upload the golden set as a LangSmith dataset for run-over-run comparison and attach judge
scores as feedback on their spans; swap in `MarkdownHeaderTextSplitter` for docs and a
symbol-aware splitter for code; measure filtered vs unfiltered retrieval; add a checkpointer
for follow-up questions.
