# Decisions & Eval Log

Engineering decisions, their rationale, and measured results. Written as I go so
the numbers behind each choice are reproducible rather than remembered.

---

## Corpus & ingestion

- **Pinned commits, third-party source not committed.** `data/corpus_manifest.yaml`
  lists three repos (docs, langgraph, langchain) with pinned SHAs and include globs.
  The corpus is reproducible from a clone without vendoring ~400 files into git.
- **`type: doc | code` drives a chunking fork.** `RecursiveCharacterTextSplitter.from_language(MARKDOWN)`
  vs `(PYTHON)` — splitting Python on prose separators cuts through function bodies.
  `chunk_size=1000, chunk_overlap=150` are first-guess values, to be tuned against
  the retrieval metrics below.
- **Deterministic `chunk_id` = `repo:rel_path:index`**, and point id = `uuid5(chunk_id)`.
  Makes re-ingestion an idempotent upsert instead of a duplicate insert, and makes
  golden-set labels stable across rebuilds.
- Result: **9997 chunks** (5119 doc / 4878 code) from 432 files.

## Retrieval

- **Hand-rolled hybrid rather than Qdrant native hybrid**: dense top-20 + BM25 top-20
  → Reciprocal Rank Fusion (k=60) → Cohere rerank-v3.5 cross-encoder → top-k.
  RRF is scale-invariant, so no score normalisation between two incomparable scoring
  systems; keying the fusion on `chunk_id` means a doc found by *both* retrievers gets
  both reciprocal terms summed — a consensus boost.
- **Why hybrid at all (anecdote, quantified below):** dense-only for
  `add_conditional_edges` returned the right chunk *but also* `add_edge` — semantic
  bleed between similar API names. BM25 matches the exact symbol; rerank demoted the
  wrong methods to ~0.5.
- Gotcha: `BedrockRerank` defaults to `top_n=3` and **silently** truncates. Set to
  `CANDIDATE_N` so the final `[:k]` slice is the real limiter.

## Models

- `get_llm()` → `us.anthropic.claude-sonnet-5` (generation).
- `get_judge_llm()` → `us.anthropic.claude-sonnet-4-6` (LLM-as-judge).
  **Why the split:** Sonnet 5 no longer supports `temperature`; `langchain-aws`
  consults a model profile and silently drops the param with a `UserWarning`. Sonnet 4.6
  still honours `temperature=0`, which the judge needs for reproducible scores.
  Retrieval metrics are unaffected either way — they make zero LLM calls.

---

## Eval: golden set (Step 6a)

**Method.** Bootstrapped synthetically, then hand-curated. `techdoc/eval/gen_goldenset.py`
samples 40 chunks (seed 42, balanced 20 doc / 20 code, filtered by `is_useful()` to skip
import-only and boilerplate chunks), then asks Sonnet 5 to write a realistic developer
question answerable *solely* from that chunk. The sampled chunk's id becomes the gold label.

Synthetic generation gives coverage cheaply; the curation pass is what makes it a
*benchmark* rather than a restatement of the generator's biases.

**Curation pass: 39 drafted → 38 final.**

| Action | n | Reason |
| --- | --- | --- |
| Kept as-is | 29 | Chunk provably answers the question |
| Dropped | 7 | 5× JS/TS content, 2× chunk truncated before the answer |
| Question rewritten | 3 | Chunk was good but the question over-claimed |
| Hand-authored added | 6 | Restore doc/code balance after the drops |

Final: **38 rows, 19 doc / 19 code, 37 distinct files**, all gold ids verified to exist
in the corpus, no duplicate questions or gold ids.

Dropped rows and why:

| Gold chunk | Reason |
| --- | --- |
| `docs:...langgraph/graph-api.mdx:101` | LangGraph.js/TypeScript fragment |
| `docs:...middleware/built-in.mdx:18` | JS API surface (camelCase params) |
| `docs:...langchain/runtime.mdx:6` | LangChain.js/TypeScript |
| `docs:...handoffs-customer-support.mdx:63` | TypeScript `tool()` code |
| `docs:...langchain/streaming.mdx:55` | Raw stdout dump, no prose — question unanswerable from it |
| `docs:...integrations/copilotkit.mdx:31` | React/TSX frontend, out of scope |
| `langgraph:...pregel/_utils.py:2` | 157-char chunk truncates before the function body |

Rewritten (question narrowed to what the chunk actually supports):

- `utils/usage.py:1` — dropped "what happens when max_depth is exceeded"; chunk ends before the body.
- `tracers/base.py:35` — dropped tool *start*; only `_on_tool_end`/`_on_tool_error` are in the chunk.
- `pregel/remote.py:38` — asked about the REST endpoints (which the docstring states) instead of
  what `subgraphs`/`headers` do (which it doesn't).

**Known limitations, deliberately not fixed:**

1. **JS leaks into the doc corpus at chunk level.** The manifest excludes
   `*-javascript.mdx` *files*, but these MDX pages interleave `:::python` and `:::js`
   blocks **inside one file**, so a char-based splitter can emit a JS-only chunk. Fixing
   this properly means stripping `:::js` regions during ingestion and re-embedding
   (~13 min). Recorded as future work; for now curation caught them in the labels.
   Note the corpus itself still contains those chunks — they are legitimate distractors
   for retrieval, which is arguably realistic.
2. **Fragment chunks.** Char-based splitting can start a chunk mid-structure (a dangling
   ` ``` ` or `:::`). Kept in the golden set where the answer is still fully present — a
   retriever has to cope with real chunk boundaries. `MarkdownHeaderTextSplitter` would
   reduce these.
3. **Single gold chunk per question.** With `chunk_overlap=150` an adjacent chunk can
   sometimes answer a question too, so recall@k is a slight *under*-estimate. `gold_chunk_ids`
   is a list precisely so this can be relaxed later.
4. **Generator and judge are both Claude.** Self-preference bias is possible; the human
   curation pass is the mitigation.

---

## Eval: retrieval metrics (Step 6b)

`techdoc/eval/retrieval_metrics.py`, 38-question golden set, zero LLM calls (so it is
deterministic and re-runnable in ~1 min). The three configs are literally prefixes of
one `HybridRetriever.retrieve(..., mode=)` code path, so the eval measures the real
pipeline rather than a reimplementation of it.

**Run 2026-07-25 (final):**

| config | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| dense only | 0.711 | 0.737 | 0.789 | 0.816 | 0.736 |
| hybrid (RRF) | 0.684 | 0.816 | 0.895 | 0.895 | 0.767 |
| **hybrid + rerank** | **0.763** | **0.921** | **0.921** | **0.921** | **0.842** |

recall@k is omitted: with a single gold chunk per question it is identical to hit@k
by definition. Both are computed and stored in `results_retrieval.json`.

Reading the table:
- **Rerank earns its latency in MRR** (0.767 → 0.842) and hit@3 (0.816 → 0.921) more
  than in hit@10, which is exactly what a cross-encoder is for: it cannot add new
  candidates, only reorder the ones fusion supplied. hit@5 == hit@10 for rerank
  because reranking a fixed candidate pool cannot improve deep recall.
- Rerank costs ~0.55s/query vs ~0.1s for hybrid (32s vs 15s for 38 queries).

### The finding that mattered: BM25's default tokenizer

**First run had hybrid WORSE than dense** — hit@1 0.711 → 0.474, MRR 0.736 → 0.597.
The per-query dump showed the signature: most regressions were `dense#1 → hybrid#2`,
i.e. fusion was *demoting correct top hits by one position*, not retrieving worse
documents. Diagnosis path:

1. Swept `RRF_K` from 1 to 120 — barely moved (MRR 0.674 at best, still under dense).
   So the fusion constant was not the problem.
2. Swept a dense:bm25 weight — monotonically better the *more* BM25 was suppressed,
   converging on dense-only. So BM25 was pure noise.
3. Measured BM25 in isolation: **recall@20 = 0.474**, and it found **zero** queries
   that dense missed entirely. A retriever contributing no unique recall can only
   dilute under RRF.
4. Read `BM25Retriever`'s default `preprocess_func`: it is **`str.split()`**.

`str.split()` keeps punctuation attached and does not case-fold, so on a Python
corpus the query token `_dict_int_op` cannot match the source token `_dict_int_op(`,
and `RunInfo` cannot match `runinfo`. BM25's whole reason for existing here — exact
symbol matching — was silently broken.

Fix: `code_tokenize()` in `techdoc/retrieval.py`, splitting on non-identifier chars
(`[A-Za-z_][A-Za-z0-9_]*|\d+`, lowercased, underscores kept so `snake_case` stays whole).

| BM25 tokenizer | BM25 recall@20 | fused hit@1 | fused hit@5 | fused MRR | fused MRR (code) |
| --- | --- | --- | --- | --- | --- |
| `str.split()` (default) | 0.474 | 0.474 | 0.737 | 0.600 | 0.504 |
| `code_tokenize()` | **0.921** | 0.684 | 0.895 | 0.767 | 0.643 |

Two lessons worth stating plainly:
- **The library default was wrong for this corpus and failed silently.** Nothing threw;
  retrieval just got worse. Only an eval caught it.
- **The aggregate number said "hybrid is bad"; the per-query dump said "hybrid is
  misconfigured."** Keeping `per_query` with `gold_rank` and `retrieved_ids` is what
  made the `dense#1 → hybrid#2` pattern visible. Aggregates tell you *that* something
  moved; per-query rows tell you *which* and *how*.

### Where retrieval still fails

Split by corpus type (rerank mode): **doc hit@5 = 1.000, code hit@5 = 0.684.**
All remaining failures are code questions. Two causes, both known and unfixed:

- **Chunk-boundary misattribution.** The gold chunk holds a function *signature* while
  the neighbouring chunk holds the body the question actually describes — retrieval
  returns the neighbour, scored as a miss. Partly a labelling artifact of picking a
  single gold chunk over a 150-char-overlap split.
- **Docstring-poor code.** Questions about symbols whose chunk is mostly code with
  little prose have weak dense signal; BM25 now carries these, which is precisely the
  division of labour hybrid is supposed to provide.

Both point at the same next experiment: tune `chunk_size` / `chunk_overlap` (done below),
or move docs to `MarkdownHeaderTextSplitter` and code to a symbol-aware splitter.

### Chunk-size sweep (`techdoc/eval/sweep_chunking.py`)

**Result: keep `chunk_size=1000, chunk_overlap=150`.** The first-guess value won.

| size/overlap | n_chunks | hit@1 | hit@3 | hit@5 | MRR | MRR doc | MRR code |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 500/75 | 19670 | 0.639 | 0.722 | 0.806 | 0.695 | 0.731 | 0.663 |
| **1000/150** | **9997** | **0.684** | **0.816** | **0.868** | **0.768** | **0.895** | 0.641 |
| 1500/225 | 6718 | 0.605 | 0.763 | 0.868 | 0.708 | 0.783 | 0.633 |

Measured on the hybrid stage (dense + RRF), no rerank: reranking is fixed
post-processing that cannot change which candidates chunking makes available, and
skipping it saves a Bedrock call per query per config.

**The methodological trap this exposed — and why the sweep needed its own harness.**
`chunk_id` is `repo:rel_path:index`, so changing `chunk_size` renumbers every chunk in
a file. Reusing the golden set's ids across configs looks fine and is completely wrong:
at `chunk_size=500`, **33/38 gold ids still resolve to a real chunk, but only 1/38 point
at the text that was actually labelled.** Nothing errors; the numbers are just garbage.
So `sweep_chunking.py` re-anchors every label by CONTENT (find the chunk in the same
file containing the first 120 chars of the labelled text) and verifies the resolved
chunk really contains the label: 38/38 re-anchored at 1000 and 1500, 36/38 at 500
(two labels straddle a boundary at that size and are excluded from scoring rather than
silently counted as misses — hence `n_scored=36` for that row).

This is the same class of bug as the BM25 tokenizer: **a silent one that produces
plausible numbers.** Content-anchored labels are the defence, and it is the reason
`gold_chunk_ids` being a list is useful — 2–3 chunks legitimately contain the label
once overlap is in play.

**Is the code-question difference at 500/75 real?** No. A paired per-query comparison
against 1000/150 (same questions, same seeds) gives:

| config | mean paired ΔMRR | doc Δ | code Δ | code better/worse |
| --- | --- | --- | --- | --- |
| 500/75 | −0.060 | −0.151 | +0.022 | 3 / 3 |
| 1500/225 | −0.060 | −0.112 | −0.008 | 3 / 5 |

The apparent code gain at 500/75 is 3 questions better and 3 worse — noise on n=19,
not a signal. Both alternatives lose clearly on docs, and there is no config where
code improves without docs regressing more.

**Why 1000 wins, mechanistically:** at 500 chars a prose explanation gets split from
the code sample it explains, so neither half is independently answerable (docs MRR
0.895 → 0.731). At 1500 each chunk covers more unrelated material, so the embedding
averages over several topics and matches queries less sharply — the classic
"washed out" failure that more context is supposed to fix but does not.

**Conclusion:** code retrieval is not chunk-size-limited. The remaining code failures
are signature-vs-body misattribution, which a *symbol-aware* splitter would fix and a
*character-count* splitter cannot, at any size. Logged as the real next experiment
instead of further size tuning.

Sweep collections (`techdoc_s{size}_o{overlap}`, ~414 MB) were deleted after recording;
`sweep_chunking.py` recreates them on demand.

## Eval: generation (Step 6c)

_Pending — faithfulness, answer relevance, citation correctness via `get_judge_llm()`._

---

## Agent graph (Step 7)

`techdoc/graph.py`. A LangGraph supervisor that routes each question to a docs agent,
a code agent, or both in parallel, then synthesises one cited answer.

```
START -> route ->  ┌── search_docs ──┐
                   ├── search_code ──┤ -> synthesize -> END
                   └── (both, parallel)
```

**Why a graph and not one prompt.** Step 6b measured the gap this exists to close:
**doc questions hit@5 = 1.000, code questions 0.684.** The two corpora fail differently,
so they get different candidate pools. The supervisor shape also buys two things a single
prompt cannot: cost tiering, and a LangSmith trace where every step is a named span.

### Model tiering

| node | model | why |
| --- | --- | --- |
| `route` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 3-way classification on one short question, runs on **every** query |
| `synthesize` | `us.anthropic.claude-sonnet-5` | the only step whose output the user reads |

Routing is where tiering pays, because it is the step that always runs and the step
that needs the least capability. Gotcha: the Haiku id needs **both** the `us.`
inference-profile prefix and the `-v1:0` suffix. Without the suffix Bedrock raises
`ValidationException: The provided model identifier is invalid` — the ARN is only
discoverable via `aws bedrock list-inference-profiles`.

### Design decisions

- **`Annotated[list[Document], operator.add]` on `docs` is the single load-bearing line.**
  On `route == "both"` two nodes write `docs` in the same superstep. Without a reducer
  LangGraph raises `InvalidUpdateError` on the concurrent write; with `operator.add` the
  two lists concatenate. Verified: the `both` route returns 7 sources (5 doc + 5 code,
  post-filtered), not 5.
- **Conditional edge returning a *list* of node names**, not `Send()`. Returning
  `["search_docs", "search_code"]` fans out to run both in one superstep. `Send()` is the
  alternative and is the right tool when each branch needs a *different payload*; here
  both branches read the same `question` off shared state, so the list form is simpler.
- **Both agents edge into `synthesize`, which still runs exactly once.** LangGraph waits
  for all active branches to finish before the next superstep (barrier), so `synthesize`
  sees the merged list rather than firing per branch. Confirmed empirically.
- **Structured routing via `with_structured_output(RouteDecision)`** — a Pydantic model
  with `route: Literal["docs","code","both"]` plus a `reason` field. No free-text parsing,
  and the field `description`s are the actual routing instructions the model sees. The
  `reason` field is not consumed by the graph; it exists to make the trace readable and
  to give the model a place to think.
- **Post-filter, not pre-filter, for corpus type.** `retrieve_filtered()` fetches `k*4`
  hybrid candidates and keeps those whose `metadata["type"]` matches. A Qdrant payload
  pre-filter would be strictly better, but BM25 here is an in-memory index with no filter
  API, so pre-filtering would mean maintaining two BM25 indexes. Documented tradeoff, not
  an oversight — see limitations below.
- **`answer()` / `astream_answer()` are the public contract.** The Streamlit UI (Step 10)
  depends only on `AnswerResult(text, sources, route)` and an async iterator of strings,
  so graph internals stay free to change.

### Streaming

`stream_mode="messages"` yields `(chunk, metadata)` for **every LLM token inside the
graph**, so the `metadata["langgraph_node"] == "synthesize"` filter is load-bearing, not
defensive. Measured on one `both` query:

| node | message chunks emitted |
| --- | --- |
| `route` | 35 |
| `synthesize` | 230 |

Without the filter the router's structured-output JSON streams into the user's answer.
The `and chunk.text` guard is also required — Bedrock emits empty content-block chunks.
Measured `both` route: 227 tokens, **TTFT ~9s**, 19.3s total; inline `[n]` citations
survive streaming intact.

### Bugs found, all three silent

Same pattern as the BM25 tokenizer and the chunk-id label rot: nothing threw where the
damage happened.

1. **`return {"code": hits}` from `search_code`.** There is no `code` key in `GraphState`,
   so LangGraph **silently discarded every code result** — no error, no warning, just a
   graph that answered code questions from docs. TypedDict is a type-checker hint; it does
   not validate at runtime. Same reason `GraphState(route="")` constructs happily despite
   `route` being a `Literal`.
2. **The retriever singleton was not thread-safe.** Guarded with try/except-`NameError`,
   so on the `both` route both branches could enter the constructor before either finished,
   and the loser hit
   `RuntimeError: Storage folder qdrant_data is already accessed by another instance of
   Qdrant client` (Qdrant local mode holds an exclusive file lock). The sync path never
   crashed **by luck** — `.invoke()` happened to schedule both branches onto the same pool
   worker (`ThreadPoolExecutor-1_0`), serialising them; `.astream()` does not. Fixed with
   double-checked locking under a `threading.Lock`: 8 concurrent callers against a
   deliberately-slow constructor now give 1 construction, 1 instance.
3. **`response.text()` vs `response.text`.** In langchain-core 1.4.9 `.text` is a
   **property**; calling it emits `LangChainDeprecationWarning`. Only visible because the
   streaming test surfaced the warning.

### Non-determinism is expected here

The same question streamed 73–227 tokens across runs. Retrieval was verified byte-identical
over three runs, so the variance is generation: **Sonnet 5 does not support `temperature`**
and `langchain-aws` silently drops the param. This is why `get_judge_llm()` is Sonnet 4.6 —
Step 6c needs reproducible scores. Not a bug, but it means generation eval must average
over runs or accept score jitter.

### Known limitations

- **No checkpointer, so no multi-turn memory.** `answer()` is single-shot. Adding
  `MemorySaver` + a `thread_id` is a small change; deferred until the UI needs it.
- **Post-filter can under-fill.** If fewer than `k` of the `k*4` candidates match the
  requested type, the agent returns fewer than 5 chunks (worst case zero, which
  `synthesize` handles with an explicit "not found" answer rather than hallucinating).
  Not yet observed on the golden set, but it is the failure mode to watch.
- **Router accuracy is unmeasured.** Cheap to fix and worth doing: the 38-row golden set
  already carries a `type: doc | code` label per question, which is a free routing
  ground-truth for the two-way cases. Folded into Step 6c.
- **Filtered retrieval is unmeasured.** Step 6b measured unfiltered `HybridRetriever`;
  the graph measures nothing about whether `retrieve_filtered` beats it on the matching
  half of the corpus. That is the experiment that would justify the whole fork.
