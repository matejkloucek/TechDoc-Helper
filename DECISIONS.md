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

`techdoc/eval/generation_metrics.py`. Step 6b asked "did we put the right chunk in front
of the model?". This asks "given those chunks, was the ANSWER any good?" — which needs a
judge, because there is no single correct string to compare against.

Five metrics, and **only three of them use an LLM**:

| metric | how | why |
| --- | --- | --- |
| faithfulness 1–5 | judge, sees context + answer | anti-hallucination |
| answer relevance 1–5 | judge, sees question + answer (**not** the context) | did it answer the question asked |
| citation support | judge, one (claim, source) pair at a time | does source [n] really back the sentence citing it |
| citation syntax | pure regex, **no LLM** | do the `[n]` markers point at sources that exist |
| router accuracy | comparison against golden `type`, **no LLM** | did the supervisor pick the right corpus |

Prefer a deterministic metric over a judge wherever ground truth already exists. The
golden set's `type` field is free routing ground truth, and marker-in-range is a regex —
paying Bedrock for either would be slower, noisier and no more correct.

**Run 2026-07-26, 38 questions, Sonnet 4.6 as judge:**

| slice | n | faithfulness | relevance | citation precision | % with citations | % invalid markers |
| --- | --- | --- | --- | --- | --- | --- |
| all | 38 | 4.789 | 4.632 | 0.876 | 0.974 | 0.026 |
| doc | 19 | 4.895 | 4.737 | 0.883 | 0.947 | 0.000 |
| code | 19 | 4.684 | 4.526 | 0.870 | 1.000 | 0.053 |

| routing | n | strict | lenient | both_rate |
| --- | --- | --- | --- | --- |
| router | 38 | 0.316 | 0.947 | 0.632 |

277 markers written, 126 distinct (claim, source) pairs judged, 18 unsupported →
**pooled citation precision 0.857** (doc 0.851, code 0.864). The per-question mean of
0.876 runs higher because it weights a 1-citation answer the same as a 6-citation one;
both are reported, since the pooled figure is the honest one for "how often is a citation
trustworthy" and the mean is the right one for "how good is a typical answer".

Generation: 14.6 min for 38 questions, median 14.0s, p90 19.1s, one 364s outlier
(Bedrock throttle-and-retry mid-run, not cold start — q001 was first and took 19s).

### Two stages on purpose: generate, then judge

    stage 1  generate  -- run the real graph over the golden set  -> answers_cache.json
    stage 2  judge     -- score the cached answers                 -> results_generation.json

Generation is the expensive half, so it is cached to disk and stage 2 reads the cache.
Two reasons, and both paid off within an hour of writing it:

- **Rubric iteration costs seconds instead of 15 minutes.** Fixing a claim-extraction bug
  and re-scoring was a cache read, not a regeneration.
- **It removes a confound.** Sonnet 5 ignores `temperature` (see Models), so regenerating
  produces *different answers*. A score that moved would be unattributable — rubric change,
  or generator noise? Judging fixed answers makes rubric changes measurable.

It also turned a crash into a recoverable event: when stage 2 died 38/38 of the way
through (below), all 38 answers were already on disk. Recovery cost ~11 min of judging
instead of a 26-minute full run.

### Rubric design: anchored, and deliberately non-overlapping

**Anchored levels.** Every score level is described concretely ("4: all claims supported,
but one is a mild over-generalisation"). An unanchored "rate this 1–5" drifts between
runs and clusters everything on 4. It held: faithfulness spread over {3,4,5} and
relevance over {1,3,4,5} rather than piling on one value.

**The relevance judge does not get the context.** It sees only `{question}` and
`{answer}`. If it saw the context it would start reasoning "the answer does reflect what
was retrieved, so it's fine" — which is groundedness, which faithfulness already measures.
Two rubrics that partly measure the same thing produce correlated scores, and a correlated
second metric costs money without adding information.

**The independence test.** A well-formed *"the docs don't cover this"* must score
**faithfulness 5, relevance 1** — it fabricates nothing, and it answers nothing. Both
prompts say so explicitly. Three questions hit it on the real run:

| qid | sources retrieved | faithfulness | relevance | what it means |
| --- | --- | --- | --- | --- |
| q006 | 0 | 5 | 1 | misrouted → post-filter left nothing → refusal |
| q024 | 10 | 5 | 1 | retrieval returned material, but not the right material |
| q025 | 5 | 5 | 3 | partial answer, honest about the gap |

The judge's own wording on q006: *"an explicit refusal making no factual claims, so there
is nothing to contradict"* against *"provides no answer to the question."* That divergence
is the diagnostic — **high faithfulness + low relevance means retrieval failed and the
generator behaved correctly.** A single blended "quality" score would have averaged all
three into a bland 3 and hidden the cause. q024 is the sharpest case: 10 sources retrieved
and still nothing to say, which no source-count metric would have flagged.

**Citation support is a boolean, not a 1–5.** "Does this source contain this fact?" has a
right answer, so a scale would only invite the judge to park at 3 on hard cases. Partial
support counts as *unsupported*, and merely being on-topic is not support.

### What the citation metric caught

The failures are **over-citation, not hallucination** — the model writes a correct sentence
and staples several markers to it, only some of which back the full claim. Faithfulness
scored 5 on most of these same answers: the facts are all in the context *somewhere*, the
*attribution* is wrong. Neither retrieval eval nor faithfulness alone could see this.

**q032 scored 0.0 — every citation in it unsupported, the single most diagnostic row**
("What file formats does
`_load_prompt_from_file` support?"):
- `[2]` — "the source shows three formats (JSON, YAML, YML), not two as the claim asserts"
- `[4]` — cited for file-format behaviour; the source is just the `load_prompt` signature
- `[1]` — claim says jinja2 is rejected for "security"; source says arbitrary code execution

**q027 wrote `[0]`** — a marker for a source that cannot exist, since `format_context`
numbers from 1. Exactly the fabricated-handle case the regex check exists for, and it is
real rather than hypothetical (1/38 = 2.6%, the only invalid marker in the run).

**An answer with no citations scores precision 1.0** — vacuously perfect. That is why
`pct_with_citations` is reported next to it; precision alone would rate a citation-free
answer as flawless. q006's refusal is the only such row.

### The router hedges rather than errs

strict 0.316 looks alarming and lenient 0.947 looks great; the truth is in `both_rate`.

    confusion (golden type -> chosen route)
      doc->both   14      code->both   10
      doc->docs    3      code->code    9
      doc->code    2

Only **2 of 38 are true misses**, both `doc->code`. The other 24 "strict failures" are
`both` — the correct corpus *was* searched, alongside a second one. So the router is
cautious, not wrong, and the cost is double retrieval on 63% of questions.

Reporting both numbers is the point, and this run demonstrates why rather than asserting
it: **strict alone under-credits a router whose only sin is caution; lenient alone would
hide a router that always says "both" and has learned nothing.** With `both_rate = 0.632`
the distinction is load-bearing. (An earlier `--limit 3` smoke run showed `both` on all
three and `both_rate = 1.000`, which looked exactly like collapse — n=3 noise, and a
reminder not to conclude from a smoke test.)

The two real misses are instructive in opposite directions:
- **q006** "…precedence order for resolving general configuration options like interpreter
  limits or themes?" — labelled `doc`, but reads as implementation vocabulary. The router's
  choice is defensible; the *consequence* is not, and it compounds: routed to `code`, the
  post-filter kept only `type == "code"` chunks, of which the top-20 had none → **0 sources
  → refusal**. The routing error and the retrieval failure are the same event.
- **q033** "What attributes does `NodeTimeoutError` expose…?" — labelled `doc`, but it names
  a class and asks for its attributes. `code` is arguably the better answer. Charged to the
  router in the table, but it is more likely a **golden-set labelling artifact**.

`TYPE_TO_ROUTE = {"doc": "docs", "code": "code"}` exists because the golden set says `doc`
and the router emits `docs`. Without it every doc question scores as a routing miss and the
router looks broken — a silent vocabulary mismatch, pre-empted rather than discovered.

### Bugs found (both silent-adjacent, both mine)

**1. `with_structured_output` does not guarantee schema conformance.** Bedrock Converse
returned `unsupported_claims` as the *string* `'["claim a", "claim b"]'` instead of a list,
and Pydantic correctly rejected it. Fixed with a `@field_validator(mode="before")` that
coerces — deliberately *not* by loosening the annotation to `list[str] | str`, so the
schema shown to the model stays a proper array and every downstream consumer still gets a
list. The lesson: structured output constrains the model, it does not *guarantee* the
provider's serialisation.

**2. One bad response destroyed all 38 judgements.** The judging loop had no per-row
`try/except`, so that single `ValidationError` propagated out of `main()` after ~20 minutes
of Bedrock calls, and the report was never printed. `generate_answers` already had the
guard; the judging loop did not. Fixed, and `aggregate` counts error rows in `n_errors` so
a partial run reports as partial instead of silently averaging over 37 of 38.

The second one is the more interesting failure. A harness that *crashes* is loud. The
version of this bug that keeps going and quietly reports on 37 questions while claiming 38
is silent — and that is the shape of every other bug in this project. Excluding error rows
from the means *and* printing the count is what makes it visible either way.

Third, minor but wasteful: `print()` without `flush=True` is block-buffered when stdout is
a pipe, so a 15-minute run showed zero progress until it exited. Both loops now flush.

**A claim-extraction bug the metric found in itself.** `_claim_for_marker` originally ran
to the end of the *line*, so on `"…saves state [1]. Then more prose [2]."` the claim for
`[1]` swallowed the sentence belonging to `[2]`. Combined with "partial support is not
support", that depressed citation precision on every multi-sentence line (0.667 → 0.750 on
the smoke set once fixed). It now stops at the sentence end *or* newline, whichever comes
first — and when a marker sits alone on a line (the generator does this after a code fence,
`"```\n[3][4]"`) it widens to the preceding paragraph, because a claim of literally `"[3][4]"`
contains no assertion and the judge rightly returned `supported=False` for it. **A metric
can fail by measuring its own preprocessing.**

### Known limitations

- **n=38 with one judge.** No inter-rater agreement, no human-labelled subset to validate
  the judge against. The scores are self-consistent, not externally calibrated.
- **Deduplicating markers loses per-instance information.** An answer citing `[1]` five
  times is judged once, using the first claim. Cuts ~250 Bedrock calls to ~126, but a
  later mis-citation of the same source is missed.
- **Judge and generator are different models** (Sonnet 4.6 vs Sonnet 5), which avoids the
  worst of self-preference bias but does not eliminate family bias.
- **Faithfulness is measured against retrieved context, not against truth.** An answer
  that faithfully reports a wrong document scores 5. That is the correct definition of
  groundedness and a real ceiling on what this metric can tell you.

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
