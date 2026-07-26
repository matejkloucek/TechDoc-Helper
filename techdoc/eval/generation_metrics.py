"""Generation eval: faithfulness, answer relevance, citation correctness, routing.

Retrieval eval asked "did we put the right chunk in front of the model?".
This asks "given those chunks, was the ANSWER any good?" -- which needs a judge,
because there is no single correct string to compare against.

TWO STAGES, ON PURPOSE
----------------------
    stage 1  generate  -- run the real graph over the golden set
    stage 2  judge     -- score the cached answers

Generation is the expensive half, so its output is cached to ANSWERS_PATH. Every
rubric tweak then costs seconds instead of a full regeneration. It also removes a
confound: Sonnet 5 ignores `temperature`, so regenerating gives DIFFERENT answers,
and a score that moved would be unattributable -- rubric change, or generator
noise? Judging fixed answers makes rubric changes measurable.

WHAT IS MEASURED
----------------
1. faithfulness      -- is every claim supported by the retrieved context? (anti-hallucination)
2. answer relevance  -- does it actually answer the question asked?
3. citation syntax   -- do the [n] markers point at sources that exist?   (NO LLM -- free)
4. citation support  -- does source [n] really back the sentence citing it? (LLM)
5. router accuracy   -- did the supervisor pick the right corpus?          (NO LLM -- free)

(3) and (5) are deterministic checks that need no judge at all. Prefer a cheap
deterministic metric over an LLM one wherever the ground truth already exists --
the golden set's `type` field is free routing ground truth.

Usage:
    PYTHONPATH=. uv run python -m techdoc.eval.generation_metrics            # full run
    PYTHONPATH=. uv run python -m techdoc.eval.generation_metrics --limit 5  # smoke test
    PYTHONPATH=. uv run python -m techdoc.eval.generation_metrics --regenerate
"""
import json
import re
import sys
import time

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

from techdoc.config import get_judge_llm
from techdoc.eval.retrieval_metrics import load_golden
from techdoc.graph import answer, format_context

ANSWERS_PATH = "techdoc/eval/answers_cache.json"
RESULTS_PATH = "techdoc/eval/results_generation.json"

# The golden set labels questions "doc" / "code"; the router emits "docs" / "code"
# / "both". Without this mapping every doc question scores as a routing miss --
# a silent off-by-one-vocabulary bug that would make the router look broken.
TYPE_TO_ROUTE = {"doc": "docs", "code": "code"}


# -- judge output schemas --
# Structured output, not free text: the score must be a parseable number, and
# `with_structured_output` makes the model retry on a schema violation instead of
# handing us "I'd rate this a solid 4/5!" to regex.

class FaithfulnessScore(BaseModel):
    """Is the answer grounded in the retrieved context?"""
    score: int = Field(
        ge=1, le=5,
        description=(
            "1 = most claims unsupported or contradicted by the context. "
            "3 = mostly grounded but contains at least one unsupported claim. "
            "5 = every claim is directly supported by the context."
        ),
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Verbatim claims from the answer that the context does not support. Empty if none.",
    )
    reason: str = Field(description="One or two sentences justifying the score.")

    @field_validator("unsupported_claims", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        """Accept a stringified array for the list field.

        `with_structured_output` does NOT guarantee schema conformance -- observed on
        a real run: Bedrock Converse returned unsupported_claims as the STRING
        '["claim a", "claim b"]' rather than a list, and Pydantic rejected it. One
        such response killed all 38 judgements. Coerce here rather than loosen the
        annotation to `list[str] | str`, so every downstream consumer still sees a
        list and the schema shown to the model stays a proper array.
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError:
                return [v] if v.strip() else []
            return parsed if isinstance(parsed, list) else [str(parsed)]
        return v


class RelevanceScore(BaseModel):
    """Does the answer address the question that was asked?"""
    score: int = Field(
        ge=1, le=5,
        description=(
            "1 = does not address the question. 3 = partially answers it or buries "
            "the answer in irrelevant material. 5 = directly and completely answers it."
        ),
    )
    reason: str = Field(description="One or two sentences justifying the score.")


class CitationSupport(BaseModel):
    """Does one specific cited source actually back the claim citing it?"""
    supported: bool = Field(description="True if the cited source substantiates the claim.")
    reason: str = Field(description="One short sentence.")


# -- prompts --
# Rubrics are ANCHORED: each score level is described concretely. An unanchored
# "rate 1-5" prompt drifts between runs and clusters everything on 4.

FAITHFULNESS_PROMPT = """You are evaluating a technical documentation assistant for \
hallucination. You are given the CONTEXT it retrieved and the ANSWER it produced.

Judge ONLY groundedness: is every factual claim in the answer supported by the context?
Do NOT reward or penalise style, completeness, or whether the answer is helpful.

Scoring:
- 5: every claim is directly supported by the context.
- 4: all claims supported, but one is a mild over-generalisation.
- 3: mostly grounded, but at least one claim is not in the context.
- 2: several unsupported claims, or an invented API name/parameter.
- 1: the answer contradicts the context, or is largely fabricated.

Treat these as NOT hallucinations:
- an explicit refusal ("the context does not cover this")
- generic connective prose that makes no factual claim
- correct restatement in different words

List any unsupported claims verbatim.

CONTEXT:
{context}

ANSWER:
{answer}"""

# Deliberately NO context placeholder. Relevance asks "did it answer the question
# asked?", which is decidable from the question and the answer alone -- and keeping
# the context out is what stops this rubric from quietly re-measuring groundedness.
RELEVANCE_PROMPT = """You are evaluating a technical documentation assistant for \
answer relevance. You are given the QUESTION a developer asked and the ANSWER the \
assistant produced.

Judge ONLY whether the answer addresses the question that was asked. Groundedness is
NOT your concern -- a separate judge scores that. Assume every factual claim is true
and score only how well the answer serves the question. Do not reward or penalise
citations, formatting, or length except where they change what the question gets.

Scoring:
- 5: directly and completely answers the question asked.
- 4: answers the question, but adds material the question did not ask for.
- 3: partially answers it -- addresses one part of a two-part question, or buries the
  answer inside mostly irrelevant material.
- 2: on-topic but does not answer the question -- discusses a neighbouring API, or
  answers a question the developer did not ask.
- 1: does not address the question at all, or states that the answer is unavailable.

A well-formed refusal ("the context does not cover this") scores 1: nothing was
answered. That is a RETRIEVAL failure surfacing here, not a hallucination -- the
faithfulness judge will score the same refusal 5, and that disagreement is the
signal, not a contradiction.

QUESTION:
{question}

ANSWER:
{answer}"""

# One (claim, source) pair per call, and a BOOLEAN rather than a 1-5 scale: "does
# this source back this sentence?" has a right answer, so a scale would only invite
# the judge to hedge at 3.
CITATION_PROMPT = """You are checking one citation in a technical documentation \
assistant's answer. The assistant wrote the CLAIM below and cited source [{n}] for \
it. You are given the full text of that source.

Decide one thing: does SOURCE [{n}] substantiate the CLAIM?

Rules:
- Partial support is NOT support. If the source backs part of the claim but not all
  of it, answer supported = false.
- Being on the same topic is NOT support. The source must contain the specific fact,
  signature, parameter, or behaviour the claim asserts.
- A correct restatement in different words IS support -- do not require matching
  wording.
- Judge only this pair. Do not consider whether some other source might support the
  claim, and do not penalise the claim for anything outside it.

CLAIM:
{claim}

SOURCE [{n}]:
{source}"""


# -- stage 1: generate --

def _serialize_sources(sources: list[Document]) -> list[dict]:
    """Keep only what the judge needs, so the cache stays readable and diffable."""
    return [
        {
            "chunk_id": d.metadata["chunk_id"],
            "type": d.metadata["type"],
            "rel_path": d.metadata["rel_path"],
            "page_content": d.page_content,
        }
        for d in sources
    ]


def _deserialize_sources(raw: list[dict]) -> list[Document]:
    """Rebuild Documents from the cache so format_context() can render them.

    Reusing the graph's own format_context() matters: the judge must see EXACTLY
    the numbered context the generator saw, or citation numbers will not line up.
    """
    return [
        Document(
            page_content=r["page_content"],
            metadata={k: v for k, v in r.items() if k != "page_content"},
        )
        for r in raw
    ]


def generate_answers(rows: list[dict]) -> list[dict]:
    """Run the real graph over every golden question and record what came back.

    Deliberately calls techdoc.graph.answer() rather than reimplementing the
    pipeline, so this eval measures the system the UI will actually ship.

    Returns one record per row:
        {"qid", "question", "type", "answer", "route", "sources": [ ... ]}

    Print progress: this is a ~13-minute loop and a silent terminal is
    indistinguishable from a hang.
    """
    records = []
    for i, row in enumerate(rows, start=1):
        t0 = time.time()
        try:
            result = answer(row["question"])
            record = {
                "qid": row["qid"],
                "question": row["question"],
                "type": row["type"],
                "answer": result.text,
                "route": result.route,
                "sources": _serialize_sources(result.sources),
                "elapsed_s": round(time.time() - t0, 1),
            }
        except Exception as e:
            record = {
                "qid": row["qid"],
                "question": row["question"],
                "type": row["type"],
                "error": str(e),
                "elapsed_s": round(time.time() - t0, 1),
            }
        records.append(record)
        # flush=True: stdout is block-buffered when piped to a file, so without this
        # a 15-minute run shows no progress at all until it exits.
        print(f"{i}/{len(rows)} route={record.get('route', 'ERROR')} "
              f"n_sources={len(record.get('sources', []))} "
              f"elapsed={record['elapsed_s']}s", flush=True)
    return records


def load_or_generate(rows: list[dict], regenerate: bool = False) -> list[dict]:
    """Return cached answers if present and complete, else generate and cache them.

    The cache is keyed by qid, so a --limit 5 smoke run followed by a full run
    regenerates rather than silently scoring 5 questions and reporting 38.
    """
    if not regenerate:
        try:
            with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except FileNotFoundError:
            cached = None
        if cached and {c["qid"] for c in cached} >= {r["qid"] for r in rows}:
            wanted = {r["qid"] for r in rows}
            print(f"using cached answers from {ANSWERS_PATH}")
            return [c for c in cached if c["qid"] in wanted]

    records = generate_answers(rows)
    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {ANSWERS_PATH}")
    return records


# -- stage 2a: deterministic checks (no LLM) --

CITATION_RE = re.compile(r"\[(\d+)\]")


def check_citation_syntax(text: str, n_sources: int) -> dict:
    """Are the [n] markers well-formed and in range? Pure string work, no LLM.

    Catches the two cheap failure modes before spending judge tokens:
      - the model cited [7] when only 5 sources were supplied (fabricated handle)
      - the model cited nothing at all despite being told to

    Returns {"markers": [ints in order of appearance], "n_markers": int,
             "invalid": [out-of-range ints], "has_citations": bool}
    """

    citation_markers = [int(m) for m in CITATION_RE.findall(text)]
    invalid_markers = [m for m in citation_markers if m < 1 or m > n_sources]
    return {
        "markers": citation_markers,
        "n_markers": len(citation_markers),
        "invalid": invalid_markers,
        "has_citations": len(citation_markers) > 0,
    }


def router_accuracy(records: list[dict]) -> dict:
    """Did the supervisor route to the right corpus? Free ground truth, no LLM.

    Every golden question was written from ONE chunk of a known `type`, so that
    type is the corpus the question is answerable from. Two scores, because "both"
    is not simply wrong:
      strict  -- route == the expected corpus exactly
      lenient -- route == expected, OR route == "both" (correct corpus was searched,
                 just alongside a second one: right answer, wasted tokens)

    Report both. Strict alone under-credits a router whose only sin is caution;
    lenient alone hides a router that always says "both" and has learned nothing.

    Returns {"n": int, "strict": float, "lenient": float, "both_rate": float,
             "confusion": {"doc->docs": n, "doc->code": n, ...}}
    """

    records = [r for r in records if "error" not in r]
    n = len(records)
    strict_hits = sum(1 for r in records if r["route"] == TYPE_TO_ROUTE[r["type"]])
    lenient_hits = sum(1 for r in records if r["route"] == TYPE_TO_ROUTE[r["type"]] or r["route"] == "both")
    both_rate = sum(1 for r in records if r["route"] == "both")
    confusion = {}
    for r in records:
        key = f'{r["type"]}->{r["route"]}'
        confusion[key] = confusion.get(key, 0) + 1
    return {
        "n": n,
        "strict": round(strict_hits / n, 3) if n > 0 else 0.0,
        "lenient": round(lenient_hits / n, 3) if n > 0 else 0.0,
        "both_rate": round(both_rate / n, 3) if n > 0 else 0.0,
        "confusion": confusion,
    }


# -- stage 2b: LLM judges --

def _trace(name: str, record: dict, **extra) -> dict:
    """LangSmith config so a judge span is identifiable in the trace UI.

    Without this every judge call lands as an anonymous `RunnableSequence` root --
    432 of them for one full run, with no way to get from a surprising score back to
    the question that produced it. `run_name` makes the span searchable and the
    metadata makes it filterable (e.g. all faithfulness judgements on code questions).
    """
    return {
        "run_name": name,
        "metadata": {"qid": record["qid"], "q_type": record.get("type"),
                     "route": record.get("route"), **extra},
        "tags": ["eval", "step-6c", name],
    }


def judge_faithfulness(record: dict) -> FaithfulnessScore:
    """Score one answer for groundedness against its own retrieved context."""
    llm = get_judge_llm().with_structured_output(FaithfulnessScore)
    context = format_context(_deserialize_sources(record["sources"]))
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=record["answer"])
    return llm.invoke(prompt, config=_trace("judge_faithfulness", record))


def judge_relevance(record: dict) -> RelevanceScore:
    """Score one answer for whether it answers the question."""

    llm = get_judge_llm().with_structured_output(RelevanceScore)
    prompt = RELEVANCE_PROMPT.format(question=record["question"], answer=record["answer"])
    return llm.invoke(prompt, config=_trace("judge_relevance", record))


def _claim_for_marker(text: str, marker_pos: int) -> str:
    """The sentence containing a citation marker, as the claim to verify.

    Crude sentence split on purpose: technical answers are full of `.` inside
    code (`state.docs`) and version numbers, so a naive `split(".")` mangles them.
    Taking a window around the marker up to the nearest newline or sentence end is
    good enough for a judge that only needs the local claim.
    """
    start = max(text.rfind("\n", 0, marker_pos), text.rfind(". ", 0, marker_pos)) + 1
    # End at whichever comes first: the next newline, or the end of THIS sentence.
    # Bounding on the newline alone let the claim swallow the following sentence, so
    # the judge saw facts belonging to [2] while asked about [1] -- and since the
    # rubric says partial support is not support, that silently depressed
    # citation_precision on every multi-sentence line.
    ends = [e for e in (text.find("\n", marker_pos),
                        text.find(". ", marker_pos) + 1) if e > 0]
    end = min(ends) if ends else len(text)
    claim = text[start:end].strip()

    # A marker can sit alone on its own line -- the generator does this after a code
    # fence, e.g. "```\n[3][4]". The line then contains no assertion at all, and the
    # judge rightly answers "nothing to evaluate" -> supported=False. That is an
    # extraction artifact scoring as a citation failure, so widen to the preceding
    # paragraph (usually the code block being cited) when nothing but markers is left.
    if not CITATION_RE.sub("", claim).strip(" \t.,;:()"):
        para_start = text.rfind("\n\n", 0, start)
        claim = text[0 if para_start == -1 else para_start + 2:end].strip()
    return claim


def judge_citations(record: dict) -> dict:
    """Check each citation marker against the source it points at.

    One judge call per (claim, source) pair. Deduplicate first: an answer citing
    [1] five times does not need five identical judgements -- judge each DISTINCT
    marker once, using the first claim that cites it. On a 38-question run that is
    the difference between ~90 and ~250 Bedrock calls.

    Returns {"n_checked": int, "n_supported": int, "precision": float,
             "details": [{"n": int, "claim": str, "supported": bool, "reason": str}]}
    where precision = n_supported / n_checked (1.0 if nothing to check).
    """
    llm = get_judge_llm().with_structured_output(CitationSupport)
    text = record["answer"]
    n_sources = len(record["sources"])

    seen: set[int] = set()
    details = []
    for m in CITATION_RE.finditer(text):
        # m.group(1) is the marker VALUE, m.start() is where it sits in the answer.
        # Both are needed: the value picks the source, the position picks the claim.
        n = int(m.group(1))
        if n in seen or not 1 <= n <= n_sources:
            # Out-of-range markers are already counted by check_citation_syntax;
            # judging them would mean asking about a source that does not exist.
            continue
        seen.add(n)

        claim = _claim_for_marker(text, m.start())
        verdict = llm.invoke(
            CITATION_PROMPT.format(
                claim=claim,
                n=n,
                source=record["sources"][n - 1]["page_content"],
            ),
            # marker=n so a failing citation is findable without opening every span
            config=_trace("judge_citation", record, marker=n),
        )
        details.append({
            "n": n,
            "claim": claim,
            "supported": verdict.supported,
            "reason": verdict.reason,
        })

    n_checked = len(details)
    n_supported = sum(d["supported"] for d in details)
    return {
        "n_checked": n_checked,
        "n_supported": n_supported,
        # An answer with no citations gets 1.0 here -- vacuously perfect. That is
        # why pct_with_citations is reported alongside it: precision alone would
        # rate a citation-free answer as flawless.
        "precision": round(n_supported / n_checked, 3) if n_checked else 1.0,
        "details": details,
    }


def judge_record(record: dict) -> dict:
    """Run every check on one answer and return a flat scored row."""
    if "error" in record:
        return {"qid": record["qid"], "error": record["error"]}
    faithfulness = judge_faithfulness(record)
    relevance = judge_relevance(record)
    citation_syntax = check_citation_syntax(record["answer"], len(record["sources"]))
    citation_judgement = judge_citations(record)    
    return {
        "qid": record["qid"],
        "type": record["type"],
        "route": record["route"],
        "faithfulness": faithfulness.score,
        "relevance": relevance.score,
        "n_markers": citation_syntax["n_markers"],
        "n_invalid_markers": len(citation_syntax["invalid"]),
        "has_citations": citation_syntax["has_citations"],
        "citation_precision": citation_judgement["precision"],
        "elapsed_s": record["elapsed_s"],
        "detail": {
            "unsupported_claims": faithfulness.unsupported_claims,
            "faithfulness_reason": faithfulness.reason,
            "relevance_reason": relevance.reason,
            "citation_details": citation_judgement["details"],
        },
    }


# -- aggregation --

def aggregate(scored: list[dict]) -> dict:
    """Mean each metric overall and split by question type.

    The doc/code split is the point: Step 6b found retrieval fails asymmetrically
    (doc hit@5 1.000 vs code 0.684), so generation quality should be checked for
    the same asymmetry rather than hidden behind one average.

    Returns {"n": int, "n_errors": int,
             "faithfulness": float, "relevance": float, "citation_precision": float,
             "pct_with_citations": float, "pct_invalid_markers": float,
             "by_type": {"doc": {...}, "code": {...}}}
    """
    scored_ok = [r for r in scored if "error" not in r]

    def slice_metrics(rows: list[dict]) -> dict:
        """Every metric for one slice. Factored out so the overall row and the
        per-type rows cannot drift apart -- if they were written twice, a later
        metric added to one and not the other would produce a table whose columns
        silently mean different things."""
        if not rows:
            # A --limit 5 run legitimately contains zero code questions (the golden
            # set is not interleaved by type), so this is a normal path, not an error.
            return {"n": 0, "faithfulness": None, "relevance": None,
                    "citation_precision": None, "pct_with_citations": None,
                    "pct_invalid_markers": None}
        n = len(rows)
        return {
            "n": n,
            "faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 3),
            "relevance": round(sum(r["relevance"] for r in rows) / n, 3),
            "citation_precision": round(sum(r["citation_precision"] for r in rows) / n, 3),
            "pct_with_citations": round(sum(r["has_citations"] for r in rows) / n, 3),
            "pct_invalid_markers": round(sum(r["n_invalid_markers"] > 0 for r in rows) / n, 3),
        }

    return {
        **slice_metrics(scored_ok),
        "n_errors": len(scored) - len(scored_ok),
        "by_type": {t: slice_metrics([r for r in scored_ok if r["type"] == t])
                    for t in ("doc", "code")},
    }


def print_report(agg: dict, routing: dict) -> None:
    """Print markdown tables ready to paste into DECISIONS.md.

    Same shape as retrieval_metrics.print_table -- markdown, because the destination
    is DECISIONS.md and hand-retyping numbers into a doc is how transcription errors
    get into a report you then defend in an interview.
    """
    cols = ["faithfulness", "relevance", "citation_precision",
            "pct_with_citations", "pct_invalid_markers"]

    def row(label: str, m: dict) -> str:
        # `-` rather than 0.000 for an empty slice: a zero would read as "scored
        # terribly" when it means "no questions of this type were run".
        cells = ["-" if m[c] is None else f"{m[c]:.3f}" for c in cols]
        return f"| {label} | {m['n']} | {' | '.join(cells)} |"

    print(f"\n| slice | n | {' | '.join(cols)} |")
    print(f"| --- | --- | {' | '.join('---' for _ in cols)} |")
    print(row("all", agg))
    for t in ("doc", "code"):
        print(row(t, agg["by_type"][t]))
    if agg["n_errors"]:
        print(f"\n{agg['n_errors']} question(s) errored (generation or judging) and were not scored.")

    print(f"\n| routing | n | strict | lenient | both_rate |")
    print(f"| --- | --- | --- | --- | --- |")
    print(f"| router | {routing['n']} | {routing['strict']:.3f} | "
          f"{routing['lenient']:.3f} | {routing['both_rate']:.3f} |")
    # The confusion counts, not just the rates: they say WHICH way the router leans.
    # "doc->code" and "code->docs" are different bugs with different fixes.
    print("\nconfusion (golden type -> chosen route):")
    for k, v in sorted(routing["confusion"].items()):
        print(f"  {k:16} {v}")


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    regenerate = "--regenerate" in sys.argv

    rows = load_golden()
    if not rows:
        raise ValueError("golden set is empty")
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} questions")

    records = load_or_generate(rows, regenerate=regenerate)

    # Routing is scored from `records` (which carries `route`), so compute it BEFORE
    # judging -- judge_record drops `route` from its output, and mutating `records`
    # in place would leave nothing to score.
    routing = router_accuracy(records)

    # Per-row try/except, same as generate_answers: a judge that returns a response
    # Pydantic rejects must cost ONE question, not the whole run. Learned the hard
    # way -- a single malformed `unsupported_claims` discarded all 38 judgements
    # after 20 minutes of Bedrock calls. `aggregate` already excludes error rows.
    scored = []
    for i, record in enumerate(records, start=1):
        print(f"judging {i}/{len(records)} qid={record['qid']}", flush=True)
        try:
            scored.append(judge_record(record))
        except Exception as e:
            print(f"  JUDGE FAILED on {record['qid']}: {type(e).__name__}: {e}", flush=True)
            scored.append({"qid": record["qid"], "error": f"judge: {e}"})

    agg = aggregate(scored)
    print_report(agg, routing)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "routing": routing, "per_query": scored}, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
