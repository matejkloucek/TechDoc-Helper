"""Retrieval eval: hit@k, recall@k, MRR over the curated golden set.

Makes ZERO LLM calls -- it only runs the retriever and compares chunk_ids against
the gold labels. That makes it deterministic, cheap, and fast enough to use as the
inner feedback loop when tuning chunk_size, candidate_n, or the fusion constant.
(The LLM-as-judge layer in Step 6c is the slow, non-deterministic outer loop.)

Compares three ablations -- dense / hybrid / hybrid+rerank -- so the "hybrid is
better" claim becomes a number instead of an anecdote.

Usage: uv run python -m techdoc.eval.retrieval_metrics
"""
import json
import time

from techdoc.retrieval import HybridRetriever

GOLDEN_PATH = "techdoc/eval/goldenset.jsonl"
RESULTS_PATH = "techdoc/eval/results_retrieval.json"
MODES = ["dense", "hybrid", "rerank"]
K_VALUES = [1, 3, 5, 10]
RETRIEVE_K = max(K_VALUES)   # retrieve once at the largest k, slice for smaller ones


# --metrics --
# Each takes the retrieved chunk_ids IN RANK ORDER plus the set of gold ids,
# and returns a float for ONE query. Averaging across queries happens later.

def hit_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """1.0 if ANY gold id appears in the top-k, else 0.0.

    "Did we get at least one right answer in front of the user?" This is the
    metric that matters most for RAG: the generator usually only needs one good
    chunk to answer correctly.
    """

    return 1.0 if set(retrieved_ids[:k]) & gold_ids else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Fraction of ALL gold ids found in the top-k.

    Differs from hit@k only when a question has multiple gold chunks. Our set is
    currently single-gold, so recall@k == hit@k -- they will print identically,
    and that is expected, not a bug. Implemented now because gold_chunk_ids is a
    list precisely so multi-gold questions can be added later.
    """

    if len(gold_ids) == 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids)


def reciprocal_rank(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """1 / (1-based rank of the FIRST gold hit); 0.0 if no gold id was retrieved.

    Averaged over queries this is MRR. Unlike hit@k it is rank-SENSITIVE: moving
    the right chunk from position 5 to position 1 raises MRR (0.2 -> 1.0) but
    leaves hit@5 unchanged. That is exactly the improvement a reranker makes, so
    MRR is the metric that should show the rerank stage earning its latency.
    """

    for i, id in enumerate(retrieved_ids, start=1):
        if id in gold_ids:
            return 1.0 / i
    return 0.0


# -- harness --

def load_golden(path: str = GOLDEN_PATH) -> list[dict]:
    """Read the curated golden set (JSONL -> list of dicts)."""

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_mode(retriever: HybridRetriever, rows: list[dict], mode: str) -> dict:
    """Run every golden question through `mode` and aggregate the metrics.

    Returns a dict shaped like:
        {"mode": mode,
         "n_queries": 38,
         "hit@1": 0.71, "hit@3": ..., "recall@1": ..., "mrr": 0.79,
         "per_query": [{"qid": "q001", "rr": 1.0, "gold_rank": 1, "type": "doc"}, ...]}

    Keep `per_query` -- aggregates tell you the score moved, per-query rows tell
    you WHICH questions moved, which is what you actually debug with. It also
    lets you split doc vs code performance afterwards.
    """
    per_query = []
    for row in rows:
        gold_ids = set(row["gold_chunk_ids"])
        docs = retriever.retrieve(row["question"], k=RETRIEVE_K, mode=mode)
        retrieved_ids = [d.metadata["chunk_id"] for d in docs]

        rr = reciprocal_rank(retrieved_ids, gold_ids)
        per_query.append({
            "qid": row["qid"],
            "type": row["type"],
            "rr": rr,
            # 1-based rank of the first gold hit, or None if missed. Handy for
            # eyeballing near-misses (rank 6 when k=5) vs total failures.
            "gold_rank": round(1 / rr) if rr else None,
            "retrieved_ids": retrieved_ids,
        })

    n = len(rows)
    result = {"mode": mode, "n_queries": n}

    for k in K_VALUES:
        result[f"hit@{k}"] = round(sum(hit_at_k(pq["retrieved_ids"], set(row["gold_chunk_ids"]), k) for row, pq in zip(rows, per_query)) / n, 3)
        result[f"recall@{k}"] = round(sum(recall_at_k(pq["retrieved_ids"], set(row["gold_chunk_ids"]), k) for row, pq in zip(rows, per_query)) / n, 3)

    result["mrr"] = round(sum(pq["rr"] for pq in per_query) / n, 3)

    result["per_query"] = per_query
    return result


def print_table(results: list[dict]) -> None:
    """Print a markdown comparison table, ready to paste into DECISIONS.md."""
    cols = [f"hit@{k}" for k in K_VALUES] + ["mrr"]
    print(f"\n| config | {' | '.join(cols)} |")
    print(f"| --- | {' | '.join('---' for _ in cols)} |")
    for r in results:
        cells = [f"{r[c]:.3f}" for c in cols]
        print(f"| {r['mode']} | {' | '.join(cells)} |")


def main():
    rows = load_golden()
    if not rows:
        raise ValueError(f"no golden questions found in {GOLDEN_PATH}")
    print(f"loaded {len(rows)} golden questions")

    # ONE retriever for all modes: constructing it twice would trip Qdrant's
    # exclusive file lock, and re-chunking for BM25 would be wasted work.
    retriever = HybridRetriever()

    results = []
    for mode in MODES:
        print(f"\n--- {mode} ---")
        t0 = time.time()
        result = evaluate_mode(retriever, rows, mode)
        results.append(result)
        print(f"{mode}: hit@5={result['hit@5']:.3f}  mrr={result['mrr']:.3f}  "
              f"({time.time() - t0:.1f}s)")

    print_table(results)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
