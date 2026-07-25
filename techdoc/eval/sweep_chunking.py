"""Sweep chunk_size / chunk_overlap and re-measure retrieval.

WHY THIS IS NOT JUST A LOOP OVER retrieval_metrics.py
-----------------------------------------------------
`chunk_id` is `repo:rel_path:index`, so changing chunk_size RENUMBERS every chunk
in a file. The golden set's `gold_chunk_ids` then point at different text while
still resolving to a valid id -- 33/38 ids still exist at chunk_size=500, but only
1/38 refers to the text that was actually labelled. Matching by id across configs
would produce plausible-looking, meaningless numbers.

Fix: re-anchor each label by CONTENT. For every config we find the chunk that best
overlaps the originally-labelled text (same file, highest character overlap with
the recorded preview) and treat that as gold. A retrieved chunk counts as a hit if
it contains the labelled snippet.

Each config also needs its OWN embedded collection (~13 min for the full corpus at
chunk_size=1000), which is the real cost of this sweep. Collections are named
`techdoc_s{size}_o{overlap}` and reused if already present, so re-runs are cheap.

Usage:
    PYTHONPATH=. uv run python -m techdoc.eval.sweep_chunking
"""
import json

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

import techdoc.ingest as ingest
from techdoc.config import get_embeddings
from techdoc.eval.retrieval_metrics import reciprocal_rank
from techdoc.ingest import chunk_records, iter_source_files, load_manifest
from techdoc.retrieval import CANDIDATE_N, code_tokenize, rrf_fuse
from techdoc.vectorstore import build_index, get_vector_store

GOLDEN_PATH = "techdoc/eval/goldenset.jsonl"
RESULTS_PATH = "techdoc/eval/results_chunking.json"

# (chunk_size, chunk_overlap). 1000/150 is the current production setting.
CONFIGS = [
    (500, 75),
    (1000, 150),
    (1500, 225),
]
# Chars of the labelled chunk used to re-anchor gold across configs. Long enough
# to be unique in the corpus, short enough to survive a smaller chunk_size.
ANCHOR_CHARS = 120


def make_splitter_factory(chunk_size: int, chunk_overlap: int):
    """Return a get_splitter replacement bound to this config's size/overlap."""
    def get_splitter(record_type: str) -> RecursiveCharacterTextSplitter:
        lang = Language.MARKDOWN if record_type == "doc" else Language.PYTHON
        return RecursiveCharacterTextSplitter.from_language(
            lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    return get_splitter


def rechunk(chunk_size: int, chunk_overlap: int):
    """Re-run ingestion at this config by swapping ingest.get_splitter.

    chunk_records() calls get_splitter() internally, so monkeypatching the module
    attribute is the least invasive way to parameterise it without changing the
    production signature.
    """
    original = ingest.get_splitter
    ingest.get_splitter = make_splitter_factory(chunk_size, chunk_overlap)
    try:
        return chunk_records(list(iter_source_files(load_manifest())))
    finally:
        ingest.get_splitter = original


def anchor_gold(rows: list[dict], docs: list[ingest.Document]) -> list[dict]:
    """Re-anchor each golden label to this config's chunking, by content.

    For each row, look at chunks from the SAME file and pick the one with the
    largest character overlap against the labelled preview. Returns rows with an
    added "anchor" (the labelled snippet) and "gold_ids" (ids that contain it).
    """
    by_path: dict[str, list] = {}
    for d in docs:
        by_path.setdefault(d.metadata["rel_path"], []).append(d)

    anchored = []
    for row in rows:
        anchor = row["source_chunk_preview"][:ANCHOR_CHARS].strip()
        candidates = by_path.get(row["rel_path"], [])
        # any chunk fully containing the anchor is an acceptable gold chunk
        gold_ids = [d.metadata["chunk_id"] for d in candidates if anchor in d.page_content]
        if not gold_ids and candidates:
            # anchor was split across a boundary: fall back to best partial overlap
            best = max(candidates, key=lambda d: _overlap(anchor, d.page_content))
            if _overlap(anchor, best.page_content) >= len(anchor) * 0.5:
                gold_ids = [best.metadata["chunk_id"]]
        anchored.append({**row, "anchor": anchor, "gold_ids": set(gold_ids)})
    return anchored


def _overlap(anchor: str, text: str) -> int:
    """Length of the longest prefix of `anchor` present in `text`."""
    for n in range(len(anchor), 0, -1):
        if anchor[:n] in text:
            return n
    return 0


def evaluate_config(chunk_size: int, chunk_overlap: int, rows: list[dict],
                    k_values=(1, 3, 5, 10)) -> dict:
    """Build (or reuse) an index for this config and measure hybrid retrieval.

    Rerank is deliberately SKIPPED here: it is a fixed post-processing step that
    cannot change which candidates chunking makes available, and it costs a
    Bedrock call per query. We compare the hybrid stage, where chunking acts.
    """
    docs = rechunk(chunk_size, chunk_overlap)
    name = f"techdoc_s{chunk_size}_o{chunk_overlap}"
    print(f"  {len(docs)} chunks -> collection '{name}'")

    anchored = anchor_gold(rows, docs)
    unanchored = [r["qid"] for r in anchored if not r["gold_ids"]]
    if unanchored:
        print(f"  WARNING: {len(unanchored)} labels could not be re-anchored: {unanchored}")

    embeddings = get_embeddings()
    store = build_index(docs, embeddings, name=name)   # reuses if it exists

    bm25 = BM25Retriever.from_documents(docs, preprocess_func=code_tokenize)
    bm25.k = CANDIDATE_N

    per_query, rr = [], []
    for row in anchored:
        if not row["gold_ids"]:
            continue          # cannot score a label we could not locate
        dense = store.similarity_search(row["question"], k=CANDIDATE_N)
        fused = rrf_fuse([dense, bm25.invoke(row["question"])])[:CANDIDATE_N]
        ids = [d.metadata["chunk_id"] for d in fused]
        v = reciprocal_rank(ids, row["gold_ids"])
        rr.append(v)
        per_query.append({"qid": row["qid"], "type": row["type"], "rr": v,
                          "gold_rank": round(1 / v) if v else None})

    n = len(rr)
    result = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
              "n_chunks": len(docs), "n_scored": n, "n_unanchored": len(unanchored),
              "mrr": round(sum(rr) / n, 3)}
    for k in k_values:
        result[f"hit@{k}"] = round(
            sum(1 for p in per_query if p["gold_rank"] and p["gold_rank"] <= k) / n, 3)
    for t in ("doc", "code"):
        sub = [p["rr"] for p in per_query if p["type"] == t]
        result[f"mrr_{t}"] = round(sum(sub) / len(sub), 3) if sub else None
        result[f"hit@5_{t}"] = round(
            sum(1 for p in per_query if p["type"] == t and p["gold_rank"]
                and p["gold_rank"] <= 5) / len(sub), 3) if sub else None
    result["per_query"] = per_query
    return result


def main():
    rows = [json.loads(l) for l in open(GOLDEN_PATH, encoding="utf-8") if l.strip()]
    print(f"loaded {len(rows)} golden questions\n")

    results = []
    for size, overlap in CONFIGS:
        print(f"=== chunk_size={size} overlap={overlap} ===")
        r = evaluate_config(size, overlap, rows)
        results.append(r)
        print(f"  scored {r['n_scored']}  hit@5={r['hit@5']:.3f}  mrr={r['mrr']:.3f}  "
              f"(doc {r['mrr_doc']} / code {r['mrr_code']})\n")

    cols = ["n_chunks", "hit@1", "hit@3", "hit@5", "mrr", "mrr_doc", "mrr_code"]
    print(f"| size/overlap | {' | '.join(cols)} |")
    print(f"| --- | {' | '.join('---' for _ in cols)} |")
    for r in results:
        cells = [str(r[c]) for c in cols]
        print(f"| {r['chunk_size']}/{r['chunk_overlap']} | {' | '.join(cells)} |")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
