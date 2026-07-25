import json
import random
from pydantic import BaseModel, Field
from techdoc.config import get_llm
from techdoc.ingest import load_manifest, iter_source_files, chunk_records

N_CANDIDATES = 40
SEED = 42
OUT_PATH = "techdoc/eval/goldenset.draft.jsonl"

class GenResult(BaseModel):
    question: str = Field(description="A natural question a developer would ask, answerable SOLELY by the given chunk.")
    answerable: bool = Field(description="True only if the chunk genuinely answers it.")

PROMPT = """You are helping build a retrieval eval set for LangChain/LangGraph docs.
Given ONE documentation or source chunk, write a single realistic question that a
developer would type, which THIS chunk specifically answers. Avoid questions so
generic that many chunks would answer them. If the chunk is boilerplate (imports,
license, __all__) and answers nothing useful, set answerable=false.

Chunk metadata: {meta}
Chunk content:
\"\"\"{content}\"\"\""""


def is_useful(d) -> bool:
      lines = [l for l in d.page_content.splitlines() if l.strip()]
      if len(lines) < 3:
          return False
      boiler = sum(1 for l in lines if l.strip().startswith(("import ", "from ", "@", "__all__")))
      if boiler / len(lines) > 0.5:
          return False
      # a code chunk with no def/class is usually a fragment, not a teachable unit
      if d.metadata["type"] == "code" and not d.metadata.get("symbol"):
          return False
      return True


def main():
    docs = chunk_records(list(iter_source_files(load_manifest())))
    rng = random.Random(SEED)

    pool = [d for d in docs if is_useful(d)]
    doc_pool  = [d for d in pool if d.metadata["type"] == "doc"]
    code_pool = [d for d in pool if d.metadata["type"] == "code"]
    half = N_CANDIDATES // 2
    sample = rng.sample(doc_pool, half) + rng.sample(code_pool, half)

    llm = get_llm().with_structured_output(GenResult)
    rows = []
    for d in sample:
        meta = {k: d.metadata.get(k) for k in ("type", "rel_path", "heading", "symbol")}
        result = llm.invoke(PROMPT.format(meta=meta, content=d.page_content))
        if result.answerable:
            rows.append({
                "question": result.question,
                "gold_chunk_ids": [d.metadata["chunk_id"]],
                "type": d.metadata["type"],
                "rel_path": d.metadata["rel_path"],
                "source_chunk_preview": d.page_content[:200]
            })

    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} candidates to {OUT_PATH}")

if __name__ == "__main__":
    main()