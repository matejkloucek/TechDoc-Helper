
"""Hybrid retrieval: BM25 (lexical) + dense (Qdrant) fused with RRF, then reranked.

Pipeline:  query
            ├── BM25 top-N  ─┐
            └── dense top-N ─┤ 
                            ├── RRF fusion  ── top-N fused
                            └── Bedrock Cohere rerank ── top-k
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_aws.document_compressors.rerank import BedrockRerank

from techdoc.config import get_embeddings
from techdoc.ingest import load_manifest, iter_source_files, chunk_records
from techdoc.vectorstore import get_client, get_vector_store

# how many candidates each retriever contributes before fusion/rerank
CANDIDATE_N = 20
RRF_K = 60                    # RRF damping constant (standard default)
RERANK_MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"


def build_bm25(k: int = CANDIDATE_N) -> BM25Retriever:
    """BM25 over the SAME chunks as the dense index (option A: re-chunk on startup).

    BM25Retriever builds its index in-memory from a document list; it does NOT
    read from Qdrant. Re-chunking is fast (no embedding) and stays in sync with
    the vector store since both derive from the same manifest.
    """
    docs = chunk_records(list(iter_source_files(load_manifest())))
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k  # how many docs BM25 returns
    return retriever


def rrf_fuse(rankings: list[list[Document]], k: int = RRF_K) -> list[Document]:
    """Reciprocal Rank Fusion of multiple ranked Document lists.

    RRF score for a doc = sum over each ranking of 1 / (k + rank), where rank is
    1-based position in that ranking. Docs are identified by metadata['chunk_id']
    (stable + unique). Return docs sorted by fused score, descending.

    Why RRF: it fuses rankings using only POSITIONS, not the raw scores — so we
    don't have to normalize BM25 scores (unbounded) against cosine sims (0..1),
    which are on totally different scales. That scale-invariance is the point.
    """
    scores: dict[str, float] = {}
    doc_by_id: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):   # 1-based
            cid = doc.metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            doc_by_id[cid] = doc # remember the Document
    sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_by_id[cid] for cid in sorted_cids]


class HybridRetriever:
    """Owns the dense store, BM25, and reranker. Construct ONCE (Qdrant lock!)."""

    def __init__(self):
        self.client = get_client()                              # single client!
        self.dense = get_vector_store(self.client, get_embeddings())
        self.bm25 = build_bm25()
        self.reranker = BedrockRerank(
            model_arn=RERANK_MODEL_ARN,
            region_name="us-east-1",
            top_n=CANDIDATE_N
        )

    def retrieve(self, query: str, k: int = 5,
                candidate_n: int = CANDIDATE_N) -> list[Document]:
        """Return top-k Documents after hybrid fusion + rerank."""
        dense_hits = self.dense.similarity_search(query, k=candidate_n)
        bm25_hits  = self.bm25.invoke(query)   # returns list[Document]
        fused = rrf_fuse([dense_hits, bm25_hits])[:candidate_n]
        reranked = self.reranker.compress_documents(fused, query=query)
        return list(reranked)[:k]