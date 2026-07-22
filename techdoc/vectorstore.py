"""Vector store: embed chunks with Cohere and persist to Qdrant (local, on-disk).

Embedding ~10k chunks is the slow/expensive step, so ingestion is designed to
run ONCE and be reused: the Qdrant collection persists to disk (qdrant_data/),
and re-running detects the existing collection instead of re-embedding.
"""
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

COLLECTION_NAME = "techdoc"
QDRANT_PATH = "qdrant_data"
EMBED_DIM = 1024   # Cohere embed-english-v3 output dim
DISTANCE = Distance.COSINE
UPSERT_BATCH = 500                    # docs per add_documents call (progress + memory)


def get_client(path: str = QDRANT_PATH) -> QdrantClient:
    """Local on-disk Qdrant. NOTE: local mode holds a lock — one process at a time."""
    return QdrantClient(path=path)


def ensure_collection(client: QdrantClient, name: str = COLLECTION_NAME) -> bool:
    """Create the collection if missing. Return True if it already existed.

    IMPORTANT: create with a BARE VectorParams (not a dict) so the vector is
    'unnamed' — that matches QdrantVectorStore's default vector_name="". A named
    vector here would make the store fail its config validation later.
    """
    if client.collection_exists(name):
        return True
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=DISTANCE),
    )
    return False


def get_vector_store(client: QdrantClient, embeddings: Embeddings,
                    name: str = COLLECTION_NAME) -> QdrantVectorStore:
    """Wrap an (existing) collection as a LangChain vector store.

    retrieval_mode=DENSE: we only store dense vectors here. BM25/sparse is
    hand-rolled in Step 5, so we deliberately do NOT use Qdrant's native sparse.
    """
    return QdrantVectorStore(
        client=client, collection_name=name,
        embedding=embeddings, retrieval_mode=RetrievalMode.DENSE,
    )


def stable_id(chunk_id: str) -> str:
    """Deterministic UUID from our string chunk_id.

    Qdrant point IDs must be uint or UUID — our 'repo:relpath:index' strings
    aren't valid. uuid5 makes re-ingesting the SAME chunk upsert (overwrite)
    rather than duplicate -> idempotent ingestion.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_index(documents: list[Document], embeddings: Embeddings,
                path: str = QDRANT_PATH, name: str = COLLECTION_NAME,
                force: bool = False) -> QdrantVectorStore:
    """Embed + load documents, reusing an existing collection unless force=True."""
    client = get_client(path)

    if force and client.collection_exists(name):
        client.delete_collection(name)

    existed = ensure_collection(client, name)
    store = get_vector_store(client, embeddings, name)

    if existed and not force: 
        print(f"Collection '{name}' already exists — skipping embedding.")
        return store

    ids = [stable_id(d.metadata["chunk_id"]) for d in documents]
    for i in range(0, len(documents), UPSERT_BATCH):
        slice_docs = documents[i:i + UPSERT_BATCH]
        slice_ids = ids[i:i + UPSERT_BATCH]
        store.add_documents(slice_docs, ids=slice_ids)
        print(f"{min(i + UPSERT_BATCH, len(documents))}/{len(documents)} embedded")

    return store