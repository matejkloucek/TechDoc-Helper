"""LangGraph supervisor: routes a question to the docs agent, the code agent, or both.

    START -> route ->  ┌── docs_agent ──┐
                       ├── code_agent ──┤ -> synthesize -> END
                       └── (both, in parallel)

WHY A GRAPH AND NOT ONE PROMPT
------------------------------
Different question types need different retrieval. "What is a checkpointer?" wants
prose; "what does _pending_interrupts return?" wants source. Step 6 measured the gap:
doc questions hit@5 = 1.000, code questions 0.684. Filtering retrieval by corpus type
gives each agent a cleaner candidate pool than one undifferentiated search.

The supervisor pattern also gives us cost tiering (Haiku routes, Sonnet writes) and a
LangSmith trace where each step is a named span.

PUBLIC CONTRACT (Step 10 depends on this, so keep it stable):
    answer(question) -> AnswerResult(text, sources, route)
    astream_answer(question) -> async iterator of token strings
"""
from dataclasses import dataclass, field
from typing import Annotated, Literal
import operator
import threading

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from techdoc.config import get_llm, get_router_llm
from techdoc.retrieval import HybridRetriever

TOP_K = 5
Route = Literal["docs", "code", "both"]


# -- state --

class GraphState(TypedDict):
    """Shared state. NOTE the reducer on `docs`.

    Both agents can write `docs` in the SAME superstep when route == "both".
    Without a reducer LangGraph raises InvalidUpdateError on the concurrent write;
    `operator.add` makes the two lists concatenate instead. This is the single most
    important line in the file -- it is what makes the fan-out legal.
    """
    question: str
    route: Route
    docs: Annotated[list[Document], operator.add]
    answer: str


class RouteDecision(BaseModel):
    """Structured router output -- no free-text parsing."""
    route: Route = Field(description=(
        "'docs' for conceptual/how-to questions answerable from prose documentation; "
        "'code' for questions about specific functions, classes, signatures or implementation details;"
        "'both' when the question needs concept AND implementation."
    ))
    reason: str = Field(description="One short sentence justifying the choice.")


@dataclass
class AnswerResult:
    """What the UI gets back. Decoupled from graph internals on purpose."""
    text: str
    sources: list[Document] = field(default_factory=list)
    route: str = ""


# -- nodes --

ROUTER_PROMPT = """You route developer questions about LangChain/LangGraph to the \
right corpus.

- docs: conceptual explanations, guides, how-to, "what is X", "when should I use X"
- code: specific symbols, function signatures, return values, implementation details
- both: needs the concept explained AND the implementation shown

Question: {question}"""

SYNTHESIZE_PROMPT = """You are a technical documentation assistant for LangChain and \
LangGraph. Answer the question using ONLY the numbered context below.

Rules:
- Cite the sources you use inline as [1], [2], matching the numbers below.
- If the context does not contain the answer, say so plainly. Do not invent APIs.
- Prefer showing a short code example when the context has one.

Context:
{context}

Question: {question}"""


_RETRIEVER: HybridRetriever | None = None
_RETRIEVER_LOCK = threading.Lock()


def _retriever() -> HybridRetriever:
    """Process-wide singleton, built at most once even under concurrency.

    Qdrant local mode holds an EXCLUSIVE FILE LOCK, so a second HybridRetriever in
    the same process throws AlreadyLocked -- and rebuilding BM25 costs ~10s anyway.
    Streamlit reruns the whole script on every interaction (Step 10), which makes
    this lazy singleton mandatory rather than just an optimisation.

    THE LOCK IS LOAD-BEARING. LangGraph runs fanned-out branches concurrently, so on
    route == "both" search_docs and search_code both call this before either has
    finished constructing, and the loser gets
    `RuntimeError: Storage folder qdrant_data is already accessed by another
    instance of Qdrant client`. Double-checked locking: the fast path takes no lock
    once built, and the slow path re-checks inside the lock so the second caller
    returns the first caller's instance instead of building its own.
    """
    global _RETRIEVER
    if _RETRIEVER is None:
        with _RETRIEVER_LOCK:
            if _RETRIEVER is None:
                _RETRIEVER = HybridRetriever()
    return _RETRIEVER


def route(state: GraphState) -> dict:
    """Classify the question into docs / code / both using the cheap model.

    Returns {"route": ...} -- a partial state update, which is how every LangGraph
    node reports its result.
    """

    llm = get_router_llm().with_structured_output(RouteDecision)
    decision = llm.invoke(ROUTER_PROMPT.format(question=state["question"]))
    return {"route": decision.route}


def search_docs(state: GraphState) -> dict:
    """Retrieve from the DOC half of the corpus only."""

    hits = retrieve_filtered(state["question"], "doc")
    return {"docs": hits}


def search_code(state: GraphState) -> dict:
    """Retrieve from the CODE half of the corpus only."""
  
    hits = retrieve_filtered(state["question"], "code")
    return {"docs": hits}


def retrieve_filtered(question: str, doc_type: str, k: int = TOP_K) -> list[Document]:
    """Hybrid retrieval restricted to one corpus type.

    Post-filtering (retrieve wide, then keep matching metadata) is the pragmatic
    choice here: our BM25 index is in-memory and has no filter API, so a pre-filter
    would mean maintaining two BM25 indexes. Documented as a known tradeoff --
    a Qdrant payload pre-filter would be more efficient at scale.
    """

    docs = _retriever().retrieve(question, k=k * 4)
    filtered = [d for d in docs if d.metadata.get("type") == doc_type]
    return filtered[:k]


def source_label(doc: Document) -> str:
    """Human-readable citation label: 'src/oss/langgraph/persistence.mdx - Checkpointers'.

    `rel_path` rather than `module` even for code, because rel_path is something the
    user can actually open. The trailing context differs by corpus:
      doc  -> heading (nearest markdown heading; always populated)
      code -> symbol, BUT ~2400 code chunks are mid-function fragments with
              symbol=None, so fall back to the dotted module path. Note `or` and not
              .get(default): the key EXISTS with value None, so a default never fires.
    """
    meta = doc.metadata
    if meta["type"] == "doc":
        context = meta.get("heading") or meta.get("title") or ""
    else:
        context = meta.get("symbol") or meta.get("module") or ""
    return f"{meta['rel_path']} - {context}" if context else meta["rel_path"]


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt.

    The numbers here are the citation handles: SYNTHESIZE_PROMPT tells the model to
    cite as [1]/[2], and `answer()` returns the same list in the same order, so the
    UI can map [1] back to a real source. enumerate(start=1) is load-bearing --
    0-indexing here would make every citation off by one.
    """
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[{i}] ({doc.metadata['type']}) {source_label(doc)}\n{doc.page_content}")
    return "\n\n".join(blocks)


def synthesize(state: GraphState) -> dict:
    """Generate the final cited answer from whatever the agent(s) retrieved."""
    docs = state["docs"]
    if not docs:
        return {"answer": "I could not find anything relevant in the indexed "
                          "LangChain/LangGraph documentation or source to answer that."}

    prompt = SYNTHESIZE_PROMPT.format(
        context=format_context(docs),
        question=state["question"],
    )
    response = get_llm().invoke(prompt)
    # `.text` (property, not a method) concatenates the content blocks that
    # ChatBedrockConverse returns; `.content` can be a list of block dicts.
    return {"answer": response.text}


def fan_out(state: GraphState) -> list[str]:
    """Conditional edge: map the route onto the node(s) to run next.

    Returning a LIST of node names from a conditional edge is how LangGraph fans out
    to run them in PARALLEL in one superstep. (Send() is the other way to fan out --
    used when each branch needs a different payload rather than the shared state.)
    """

    return {
        "docs": ["search_docs"],
        "code": ["search_code"],
        "both": ["search_docs", "search_code"]
    }[state["route"]]


# -- graph --

def build_graph():
    """Wire the nodes together and compile.

    Structure:
        START -> route
        route -conditional-> search_docs and/or search_code
        search_docs -> synthesize
        search_code -> synthesize
        synthesize -> END

    Both agents point at synthesize; LangGraph waits for ALL active branches to
    finish before running it (superstep barrier), so synthesize sees the merged
    `docs` list rather than running twice.
    """
    builder = StateGraph(GraphState)

    builder.add_node("route", route)
    builder.add_node("search_docs", search_docs)
    builder.add_node("search_code", search_code)
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", fan_out, ["search_docs", "search_code"])
    builder.add_edge("search_docs", "synthesize")
    builder.add_edge("search_code", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


_GRAPH = None


def get_graph():
    """Compile once, reuse. Compilation is cheap but the retriever inside is not."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# -- public API --

def answer(question: str) -> AnswerResult:
    """Run the graph to completion and return the answer plus its sources."""

    initial_state = GraphState(
        question=question,
        route="",
        docs=[],
        answer=""
    )
    final_state = get_graph().invoke(initial_state)
    return AnswerResult(
        text=final_state["answer"],
        sources=final_state["docs"],
        route=final_state["route"]
    )


async def astream_answer(question: str):
    """Yield answer tokens as they are generated, for st.write_stream.

    Uses stream_mode="messages", which emits (message_chunk, metadata) tuples for
    every LLM token inside the graph. We filter to the synthesize node so the
    router's tokens never leak into the user-visible answer.
    """
    initial_state = GraphState(
        question=question,
        route="",
        docs=[],
        answer=""
    )

    async for chunk, meta in get_graph().astream(
        initial_state,
        stream_mode="messages",
    ):
        if meta.get("langgraph_node") == "synthesize" and chunk.text:
            yield chunk.text


async def astream_events(question: str):
    """Stream tokens AND the retrieved sources from ONE graph run.

    `astream_answer` yields only text, so a UI that also wants to show citations
    would have to run the graph a second time -- paying for retrieval and synthesis
    twice to display something the first run already computed.

    Passing a LIST of stream modes fixes that: LangGraph then yields
    `(mode, payload)` tuples, where the payload shape depends on the mode --
        "messages" -> (message_chunk, metadata)
        "values"   -> the full state dict after each superstep
    Verified on langgraph 1.2.9: `docs` is populated in the values event BEFORE
    synthesize finishes streaming, so the sources are known by the time the answer
    is complete.

    Yields ("token", str) and ("sources", list[Document]) -- a tagged union, so the
    caller can render tokens live and stash the sources for afterwards.
    """
    initial_state = GraphState(
        question=question,
        route="",
        docs=[],
        answer="",
    )

    sources_sent = False

    async for mode, payload in get_graph().astream(
        initial_state,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk, meta = payload
            # Same filter as astream_answer, and just as load-bearing: the router
            # emits ~39 chunks of structured-output JSON on its way to a RouteDecision.
            if meta.get("langgraph_node") == "synthesize" and chunk.text:
                yield "token", chunk.text
        elif mode == "values":
            # One values event per superstep, so `docs` is empty for the first two
            # (START -> route) and identical for the rest. Emit the first non-empty
            # snapshot only -- re-yielding would make the UI append the same sources
            # once per remaining superstep.
            docs = payload.get("docs")
            if docs and not sources_sent:
                sources_sent = True
                yield "sources", docs


if __name__ == "__main__":
    for q in [
        "What is a checkpointer in LangGraph?",                    # expect docs
        "What does the _pending_interrupts method return?",        # expect code
        "How do interrupts work and how are they implemented?",    # expect both
    ]:
        r = answer(q)
        print(f"\n{'=' * 70}\nQ: {q}\nROUTE: {r.route}  SOURCES: {len(r.sources)}")
        print(r.text[:400])
        for i, d in enumerate(r.sources, 1):
            print(f"  [{i}] {d.metadata['chunk_id']}")
