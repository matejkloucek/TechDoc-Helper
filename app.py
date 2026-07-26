"""Streamlit chat UI for TechDoc-Helper.

    uv run streamlit run app.py

Thin by design: every question goes through `techdoc.graph.astream_events`, so the
UI measures the same pipeline Step 6c evaluated. No retrieval, prompting or model
config lives here -- if it did, the eval numbers would stop describing the product.

TWO THINGS THIS FILE HAS TO SOLVE
---------------------------------
1. The graph is async, `st.write_stream` is sync. Bridged below with a worker thread
   feeding a queue (`_sync_stream`).
2. Streamlit reruns this entire script on every interaction, so anything expensive
   must be cached or a singleton. The retriever singleton in graph.py is what makes
   that survivable -- otherwise each keystroke would rebuild BM25 and trip Qdrant's
   exclusive file lock.
"""
import asyncio
import queue
import threading

import streamlit as st

from techdoc.graph import astream_events, source_label

# A sentinel object rather than None: None is a legitimate value to put on a queue,
# and using it as "done" would make an empty yield indistinguishable from the end.
_DONE = object()

# Chunks are ~1000 chars (see DECISIONS.md); truncate the preview so a 5-source
# expander stays scannable rather than becoming a wall of text.
PREVIEW_CHARS = 600

EXAMPLES = [
    "What is a checkpointer in LangGraph?",
    "How do interrupts work and how are they implemented?",
    "What does the _pending_interrupts method return?",
]


def _sync_stream(question: str):
    """Run the async graph on a worker thread, yield its events synchronously.

    st.write_stream needs a plain sync iterator. asyncio.run() in the Streamlit
    script thread is not enough on its own -- we need to *interleave* consuming the
    async generator with yielding to Streamlit, so the tokens appear as they arrive
    instead of all at once at the end.

    So: a worker thread owns its own event loop and pushes each event onto a queue;
    this generator (on the Streamlit thread) drains that queue. Exceptions are put
    on the queue too and re-raised here, otherwise a Bedrock failure would surface
    as a silent hang rather than an error in the UI.
    """
    q: queue.Queue = queue.Queue()

    def worker():
        async def pump():
            async for event in astream_events(question):
                q.put(event)

        try:
            asyncio.run(pump())
        except Exception as e:      # noqa: BLE001 -- re-raised on the main thread
            q.put(e)
        finally:
            q.put(_DONE)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = q.get()
        if item is _DONE:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def render_sources(sources: list) -> None:
    """Show the numbered sources backing an answer, in an expander.

    The numbers MUST match `format_context`'s `enumerate(docs, start=1)` -- that is
    what makes the [1]/[2] markers in the answer text meaningful. Any renumbering
    here would silently mislabel every citation.
    """
    if not sources:
        st.caption("_No sources retrieved — the answer above should say so._")
        return

    with st.expander(f"Sources ({len(sources)})"):
        for i, doc in enumerate(sources, start=1):
            is_code = doc.metadata.get("type") == "code"
            st.markdown(f"**[{i}]** `{doc.metadata.get('type', '?')}` — {source_label(doc)}")
            body = doc.page_content
            st.code(body[:PREVIEW_CHARS] + ("\n…" if len(body) > PREVIEW_CHARS else ""),
                    language="python" if is_code else "markdown")


def main():
    st.set_page_config(page_title="TechDoc-Helper", page_icon="📚", layout="centered")
    st.title("📚 TechDoc-Helper")
    st.caption("Ask about LangChain / LangGraph. Answers are grounded in the indexed "
               "docs and source, with inline [n] citations.")

    # Streamlit reruns this whole script on every interaction, so the transcript must
    # live in session_state or the conversation resets with each message.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Try one")
        for ex in EXAMPLES:
            # A button click sets `pending` and reruns; the input handler below picks
            # it up. Buttons cannot return a value across a rerun, hence the relay.
            if st.button(ex, use_container_width=True):
                st.session_state.pending = ex
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Replay history BEFORE handling new input, so turns render top-to-bottom.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []))

    question = st.chat_input("e.g. How do interrupts work in LangGraph?")
    if not question:
        question = st.session_state.pop("pending", None)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources: list = []

        def tokens():
            """Yield only the text, capturing the sources event as a side effect.

            `sources[:] = payload` mutates the list the enclosing scope holds.
            Rebinding with `sources = payload` would create a local and leave the
            outer list empty -- the classic closure trap, and a silent one: the
            answer would stream fine and the citations would just vanish.
            """
            for kind, payload in _sync_stream(question):
                if kind == "sources":
                    sources[:] = payload
                else:
                    yield payload

        try:
            # write_stream renders incrementally and returns the joined string.
            text = st.write_stream(tokens())
        except Exception as e:      # noqa: BLE001 -- surface it, do not hang
            text = f"⚠️ **Error:** {e}"
            st.error(text)

        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources}
    )


if __name__ == "__main__":
    main()
