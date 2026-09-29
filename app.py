import os

import streamlit as st

try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass

from src.graph import build_rag_graph, run, to_payload

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def load_graph():
    return build_rag_graph()


st.title("Agentic AI eBook Chatbot")
st.caption("LangGraph + Pinecone RAG. Answers come strictly from the Agentic AI eBook.")

if not os.environ.get("OPENROUTER_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
    st.error("Set OPENROUTER_API_KEY and PINECONE_API_KEY in .env or Streamlit secrets.")
    st.stop()

session = st.session_state
session.setdefault("history", [])
session.setdefault("last", None)

query = st.chat_input("Ask a question about the eBook")

if query:
    try:
        with st.spinner("Retrieving and generating..."):
            result = run(load_graph(), query)
        payload = to_payload(query, result)
        session.history.append(payload)
        session.last = {"payload": payload, "pages": result["pages"]}
    except Exception as exc:
        st.error(f"Request failed: {exc}")

for item in session.history:
    st.chat_message("user").write(item["query"])
    st.chat_message("assistant").write(item["final_answer"])

if session.last:
    last = session.last
    with st.sidebar:
        st.divider()
        st.subheader("Confidence")
        st.metric("Confidence score", last["payload"]["confidence_score"])
        st.progress(min(1.0, max(0.0, last["payload"]["confidence_score"])))
        st.subheader("Retrieved context")
        for i, chunk in enumerate(last["payload"]["retrieved_context_chunks"]):
            page = last["pages"][i] if i < len(last["pages"]) else "?"
            with st.expander(f"Chunk {i + 1} (page {page})"):
                st.write(chunk)
        st.subheader("JSON response")
        st.json(last["payload"])
