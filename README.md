# Agentic AI eBook RAG Chatbot

A Retrieval-Augmented Generation chatbot built with **LangGraph**, **Pinecone** and **Streamlit**. It answers questions strictly from the *Agentic AI for Executives* eBook and refuses anything the document does not cover.

LLM and embeddings are called through **OpenRouter** (OpenAI-compatible API): `openai/text-embedding-3-small` for embeddings (OpenAI embeddings, routed via OpenRouter) and `openai/gpt-4o-mini` for generation and grading.

## Architecture

```
START -> retrieve -+-> generate -+-> grade -+-> END
                   |             |          |
                   |             |          +-> generate (retry once if groundedness < 0.6)
                   |             +-> refuse -> END   (model says "not in document")
                   +-> refuse -> END                 (top similarity below threshold)
```

| Component | Technology | Responsibility |
|---|---|---|
| Ingestion | PyPDFLoader + RecursiveCharacterTextSplitter | Parse the PDF, chunk (800 chars, 100 overlap), attach page metadata |
| Embeddings + vector DB | OpenAI `text-embedding-3-small` (via OpenRouter) + Pinecone (cosine, 1536 dims) | Store and search chunk vectors |
| Orchestration | LangGraph `StateGraph` | retrieve, generate, grade, refuse nodes with conditional routing |
| UI | Streamlit | Chat, retrieved chunks with page numbers, confidence score, raw JSON |

**Nodes**
- `retrieve`: top-4 similarity search in Pinecone. If the best cosine similarity is below `RELEVANCE_THRESHOLD` (0.25), the graph routes to `refuse`.
- `generate`: LLM answers using only the retrieved context; if the context lacks the answer it must output a fixed refusal sentence.
- `grade`: a second LLM call scores how well the answer is supported by the context (0 to 1). If below 0.6 the graph loops back to `generate` once with a stricter prompt.
- `refuse`: returns the refusal message with confidence 0.

**Confidence score** = `0.7 * groundedness (LLM grader) + 0.3 * min(1, top_similarity / 0.6)`, rounded to 2 decimals. Refusals score 0.

## Output format

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["Chunk 1 text...", "Chunk 2 text..."],
  "confidence_score": 0.92
}
```

## Project structure

```
rag-agentic-ai/
├── data/Ebook-Agentic-AI.pdf
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── graph.py
│   └── config.py
├── app.py
├── requirements.txt
├── .env.example
├── README.md
└── tests_sample_queries.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in OPENROUTER_API_KEY and PINECONE_API_KEY
```

You need an [OpenRouter](https://openrouter.ai) key with a few dollars of credit and a free [Pinecone](https://www.pinecone.io) key. Avoid `:free` OpenRouter models for the LLM, they are heavily rate-limited. Set `FALLBACK_MODELS` in `.env` to let OpenRouter fail over to another model automatically.

## Ingest the eBook (run once)

```bash
python -m src.ingestion            # creates the Pinecone index if needed and upserts all chunks
python -m src.ingestion --reset    # wipe and re-ingest
```

## Run the app

```bash
streamlit run app.py
```

## Run the sample queries

```bash
python tests_sample_queries.py
```

Runs the six validation queries (definition, architecture, use cases, comparison, challenges, and the out-of-scope "capital of France" check), prints each JSON payload and saves them to `sample_outputs.json`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (keep `data/Ebook-Agentic-AI.pdf` in the repo). Never commit `.env`.
2. Run the ingestion command locally once so the Pinecone index is populated.
3. On [share.streamlit.io](https://share.streamlit.io) create a new app from the repo with `app.py` as the entry file.
4. In **Advanced settings > Secrets** paste the contents of `.streamlit/secrets.toml.example` with your real keys.

## Configuration

| Variable | Default |
|---|---|
| `OPENROUTER_API_KEY` | required |
| `PINECONE_API_KEY` | required |
| `PINECONE_INDEX_NAME` | `agentic-ai-index` |
| `LLM_MODEL` | `openai/gpt-4o-mini` |
| `FALLBACK_MODELS` | empty (comma-separated OpenRouter model IDs) |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` |
| `RELEVANCE_THRESHOLD` | `0.25` |
