import os

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PDF_PATH = "data/Ebook-Agentic-AI.pdf"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
FALLBACK_MODELS = [m.strip() for m in os.getenv("FALLBACK_MODELS", "nvidia/nemotron-3-ultra-550b-a55b:free").split(",") if m.strip()]
EMBEDDING_DIM = 384
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.25"))
GROUNDED_THRESHOLD = 0.6
MAX_ATTEMPTS = 2
REFUSAL_TEXT = "I cannot answer based on the provided document."

SAMPLE_QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]


def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def get_llm():
    extra = {"models": [LLM_MODEL, *FALLBACK_MODELS]} if FALLBACK_MODELS else {}
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        max_retries=3,
        timeout=60,
        extra_body=extra,
    )
