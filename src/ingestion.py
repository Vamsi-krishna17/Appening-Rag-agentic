import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src import config


def load_chunks(pdf_path):
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata = {
            "page": int(chunk.metadata.get("page", 0)) + 1,
            "source": os.path.basename(pdf_path),
        }
    return [c for c in chunks if len(c.page_content.strip()) >= 40]


def ensure_index(pc, index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)


def run_ingestion(pdf_path=config.PDF_PATH, index_name=config.INDEX_NAME, reset=False):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    ensure_index(pc, index_name)
    index = pc.Index(index_name)
    existing = index.describe_index_stats().total_vector_count
    if existing and not reset:
        print(f"Index '{index_name}' already holds {existing} vectors. Use --reset to re-ingest.")
        return None
    if existing:
        index.delete(delete_all=True)
    chunks = load_chunks(pdf_path)
    store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=config.get_embeddings(),
        index_name=index_name,
        batch_size=64,
    )
    print(f"Ingested {len(chunks)} chunks into '{index_name}'.")
    return store


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=config.PDF_PATH)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run_ingestion(args.pdf, config.INDEX_NAME, args.reset)
