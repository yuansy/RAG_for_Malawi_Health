import os
import pickle
import shutil

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

INDEX_DIR = "./my_hybrid_index"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def build_index(chunks, index_dir=INDEX_DIR):
    """
    Converts a list of chunk dicts into LangChain Documents,
    builds the Hybrid Index (Chroma vector store + BM25), and saves to disk.

    Parameters
    ----------
    chunks    : list of dicts with keys chunk_id, doc_id, section, text
    index_dir : directory where the index is persisted

    Returns
    -------
    db dict  : {"vectorstore": ..., "bm25_retriever": ..., "raw_docs": ...}
    """
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    print("Building and Saving Hybrid Index (Vector + BM25)...")

    # Convert chunks into LangChain Documents
    docs = [
        Document(
            page_content=c['text'],
            metadata={
                "chunk_id": c.get("chunk_id", ""),
                "doc_id":   c.get("doc_id", ""),
                "section":  c.get("section", ""),
            }
        )
        for c in chunks
    ]

    # 1. Vector Store (Semantic Search)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=index_dir,
    )

    # 2. BM25 Store (Keyword Search)
    bm25_retriever = BM25Retriever.from_documents(docs)

    # Save BM25 + raw docs to disk
    with open(os.path.join(index_dir, "bm25_and_docs.pkl"), "wb") as f:
        pickle.dump({"bm25_retriever": bm25_retriever, "raw_docs": docs}, f)

    print(f"Index successfully built and saved to '{index_dir}'.")

    return {
        "vectorstore":    vectorstore,
        "bm25_retriever": bm25_retriever,
        "raw_docs":       docs,
    }


def is_index_valid(index_dir=INDEX_DIR):
    """Return True if a previously built index exists on disk."""
    required_files = [
        os.path.join(index_dir, "bm25_and_docs.pkl"),
        os.path.join(index_dir, "chroma.sqlite3"),
    ]
    return all(os.path.exists(f) for f in required_files)


def load_index(index_dir=INDEX_DIR):
    """
    Load the Chroma vector store and BM25 retriever from disk.

    Returns
    -------
    db dict  : {"vectorstore": ..., "bm25_retriever": ..., "raw_docs": ...}
    """
    if not os.path.exists(index_dir):
        raise FileNotFoundError(
            f"Index directory '{index_dir}' does not exist. "
            "Please build the index first by running ingest + build_index."
        )

    print(f"Loading Hybrid Index from '{index_dir}'...")

    # 1. Load Vector Store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=index_dir,
        embedding_function=embeddings,
    )

    # 2. Load BM25 Store and raw_docs
    with open(os.path.join(index_dir, "bm25_and_docs.pkl"), "rb") as f:
        data = pickle.load(f)

    print("Index successfully loaded.")

    return {
        "vectorstore":    vectorstore,
        "bm25_retriever": data["bm25_retriever"],
        "raw_docs":       data["raw_docs"],
    }
