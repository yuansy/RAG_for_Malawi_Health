import argparse
import sys
import time

from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaLLM

from ingest import load_booklets, chunk_documents
from index import build_index, is_index_valid, load_index
from rag import retrieve, generate, format_answer

# ---------------------------------------------------------------------------
# Configuration — edit these to match your environment
# ---------------------------------------------------------------------------

INDEX_DIR      = "./my_hybrid_index"
BOOKLET_GLOB   = "MWTGBookletsExcel/TG Booklet*.xlsx"
OLLAMA_MODEL   = "llama3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEMO_QUERIES = [
    # Correct samples
    "How can the laboratory results be reviewed during an outbreak investigation?",
    "Can you provide an example of a district spot map in disease surveillance?",
    "Why is it important to estimate carrier numbers for viral hepatitis B and C?",
    "Where do large outbreaks of Bacterial Meningitis occur, and how do they differ from smaller outbreaks outside the meningitis belt?",
    "What are the surveillance goals for lymphatic filariasis, and how can they be implemented at the local and international levels?",
    "How should Monkeypox specimens be prepared, stored, and transported?",

    # Correct samples (multiple chunks)
    "Compare the purposes and criteria for conducting a register review with those of the District log of suspected outbreaks and alerts.",
    "What is the role of community networks in Community Based Surveillance (CBS)?",

    # Abstain samples
    "What is the recommended treatment for COVID-19 in the 2024 guidelines?",
    "Who won the presidential election in Malawi in 2012?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def show_thinking_animation():
    print(f"\nThinking", end="")
    sys.stdout.flush()
    for _ in range(3):
        time.sleep(2)
        print(".", end="")
        sys.stdout.flush()


def get_db(rebuild=False):
    """Return a loaded (or freshly built) index db dict."""
    if rebuild or not is_index_valid(INDEX_DIR):
        docs   = load_booklets(BOOKLET_GLOB)
        chunks = chunk_documents(docs)
        db     = build_index(chunks, INDEX_DIR)
    else:
        db = load_index(INDEX_DIR)
    return db


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def run_interactive(db, reranker_model, llm, trace=False):
    print("\n===========================================================")
    print(" RAG SYSTEM for MALAWI HEALTH DATASET (Type 'exit' to quit)")
    print("===========================================================")

    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not query:
            continue

        show_thinking_animation()

        top_docs = retrieve(query, db, reranker_model)
        answer   = generate(query, top_docs, llm)
        format_answer(query, answer, top_docs, trace_mode=trace)


# ---------------------------------------------------------------------------
# Demo batch runner
# ---------------------------------------------------------------------------

def run_demo(db, reranker_model, llm, trace=False):
    for query in DEMO_QUERIES:
        top_docs = retrieve(query, db, reranker_model)
        answer   = generate(query, top_docs, llm)
        format_answer(query, answer, top_docs, trace_mode=trace)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="RAG system for the Malawi IDSR public health dataset.",
    )

    # ── Mode (mutually exclusive) ──────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="Start the interactive question-answering CLI.",
    )
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Run the 10 pre-defined demo queries and print results.",
    )

    # ── Index options ──────────────────────────────────────────────────────
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Force a full re-ingest and rebuild the index even if one already exists.",
    )

    # ── Display options ────────────────────────────────────────────────────
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Show retrieved chunks and relevance scores alongside each answer.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load index ────────────────────────────────────────────────────────
    db = get_db(rebuild=args.rebuild)

    # ── Load models ───────────────────────────────────────────────────────
    reranker_model = CrossEncoder(RERANKER_MODEL)
    llm            = OllamaLLM(model=OLLAMA_MODEL, temperature=0)

    # ── Run ───────────────────────────────────────────────────────────────
    if args.interactive:
        run_interactive(db, reranker_model, llm, trace=args.trace)
    elif args.demo:
        run_demo(db, reranker_model, llm, trace=args.trace)


if __name__ == "__main__":
    main()
