import re

import httpx
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

prompt_template = PromptTemplate.from_template("""
You are an expert public health assistant for Malawi.

Answer the question concisely using only the provided context.

If the context is insufficient, say exactly:
"cannot find in sources"

Do not guess or add outside knowledge.

Each context passage is labeled as:
[chunk_id | paragraph_id]

Provide:
1. A concise answer
2. A separate citation section at the end

Format your response exactly like this:

Answer:
<concise answer>

Citation:
[chunk_x | para_y]
[chunk_z | para_a]

Context:
{context}

Question:
{question}
""")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query, db, reranker_model, fetch_k=15, top_k=4):
    """
    Retrieves fetch_k chunks separately using Chroma (vector) and BM25,
    fuses them via simple deduplication, reranks with a CrossEncoder,
    and returns the top_k results.

    Parameters
    ----------
    query          : user question string
    db             : index dict from index.load_index() or index.build_index()
    reranker_model : CrossEncoder instance
    fetch_k        : number of candidates to fetch from each retriever
    top_k          : number of final chunks to return after reranking

    Returns
    -------
    list of LangChain Document objects with relevance_score in metadata
    """
    # Vector retriever
    vector_retriever = db["vectorstore"].as_retriever(search_kwargs={"k": fetch_k})

    # BM25 retriever
    bm25_retriever = db["bm25_retriever"]
    bm25_retriever.k = fetch_k

    # 1. Retrieve separately and merge with simple deduplication
    vector_docs = vector_retriever.invoke(query)
    bm25_docs   = bm25_retriever.invoke(query)

    unique_docs_dict = {doc.metadata.get("chunk_id"): doc for doc in vector_docs + bm25_docs}
    merged_docs = list(unique_docs_dict.values())

    # 2. Cross-encoder rerank
    pairs  = [(query, doc.page_content) for doc in merged_docs]
    scores = reranker_model.predict(pairs)

    for doc, score in zip(merged_docs, scores):
        doc.metadata['relevance_score'] = float(score)

    # 3. Top-K filter
    ranked_docs = sorted(merged_docs, key=lambda x: x.metadata['relevance_score'], reverse=True)
    top_docs    = ranked_docs[:top_k]

    return top_docs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def extract_paragraphs(text):
    """Parse <PARA_NUM:N> tags from a chunk's text and return (para_id, content) pairs."""
    pattern = r"<PARA_NUM:(\d+)>\s*(.*?)(?=<PARA_NUM:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(f"para_{num}", content.strip()) for num, content in matches]


def generate(query, top_docs, llm, print_llm=False):
    """
    Build the formatted prompt from retrieved chunks and call the local Ollama LLM.

    Parameters
    ----------
    query      : user question string
    top_docs   : list of LangChain Documents from retrieve()
    llm        : OllamaLLM instance
    print_llm  : if True, print the full prompt and raw LLM response

    Returns
    -------
    str : raw LLM answer
    """
    # 1. Format Context
    context_text = ""
    for doc in top_docs:
        chunk_id   = doc.metadata.get("chunk_id")
        paragraphs = extract_paragraphs(doc.page_content)
        for para_id, text in paragraphs:
            context_text += f"[{chunk_id} | {para_id}] {text}\n"

    formatted_prompt = prompt_template.format(context=context_text, question=query)

    # 2. Generate Answer
    try:
        answer = llm.invoke(formatted_prompt)
    except httpx.ConnectError:
        raise RuntimeError(
            "[ERROR] Ollama server is not running. "
            "Start it with: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"[ERROR] Generation failed: {e}")

    if print_llm:
        print(f"\n{'='*20} QUERY {'='*20}")
        print(query)
        print(f"\n{'='*20} PROMPT {'='*19}")
        print(formatted_prompt)
        print(f"\n{'='*20} ANSWER {'='*19}")
        print(answer)

    return answer


# ---------------------------------------------------------------------------
# Answer formatting
# ---------------------------------------------------------------------------

def compress_ranges(nums):
    """Convert a list of paragraph number strings into a compact range string, e.g. '3-5, 8'."""
    nums = sorted(set(map(int, nums)))

    ranges = []
    start = prev = nums[0]

    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = n

    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def format_answer(query, answer, top_docs, print_answer=True, trace_mode=False):
    """
    Parse the raw LLM answer, expand citation metadata, and optionally print a
    formatted result to the console.

    Parameters
    ----------
    query        : original user question
    answer       : raw string returned by generate()
    top_docs     : list of LangChain Documents from retrieve()
    print_answer : if True, print the formatted answer + citations to stdout
    trace_mode   : if True, also print the retrieved chunk details

    Returns
    -------
    tuple : (answer_text: str, citation_ret: dict {doc_id: para_range_str})
    """
    # Build metadata lookup: chunk_id → {doc_id, section}
    chunk_lookup = {
        doc.metadata.get("chunk_id"): {
            "doc_id":  doc.metadata.get("doc_id"),
            "section": doc.metadata.get("section"),
        }
        for doc in top_docs
    }

    if print_answer:
        print(f"\n{'='*26} QUERY {'='*26}")
        print(query)

    # Extract answer text (everything before "Citation:")
    answer_text = answer.split("Citation:")[0]
    answer_text = answer_text.replace("Answer:", "").strip()

    # Extract citations from the full answer
    citations = re.findall(r"\[(chunk_\d+)\s*\|\s*(para_\d+)\]", answer)
    citation_grouped = {}

    # If model says cannot find OR no citations exist, treat as abstain
    if "cannot find in sources" in answer_text.lower() or len(citations) == 0:
        answer_text = "Cannot find in sources"

        if print_answer:
            print(f"\n{'='*26} ANSWER {'='*25}")
            print(answer_text)
            print(f"\n{'='*24} CITATIONS {'='*24}")
            print("None")

    else:
        if print_answer:
            print(f"\n{'='*26} ANSWER {'='*25}")
            print(answer_text)
            print(f"\n{'='*24} CITATIONS {'='*24}")

        # Group by (doc_id, section)
        grouped = {}
        for chunk_id, para_id in citations:
            meta     = chunk_lookup.get(chunk_id, {})
            doc_id   = meta.get("doc_id")
            section  = meta.get("section")
            para_num = para_id.replace("para_", "")

            if (doc_id, section) not in grouped:
                grouped[(doc_id, section)] = []
            grouped[(doc_id, section)].append(para_num)

            if doc_id not in citation_grouped:
                citation_grouped[doc_id] = []
            citation_grouped[doc_id].append(para_num)

        if print_answer:
            for (doc_id, section), para_nums in grouped.items():
                para_text = compress_ranges(para_nums)
                print(f"{doc_id} | Section: {section} | Paragraph: {para_text}")

    if print_answer and trace_mode:
        print(f"\n{'='*20} RETRIEVED CHUNKS {'='*20}")
        for d in top_docs:
            print(f"Score: {d.metadata.get('relevance_score'):.2f}")
            print(f"[{d.metadata['chunk_id']} - {d.metadata['doc_id']} - {d.metadata['section']}]")
            print(f"{d.page_content[:300]}...\n")

    citation_ret = {
        doc_id: compress_ranges(para_nums)
        for doc_id, para_nums in citation_grouped.items()
    }

    return answer_text, citation_ret
