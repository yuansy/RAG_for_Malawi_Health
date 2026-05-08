import glob
import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Hardcoded section matching rules for known inconsistencies in specific booklets.
# These are used when automatic text matching fails due to formatting issues.
# ---------------------------------------------------------------------------
HARDCODED_SECTION_RULES = {
    "TG Booklet 1": {
        "Annex A: IDSR matrix: Core functions and activities by health system level": 309,  # Section title is concatenated with additional content on the same row.
    },
    "TG Booklet 2": {
        "2.5 Data protection and security to protect patients confidentially": 655,  # Section appears without numbering prefix ("2.5").
        "Annex 2H: IDSR weekly/monthly summary reporting form": 693,  # Minor text mismatch: "form" (TOC) vs "forms" (content).
        "3.2.3 Analyse data by person": 820,  # Spelling difference: "Analyse" (TOC) vs "Analyze" (content).
    },
    "TG Booklet 3": {
        "Annex 6F: Sample messages for community education": 1200,  # Section title is appended at the end of the previous paragraph instead of starting a new row.
    },
    "TG Booklet 4": {
        "Annex 8A: Indicators for monitoring IDSR core functions at the health facility level": 408,  # section text is appended to the row
        "Annex 8B: Indicators for monitoring IDSR core functions at the district level": 410,  # section text is appended to the row
        "ANNEX 8E: Monitoring chart IDSR performance indicators at district": 418,  # section text is appended to the row
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_booklets(folder_pattern):
    """
    Load the booklets from the provided paths.
    Assumes the Excel files have columns that map to 'para_num' and 'text'.

    Parameters
    ----------
    folder_pattern : glob pattern used to locate the Excel booklet files

    Returns
    -------
    list of booklet dicts
    """
    booklet_paths = sorted(glob.glob(folder_pattern))
    if not booklet_paths:
        raise ValueError(f"No Excel booklet files found. Please check {folder_pattern}.")

    docs = []
    for path in booklet_paths:
        try:
            df = pd.read_excel(path, header=None)
            df.columns = ['para_num', 'text']

            # Clean text column
            df['text'] = (
                df['text']
                .astype(str)
                .str.replace(r'\s+', ' ', regex=True)
                .str.replace(r'[^\x20-\x7E]', '', regex=True)  # keep only standard printable ASCII
                .str.strip()
                .str.strip('"')
            )

            doc_id = os.path.basename(path).replace('.xlsx', '')
            docs.append({
                "doc_id": doc_id,
                "rows": df.to_dict('records')
            })

            print(f"Successfully loaded: {doc_id}")

        except Exception as e:
            print(f"Error loading {path}: {e}")

    return docs


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def extract_section_names(rows):
    """
    Extract the table of contents.
    Stops when the actual document content begins (i.e., when the first TOC item repeats).

    Parameters
    ----------
    rows : row dicts for a single booklet

    Returns
    -------
    toc_sections : list of section title strings extracted from the TOC
    start_idx    : index into `rows` where the actual document content begins
    """
    toc_sections = []
    in_toc = False
    start_idx = 0

    for i, row in enumerate(rows):
        text = str(row.get('text', ''))

        if text.lower() in ["table of contents", "contents"]:
            in_toc = True
            continue

        if in_toc:
            # Remove trailing numbers or Roman numerals (and any dots/spaces before them)
            clean_text = re.sub(r'[\.\t\s]+([ivxlcdm]+|\d+)$', '', text, flags=re.IGNORECASE).strip()

            # If we see the very first TOC entry again, it means the TOC is over
            if toc_sections and clean_text.lower() == toc_sections[0].lower():
                start_idx = i
                break

            # Skip figure entries (not actual sections)
            if clean_text.startswith("Figure"):
                continue

            toc_sections.append(clean_text)

    if not toc_sections:
        raise ValueError("No table of contents sections were extracted.")

    return toc_sections, start_idx


def get_metadata_chunk(rows, start_idx, doc_id):
    """
    Extract everything before the content starts into a single Metadata chunk.

    Parameters
    ----------
    rows      : row dicts for a single booklet
    start_idx : index marking where content begins
    doc_id    : booklet identifier

    Returns
    -------
    chunk dict of metadata and TOC
    """
    metadata_text = []
    for row in rows[:start_idx]:
        para_num = row.get('para_num', '')
        text = str(row.get('text', ''))
        metadata_text.append(f"<PARA_NUM:{para_num}> {text}")

    if metadata_text:
        return {
            "doc_id": doc_id,
            "section": "Metadata & Table of Contents",
            "text": "\n".join(metadata_text),
        }
    return None


def get_section_chunks(rows, start_idx, section_names, doc_id):
    """
    Process the actual document content sequentially against the TOC.

    Iterates through every row from start_idx onwards and advances to the
    next expected section title using exact text matching or a hardcoded
    paragraph-number rule (via HARDCODED_SECTION_RULES).

    Parameters
    ----------
    rows          : row dicts for a single booklet
    start_idx     : index marking where content begins
    section_names : ordered TOC section titles returned by extract_section_names
    doc_id        : booklet identifier

    Returns
    -------
    list of chunk dict per TOC section
    """
    chunks = []
    current_section = section_names[0]
    next_section_idx = 1
    current_chunk_text = []

    def save_chunk():
        nonlocal current_chunk_text, current_section
        if current_chunk_text:
            chunks.append({
                "doc_id": doc_id,
                "section": current_section,
                "text": "\n".join(current_chunk_text),
            })
            current_chunk_text = []

    # Start iterating exactly where the TOC ends
    for row in rows[start_idx:]:
        para_num = str(row.get('para_num', ''))
        text = str(row.get('text', ''))

        # Sequential matching: Only check against the next expected section
        if next_section_idx < len(section_names):
            expected_next_section = section_names[next_section_idx]

            # Condition 1: If there's a hardcoded rule, rely strictly on the para_num
            in_hardcoded = (
                doc_id in HARDCODED_SECTION_RULES
                and expected_next_section in HARDCODED_SECTION_RULES[doc_id]
                and para_num == str(HARDCODED_SECTION_RULES[doc_id][expected_next_section])
            )

            # Condition 2: Rely on the exact text matching
            match_text = text.strip(".") == expected_next_section

            # If we found the next section, save current chunk and move pointers forward
            if in_hardcoded or match_text:
                save_chunk()
                current_section = expected_next_section
                next_section_idx += 1

        current_chunk_text.append(f"<PARA_NUM:{para_num}> {text}")

    save_chunk()
    return chunks


# ---------------------------------------------------------------------------
# Chunking orchestration
# ---------------------------------------------------------------------------

def chunk_document_sections(docs):
    """
    Process the loaded documents, extract TOC, and chunk by section.

    Parameters
    ----------
    docs : list of booklet dicts

    Returns
    -------
    list of chunk dict with keys "doc_id", "section", and "text"

    """
    all_chunks = []

    for doc in docs:
        doc_id = doc['doc_id']
        rows = doc['rows']

        # Extract section names and where the actual content starts
        section_names, start_idx = extract_section_names(rows)
        print(f"\nExtracted {len(section_names)} sections from {doc_id}.")

        doc_chunks = []

        # Step A: Create a single chunk for everything before the content
        metadata_chunk = get_metadata_chunk(rows, start_idx, doc_id)
        if metadata_chunk:
            doc_chunks.append(metadata_chunk)

        # Step B: Process the actual document content
        section_chunks = get_section_chunks(rows, start_idx, section_names, doc_id)
        doc_chunks.extend(section_chunks)

        # Validation: Check if chunk count exactly matches expected
        expected_count = len(section_names) + 1
        if len(doc_chunks) != expected_count:
            raise ValueError(
                f"\n[ERROR] Section mismatch in '{doc_id}'!\n"
                f"Expected {expected_count} chunks (Metadata + {len(section_names)} sections).\n"
                f"Generated {len(doc_chunks)} chunks instead.\n"
                f"ACTION REQUIRED: Please check the document manually and add the missing "
                f"section to 'HARDCODED_SECTION_RULES' in ingest.py."
            )

        print(f"Verified: {doc_id} generated exactly {len(doc_chunks)} section chunks.")
        all_chunks.extend(doc_chunks)

    return all_chunks


def subchunk_large_sections(chunks, max_words=800, min_words=200, overlap_words=100):
    """
    Split oversized section chunks into smaller sub-chunks with overlap.

    Sections at or below max_words are kept intact. Larger sections are
    broken down paragraph by paragraph according to three cases:

    - Case 1 – Long paragraph (> 800 words):
        Split internally using a sliding window with a 100-word overlap
    - Case 2 – Medium paragraph (200-800 words):
        Becomes its own standalone chunk
    - Case 3 – Short paragraph (< 200 words):
        Accumulated into a buffer with neighbouring paragraphs; flushed
        with overlap when the buffer would exceed max_words

    Parameters
    ----------
    chunks        : list of chunk dicts from Phase 1
    max_words     : maximum word count per output chunk
    min_words     : minimum word count that qualifies a paragraph as "medium"
    overlap_words : number of words carried over as overlap between consecutive sub-chunks to preserve context at boundaries

    Returns
    -------
    list of chunk dict with keys "doc_id", "section", and "text"

    """
    final_chunks = []
    large_chunks_count = 0

    for chunk in chunks:
        text = chunk['text']
        total_words = len(text.split())

        # If the entire section is small, keep it intact
        if total_words <= max_words:
            final_chunks.append(chunk)
            continue

        # If section is too large, break it down by paragraph blocks
        large_chunks_count += 1
        paragraphs = [p.strip() for p in re.split(r'(?=<PARA_NUM:.*?>)', text) if p.strip()]
        section_name = f"<SECTION:{chunk['section']}>"

        buffer = []
        buffer_words = 0

        def flush_buffer(keep_overlap=False):
            nonlocal buffer, buffer_words, final_chunks
            if buffer:
                new_chunk = chunk.copy()
                joined_text = "\n".join(buffer)
                new_chunk['text'] = f"{section_name}\n{joined_text}"
                final_chunks.append(new_chunk)

                if keep_overlap:
                    overlap_buffer = []
                    overlap_words_count = 0

                    # Walk backwards through the flushed buffer to extract overlap
                    for para in reversed(buffer):
                        p_words = len(para.split())
                        # Guarantee at least 1 paragraph overlaps, or as many as fit in overlap_words
                        if overlap_words_count + p_words <= overlap_words or not overlap_buffer:
                            overlap_buffer.insert(0, para)
                            overlap_words_count += p_words
                        else:
                            break

                    buffer = overlap_buffer
                    buffer_words = overlap_words_count
                else:
                    buffer = []
                    buffer_words = 0

        for para_block in paragraphs:
            p_words = len(para_block.split())

            # Extract para_num strictly, supporting internal newlines with re.DOTALL
            match = re.match(r'^<PARA_NUM:(.*?)>\s*(.*)', para_block, re.DOTALL)
            if not match:
                raise ValueError(
                    f"Paragraph does not match the <PARA_NUM:para_num> pattern. "
                    f"Text snippet: {para_block[:100]}"
                )

            para_num = match.group(1)
            para_content = match.group(2)

            # Case 1: Long paragraph
            if p_words > max_words:
                flush_buffer()  # Flush any accumulated short paragraphs first

                words = para_content.split()
                step = max_words - overlap_words
                for i in range(0, len(words), step):
                    sub_text = " ".join(words[i:i + max_words])
                    new_chunk = chunk.copy()
                    new_chunk['text'] = f"{section_name}\n<PARA_NUM:{para_num}> {sub_text}"
                    final_chunks.append(new_chunk)

            # Case 2: Medium paragraph
            elif min_words <= p_words <= max_words:
                flush_buffer()  # Flush any accumulated short paragraphs first

                new_chunk = chunk.copy()
                new_chunk['text'] = f"{section_name}\n{para_block}"
                final_chunks.append(new_chunk)

            # Case 3: Short paragraph
            else:
                # Flush if adding this paragraph to the buffer would blow past the max limit
                if buffer_words > 0 and (buffer_words + p_words) > max_words:
                    flush_buffer(keep_overlap=True)

                buffer.append(para_block)
                buffer_words += p_words

        # Flush any remaining small paragraphs at the end of the section
        flush_buffer()

    print(f"\nInitial total chunks: {len(chunks)}")
    print(f"Large chunks (> {max_words} words) that were sub-chunked: {large_chunks_count}")

    return final_chunks


def chunk_documents(docs, print_chunks=False):
    """
    Full chunking pipeline: section-level chunking followed by sub-chunking.

    Phase 1 – split each booklet by TOC section.
    Phase 2 – break oversized sections down further.
    Finally assigns a sequential "chunk_id" (e.g. "chunk_1") to every output chunk.

    Parameters
    ----------
    docs         : list of booklet dicts from load_booklets()
    print_chunks : if True, prints the first 10 chunks to stdout for a quick sanity check

    Returns
    -------
    final list of chunk dict with keys "chunk_id", "doc_id", "section", and "text"

    """
    # Phase 1: Initial Section Chunking
    base_chunks = chunk_document_sections(docs)

    # Phase 2: Sub-chunking Long Sections
    final_chunks = subchunk_large_sections(base_chunks, max_words=800, min_words=200, overlap_words=100)

    # Assign Final Chunk IDs
    for i, chunk in enumerate(final_chunks, 1):
        chunk['chunk_id'] = f"chunk_{i}"

    print(f"\nFinal Total Chunks Generated: {len(final_chunks)}\n")

    # Print the first few chunks to verify the structure and logic
    if print_chunks:
        for chunk in final_chunks[:10]:
            print("-" * 60)
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Doc ID:   {chunk['doc_id']}")
            print(f"Section:  {chunk['section']}")
            print(f"Text:\n{chunk['text'][:150]}...\n")

    return final_chunks
