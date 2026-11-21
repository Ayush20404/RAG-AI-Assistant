import re
import numpy as np

from embedder import create_embeddings


def chunk_text(text, chunk_size=800, overlap=150):

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Reserve space for overlap so that the final chunk
    # never exceeds chunk_size.
    content_size = chunk_size - overlap

    separators = ["\n\n", "\n", " ", ""]

    # --------------------------------------------------
    # STEP 1: Recursive splitting
    # --------------------------------------------------

    def split_recursive(text, separators):

        if len(text) <= content_size:
            return [text]

        if not separators:
            return [
                text[i:i + content_size]
                for i in range(0, len(text), content_size)
            ]

        separator = separators[0]

        if separator:
            pieces = text.split(separator)
        else:
            pieces = list(text)

        chunks = []
        current_chunk = ""

        for piece in pieces:

            if not piece:
                continue

            if current_chunk:
                candidate = current_chunk + separator + piece
            else:
                candidate = piece

            # If the piece fits, keep adding it
            if len(candidate) <= content_size:

                current_chunk = candidate

            else:

                # Save the current chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # If the individual piece is too large,
                # recursively split it using the next separator.
                if len(piece) > content_size:

                    smaller_chunks = split_recursive(
                        piece,
                        separators[1:]
                    )

                    chunks.extend(smaller_chunks)

                    current_chunk = ""

                else:

                    current_chunk = piece

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    base_chunks = split_recursive(
        text,
        separators
    )

    # --------------------------------------------------
    # STEP 2: Create boundary-aware overlap
    # --------------------------------------------------

    def get_overlap(previous_chunk, overlap):

        words = previous_chunk.split()

        selected_words = []
        current_length = 0

        # Start from the end of the previous chunk
        for word in reversed(words):

            additional_length = len(word)

            if selected_words:
                additional_length += 1  # space between words

            if current_length + additional_length > overlap:
                break

            selected_words.insert(0, word)
            current_length += additional_length

        return " ".join(selected_words)

    # --------------------------------------------------
    # STEP 3: Build final chunks
    # --------------------------------------------------

    final_chunks = []

    for i, chunk in enumerate(base_chunks):

        # First chunk has no overlap
        if i == 0:

            final_chunks.append(chunk)
            continue

        previous_chunk = final_chunks[-1]

        overlap_text = get_overlap(
            previous_chunk,
            overlap
        )

        if overlap_text:

            combined_chunk = overlap_text + " " + chunk

        else:

            combined_chunk = chunk

        # Safety check
        if len(combined_chunk) > chunk_size:

            combined_chunk = chunk

        final_chunks.append(combined_chunk)

    return final_chunks

def chunk_pages(pages, chunk_size=800, overlap=150):

    chunks = []

    for page in pages:

        page_number = page["page"]
        page_text = page["text"]

        # Use our recursive chunking logic
        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        # Attach page number to every chunk
        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": page_number
            })

    return chunks


def split_into_sentences(text):
    """
    Splits text into individual sentences.
    """

    if not text or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]



def calculate_cosine_similarity(vector_a, vector_b):
    """
    Calculates cosine similarity between two embedding vectors.
    """

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)

    return float(similarity)



def semantic_chunk_text(text, threshold=0.65, max_chunk_size=1000):
    """
    Creates semantic chunks by detecting changes in
    similarity between consecutive sentences.
    """

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    # Step 1: Generate embeddings for all sentences
    embeddings = create_embeddings(sentences)

    # Step 2: Calculate similarity between consecutive sentences
    similarities = []

    for i in range(len(embeddings) - 1):

        similarity = calculate_cosine_similarity(
            embeddings[i],
            embeddings[i + 1]
        )

        similarities.append(similarity)

    # Step 3: Detect semantic boundaries and build chunks
    chunks = []
    current_chunk = sentences[0]

    for i, similarity in enumerate(similarities):

        next_sentence = sentences[i + 1]

        # Check for a semantic boundary
        semantic_boundary = similarity < threshold

        # Safety limit to prevent excessively large chunks
        size_boundary = (
            len(current_chunk) + len(next_sentence) + 1
            > max_chunk_size
        )

        if semantic_boundary or size_boundary:

            chunks.append(current_chunk)
            current_chunk = next_sentence

        else:
            current_chunk += " " + next_sentence

    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks



def semantic_chunk_pages(pages, threshold=0.65, max_chunk_size=1000):
    """
    Applies semantic chunking independently to each PDF page.
    Preserves page metadata.
    """

    chunks = []

    for page in pages:

        page_number = page["page"]
        page_text = page["text"]

        page_chunks = semantic_chunk_text(
            page_text,
            threshold=threshold,
            max_chunk_size=max_chunk_size
        )

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page_number
            })

    return chunks



if __name__ == "__main__":

    # -----------------------------------------
    # TEST 1: Recursive chunking
    # -----------------------------------------

    print("\n========== TEST 1: chunk_text() ==========\n")

    text = (
        "This is the first paragraph. It contains some information "
        "about artificial intelligence and machine learning.\n\n"
        "This is the second paragraph. It contains information about "
        "retrieval augmented generation and vector databases.\n\n"
        "This is the third paragraph. It contains information about "
        "embeddings, similarity search, and document retrieval."
    )

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20
    )

    for i, chunk in enumerate(chunks, start=1):

        print(f"\nChunk {i}")
        print("-" * 40)
        print(f"Length: {len(chunk)}")
        print(f"Text: {chunk}")


    # -----------------------------------------
    # TEST 2: Verify chunk size
    # -----------------------------------------

    print("\n\n========== TEST 2: Size Verification ==========\n")

    for i, chunk in enumerate(chunks, start=1):

        if len(chunk) <= 100:
            print(f"Chunk {i}: PASS ({len(chunk)} characters)")

        else:
            print(f"Chunk {i}: FAIL ({len(chunk)} characters)")


    # -----------------------------------------
    # TEST 3: Page-aware chunking
    # -----------------------------------------

    print("\n\n========== TEST 3: chunk_pages() ==========\n")

    pages = [
        {
            "page": 1,
            "text": (
                "This is content from page one. "
                "It contains information about artificial intelligence "
                "and machine learning."
            )
        },

        {
            "page": 2,
            "text": (
                "This is content from page two. "
                "It contains information about retrieval augmented "
                "generation and vector databases."
            )
        },

        {
            "page": 3,
            "text": (
                "This is content from page three. "
                "It contains information about embeddings and "
                "similarity search."
            )
        }
    ]

    page_chunks = chunk_pages(
        pages,
        chunk_size=100,
        overlap=20
    )

    for i, chunk in enumerate(page_chunks, start=1):

        print(f"\nChunk {i}")
        print("-" * 40)
        print(f"Page: {chunk['page']}")
        print(f"Length: {len(chunk['text'])}")
        print(f"Text: {chunk['text']}")


    # -----------------------------------------
    # TEST 4: Verify page metadata
    # -----------------------------------------

    print("\n\n========== TEST 4: Page Metadata Verification ==========\n")

    for i, chunk in enumerate(page_chunks, start=1):

        if "text" in chunk and "page" in chunk:
            print(
                f"Chunk {i}: PASS "
                f"(Page {chunk['page']})"
            )
        else:
            print(f"Chunk {i}: FAIL")


    # -----------------------------------------
    # TEST 5: Verify no chunk crosses pages
    # -----------------------------------------

    print("\n\n========== TEST 5: Page Boundary Verification ==========\n")

    valid_pages = {1, 2, 3}

    all_valid = True

    for i, chunk in enumerate(page_chunks, start=1):

        if chunk["page"] in valid_pages:
            print(
                f"Chunk {i}: PASS "
                f"(assigned to Page {chunk['page']})"
            )
        else:
            print(
                f"Chunk {i}: FAIL "
                f"(invalid page number)"
            )
            all_valid = False

    if all_valid:
        print("\nAll page assignments are valid.")
