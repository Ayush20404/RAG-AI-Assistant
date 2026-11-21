
from chunker import (
    chunk_text,
    semantic_chunk_pages
)

sample_pages = [
    {
        "page": 1,
        "text": """
        Machine learning enables computers to learn from data.
        Deep learning uses neural networks with multiple layers.
        Neural networks are widely used in image recognition.

        Banks use artificial intelligence to detect fraudulent transactions.
        Financial institutions monitor suspicious payments.
        """
    },
    {
        "page": 2,
        "text": """
        Retrieval augmented generation combines information retrieval
        with language model generation.

        Vector databases store embeddings for efficient similarity search.
        """
    }
]

# Test semantic page chunking
semantic_chunks = semantic_chunk_pages(
    sample_pages,
    threshold=0.65,
    max_chunk_size=1000
)

print("\nSEMANTIC CHUNKS")

for i, chunk in enumerate(semantic_chunks, start=1):
    print(f"\nChunk {i} | Page {chunk['page']}")
    print(chunk["text"])

# Validate page metadata
assert all(
    "text" in chunk and "page" in chunk
    for chunk in semantic_chunks
)

assert all(
    chunk["page"] in [1, 2]
    for chunk in semantic_chunks
)

print("\nPage metadata validation passed!")

# Compare with recursive chunking
recursive_chunks = []

for page in sample_pages:
    page_chunks = chunk_text(
        page["text"],
        chunk_size=150,
        overlap=30
    )

    recursive_chunks.extend(page_chunks)

print("\nRECURSIVE CHUNK COUNT:", len(recursive_chunks))
print("SEMANTIC CHUNK COUNT:", len(semantic_chunks))

print("\nFinal validation completed!")
