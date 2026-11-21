
from chunker import (
    split_into_sentences,
    calculate_cosine_similarity,
    semantic_chunk_text
)

# Test 1: Sentence splitting
text = """
Machine learning helps computers learn from data.
Deep learning uses neural networks.
Banks use AI to detect fraudulent transactions.
Financial institutions monitor suspicious payments.
"""

sentences = split_into_sentences(text)

print("\nSentences:")
for sentence in sentences:
    print("-", sentence)

# Test 2: Cosine similarity
vector_a = [1, 0, 0]
vector_b = [1, 0, 0]
vector_c = [0, 1, 0]

print("\nIdentical vectors:", calculate_cosine_similarity(vector_a, vector_b))
print("Orthogonal vectors:", calculate_cosine_similarity(vector_a, vector_c))

# Test 3: Semantic chunking
chunks = semantic_chunk_text(
    text,
    threshold=0.65
)

print("\nSemantic chunks:")

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)

print("\nSemantic chunking test completed!")
