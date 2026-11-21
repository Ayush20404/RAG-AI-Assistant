
from embedder import create_embeddings

texts = [
    "Machine learning enables computers to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Financial institutions use AI to detect fraudulent transactions."
]

embeddings = create_embeddings(texts)

print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))

assert len(embeddings) == len(texts)
assert all(len(vector) == len(embeddings[0]) for vector in embeddings)

print("Batch embedding test passed!")
