import faiss
import numpy as np


def create_vector_store(embeddings):
    """
    Creates and returns a FAISS vector index.
    """

    vectors = np.array(embeddings).astype("float32")

    print("Embeddings type:", type(embeddings))
    print("Number of embeddings:", len(embeddings))
    print("Vectors shape:", vectors.shape)
    print("Vectors ndim:", vectors.ndim)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index