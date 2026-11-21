
from config import client
from models import EMBEDDING_MODEL


def create_embeddings(chunks, batch_size=32):
    """
    Generates embeddings for multiple text chunks in batches.
    Returns a list of embedding vectors in the original order.
    """

    embeddings = []

    if not chunks:
        return embeddings

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i + batch_size]

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch
        )

        batch_embeddings = [
            embedding.values
            for embedding in response.embeddings
        ]

        embeddings.extend(batch_embeddings)

    return embeddings
