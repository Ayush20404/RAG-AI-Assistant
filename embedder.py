from config import client
from models import EMBEDDING_MODEL




def create_embeddings(chunks):
    """
    Generates embeddings for all chunks.
    """

    embeddings = []

    total = len(chunks)


    for i, chunk in enumerate(chunks, start=1):

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk
        )

        embeddings.append(
            response.embeddings[0].values
        )


    return embeddings