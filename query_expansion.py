
from config import client
from models import CHAT_MODEL


def expand_query(query):

    prompt = f"""
    You are a query expansion assistant for a RAG system.

    Generate exactly 2 alternative search queries
    that preserve the original question's intent.

    Use different wording while keeping the meaning unchanged.

    Return ONLY the alternative queries, one per line.
    Do not include numbering or explanations.

    Original query:
    {query}
    """

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )

    expanded_queries = [
        line.strip()
        for line in response.text.splitlines()
        if line.strip()
    ]

    return expanded_queries[:2]
