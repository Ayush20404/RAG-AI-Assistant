from config import client


def rewrite_query(query, history=None):
    """
    Rewrites the user's query into a clear standalone question.

    If conversation history is provided, it is used to understand
    references to previous messages.
    """

    if history is None:
        history = []

    conversation = ""

    for message in history:
        conversation += f"{message['role']}: {message['content']}\n"

    prompt = f"""
You are a query rewriting assistant for a PDF-based RAG chatbot.

Your job is to rewrite the user's latest question into a clear,
standalone question that can be used for document retrieval.

Use the conversation history when necessary.

Rules:
- If the question is already clear, keep its meaning unchanged.
- Resolve references such as "it", "they", "this", "that", etc.
  using the conversation history.
- Do not answer the question.
- Return ONLY the rewritten question.

Conversation history:
{conversation}

Latest user question:
{query}

Rewritten question:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip()