from config import client
from models import CHAT_MODEL


def decide_action(query):

    prompt = f"""
You are the decision-making agent of a RAG chatbot.

Decide what should be done with the user's question.

Return ONLY one of these two labels:

PDF_QUERY
GENERAL_QUERY

Use PDF_QUERY when the question asks for information that is likely
to be present in the uploaded PDF.

Use GENERAL_QUERY when the question is unrelated to the uploaded PDF
or can be answered without searching the PDF.

User question:
{query}
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )

    return response.text.strip()

