from config import MODEL, TEMPERATURE
from retrieval import search_index
from embeddings import create_embeddings


def qa_prompt(context, question):
    return f"""
You are a research assistant. Use ONLY the context below to answer the question. Add citations to the answer.
Context:
{context}

Question: {question}

Answer:
"""

class LLM:
    def __init__(self, client):
        self.client = client

    def response(self, query, context, history=[], streaming = False):
        # Prompt LLM
        prompt = qa_prompt(context, query)

        response = self.client.chat.completions.create(
            model = MODEL,
            temperature = TEMPERATURE,
            messages = [*history, {"role": "user", "content": prompt}],
            stream = streaming
        )
        return response
