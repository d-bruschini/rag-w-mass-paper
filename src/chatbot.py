from config import MODEL, TEMPERATURE, TOP_K, MAX_CONTEXT_LENGTH
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

class Chatbot:
    def __init__(self, client, index, chunks, pages, title):
        self.client = client
        self.index = index
        self.chunks = chunks
        self.pages = pages
        self.title = title
        self.context = []

    def maintain_context(self):
        if len(self.context) > MAX_CONTEXT_LENGTH * 2:  # 2 for question and answer pairs
            self.context = self.context[-(MAX_CONTEXT_LENGTH * 2):]

    def chat(self, query):
        # Process query
        query_emb = create_embeddings([query])[0]
        top_idx = search_index(self.index, query_emb, top_k=TOP_K)
        context = f"Title: {self.title}\n\n"
        for i in top_idx:
            context += f"[Page {self.pages[i]}]\n{self.chunks[i]}\n\n" # we could also pass the chunk index, but this would be mostly for debugging, rather than to cross-check on the actual original paper

        # Prompt LLM
        prompt = qa_prompt(context, query)

        response = self.client.chat.completions.create(
            model = MODEL,
            temperature = TEMPERATURE,
            messages = [{"role": "user", "content": prompt}]
        )
        response_content = response.choices[0].message.content.strip() if response.choices else "Sorry, I couldn't generate an answer."
        self.maintain_context()
        self.context.append({"role":"user", "content": query}) # we retain only the query, and not the whole prompt with the context
        self.context.append({"role":"assistant", "content": response_content})
        print(f"CHATBOT: {response_content}")
        return response_content
