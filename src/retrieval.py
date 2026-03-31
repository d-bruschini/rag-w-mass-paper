import faiss
import numpy as np
from config import TOP_K
from embeddings import create_embeddings

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, top_k=3):
    D, I = index.search(np.array([query_embedding]), top_k)
    return I[0]

def create_context(query, index, chunks, pages, title):
    # Process query
    query_emb = create_embeddings([query])[0]
    top_idx = search_index(index, query_emb, top_k=TOP_K)
    context = f"Title: {title}\n\n"
    for i in top_idx:
        context += f"[Page {pages[i]}]\n{chunks[i]}\n\n" # we could also pass the chunk index, but this would be mostly for debugging, rather than to cross-check on the actual original paper
    return context
