import faiss
import numpy as np

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, top_k=3):
    D, I = index.search(np.array([query_embedding]), top_k)
    return I[0]
