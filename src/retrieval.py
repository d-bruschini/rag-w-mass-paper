import faiss
import numpy as np
from config import TOP_K
from embeddings import create_embeddings

def format_page(page):
    roman_map = {-1: "i", 0: "ii"}
    
    return roman_map.get(page - 1, page - 1)

def normalize(embeddings):
    # Normalize embeddings to 1 and use float32 precision (normalize_L2 works for dtype float32. Also, no change was observed when the precision is reduced from float64 in the default configuration)
    embeddings_f32 = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings_f32)
    return embeddings_f32

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    # Normalize embeddings to 1 (and use float32)
    embeddings_f32 = normalize(embeddings)
    index.add(embeddings_f32)
    return index

def search_index(index, query_embedding, top_k=3):
    D, I = index.search(np.array([query_embedding]), top_k)
    return I[0]

def create_context(query, index, docs):
    # Process query
    query_emb = create_embeddings([query])
    # Normalize query embedding to 1 (and use float32)
    query_emb_f32 = normalize(query_emb)
    top_idx = search_index(index, query_emb_f32[0], top_k=TOP_K)
    title = docs[0].metadata.get("title","unknown")
    context = f"Title: {title}\n\n"
    for i in top_idx:
        page = docs[i].metadata.get("page","unknown")
        page = format_page(page)
        text = docs[i].page_content
        context += f"[Page {page}]\n{text}\n\n" # we could also pass the chunk index, but this would be mostly for debugging, rather than to cross-check on the actual original paper
    return context
