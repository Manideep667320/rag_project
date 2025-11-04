import faiss
import numpy as np
import pickle

def build_faiss_index(embeddings, save_path="embeddings/faiss.index"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, save_path)
    print(f"FAISS index stored at {save_path}")
    return index

def load_faiss_index(embedding_path="embeddings/embeddings.pkl", index_path="embeddings/faiss.index"):
    with open(embedding_path, "rb") as f:
        chunks, embeddings = pickle.load(f)
    index = faiss.read_index(index_path)
    return chunks, embeddings, index
