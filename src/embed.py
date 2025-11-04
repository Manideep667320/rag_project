from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2", save_path="embeddings/embeddings.pkl"):
    os.makedirs("embeddings", exist_ok=True)
    model = SentenceTransformer(model_name)
    # `chunks` may be a list of strings or a list of dicts with a 'text' field.
    if len(chunks) > 0 and isinstance(chunks[0], dict):
        texts = [c.get("text", "") for c in chunks]
    else:
        texts = chunks
    embeddings = model.encode(texts, show_progress_bar=True)
    with open(save_path, "wb") as f:
        pickle.dump((chunks, embeddings), f)
    print(f"Embeddings saved at {save_path}")
    return embeddings
