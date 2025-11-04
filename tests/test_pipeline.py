from src.ingest import process_documents
from src.embed import generate_embeddings
from src.vectordb import build_faiss_index
from src.retrieve import retrieve

def test_pipeline():
    chunks = process_documents()
    embeddings = generate_embeddings(chunks)
    index = build_faiss_index(embeddings)
    results = retrieve("What is AI?")
    assert len(results) > 0
    print("✅ RAG pipeline test passed!")
