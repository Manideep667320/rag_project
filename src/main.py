from .ingest import process_documents
from .embed import generate_embeddings
from .vectordb import build_faiss_index
from .retrieve import retrieve
from .answer import generate_answer

def main():
    # Step 1: Ingest & split text
    chunks = process_documents()

    # Step 2: Embed chunks
    embeddings = generate_embeddings(chunks)

    # Step 3: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 4: Query the system
    query = input("Enter your query: ")
    results = retrieve(query)

    # Step 5: Generate an answer
    answer = generate_answer(query, results)
    print("\n🧠 RAG Answer:\n", answer)

if __name__ == "__main__":
    main()
