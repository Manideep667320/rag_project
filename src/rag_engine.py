"""RAG engine: build retriever and QA chain using LangChain + FAISS.

Provides helpers to load a saved FAISS index, create a Retriever, compute
confidence scores (cosine similarity) for the top chunks, and construct a
RetrievalQA chain using ChatOpenAI (requires OpenAI API key in env).
"""
import os
from typing import List, Dict

# Import langchain chain/llm classes lazily inside functions to remain
# compatible with environments that have different langchain versions.
ChatOpenAI = None
ConversationalRetrievalChain = None

import chromadb
from chromadb.config import Settings
try:
    from langchain.schema import Document as LC_Document
except Exception:
    # fallback to a simple local document if langchain.schema is unavailable
    class LC_Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class ChromaStore:
    """A lightweight wrapper around a chromadb collection providing
    similarity_search and as_retriever() compatibility used by the app.
    """
    def __init__(self, client, collection, embed_model_name: str = "all-MiniLM-L6-v2"):
        self.client = client
        self.collection = collection
        self._embed_model_name = embed_model_name
        self.embed_model = None  # created lazily on first encode

    def _ensure_embedder(self):
        if self.embed_model is None:
            try:
                from sentence_transformers.SentenceTransformer import SentenceTransformer  # avoid heavy __init__
                self.embed_model = SentenceTransformer(self._embed_model_name)
            except Exception as e:
                # Fallback to fastembed (no torch)
                try:
                    from fastembed import TextEmbedding
                    _fe = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

                    class _FastEmbedWrapper:
                        def __init__(self, fe):
                            self._fe = fe
                        def encode(self, texts, convert_to_numpy=True):
                            if isinstance(texts, str):
                                texts = [texts]
                            vecs = [v for v in self._fe.embed(texts)]
                            if convert_to_numpy:
                                return vecs if len(texts) > 1 else vecs[0]
                            return vecs
                    self.embed_model = _FastEmbedWrapper(_fe)
                except Exception as e2:
                    raise RuntimeError("Embedding backend not available. Install either:\n"
                                       " - PyTorch CPU + sentence-transformers, or\n"
                                       " - fastembed (pip install fastembed)\n") from e2

    def similarity_search(self, query: str, k: int = 3):
        # compute query embedding
        self._ensure_embedder()
        q_emb = self.embed_model.encode(query, convert_to_numpy=True).tolist()
        resp = self.collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
        docs = []
        if resp and isinstance(resp, dict):
            documents = resp.get("documents", [[]])[0]
            metadatas = resp.get("metadatas", [[]])[0]
            for doc_text, md in zip(documents, metadatas):
                class _D:
                    pass
                d = _D()
                d.page_content = doc_text
                d.metadata = md or {}
                docs.append(d)
        return docs

    def as_retriever(self, search_type: str = "similarity", search_kwargs: dict = None):
        # Return an object with get_relevant_documents(query) for LangChain
        search_kwargs = search_kwargs or {"k": 3}
        wrapper = self

        class Retriever:
            def get_relevant_documents(self, query: str):
                k = search_kwargs.get("k", 3)
                docs = wrapper.similarity_search(query, k=k)
                # convert to LangChain Document objects
                lc_docs = []
                for d in docs:
                    lc_docs.append(LC_Document(page_content=d.page_content, metadata=d.metadata))
                return lc_docs

        return Retriever()


def load_vectorstore(index_path: str = "data/index") -> ChromaStore:
    """Load or open a Chroma collection persisted at the given path and return a ChromaStore."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index path not found: {index_path}")
    client = chromadb.PersistentClient(path=os.path.abspath(index_path))
    collection = client.get_collection(name="rag_collection")
    return ChromaStore(client, collection)


def get_retriever(vectorstore: ChromaStore, k: int = 3):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_qa_chain(retriever, model_name: str = "gpt-3.5-turbo"):
    """Create a RetrievalQA chain using ChatOpenAI when available, otherwise OpenAI LLM wrapper.

    Note: ChatOpenAI requires environment configuration (OPENAI_API_KEY) and langchain support.
    """
    # import langchain LLM/chain classes lazily
    try:
        from langchain.chains import RetrievalQA
    except Exception:
        RetrievalQA = None

    # Prefer modern provider packages first
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception:
        try:
            from langchain.llms import ChatOpenAI  # legacy
        except Exception:
            ChatOpenAI = None

    try:
        from langchain.llms import OpenAI  # legacy fallback
    except Exception:
        OpenAI = None

    if ChatOpenAI is not None:
        llm = ChatOpenAI(model=model_name)
    elif OpenAI is not None:
        llm = OpenAI(model_name=model_name)
    else:
        raise RuntimeError("No supported LLM wrapper found in LangChain (ChatOpenAI/OpenAI).")

    if RetrievalQA is None:
        raise RuntimeError("RetrievalQA not available in this LangChain version.")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain


def retrieve_with_confidence(query: str, vectorstore: ChromaStore, top_k: int = 3) -> List[Dict]:
    """Retrieve top_k documents for `query` and compute cosine similarity confidence.

    Returns a list of dicts: { 'text', 'source', 'score' } where score is a float in [0,1].
    """
    # get top docs using the vectorstore (fast) then compute accurate similarity
    docs = vectorstore.similarity_search(query, k=top_k)

    import numpy as np
    # Try sentence-transformers first, then fastembed
    try:
        from sentence_transformers.SentenceTransformer import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_vec = model.encode(query, convert_to_numpy=True)
        def enc(txt):
            return model.encode(txt, convert_to_numpy=True)
    except Exception:
        from fastembed import TextEmbedding
        _fe = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        q_vec = [v for v in _fe.embed([query])][0]
        def enc(txt):
            return [v for v in _fe.embed([txt])][0]
    # normalize
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-12)

    results = []
    for d in docs:
        chunk_text = d.page_content
        # embed chunk and compute cosine similarity
        c_vec = enc(chunk_text)
        c_norm = c_vec / (np.linalg.norm(c_vec) + 1e-12)
        sim = float(np.clip(np.dot(q_norm, c_norm), -1.0, 1.0))
        # normalize to 0..1
        conf = (sim + 1.0) / 2.0
        results.append({
            "text": chunk_text,
            "source": d.metadata.get("source"),
            "score": float(conf),
        })

    return results


def create_conversational_chain(vectorstore: ChromaStore, memory=None, model_name: str = "gpt-3.5-turbo"):
    """Create a ConversationalRetrievalChain wired to the provided vectorstore and memory.

    Memory should be a LangChain memory object (e.g., ConversationBufferWindowMemory).
    """
    try:
        from langchain.chains import ConversationalRetrievalChain
    except Exception:
        ConversationalRetrievalChain = None

    try:
        from langchain.llms import ChatOpenAI
    except Exception:
        ChatOpenAI = None

    try:
        from langchain.llms import OpenAI
    except Exception:
        OpenAI = None

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # LangChain-first path (with graceful failure to fallback)
    if ConversationalRetrievalChain is not None and (ChatOpenAI is not None or OpenAI is not None):
        try:
            if ChatOpenAI is not None:
                llm = ChatOpenAI(model=model_name)
            else:
                llm = OpenAI(model_name=model_name)  # type: ignore
            chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
            return chain
        except Exception:
            # e.g., provider import present but API key missing; fall back locally
            pass

    # Local fallback conversational chain
    class LocalConversationalChain:
        def __init__(self, retriever, memory):
            self.retriever = retriever
            self.memory = memory

        def __call__(self, inputs: dict):
            question = inputs.get("question") or inputs.get("input")
            docs = self.retriever.get_relevant_documents(question)
            # naive answer: return most relevant chunk text or a friendly message
            answer = docs[0].page_content if docs else "I couldn't find relevant context to answer that yet."
            # update memory if it supports save_context
            try:
                if hasattr(self.memory, "save_context"):
                    self.memory.save_context({"question": question}, {"answer": answer})
            except Exception:
                pass
            return {"answer": answer, "source_documents": docs}

    return LocalConversationalChain(retriever, memory)


def run_conversational_query(chain, vectorstore: ChromaStore, question: str, top_k: int = 3) -> Dict:
    """Run a conversational query through the given chain and return structured results.

    Returns: { 'answer': str, 'source_docs': [{text, source, score}], 'raw': chain_output }
    """
    # Execute conversational chain (it will update memory internally)
    out = chain.ask(question) if hasattr(chain, 'ask') else chain({'question': question})
    # LangChain versions vary: try common keys
    answer = None
    source_docs = []
    if isinstance(out, dict):
        answer = out.get('answer') or out.get('output_text') or out.get('result')
        docs = out.get('source_documents') or out.get('docs') or []
    else:
        # fallback: string
        answer = str(out)
        docs = []

    # Compute per-chunk confidence using the vectorstore
    if True:
        # Always compute per-chunk confidence for the asked question using current vectorstore
        source_docs = retrieve_with_confidence(question, vectorstore, top_k=top_k)

    return {"answer": answer, "source_docs": source_docs, "raw": out}
