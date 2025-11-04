import json
import sys
from pathlib import Path

# add project root to sys.path so `import src.*` works when executed directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest import process_documents
from src.rag_engine import load_vectorstore, create_conversational_chain, run_conversational_query
from src.conversation_manager import create_memory


def main():
    out = {"index": None, "query": None, "errors": []}
    try:
        proc = process_documents()
        out["index"] = True if proc is not None else False
    except Exception as e:
        out["index"] = False
        out["errors"].append(f"index_error: {e}")

    try:
        vs = load_vectorstore('data/index')
        mem = create_memory(k=3)
        chain = create_conversational_chain(vs, memory=mem)
        res = run_conversational_query(chain, vs, 'What is in the docs?', top_k=3)
        out["query"] = {
            "answer_prefix": (res.get("answer") or "")[:120],
            "num_sources": len(res.get("source_docs", [])),
            "confidence": res.get("answer") is not None,
        }
    except Exception as e:
        out["query"] = False
        out["errors"].append(f"query_error: {e}")

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
