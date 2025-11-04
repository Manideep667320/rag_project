"""Topic tracking utilities.

Clusters queries into topics using semantic similarity from the same
all-MiniLM-L6-v2 embedding model. Topics are persisted to a JSON file.
"""
import os
import json
from typing import Dict, List


class TopicTracker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", store_path: str = "data/topics.json", threshold: float = 0.7):
        self._model_name = model_name
        self._model = None  # lazy init to avoid torch import at module import time
        self.store_path = store_path
        self.threshold = threshold
        # topics: Dict[str, Dict] -> {topic_name: { 'emb': <tensor>, 'conversations': [ ... ] }}
        self.topics: Dict[str, Dict] = {}
        self._load()

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers.SentenceTransformer import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                self._enc = lambda t: self._model.encode(t, convert_to_tensor=True)
                self._sim = None
            except Exception:
                # Fallback to fastembed (no torch)
                from fastembed import TextEmbedding
                import numpy as np
                _fe = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

                class _Wrapper:
                    def __init__(self, fe):
                        self._fe = fe
                    def encode(self, text, convert_to_tensor=False):
                        if isinstance(text, str):
                            vec = [v for v in self._fe.embed([text])][0]
                            return vec
                        else:
                            return [v for v in self._fe.embed(text)]
                self._model = _Wrapper(_fe)
                self._enc = lambda t: self._model.encode(t)
                self._sim = lambda a, b: float(np.dot(a/ (np.linalg.norm(a)+1e-12), b/ (np.linalg.norm(b)+1e-12)))

    def _load(self):
        if os.path.exists(self.store_path):
            try:
                raw = json.load(open(self.store_path, "r", encoding="utf-8"))
                # stored as text; we keep only conversation lists and placeholder embeddings will be recomputed
                for name, payload in raw.items():
                    self.topics[name] = {"conversations": payload.get("conversations", []), "query": payload.get("query", "")}
            except Exception:
                self.topics = {}

    def _save(self):
        out = {name: {"conversations": data["conversations"], "query": data.get("query","") } for name, data in self.topics.items()}
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    def assign_topic(self, user_query: str, user_text: str, bot_text: str) -> str:
        """Assign the (user,bot) pair to an existing topic or create a new one.

        Returns the topic name assigned.
        """
        self._ensure_model()
        import numpy as np
        q_emb = self._enc(user_query)

        # compute similarity to existing topics by comparing to their representative query
        best_name = None
        best_sim = -1.0
        for name, data in self.topics.items():
            rep = data.get("query") or ""
            if not rep:
                continue
            rep_emb = self._enc(rep)
            if hasattr(self, '_sim') and self._sim is not None:
                sim = self._sim(q_emb, rep_emb)
            else:
                from sentence_transformers import util
                sim = util.cos_sim(q_emb, rep_emb).item()
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_name is not None and best_sim >= self.threshold:
            # assign
            self.topics[best_name]["conversations"].append({"user": user_text, "bot": bot_text})
            self._save()
            return best_name

        # create a new topic (use short user_query as title, truncated)
        title = (user_query[:40] + "...") if len(user_query) > 40 else user_query
        self.topics[title] = {"query": user_query, "conversations": [{"user": user_text, "bot": bot_text}]}
        self._save()
        return title
