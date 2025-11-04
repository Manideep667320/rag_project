import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os


def _normalize(scores):
    scores = np.array(scores, dtype=float)
    minv = float(np.min(scores))
    maxv = float(np.max(scores))
    if maxv - minv < 1e-12:
        return [0.0 for _ in scores]
    return [(float(s) - minv) / (maxv - minv) for s in scores]


def retrieve(query, top_k=3, re_rank='cosine', bm25_weight=0.0, cross_encoder_model=None):
    """Retrieve relevant chunks for `query`.

    Parameters:
    - query: str
    - top_k: number of final results to return
    - re_rank: 'cosine' | 'bm25' | 'cross_encoder' (default 'cosine')
    - bm25_weight: when >0, combine BM25 score with cosine (weight in [0..1])
    - cross_encoder_model: optional model name for CrossEncoder re-ranker

    Returns: list of dicts: {'id': idx, 'text': str, 'score': float, 'source': str, 'chunk_id': str}
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # load stored chunks and embeddings
    with open(os.path.join("embeddings", "embeddings.pkl"), "rb") as f:
        chunks, embeddings = pickle.load(f)

    index = faiss.read_index(os.path.join("embeddings", "faiss.index"))

    query_embedding = model.encode([query])[0]

    # choose how many candidates to fetch initially
    if re_rank == 'cross_encoder':
        initial_k = max(top_k, 50)
    else:
        initial_k = top_k

    distances, indices = index.search(np.array([query_embedding]), initial_k)
    ids = indices[0].tolist()

    # build candidate list
    candidates = []
    candidate_embeddings = []
    for i in ids:
        # defensive: skip invalid ids
        if i < 0 or i >= len(chunks):
            continue
        c = chunks[i]
        # chunk text may be stored as dict or raw string
        if isinstance(c, dict):
            text = c.get('text', '')
            source = c.get('source')
            chunk_id = c.get('chunk_id')
        else:
            text = str(c)
            source = None
            chunk_id = None
        candidates.append({'id': i, 'text': text, 'source': source, 'chunk_id': chunk_id})
        candidate_embeddings.append(embeddings[i])

    # compute cosine similarities
    if len(candidate_embeddings) > 0:
        cand_embs = np.vstack(candidate_embeddings)
        # cosine = dot / (||a||*||b||)
        q_norm = np.linalg.norm(query_embedding)
        cand_norms = np.linalg.norm(cand_embs, axis=1)
        dots = cand_embs.dot(query_embedding)
        # avoid divide by zero
        cosines = [0.0 if (q_norm == 0 or n == 0) else float(d / (q_norm * n)) for d, n in zip(dots, cand_norms)]
    else:
        cosines = []

    final_scores = cosines

    # optional BM25 re-ranking over the small candidate set
    if (re_rank == 'bm25' or bm25_weight > 0.0):
        try:
            from rank_bm25 import BM25Okapi
            import re

            tokenized = [re.findall(r"\w+", c['text'].lower()) for c in candidates]
            bm25 = BM25Okapi(tokenized)
            bm25_scores = bm25.get_scores(re.findall(r"\w+", query.lower()))
            # combine scores
            if len(final_scores) == len(bm25_scores):
                # weighted sum: (1 - bm25_weight)*cos + bm25_weight*norm(bm25)
                bm25_norm = _normalize(bm25_scores)
                final_scores = [(1 - bm25_weight) * c + bm25_weight * b for c, b in zip(final_scores, bm25_norm)]
            else:
                # fallback: use bm25 as final_scores
                final_scores = _normalize(bm25_scores)
        except Exception as e:
            print(f"BM25 re-ranker not available or failed: {e}. Continuing with cosine scores.")

    # optional cross-encoder re-ranker (high-quality, heavier)
    if re_rank == 'cross_encoder':
        try:
            from sentence_transformers import CrossEncoder

            model_name = cross_encoder_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ce = CrossEncoder(model_name)
            pairs = [[query, c['text']] for c in candidates]
            ce_scores = ce.predict(pairs)
            # use CE scores as final ranking signal
            final_scores = _normalize(ce_scores)
        except Exception as e:
            print(f"Cross-encoder re-ranker failed or not installed: {e}. Falling back to previous scores.")

    # normalize final scores to 0..1
    if final_scores:
        final_scores = _normalize(final_scores)
    else:
        final_scores = [0.0] * len(candidates)

    # attach scores and trim to requested top_k
    results = []
    for c, s in zip(candidates, final_scores):
        results.append({
            'id': c['id'],
            'text': c['text'],
            'score': float(s),
            'source': c.get('source'),
            'chunk_id': c.get('chunk_id'),
        })

    # sort by score desc and return top_k
    results = sorted(results, key=lambda r: r['score'], reverse=True)[:top_k]
    return results
