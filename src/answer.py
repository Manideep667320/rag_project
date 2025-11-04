import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Hugging Face Inference API integration
# Add HF_API_TOKEN and optional HF_MODEL to your .env. Example:
# HF_API_TOKEN=...
# HF_MODEL=gpt2

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "gpt2")

# Local fallback model name
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gpt2")

# If set, force using the extractive fallback only (no HF API, no local generation)
FORCE_EXTRACTIVE = os.getenv("FORCE_EXTRACTIVE", "").lower() in ("1", "true", "yes")

_local_generator = None

def _call_hf_inference(prompt: str, model: str = HF_MODEL, token: str = HF_API_TOKEN):
    if not token:
        raise RuntimeError("HF_API_TOKEN not set. Add HF_API_TOKEN to your .env or export it in the environment.")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}, "parameters": {"max_new_tokens": 128}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # surface HF error message
        msg = resp.text
        raise RuntimeError(f"Hugging Face inference API error: {e}\nResponse: {msg}")
    data = resp.json()
    # depending on model, response may be list of generations or dict
    if isinstance(data, list):
        # common: [{'generated_text': '...'}]
        first = data[0]
        return first.get("generated_text") or str(first)
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    return str(data)

def generate_answer(query, context_chunks):
    # Prepare context chunks so they fit within the target model's context window.
    model_name = HF_MODEL if HF_API_TOKEN else LOCAL_MODEL
    # If user explicitly wants extractive-only behaviour, skip any generation calls.
    if FORCE_EXTRACTIVE:
        answer = _extractive_answer(context_chunks, query)
        return {"text": answer, "confidence": 0.7, "method": "extractive"}  # Base confidence for extractive method
    max_new_tokens = 128
    prepared_chunks = _truncate_chunks_for_model(context_chunks, query, model_name, max_new_tokens=max_new_tokens)

    # Build a strict RAG prompt that instructs the model to ONLY output the answer.
    # We delimit the context with triple backticks to avoid accidental repetition.
    context = "\n".join(prepared_chunks)
    prompt = (
        "You are a helpful assistant that answers questions using only the provided context.\n"
        "Do NOT repeat the context. Do NOT add commentary. Output ONLY the answer. Be concise.\n\n"
        f"Context:\n```\n{context}\n```\n\nQuestion: {query}\nAnswer:"
    )

    # Try HF Inference API first if token is available; otherwise use local model.
    raw = None
    confidence = 0.0
    method = "unknown"
    
    if HF_API_TOKEN:
        try:
            raw = _call_hf_inference(prompt)
            confidence = 0.9  # High confidence for HF API
            method = "huggingface"
        except Exception as e:
            print(f"Hugging Face inference failed: {e}. Falling back to local model.")

    if raw is None:
        try:
            raw = _local_generate(prompt)
            confidence = 0.8  # Good confidence for local model
            method = "local"
        except Exception as e:
            # Local generation failed (could be missing 'transformers' or backend). Fall back to
            # a simple extractive answer that doesn't require any model or API keys.
            print(f"Local generation failed: {e}. Falling back to extractive answer (no API, no transformers).")
            answer = _extractive_answer(prepared_chunks, query)
            return {"text": answer, "confidence": 0.7, "method": "extractive"}

    # Parse the model output: prefer the text after the last occurrence of 'Answer:'
    if not isinstance(raw, str):
        raw = str(raw)
    marker = "Answer:"
    idx = raw.rfind(marker)
    if idx != -1:
        ans = raw[idx + len(marker) :].strip()
        # Sometimes models continue with extra 'Question' tokens; stop at 'Question' if present
        qidx = ans.find("Question:")
        if qidx != -1:
            ans = ans[:qidx].strip()
        return {"text": ans, "confidence": confidence, "method": method}

    # Fallback: if marker not found, adjust confidence and return with metadata
    if raw.startswith(prompt):
        ans = raw[len(prompt) :].strip()
        confidence *= 0.9  # Slightly reduce confidence for fallback case
    else:
        ans = raw.strip()
        confidence *= 0.8  # Further reduce confidence for unexpected format
        
    return {"text": ans, "confidence": confidence, "method": method}


def _extractive_answer(chunks, query):
    """Return a concise extractive answer from chunks using simple heuristics.

    This avoids requiring any ML generation library. It selects the chunk with the
    highest overlap of query tokens and returns the most relevant sentence or a short
    prefix of the chunk.
    """
    import re

    if not chunks:
        return "I don't know based on the provided documents."

    # tokenise query into words (ignore short words)
    q_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    if not q_tokens:
        # nothing to match on — return the first short summary of the first chunk
        first = chunks[0].strip()
        return (first.split("\n")[0])[:500]

    best = None
    best_score = -1
    for c in chunks:
        words = set(re.findall(r"\w+", c.lower()))
        score = sum(1 for t in q_tokens if t in words)
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        best = chunks[0]

    # split into sentences and prefer the one containing query terms
    sentences = re.split(r'(?<=[.!?])\s+', best.strip())
    for s in sentences:
        s_low = s.lower()
        if any(t in s_low for t in q_tokens):
            return s.strip()

    # fallback: return the first sentence or a short prefix
    if sentences:
        return sentences[0].strip()[:500]
    return best.strip()[:500]


def _local_generate(prompt: str, model: str = LOCAL_MODEL):
    """Generate text locally using Hugging Face Transformers pipeline.

    This requires the `transformers` package (and a backend like torch or tensorflow).
    It will download the model the first time it's used.
    """
    global _local_generator
    if _local_generator is None:
        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError("Local generation requires the 'transformers' package. Install it with 'pip install transformers'.") from exc
        # Use CPU by default (device=-1). If you have a GPU and torch with CUDA, set device=0.
        _local_generator = pipeline("text-generation", model=model, device=-1)
    # Use max_new_tokens to avoid conflicts when prompt is already long
    out = _local_generator(prompt, max_new_tokens=128, do_sample=True, top_k=50, num_return_sequences=1)
    # `out` is a list with a dict containing 'generated_text'
    if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
        return out[0]["generated_text"]
    return str(out)


def _estimate_tokens(text: str, tokenizer):
    try:
        return len(tokenizer.encode(text))
    except Exception:
        # fallback estimation: 1 token ~= 4 chars
        return max(1, int(len(text) / 4))


def _get_model_max_context(model_name: str) -> int:
    # approximate known context sizes for common models; default to 2048
    mapping = {
        "gpt2": 1024,
        "gpt-neo-2.7B": 2048,
        "EleutherAI/gpt-neo-2.7B": 2048,
        "gpt-j": 2048,
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
    }
    for k, v in mapping.items():
        if k in model_name:
            return v
    return 2048


def _truncate_chunks_for_model(chunks, query, model_name, max_new_tokens=128):
    """Truncate the list of chunks so the assembled prompt stays within the model's context window.

    This function tries to use the model tokenizer to compute token counts. If unavailable,
    it falls back to a character-based heuristic.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tokenizer = None

    model_ctx = _get_model_max_context(model_name)
    # estimate header tokens (prompt instructions + query + wrappers)
    header = (
        "You are a helpful assistant that answers questions using only the provided context.\n"
        "Do NOT repeat the context. Do NOT add commentary. Output ONLY the answer. Be concise.\n\n"
        f"Context:\n```\n\n```\n\nQuestion: {query}\nAnswer:"
    )
    header_tokens = _estimate_tokens(header, tokenizer) if tokenizer else max(1, int(len(header) / 4))
    # leave room for generation
    allowed_for_context = model_ctx - header_tokens - max_new_tokens
    if allowed_for_context <= 0:
        # nothing can be included; return an empty list to let model answer from no context
        print("Warning: model context too small for header + generation; sending no context chunks.")
        return []

    kept = []
    acc = 0
    for chunk in chunks:
        tok = _estimate_tokens(chunk, tokenizer) if tokenizer else max(1, int(len(chunk) / 4))
        if acc + tok <= allowed_for_context:
            kept.append(chunk)
            acc += tok
            continue
        # need to truncate this chunk
        remaining = allowed_for_context - acc
        if remaining <= 0:
            break
        # approximate characters to keep
        if tokenizer and tok > 0:
            ratio = remaining / tok
            keep_chars = max(50, int(len(chunk) * ratio))
        else:
            # approx 4 chars per token
            keep_chars = max(50, int(remaining * 4))
        truncated = chunk[:keep_chars]
        kept.append(truncated)
        print(f"Truncated chunk to {keep_chars} chars to fit model context ({model_name}).")
        break

    return kept
