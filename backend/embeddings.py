"""
Embeddings via the Hugging Face Inference API.
================================================
Why an API instead of loading a model locally:

  Loading a sentence-transformers model directly (via sentence-transformers
  + torch) needs the model's weights held in RAM the whole time your server
  runs — for a good multilingual model, that's 1-2GB, on top of a slow cold
  start and a large deploy (torch alone is well over 100MB to download).
  On a tight or free hosting tier (Render's free instance is 512MB RAM),
  that alone is enough to crash the service on startup.

  Using Hugging Face's hosted Inference API moves that memory and compute
  onto Hugging Face's servers instead of yours. Your server just sends text
  and gets back a vector. The trade-off is an internet round-trip per
  request, a free-tier rate limit on Hugging Face's side, and an API key to
  manage — a good trade for a hosted, judge-facing prototype.

Setup:
  1. Create a free account at https://huggingface.co
  2. Get a token at https://huggingface.co/settings/tokens — choose "Read"
     access, which includes permission to call Inference Providers. If you
     later get a 403 error, regenerate the token as "Fine-grained" and
     explicitly check "Make calls to Inference Providers".
  3. Put it in backend/.env as: HF_API_KEY=hf_xxxxxxxxxxxx
     (.env is already gitignored — never commit real keys)
  4. On Render, set the same variable in the dashboard under
     Environment (render.yaml marks it sync:false so it prompts you there
     instead of storing the secret in the repo).

NOTE ON THE URL BELOW: Hugging Face retired their old serverless endpoint
(api-inference.huggingface.co) in favor of a unified router at
router.huggingface.co, under their newer "Inference Providers" system —
this changed after this module was first written, and was caught only by
actually running the code against the live API rather than assumed correct.
If Hugging Face changes this again, the error will be a connection/DNS
failure (old host) or a 404/410 with a message naming the new host, not a
silent wrong result — check huggingface.co/docs/inference-providers if so.
"""

import os
import time
from typing import List

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

HF_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/pipeline/feature-extraction"
HF_API_KEY = os.environ.get("HF_API_KEY", "")

_MAX_RETRIES = 5
_RETRY_DELAY_FALLBACK_SECONDS = 5


def _headers() -> dict:
    if not HF_API_KEY:
        raise RuntimeError(
            "HF_API_KEY is not set. Create a free token at "
            "https://huggingface.co/settings/tokens and put it in "
            "backend/.env as HF_API_KEY=hf_xxx (see backend/.env.example)."
        )
    return {"Authorization": f"Bearer {HF_API_KEY}"}


def embed_texts(texts: List[str]) -> np.ndarray:
    """Returns an (N, dim) float32 array of L2-normalized sentence embeddings
    for the given texts, computed by the Hugging Face Inference API.
    Normalized so a FAISS inner-product index behaves as cosine similarity."""
    if not texts:
        return np.zeros((0, 1), dtype="float32")

    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    last_error = None

    for _ in range(_MAX_RETRIES):
        try:
            resp = requests.post(HF_API_URL, headers=_headers(), json=payload, timeout=30)
            if resp.status_code == 503:
                # Free-tier models "cold start" on Hugging Face's side the
                # first time they're called in a while. HF tells us how
                # long to wait rather than making us guess.
                wait = resp.json().get("estimated_time", _RETRY_DELAY_FALLBACK_SECONDS)
                print(f"Embedding model is warming up on Hugging Face, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 403:
                raise RuntimeError(
                    "Hugging Face returned 403 Forbidden. Your token likely "
                    "lacks the 'Inference Providers' permission — regenerate "
                    "it at huggingface.co/settings/tokens as a Fine-grained "
                    "token with 'Make calls to Inference Providers' checked."
                )
            if resp.status_code in (404, 410):
                raise RuntimeError(
                    f"Hugging Face returned {resp.status_code} for this URL — "
                    "the endpoint may have changed again. Response: "
                    f"{resp.text[:300]}"
                )
            resp.raise_for_status()
            embeddings = np.array(resp.json(), dtype="float32")
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return embeddings / norms
        except requests.RequestException as e:
            last_error = e
            time.sleep(2)

    raise RuntimeError(
        f"Hugging Face embedding request failed after {_MAX_RETRIES} attempts: {last_error}"
    )
