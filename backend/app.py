"""
FastAPI backend for the AI-Powered Recommendation Engine for Indian Standards.

Endpoints:
  POST /recommend   { "query": "<free text spec / product description>", "top_k": 3 }
      -> primary recommended standard(s) + allied/normative standards (via
         graph traversal) + latest version/amendment + certification flags.

Run:
  uvicorn app:app --reload --port 8000
(after running build_index.py at least once)
"""

import io
import json
import os
import pickle
import re

import faiss
import numpy as np
import secrets

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from i18n import normalize_query
import embeddings

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index")

app = FastAPI(title="Indian Standards Recommendation Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {}

# ---------------------------------------------------------------------------
# Portal integration: API-key auth so this can be safely called by external
# e-procurement portals (GeM-style), not just the demo frontend.
# In-memory for the prototype — swap for a DB-backed key store in production.
# ---------------------------------------------------------------------------
_state["api_keys"] = {}  # api_key -> portal_name


class PortalRegistration(BaseModel):
    portal_name: str


@app.post("/portal/register")
def register_portal(reg: PortalRegistration):
    """A procurement portal (e.g. GeM, a state e-procurement system) calls
    this once to obtain an API key for calling /recommend on behalf of its
    users. Demonstrates the integration contract a real portal onboarding
    would follow."""
    api_key = secrets.token_hex(16)
    _state["api_keys"][api_key] = reg.portal_name
    return {"portal_name": reg.portal_name, "api_key": api_key,
            "usage": "Pass this as header 'X-API-Key' on /recommend and /recommend_document"}


def require_api_key(x_api_key: str = Header(default=None)):
    # Demo mode: if no keys have been registered yet, allow unauthenticated
    # calls (so the plain frontend demo keeps working out of the box).
    if not _state["api_keys"]:
        return "demo"
    if x_api_key not in _state["api_keys"]:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key. "
                             "Register via POST /portal/register first.")
    return _state["api_keys"][x_api_key]


@app.on_event("startup")
def load_artifacts():
    # No local model to load anymore — embeddings are fetched from the
    # Hugging Face Inference API on demand (see embeddings.py). This is why
    # startup is fast and RAM usage is low even on a free hosting tier.
    _state["index"] = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "id_map.json"), encoding="utf-8") as f:
        _state["id_map"] = json.load(f)
    with open(os.path.join(INDEX_DIR, "graph.gpickle"), "rb") as f:
        _state["graph"] = pickle.load(f)
    with open(os.path.join(INDEX_DIR, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
        _state["corpus_by_id"] = {rec["is_number"]: rec for rec in corpus}
    print("Artifacts loaded. Ready.")


class Query(BaseModel):
    query: str
    top_k: int = 3


def standard_summary(is_number: str) -> dict:
    rec = _state["corpus_by_id"].get(is_number, {})
    return {
        "is_number": is_number,
        "title": rec.get("title", "(not in corpus — referenced only)"),
        "status": rec.get("status", "unknown"),
        "latest_amendment": rec.get("latest_amendment", "unknown"),
        "certification": rec.get("certification", "None"),
    }


def allied_standards(is_number: str, max_depth: int = 1) -> list:
    """Pull cross-referenced standards from the knowledge graph, grouped by
    relation type (normative, test_method, related_product, etc.)."""
    graph = _state["graph"]
    if is_number not in graph:
        return []
    allied = []
    for _, target, data in graph.out_edges(is_number, data=True):
        allied.append({
            **standard_summary(target),
            "relation": data.get("relation", "related"),
        })
    return allied


@app.post("/recommend")
def recommend(q: Query, caller: str = Depends(require_api_key)):
    index = _state["index"]
    id_map = _state["id_map"]

    norm = normalize_query(q.query)
    search_text = norm["english"]

    query_vec = embeddings.embed_texts([search_text])
    scores, indices = index.search(query_vec, q.top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        is_number = id_map[idx]
        primary = standard_summary(is_number)
        primary["semantic_score"] = round(float(score), 4)
        primary["allied_standards"] = allied_standards(is_number)
        results.append(primary)

    return {
        "query": q.query,
        "detected_language": norm["detected_lang"],
        "searched_as": search_text if norm["detected_lang"] != "en" else None,
        "recommendations": results,
        "note": (
            "Semantic ranking is a similarity score (0-1), not a certainty "
            "guarantee — a procurement official should confirm the final "
            "choice, especially for safety-critical specifications."
        ),
    }


@app.get("/health")
def health():
    status_path = os.path.join(INDEX_DIR, "..", "data", "refresh_status.json")
    refresh_status = None
    if os.path.exists(status_path):
        with open(status_path, encoding="utf-8") as f:
            refresh_status = json.load(f)
    return {
        "status": "ok",
        "standards_indexed": len(_state.get("id_map", [])),
        "last_data_refresh": refresh_status,
    }


# ---------------------------------------------------------------------------
# Tender document input (PDF / DOCX / TXT)
# ---------------------------------------------------------------------------

def extract_text(filename: str, raw_bytes: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    elif name.endswith(".docx"):
        import docx
        doc = docx.Document(io.BytesIO(raw_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    else:  # plain text fallback
        return raw_bytes.decode("utf-8", errors="ignore")


def extract_spec_clauses(text: str, min_words: int = 4, max_clauses: int = 40) -> list:
    """
    Tender documents mix boilerplate (terms & conditions, eligibility, page
    numbers) with actual technical specification lines. This is a simple
    heuristic splitter — split on lines/bullets/semicolons, keep lines that
    look like a technical requirement (has a noun-ish product term and
    enough words), and cap the count so a 40-page tender doesn't fire 500
    embedding calls. Swap for a proper clause-classifier model in production
    (e.g. a small fine-tuned classifier trained on labelled tender lines).
    """
    raw_lines = re.split(r"[\n;]|(?:^\s*[\u2022\-\*]\s*)", text, flags=re.MULTILINE)
    candidates = []
    boilerplate_markers = (
        "tender", "eligibility", "emd", "bid", "clause", "page", "annexure",
        "signature", "seal", "gst", "pan no", "earnest money"
    )
    for line in raw_lines:
        line = line.strip()
        if not line or len(line.split()) < min_words:
            continue
        low = line.lower()
        if any(marker in low for marker in boilerplate_markers):
            continue
        candidates.append(line)
        if len(candidates) >= max_clauses:
            break
    return candidates


@app.post("/recommend_document")
async def recommend_document(file: UploadFile = File(...), top_k: int = 2,
                              caller: str = Depends(require_api_key)):
    raw_bytes = await file.read()
    text = extract_text(file.filename, raw_bytes)
    clauses = extract_spec_clauses(text)

    if not clauses:
        return {"filename": file.filename, "clauses_found": 0, "results": [],
                "note": "No candidate specification lines were detected — "
                        "try a plainer .txt export or check the PDF has "
                        "extractable text (not a scanned image)."}

    index = _state["index"]
    id_map = _state["id_map"]

    # Batch all clauses into a single Hugging Face request rather than one
    # request per clause — far fewer API calls, and stays under HF's
    # free-tier rate limit even for a many-clause tender document.
    search_texts = [normalize_query(clause)["english"] for clause in clauses]
    query_vecs = embeddings.embed_texts(search_texts)
    scores_all, indices_all = index.search(query_vecs, top_k)

    all_results = []
    for clause, scores, indices in zip(clauses, scores_all, indices_all):
        clause_results = []
        for score, idx in zip(scores, indices):
            if idx == -1 or score < 0.35:  # skip weak/irrelevant matches
                continue
            is_number = id_map[idx]
            primary = standard_summary(is_number)
            primary["semantic_score"] = round(float(score), 4)
            primary["allied_standards"] = allied_standards(is_number)
            clause_results.append(primary)
        if clause_results:
            all_results.append({"clause": clause, "recommendations": clause_results})

    return {
        "filename": file.filename,
        "clauses_scanned": len(clauses),
        "clauses_with_matches": len(all_results),
        "results": all_results,
        "note": "Heuristic clause extraction — review flagged clauses against "
                "the original document before finalizing the tender.",
    }


# ---------------------------------------------------------------------------
# Serve the demo frontend from this same service, so a single deployed URL
# (e.g. on Render) gives you both the API and the working demo page — no
# separate static-site deployment or CORS juggling needed.
# IMPORTANT: this mount must be the LAST thing in the file. Starlette matches
# routes in registration order, so every API route above (/recommend,
# /health, etc.) must be defined before this catch-all "/" mount, or the
# mount would swallow those requests instead of the frontend's own files.
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
