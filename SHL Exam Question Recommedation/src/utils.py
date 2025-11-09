# src/utils.py
"""
Robust utilities for retrieval. Attempts to use sentence-transformers (torch).
If torch / sentence-transformers isn't available, falls back to a TF-IDF retriever
so app can run without installing PyTorch.
"""

import os, pickle, math
import numpy as np
from typing import List, Dict

EMB_PKL = os.path.join("data", "catalog_embeddings.pkl")
FAISS_IDX = os.path.join("data", "catalog_faiss.index")
CLEAN_CSV = os.path.join("data", "catalog_clean.csv")

# Lazy-loaded objects
_df = None
_embs = None
_index = None
_model = None
_use_sbert = False

# Try to import sentence-transformers (torch). If it fails, we'll use TF-IDF fallback.
try:
    from sentence_transformers import SentenceTransformer
    _use_sbert = True
except Exception:
    _use_sbert = False

# TF-IDF fallback (no torch dependency)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _use_tfidf = True
except Exception:
    _use_tfidf = False

# -----------------------
# Load resources
# -----------------------
def load_resources():
    """
    Loads precomputed catalog dataframe and embeddings (if present), and FAISS index (if present).
    If SBERT is available, loads the model lazily.
    If not, ensures TF-IDF vectorizer is built for fallback.
    """
    global _df, _embs, _index, _model, _use_sbert, _use_tfidf

    # load df and embeddings pkl if exists
    if _df is None:
        if os.path.exists(EMB_PKL):
            _df, _embs = pickle.load(open(EMB_PKL, "rb"))
        else:
            # If no EMB_PKL, try to load the cleaned CSV
            import pandas as pd
            _df = pd.read_csv(CLEAN_CSV)
            _embs = None

    # load FAISS index if exists
    if _index is None and os.path.exists(FAISS_IDX):
        try:
            import faiss
            _index = faiss.read_index(FAISS_IDX)
        except Exception:
            _index = None

    # sbert model if available
    if _use_sbert and _model is None:
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _model = None
            _use_sbert = False

    # prepare TF-IDF fallback if available
    if not _use_sbert and _use_tfidf:
        # Build TF-IDF on combined text
        if _df is not None:
            texts = (_df["assessment_name"].fillna("") + " " + _df["description"].fillna("") + " " + _df["tags"].fillna("")).tolist()
            # build vectorizer only once and store in module state
            if not hasattr(load_resources, "_tfidf"):
                vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
                tfidf_matrix = vectorizer.fit_transform(texts)
                load_resources._tfidf = vectorizer
                load_resources._tfidf_matrix = tfidf_matrix
    return _df, _index, _embs

# -----------------------
# Helper functions
# -----------------------
TECH_KEYWORDS = {"java","python","sql","technical","coding","developer","data","aptitude","cognitive"}
BEHAV_KEYWORDS = {"personality","behaviour","behavior","teamwork","collaboration","leadership","communication"}

def classify_query_type(text: str) -> str:
    txt = (text or "").lower()
    has_tech = any(k in txt for k in TECH_KEYWORDS)
    has_beh = any(k in txt for k in BEHAV_KEYWORDS)
    if has_tech and has_beh:
        return "K&P"
    if has_tech:
        return "K"
    if has_beh:
        return "P"
    return "Unknown"

def _embed_query_with_sbert(query: str):
    """Embed the query using loaded SBERT (if available)."""
    global _model
    if _model is None:
        # try to load
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None
    emb = _model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return emb

def _tfidf_rank(query: str, topk: int = 10):
    """Fallback ranking using TF-IDF cosine similarity (no torch)."""
    if not hasattr(load_resources, "_tfidf"):
        raise RuntimeError("TF-IDF vectorizer is not available. Install scikit-learn or enable SBERT.")
    vectorizer = load_resources._tfidf
    tfidf_matrix = load_resources._tfidf_matrix
    q_vec = vectorizer.transform([query])
    # compute cosine similarity between q_vec and tfidf_matrix
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx = sims.argsort()[::-1][:topk]
    df, _, _ = load_resources()
    results = []
    for i in idx:
        row = df.iloc[i]
        results.append({
            "assessment_name": row["assessment_name"],
            "url": row["url"],
            "test_type": row.get("test_type", "Unknown"),
            "score": float(sims[i])
        })
    return results

def retrieve_candidates(query: str, topk: int = 10) -> List[Dict]:
    """
    Primary retrieval function:
    - If SBERT + FAISS available: embed query with SBERT and use FAISS search.
    - If only FAISS + precomputed embeddings and no SBERT: we cannot embed query => fallback to TF-IDF.
    - If FAISS missing but precomputed embeddings present: use cosine with precomputed embeddings (needs query embedding).
    - Ultimately fallback to TF-IDF ranking if SBERT not available.
    """
    df, index, embs = load_resources()

    # Case 1: SBERT available -> use it + FAISS if available
    if _use_sbert and _model is not None and index is not None:
        q_emb = _embed_query_with_sbert(query)
        D, I = index.search(q_emb, topk)
        results = []
        for score, idx in zip(D[0], I[0]):
            row = df.iloc[idx]
            results.append({
                "assessment_name": row["assessment_name"],
                "url": row["url"],
                "test_type": row.get("test_type", "Unknown"),
                "score": float(score)
            })
        return results

    # Case 2: SBERT not available, use TF-IDF fallback (if prepared)
    if not _use_sbert and _use_tfidf and hasattr(load_resources, "_tfidf"):
        return _tfidf_rank(query, topk=topk)

    # Case 3: as last resort, if embeddings available but no model, we cannot embed query -> use TF-IDF
    if embs is not None and not _use_sbert:
        if _use_tfidf and hasattr(load_resources, "_tfidf"):
            return _tfidf_rank(query, topk=topk)
        else:
            raise RuntimeError("Cannot embed query: sentence-transformers (torch) not available and TF-IDF fallback is not present.")

    # If nothing else, raise
    raise RuntimeError("No retrieval method available. Install sentence-transformers or scikit-learn.")

def balance_results(results: List[Dict], query_type: str, k: int = 5) -> List[Dict]:
    if k <= 0:
        return []
    if query_type != "K&P":
        return results[:k]
    k_list = [r for r in results if str(r.get("test_type","")).startswith("K")]
    p_list = [r for r in results if str(r.get("test_type","")).startswith("P")]
    final = []
    if k_list: final.append(k_list.pop(0))
    if p_list: final.append(p_list.pop(0))
    for r in results:
        if r in final: continue
        final.append(r)
        if len(final) >= k: break
    return final[:k]


def _get_hf_token():
    import os
    tok = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    if tok:
        return tok
    try:
        import streamlit as st
        return st.secrets.get("HF_TOKEN") or st.secrets.get("HF_API_KEY")
    except Exception:
        return None


def _build_explanation_source_text(query, recs, max_items=5):
    """
    Build a well-structured explanation input that summarization models understand.
    """

    rec_list = []
    for i, r in enumerate(recs[:max_items], start=1):
        name = r.get("assessment_name", "")
        ttype = r.get("test_type", "")
        rec_list.append(f"{i}. {name} (Type: {ttype})")

    recommendations_text = "\n".join(rec_list)

    text = f"""
You are an SHL assessment expert. Rewrite a clear, concise explanation (2–4 sentences)
describing WHY these assessments match the job description.

Job Description Summary:
{query}

Relevant SHL Assessments:
{recommendations_text}

Write the final explanation in a clean paragraph. DO NOT repeat the job description.
Explain the alignment between job requirements and the assessments.
"""
    # limit to 1500–2000 chars (BART's sweet spot)
    return text[:1800]

def generate_explanation_hf(query: str, recs: list, model: str = "facebook/bart-large-cnn"):
    """
    Fixes the summarizer issue by:
    1. Generating our OWN draft explanation (no instructions)
    2. Sending only the explanation text to BART for polishing
    """

    import os
    from huggingface_hub import InferenceClient

    hf_token = _get_hf_token()
    if not hf_token:
        return "(⚠ HF_TOKEN missing — add it in .streamlit/secrets.toml)"

    # ----------- STEP 1: Build draft explanation WITHOUT INSTRUCTIONS -----------
    points = []

    for r in recs:
        name = r.get("assessment_name", "")
        ttype = r.get("test_type", "")

        if "SQL" in name or "Data" in name:
            points.append(f"{name} evaluates SQL and analytical data skills aligned with core responsibilities.")
        elif "Inductive" in name or "Deductive" in name:
            points.append(f"{name} measures critical thinking and problem-solving required for data-driven tasks.")
        elif "Verbal" in name:
            points.append(f"{name} assesses communication and comprehension important for cross-team collaboration.")
        else:
            points.append(f"{name} supports role-relevant capabilities such as reasoning or accuracy.")

    bullet_text = " ".join(points)

    # Make a natural paragraph to polish
    draft_para = (
        f"The role requires: {query[:180]}. "
        f"The recommended assessments align with the job because {bullet_text}"
    )

    # ----------- STEP 2: Send ONLY the paragraph to the summarizer -----------
    try:
        client = InferenceClient(provider="hf-inference", api_key=hf_token)

        summary = client.summarization(
            draft_para,
            model=model,
        )

        if isinstance(summary, dict) and "summary_text" in summary:
            return summary["summary_text"].strip()

        return str(summary)

    except Exception as e:
        return f"(⚠ HF summarization error: {e})"
