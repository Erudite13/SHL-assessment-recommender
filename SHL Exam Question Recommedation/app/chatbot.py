# app/chatbot.py
# SHL Assessment Recommender — Streamlit Chatbot (RAG with HF summarization)
# - Retrieval: Sentence-Transformers + FAISS (from src/utils.py)
# - Generation: Hugging Face (facebook/bart-large-cnn) summarization for concise explanation

import os
import re
import sys
import pandas as pd
import streamlit as st

# --- Make project root importable ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import project utilities ---
try:
    from src.utils import (
        load_resources,
        retrieve_candidates,
        balance_results,
        classify_query_type,
        generate_explanation_hf,  # HF summarization-based explanation
    )
except Exception as e:
    st.error(f"Failed to import from src.utils: {e}")
    raise

# --- Optional URL fetch support ---
try:
    import requests
    from readability import Document
    CAN_FETCH = True
except Exception:
    CAN_FETCH = False


# ----------------- Streamlit Page Setup -----------------
st.set_page_config(
    page_title="SHL Assessment Recommender — RAG (Hugging Face)",
    layout="wide",
    page_icon="🧠",
)

st.title("💬 SHL Assessment Recommender — RAG (Hugging Face)")
st.write(
    "Paste a job description or a JD URL. I’ll retrieve relevant SHL assessments "
    "and (optionally) generate a short explanation using Hugging Face."
)

# ----------------- Sidebar Controls -----------------
with st.sidebar:
    st.header("⚙️ Controls")
    rag_mode = st.toggle("💡 Generate AI Explanation", value=True)
    # Let you choose a model string if you want to swap summarizer later
    hf_model = st.text_input(
        "HF summarization model",
        value="facebook/bart-large-cnn",
        help="Default: facebook/bart-large-cnn"
    )
    st.caption("Tip: Keep the default for best stability.")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.clear()
        st.experimental_rerun()

    st.divider()
    if st.session_state.get("recommendations"):
        csv = pd.DataFrame(st.session_state["recommendations"])[
            ["assessment_name", "url", "test_type", "score"]
        ].to_csv(index=False)
        st.download_button(
            label="Download last recommendations (CSV)",
            data=csv,
            file_name="recommendations.csv",
            mime="text/csv",
        )

# ----------------- Load models / indices -----------------
with st.spinner("Loading model & index…"):
    try:
        load_resources()
    except Exception as e:
        st.error(f"Resource load failed. Check your data folder & FAISS index.\n\n{e}")
        st.stop()

# ----------------- Session State -----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = []
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

# ----------------- Conversation History -----------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"])
        else:
            st.write(msg["content"])

# ----------------- Helpers -----------------
def is_url(text: str) -> bool:
    return bool(re.match(r"https?://", text.strip()))

def fetch_text_from_url(url: str) -> str:
    if not CAN_FETCH:
        raise RuntimeError("requests/readability not installed.")
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    doc = Document(r.text)
    html = doc.summary()
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 50:  # fallback to raw if cleaned too short
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]  # guard length


# ----------------- Chat Input -----------------
user_input = st.chat_input("Paste JD text or a JD URL here…")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # --- Retrieval flow ---
    with st.chat_message("assistant"):
        with st.spinner("Retrieving recommendations…"):
            try:
                query_text = user_input.strip()
                if is_url(query_text):
                    try:
                        query_text = fetch_text_from_url(query_text)
                    except Exception as e:
                        st.warning(f"Could not fetch URL content; using URL text. ({e})")

                st.session_state["last_query"] = query_text

                qtype = classify_query_type(query_text)            # "K", "P", or "K&P"
                candidates = retrieve_candidates(query_text, topk=50)
                final = balance_results(candidates, qtype, k=5)
                st.session_state["recommendations"] = final

                if final:
                    lines = [f"**Query type:** `{qtype}`", "", f"Top {len(final)} recommendations:"]
                    for i, r in enumerate(final, start=1):
                        lines.append(
                            f"{i}. [{r['assessment_name']}]({r['url']}) — *{r['test_type']}* — "
                            f"`score: {r['score']:.4f}`"
                        )
                    reply = "\n\n".join(lines)
                else:
                    reply = "Sorry — I couldn't find relevant assessments for that description."

                st.markdown(reply)
                st.session_state["messages"].append({"role": "assistant", "content": reply})

            except Exception as e:
                err = f"⚠️ Retrieval failed: {e}"
                st.error(err)
                st.session_state["messages"].append({"role": "assistant", "content": err})
                final = []

        # --- HF Explanation (summarization) ---
        if rag_mode and final:
            with st.spinner("Generating AI explanation (Hugging Face)…"):
                try:
                    explanation = generate_explanation_hf(
                        st.session_state["last_query"],
                        final,
                        model=hf_model,   # default: facebook/bart-large-cnn
                    )
                    st.markdown("**💡 AI Explanation:**")
                    st.write(explanation)
                    st.session_state["messages"].append({"role": "assistant", "content": explanation})
                except Exception as e:
                    err2 = f"(⚠️ HF explanation failed: {e})"
                    st.error(err2)
                    st.session_state["messages"].append({"role": "assistant", "content": err2})

# ----------------- Results Table -----------------
st.divider()
st.subheader("Latest recommendations (interactive)")

recs = st.session_state.get("recommendations", [])
if recs:
    df = pd.DataFrame(recs)[["assessment_name", "test_type", "score", "url"]].copy()
    df["score"] = df["score"].map(lambda x: f"{x:.4f}")
    st.dataframe(df, use_container_width=True)

    st.markdown("**Actions** — More info / Select")
    for idx, rec in enumerate(recs):
        c1, c2, c3 = st.columns([6, 1, 1])
        c1.markdown(f"**{rec['assessment_name']}**  \n*{rec['test_type']}*  \n[{rec['url']}]({rec['url']})")
        if c2.button("More info", key=f"more_{idx}"):
            detail = (
                f"### Details: {rec['assessment_name']}\n\n"
                f"- URL: {rec['url']}\n"
                f"- Type: {rec['test_type']}\n"
                f"- Similarity score: `{rec['score']:.4f}`\n"
            )
            st.session_state["messages"].append({"role": "assistant", "content": detail})
            st.experimental_rerun()
        if c3.button("Select", key=f"select_{idx}"):
            sel = f"Selected: **{rec['assessment_name']}** — {rec['url']}"
            st.session_state["messages"].append({"role": "assistant", "content": sel})
            st.experimental_rerun()

    if st.button("Prepare submission CSV"):
        rows = [{"Query": st.session_state["last_query"], "Assessment_url": r["url"]} for r in recs]
        csv = pd.DataFrame(rows).to_csv(index=False)
        st.download_button("Download submission CSV", data=csv, file_name="submission.csv", mime="text/csv")
else:
    st.info("No recommendations yet — ask a question above to get started.")

st.caption("Built with 🧠 Sentence-Transformers + FAISS + Hugging Face (BART) + Streamlit")
