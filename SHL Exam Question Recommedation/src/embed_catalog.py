# src/embed_catalog.py
import pandas as pd, pickle, os
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

CLEAN="data/catalog_clean.csv"
EMB_PKL="data/catalog_embeddings.pkl"
FAISS_IDX="data/catalog_faiss.index"
df=pd.read_csv(CLEAN)
texts=(df["assessment_name"].fillna("")+" "+df["description"].fillna("")+" "+df["tags"].fillna("")).tolist()
model=SentenceTransformer("all-MiniLM-L6-v2")
embs=model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
pickle.dump((df, embs), open(EMB_PKL,"wb"))
dim=embs.shape[1]
index=faiss.IndexFlatIP(dim)
index.add(embs)
faiss.write_index(index, FAISS_IDX)
print("Saved:", EMB_PKL, FAISS_IDX)
