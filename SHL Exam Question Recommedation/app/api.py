# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.utils import retrieve_candidates, balance_results, classify_query_type
import requests, re
from readability import Document

app=FastAPI(title="SHL Recommender")

class Req(BaseModel):
    query: str=None
    url: str=None
    top_k: int=5

def extract_text_from_url(url):
    try:
        r=requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        doc=Document(r.text)
        text=re.sub(r"<[^>]+>"," ", doc.summary())
        return re.sub(r"\s+"," ",text)
    except Exception as e:
        raise RuntimeError(str(e))

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/recommend")
def recommend(req: Req):
    if not req.query and not req.url: raise HTTPException(status_code=400, detail="Provide query or url")
    top_k=max(1, min(10, int(req.top_k)))
    text=req.query or extract_text_from_url(req.url)
    qtype=classify_query_type(text)
    pool=retrieve_candidates(text, topk=50)
    final=balance_results(pool, qtype, top_k)
    return {"query": req.query or req.url, "query_type": qtype, "recommendations": final}
