# src/clean_catalog.py
import pandas as pd, re, os
RAW="data/catalog_raw.csv"; OUT="data/catalog_clean.csv"
df=pd.read_csv(RAW)
df=df.drop_duplicates(subset=["url"]).dropna(subset=["assessment_name","url"])
df["description"]=df["description"].fillna("").astype(str)
df["tags"]=df["tags"].fillna("").astype(str)
def norm(t): return re.sub(r"\s+"," ",str(t)).strip()
df["assessment_name"]=df["assessment_name"].apply(norm)
df["description"]=df["description"].apply(lambda x: re.sub(r"[\n\r\t]+"," ", x))
def infer_type(row):
    txt=(row["assessment_name"]+" "+row["description"]+" "+row["tags"]).lower()
    tech_kw=["technical","coding","java","python","sql","cognitive","aptitude","skill"]
    beh_kw=["personality","behaviour","behavior","teamwork","collaboration","leadership","communication"]
    if any(k in txt for k in tech_kw) and not any(k in txt for k in beh_kw): return "K"
    if any(k in txt for k in beh_kw) and not any(k in txt for k in tech_kw): return "P"
    if any(k in txt for k in tech_kw) and any(k in txt for k in beh_kw): return "K&P"
    return "Unknown"
df["test_type"]=df.apply(lambda r: infer_type(r) if pd.isna(r.get("test_type")) or r.get("test_type")=="Unknown" else r.get("test_type"), axis=1)
df=df[df["description"].str.len()>20]
df.to_csv(OUT,index=False)
print("Saved",OUT)
