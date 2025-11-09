# src/crawl_catalog.py
import requests, time, csv, os, re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE = "https://www.shl.com"
START = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {"User-Agent":"Mozilla/5.0"}
OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "catalog_raw.csv")
os.makedirs(OUT_DIR, exist_ok=True)

def get_soup(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print("GET error:", url, e)
        return None

def is_prepackaged(soup):
    txt = soup.get_text(separator=" ").lower()
    return "pre-packaged job solutions" in txt or "pre packaged" in txt

def extract_assessment_details(url):
    soup = get_soup(url)
    if not soup: return None
    if is_prepackaged(soup): return None
    title_tag = soup.find(["h1","h2"])
    name = title_tag.get_text(strip=True) if title_tag else (soup.title.string if soup.title else "")
    desc = ""
    meta = soup.find("meta", attrs={"name":"description"}) or soup.find("meta", attrs={"property":"og:description"})
    if meta and meta.get("content"): desc = meta["content"].strip()
    else:
        p = soup.find("p")
        if p: desc = p.get_text(strip=True)
    tags = []
    for el in soup.select(".tag, .tags, .product-tags, .taxonomy, a[href*='/solutions/']"):
        t = el.get_text(strip=True)
        if t and len(t)<60: tags.append(t)
    text = (name + " " + desc + " " + " ".join(tags)).lower()
    test_type = "Unknown"
    tech = ["skill","knowledge","technical","python","java","sql","cognitive","aptitude","coding","developer"]
    beh = ["personality","behaviour","behavior","collaboration","teamwork","leadership","communication"]
    if any(k in text for k in tech) and not any(k in text for k in beh): test_type="K"
    elif any(k in text for k in beh) and not any(k in text for k in tech): test_type="P"
    elif any(k in text for k in tech) and any(k in text for k in beh): test_type="K&P"
    return {"assessment_name":name, "url":url, "description":desc, "tags":"|".join(dict.fromkeys(tags)), "test_type":test_type}

def crawl(start=START, max_pages=500):
    from collections import deque
    q = deque([start])
    seen = set(); results=[]
    pages=0
    while q and pages<max_pages:
        url = q.popleft()
        if url in seen: continue
        seen.add(url); pages+=1
        print(f"[{pages}] {url}")
        soup = get_soup(url)
        if not soup: continue
        for a in soup.find_all("a", href=True):
            href = a["href"]; abs_href = urljoin(BASE, href)
            parsed = urlparse(abs_href)
            if parsed.netloc.endswith("shl.com"):
                if re.search(r"/product|/products/|/solutions/products", parsed.path, re.I):
                    if abs_href not in seen: q.append(abs_href)
                else:
                    if "/solutions/" in parsed.path and abs_href not in seen: q.append(abs_href)
        det = extract_assessment_details(url)
        if det and det["assessment_name"] and det["url"] not in [r["url"] for r in results]:
            results.append(det)
        time.sleep(1.0)
    print("Found", len(results))
    keys=["assessment_name","url","description","tags","test_type"]
    with open(OUT_FILE,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader()
        for r in results: writer.writerow(r)
    print("Saved:", OUT_FILE)

if __name__=="__main__":
    crawl()
