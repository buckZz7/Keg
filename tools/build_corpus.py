#!/usr/bin/env python3
"""Keg — build the public fidelity corpus (prose + knowledge + technical).

Pulls public-domain / permissively-licensed text across several domains so the
top-1 gate measures "is the model still itself" broadly. Each document is a
full paragraph/abstract (kept long so the stride-64 sampler yields enough
positions); one document per line of the corpus file.

Usage: python3 tools/build_corpus.py [--out corpus/production.txt]
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import time
from pathlib import Path

import requests

HDRS = {"User-Agent": "Keg-corpus-builder/0.1 (research; contact buck@keg)"}

# Public-domain books (Project Gutenberg). Paragraph = one document.
GUTENBERG = [
    (1342, "pride-prejudice"),
    (2701, "moby-dick"),
    (1661, "sherlock"),
    (244, "study-scarlet"),
]

WIKI_TOPICS = [
    "Light", "Photosynthesis", "Gravity", "Telegraph", "Immune system",
    "Steam engine", "Telescope", "Honey bee", "Renaissance", "Printing press",
]

ARXIV_CATS = ["cs.CL", "cs.LG", "cs.CV", "cs.SE", "math.NA"]

# Real source files with clear permissive licenses (MIT/BSD/Apache-2.0),
# fetched from GitHub raw. Used to add a code component to the corpus.
CODE_FILES = [
    ("https://raw.githubusercontent.com/psf/requests/main/src/requests/models.py", "apache2 python"),
    ("https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py", "bsd3 python"),
    ("https://raw.githubusercontent.com/expressjs/express/master/lib/application.js", "mit js"),
    ("https://raw.githubusercontent.com/gorilla/mux/main/mux.go", "bsd3 go"),
    ("https://raw.githubusercontent.com/serde-rs/json/master/src/ser.rs", "mit rust"),
]


def fetch(url: str, timeout: int = 60, retries: int = 3) -> str | None:
    for i in range(retries):
        try:
            r = requests.get(url, headers=HDRS, timeout=timeout)
            if r.status_code == 429:
                time.sleep(8 * (i + 1)); continue
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(f"  ! {url}: {e}")
            if i < retries - 1:
                time.sleep(3 * (i + 1))
    return None


def keep_doc(t: str, lo: int = 120, hi: int = 3000) -> str | None:
    t = re.sub(r"\s+", " ", t).strip()
    if lo <= len(t) <= hi:
        return t
    return None


def gutenberg_docs() -> list[str]:
    docs = []
    for gid, label in GUTENBERG:
        txt = fetch(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt")
        if not txt:
            continue
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")  # CRLF -> LF
        body = txt.split("*** START OF", 1)[-1].split("*** END OF", 1)[0]
        paras = [re.sub(r"\s+", " ", p).strip()
                 for p in body.split("\n\n") if p.strip()]
        kept = [p for p in paras if (d := keep_doc(p, hi=2200)) is not None]
        print(f"  gutenberg/{label}: {len(paras)} paras -> {len(kept)} docs")
        docs.extend(kept)
        time.sleep(0.5)
    return docs


def wiki_docs() -> list[str]:
    docs = []
    for topic in WIKI_TOPICS:
        url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
               f"&prop=extracts&explaintext=1&exintro=1&titles={requests.utils.quote(topic)}&redirects=1")
        data = fetch(url)
        if not data:
            continue
        try:
            pages = requests.get(url, headers=HDRS, timeout=60).json()["query"]["pages"]
        except Exception:
            continue
        for page in pages.values():
            d = keep_doc(page.get("extract", ""))
            if d:
                docs.append(d)
                print(f"  wiki/{topic}: {len(d)} chars")
        time.sleep(1.0)
    return docs


def arxiv_docs() -> list[str]:
    docs = []
    for cat in ARXIV_CATS:
        url = ("https://export.arxiv.org/api/query?search_query="
               f"cat:{cat}&start=0&max_results=40&sortBy=submittedDate&sortOrder=descending")
        txt = fetch(url)
        if not txt:
            continue
        titles = re.findall(r"<title>(.*?)</title>", txt, re.S)
        sums = re.findall(r"<summary>(.*?)</summary>", txt, re.S)
        for s in sums:
            d = keep_doc(s)
            if d:
                docs.append(d)
        print(f"  arxiv/{cat}: {len(sums)} abstracts -> {len([1 for s in sums if keep_doc(s)])} docs")
        time.sleep(1.2)
    return docs


def code_files() -> list[str]:
    """Real source (MIT/BSD/Apache) chunked into code documents.

    Each source file is split into blocks of ~300 chars (lines joined with a
    space so each block is one corpus line/document), giving many code-like
    documents of manageable length for next-token sampling.
    """
    docs = []
    for url, lic in CODE_FILES:
        txt = fetch(url)
        if not txt:
            continue
        lines = [ln.strip() for ln in txt.splitlines()]
        lines = [ln for ln in lines
                 if ln and not ln.lstrip().startswith(("//", "#", "*", "/*"))]
        blocks = []; cur = []; curlen = 0
        for ln in lines:
            cur.append(ln); curlen += len(ln) + 1
            if curlen >= 300:
                blocks.append(" ".join(cur)); cur = []; curlen = 0
        if cur:
            blocks.append(" ".join(cur))
        kept = [b for b in blocks if (d := keep_doc(b, lo=120, hi=3000)) is not None]
        print(f"  code/{lic.split()[-1]}: {len(blocks)} blocks -> {len(kept)} docs")
        docs.extend(kept)
        time.sleep(0.3)
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus/production.txt")
    args = ap.parse_args()

    print("building production corpus (prose + knowledge + technical + code) ...")
    docs = []
    docs += gutenberg_docs()
    docs += wiki_docs()
    docs += arxiv_docs()
    docs += code_files()

    if not docs:
        print("no documents fetched — check network"); return 1

    seen = set(); uniq = []
    for d in docs:
        if d not in seen:
            seen.add(d); uniq.append(d)
    random.Random(0).shuffle(uniq)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(uniq) + "\n")

    total_chars = sum(len(d) for d in uniq)
    est = sum(max(0, (len(d) - 8 - 64) // 64 + 1) for d in uniq)
    print(f"\nwrote {out}: {len(uniq)} docs, {total_chars} chars "
          f"(~{total_chars//4} tokens), ~{est} candidate positions")
    print(f"corpus sha256: {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
