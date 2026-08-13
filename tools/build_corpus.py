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
import json
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

# Multilingual component — glimmer is trained on 100+ languages, so the corpus
# must cover non-English fidelity. We pick a set of CONCRETE *concept* articles
# and resolve each concept to its canonical title in every language via
# Wikipedia's interlanguage links (langlinks). This guarantees the science
# article in each language, not a band/album/song that happens to share the
# English word. (Passing the bare English word "Chemistry" to non-English wikis
# lands on the Kelly Clarkson album or a J-pop duo — not the concept.)
MULTI_LANGS = ["de", "fr", "es", "pt", "it", "nl", "ru", "pl", "uk", "ar",
               "fa", "tr", "he", "hi", "bn", "zh", "ja", "ko", "vi", "th",
               "id", "sw"]
# Concepts are English article names; each is resolved to per-language titles.
MULTI_TITLES = ["Physics", "Chemistry", "Biology", "Astronomy", "Mathematics",
                "Geology", "Medicine", "Geography", "Agriculture", "Engineering"]

# Language-family markers of disambiguation pages ("X can refer to:"). This is
# the real safety net (independent of title choice): any intro that reads as a
# list-of-meanings is dropped, whatever the language. Patterns are per-family;
# unknown languages fall through to the structural heuristic below.
_DISAMB_MARKERS = [
    # Germanic (de/nl)  "X steht für / verwijzen naar / bezeichnet"
    "steht für", "verweist auf", "verwijzen naar", "kan verwijzen", "wijst",
    # Romance (fr/es/pt/it)  "X peut désigner / puede referirse / può riferirsi"
    "peut désigner", "peut faire référence", "peut renvoyer",
    "puede referirse", "puede hacer referencia", "puede ser",
    "pode referir", "pode se referir", "può riferirsi", "può essere",
    # Slavic (ru/pl/uk)  "X может означать / może oznaczać / може означати"
    "может означать", "может обозначать", "може означати",
    "może oznaczać", "może odnosić", "mоже да се отнася",
    # Semitic (ar/he/fa)  "X قد تشير / עשוי להתייחס / میتواند اشاره"
    "قد تشير", "قد يعني", "עשוי להתייחס", "יכול להתייחס", "میتواند اشاره",
    # South/Southeast Asian + CJK
    "dapat mengacu", "mungkin merujuk", "menunjuk kepada", "chỉ đến",
    "có thể là", "đề cập", "може да се отнася",
]
_DISAMB_RE = __import__("re").compile("|".join(_DISAMB_MARKERS), __import__("re").I)


def is_disambiguation(t: str) -> bool:
    """Heuristically detect a Wikipedia disambiguation page.

    Two independent signals, either of which drops the doc:
      1. A language-family disambiguation marker ("X may refer to:").
      2. A structural one: a short intro dominated by parenthetical qualifiers
         and title-case tokens (album titles, band names) — the shape of a
         list-of-meanings page even when no marker regex matched.
    """
    if _DISAMB_RE.search(t):
        return True
    # Structural: parenthetical-heavy + heavy title-case = catalogue of titles.
    parens = len(__import__("re").findall(r"\(", t))
    if len(t) < 200 and parens >= 3:
        return True
    # Many "X (qualifier)" patterns early in the text → list page.
    if __import__("re").search(r"\([^)]{2,40}\),?\s+[A-Z\u00C0-\u017F]+\(", t):
        return True
    return False

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


def multilingual_wiki_docs() -> list[str]:
    """Non-English Wikipedia article intros (multilingual fidelity for glimmer).

    Each concept (e.g. Mathematics) is resolved to its canonical article title
    in every target language via Wikipedia's interlanguage links (langlinks),
    so we fetch the *concept*, not a same-named band/album/song. Disambiguation
    pages are filtered as a safety net.
    """
    # 1) Resolve each concept -> {lang: local_title} via langlinks. Queried one
    #    concept at a time: a batched multi-title langlinks query returns empty
    #    langlinks for most pages (API continuation/limit quirk).
    concept_titles = {}
    for concept in MULTI_TITLES:
        ll_url = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
                  f"&prop=langlinks&lllimit=500&titles={requests.utils.quote(concept)}")
        data = fetch(ll_url)
        if data:
            try:
                pages = json.loads(data)["query"]["pages"]
            except Exception:
                pages = {}
            for page in pages.values():
                title = page.get("title", "")
                langs = {l["lang"]: l.get("*", "")
                         for l in page.get("langlinks", [])
                         if l.get("lang") in MULTI_LANGS}
                if langs:
                    concept_titles[title] = langs
        time.sleep(0.3)
    if not concept_titles:
        print("  ! langlinks failed — falling back to raw English titles")
        for t in MULTI_TITLES:
            concept_titles[t] = {lang: t for lang in MULTI_LANGS}

    # 2) Fetch each concept's intro in each language that has a local title.
    docs = []
    for concept, langs in concept_titles.items():
        for lang, local_title in langs.items():
            url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json"
                   f"&prop=extracts&explaintext=1&exintro=1&redirects=1"
                   f"&titles={requests.utils.quote(local_title)}")
            data = fetch(url, retries=4)
            if not data:
                continue
            try:
                pages = json.loads(data)["query"]["pages"]
            except Exception:
                continue
            for page in pages.values():
                d = keep_doc(page.get("extract", ""), hi=2200)
                if not d:
                    continue
                if is_disambiguation(d):
                    print(f"  wiki/{lang}:{concept} dropped (disamb)")
                    continue
                docs.append(d)
        time.sleep(0.5)
    print(f"  multilingual: {len(docs)} docs across {len(concept_titles)} concepts")
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

    print("building production corpus (prose + knowledge + technical + code + multilingual) ...")
    # (component, text) — component labels enable stratified sampling and the
    # per-component gate (no component can be a thin weak spot to overfit).
    docs: list[tuple[str, str]] = []
    for t in gutenberg_docs():
        docs.append(("prose", t))
    for t in wiki_docs():
        docs.append(("prose", t))
    for t in multilingual_wiki_docs():
        docs.append(("multilingual", t))
    for t in arxiv_docs():
        docs.append(("technical", t))
    for t in code_files():
        docs.append(("code", t))

    if not docs:
        print("no documents fetched — check network"); return 1

    seen: dict[str, str] = {}; uniq: list[tuple[str, str]] = []
    for comp, d in docs:
        if d not in seen:
            seen[d] = comp; uniq.append((comp, d))
    random.Random(0).shuffle(uniq)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # text docs (the stream basis) ...
    out.write_text("\n".join(t for _, t in uniq) + "\n")
    # ... and one component label per doc, aligned with the output order
    comp_out = out.with_suffix(".components.txt")
    comp_out.write_text("\n".join(c for c, _ in uniq) + "\n")

    total_chars = sum(len(d) for _, d in uniq)
    est = sum(max(0, (len(d) - 8 - 64) // 64 + 1) for _, d in uniq)
    print(f"\nwrote {out}: {len(uniq)} docs, {total_chars} chars "
          f"(~{total_chars//4} tokens), ~{est} candidate positions")
    print(f"corpus sha256: {hashlib.sha256(out.read_bytes()).hexdigest()}")
    print(f"components sha256: {hashlib.sha256(comp_out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
