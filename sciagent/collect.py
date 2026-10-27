"""Automated paper collection: arXiv + OpenAlex, seeded by an author and topics.

Pipeline
1. Seed     - the author's works (OpenAlex author search, arXiv `au:` query)
              plus topic queries on arXiv.
2. Expand   - one hop along OpenAlex references of the seed works, so the
              corpus also contains the papers the author builds on / argues
              with (e.g. RotatE for the "loss-aware pattern inference" paper).
3. Filter   - keep works whose title+abstract match the topic vocabulary
              (cheap lexical relevance score; threshold configurable).
4. Dedupe   - by arXiv id, DOI and normalised title.
5. Download - open-access PDFs (arXiv first), polite rate limit + retries.
              Failures degrade to abstract-only records instead of crashing.

Google Scholar is not used: it forbids automated access (robots.txt).
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

UA = "sciagent/0.1 (research prototype; mailto:you@example.com)"
ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass
class CollectConfig:
    author: str = "Sahar Vahdati"
    topics: list[str] = field(default_factory=lambda: [
        "knowledge graph embedding link prediction",
        "knowledge graph large language model factual accuracy hallucination",
    ])
    vocabulary: list[str] = field(default_factory=lambda: [
        "knowledge graph", "embedding", "link prediction", "language model", "llm", "hallucination", "factual",
        "reasoning", "transe", "rotate", "relation", "entity", "scholarly", "abstraction", "arc"])
    max_per_query: int = 25
    expand_refs: int = 40
    min_relevance: int = 2
    download_pdfs: bool = True
    sleep: float = 3.0  # arXiv asks for >= 3 s between requests


def _get(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network errors, 429/503
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"GET failed after {retries} tries: {url}: {last}")


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


# ---------------------------------------------------------------- arXiv
def parse_arxiv_atom(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        aid = e.findtext(f"{ATOM}id", "").rsplit("/abs/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        pdf = next((l.get("href") for l in e.findall(f"{ATOM}link") if l.get("title") == "pdf"), None)
        out.append({
            "id": f"arxiv:{aid}", "arxiv_id": aid,
            "title": " ".join(e.findtext(f"{ATOM}title", "").split()),
            "abstract": " ".join(e.findtext(f"{ATOM}summary", "").split()),
            "authors": [a.findtext(f"{ATOM}name", "") for a in e.findall(f"{ATOM}author")],
            "year": int(e.findtext(f"{ATOM}published", "0000")[:4] or 0),
            "pdf_url": pdf or f"https://arxiv.org/pdf/{aid}", "source": "arxiv",
        })
    return out


def arxiv_search(query: str, n: int) -> list[dict]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": n, "sortBy": "relevance"})
    return parse_arxiv_atom(_get(url))


# ---------------------------------------------------------------- OpenAlex
def _abstract_from_inverted(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def openalex_work_to_record(w: dict) -> dict:
    ids = w.get("ids", {}) or {}
    arx = None
    for loc in w.get("locations", []) or []:
        u = (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or "")
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", u)
        if m:
            arx = m.group(1)
            break
    oa = (w.get("best_oa_location") or {}).get("pdf_url")
    return {
        "id": f"arxiv:{arx}" if arx else f"openalex:{w['id'].rsplit('/', 1)[-1]}",
        "arxiv_id": arx, "doi": ids.get("doi"), "openalex_id": w["id"],
        "title": w.get("display_name") or "", "abstract": _abstract_from_inverted(w.get("abstract_inverted_index")),
        "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
        "year": w.get("publication_year"), "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "cited_by": w.get("cited_by_count"), "references": w.get("referenced_works", []),
        "pdf_url": f"https://arxiv.org/pdf/{arx}" if arx else oa, "source": "openalex",
    }


def openalex_author_works(name: str, n: int) -> list[dict]:
    a = json.loads(_get("https://api.openalex.org/authors?" + urllib.parse.urlencode({"search": name, "per-page": 1})))
    if not a.get("results"):
        return []
    aid = a["results"][0]["id"].rsplit("/", 1)[-1]
    w = json.loads(_get("https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"filter": f"author.id:{aid}", "sort": "publication_date:desc", "per-page": n})))
    return [openalex_work_to_record(x) for x in w.get("results", [])]


def openalex_works_by_ids(ids: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(ids), 50):
        chunk = "|".join(x.rsplit("/", 1)[-1] for x in ids[i:i + 50])
        w = json.loads(_get("https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": f"openalex_id:{chunk}", "per-page": 50})))
        out += [openalex_work_to_record(x) for x in w.get("results", [])]
    return out


def openalex_search(query: str, n: int = 10) -> list[dict]:
    w = json.loads(_get("https://api.openalex.org/works?" + urllib.parse.urlencode({"search": query, "per-page": n})))
    return [openalex_work_to_record(x) for x in w.get("results", [])]


# ---------------------------------------------------------------- pipeline
def relevance(rec: dict, vocab: list[str]) -> int:
    text = (rec.get("title", "") + " " + rec.get("abstract", "")).lower()
    return sum(1 for v in vocab if v in text)


def dedupe(recs: list[dict]) -> list[dict]:
    seen, out = {}, []
    for r in recs:
        keys = [k for k in (r.get("arxiv_id"), r.get("doi"), norm_title(r.get("title", ""))) if k]
        hit = next((seen[k] for k in keys if k in seen), None)
        if hit is not None:  # merge: keep richer fields
            for f in ("abstract", "pdf_url", "venue", "doi", "cited_by"):
                if not hit.get(f) and r.get(f):
                    hit[f] = r[f]
            hit["seed"] = hit.get("seed") or r.get("seed")
            continue
        for k in keys:
            seen[k] = r
        out.append(r)
    return out


def collect(cfg: CollectConfig, corpus_dir: Path, log=print) -> list[dict]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    try:
        seed = openalex_author_works(cfg.author, cfg.max_per_query)
        for r in seed:
            r["seed"] = "author"
        recs += seed
        log(f"openalex author works: {len(seed)}")
    except Exception as e:
        log(f"[warn] OpenAlex author lookup failed: {e}")
        seed = []
    last = cfg.author.split()[-1]
    first = cfg.author.split()[0][0]
    for q in [f'au:"{last}_{first}"'] + [f'all:"{t}"' if len(t.split()) < 4 else "all:" + " AND all:".join(t.split()[:4])
                                         for t in cfg.topics]:
        try:
            got = arxiv_search(q, cfg.max_per_query)
            for r in got:
                r["seed"] = "author" if q.startswith("au:") else "topic"
            recs += got
            log(f"arxiv {q!r}: {len(got)}")
        except Exception as e:
            log(f"[warn] arXiv query failed ({q}): {e}")
        time.sleep(cfg.sleep)
    # one-hop expansion along references of the author's works
    refs = [x for r in seed for x in r.get("references", [])]
    if refs and cfg.expand_refs:
        from collections import Counter
        top = [x for x, _ in Counter(refs).most_common(cfg.expand_refs)]
        try:
            exp = openalex_works_by_ids(top)
            for r in exp:
                r["seed"] = "reference"
            recs += exp
            log(f"reference expansion: {len(exp)}")
        except Exception as e:
            log(f"[warn] reference expansion failed: {e}")
    recs = dedupe(recs)
    kept = [r for r in recs if r.get("seed") == "author" and cfg.author.split()[-1] in " ".join(r.get("authors", []))
            or relevance(r, cfg.vocabulary) >= cfg.min_relevance]
    log(f"after dedupe: {len(recs)}, after relevance filter: {len(kept)}")
    if cfg.download_pdfs:
        (corpus_dir / "pdfs").mkdir(exist_ok=True)
        for r in kept:
            if not r.get("pdf_url"):
                continue
            fn = corpus_dir / "pdfs" / (re.sub(r"[^\w.-]", "_", r["id"]) + ".pdf")
            if not fn.exists():
                try:
                    data = _get(r["pdf_url"], timeout=60)
                    if not data.startswith(b"%PDF"):
                        raise ValueError("not a PDF")
                    fn.write_bytes(data)
                    time.sleep(cfg.sleep)
                except Exception as e:
                    log(f"[warn] PDF download failed for {r['id']}: {e} (keeping abstract only)")
                    continue
            r["pdf"] = str(fn.relative_to(corpus_dir))
    for r in kept:
        r.pop("references", None)
    with open(corpus_dir / "collected.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    return kept
