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
    author: str | None = "Sahar Vahdati"  # None/"" = topic-only collection
    topics: list[str] = field(default_factory=lambda: [
        "knowledge graph embedding link prediction",
        "knowledge graph large language model factual accuracy hallucination",
    ])
    vocabulary: list[str] | None = None  # None = derived from the topics
    max_per_query: int = 25
    expand_refs: int = 40
    min_relevance: int = 2
    min_similarity: float = 0.08  # TF-IDF cosine between paper and topics (no training)
    max_papers: int = 80
    download_pdfs: bool = True
    sleep: float = 3.0  # arXiv asks for >= 3 s between requests

    def vocab(self) -> list[str]:
        if self.vocabulary:
            return self.vocabulary
        stop = {"and", "or", "of", "for", "the", "a", "an", "in", "on", "with", "to", "via", "using"}
        pairs, words = [], []
        for t in self.topics:  # topic phrases as adjacent word pairs, e.g. "knowledge graph", "link prediction"
            toks = [w for w in re.findall(r"[a-z0-9-]+", t.lower()) if w not in stop]
            pairs += [f"{x} {y}" for x, y in zip(toks, toks[1:])]
            words += toks
        return sorted(set(pairs)) or sorted(set(words))


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


def topic_similarity(recs: list[dict], topics: list[str]) -> list[float]:
    """TF-IDF cosine between each paper (title + abstract) and the topic descriptions. Unsupervised: no training."""
    if not recs or not topics:
        return [0.0] * len(recs)
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [(r.get("title", "") + " " + r.get("abstract", "")) for r in recs]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(docs + topics)
    D, T = vec.transform(docs), vec.transform(topics)
    return (D @ T.T).toarray().max(axis=1).tolist()


def collect(cfg: CollectConfig, corpus_dir: Path, log=print) -> list[dict]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    recs, seed = [], []
    report = {"author": cfg.author or None, "topics": cfg.topics, "found": {}, "warnings": []}

    def warn(msg):
        report["warnings"].append(msg)
        log(f"[warn] {msg}")

    author = (cfg.author or "").strip()
    if author:
        try:
            seed = openalex_author_works(author, cfg.max_per_query)
            for r in seed:
                r["seed"] = "author"
            recs += seed
            report["found"]["openalex_author"] = len(seed)
            log(f"openalex author works: {len(seed)}")
        except Exception as e:
            warn(f"OpenAlex author lookup failed: {e}")
    queries = []
    if author:
        queries.append(f'au:"{author.split()[-1]}_{author.split()[0][0]}"')
    queries += [f'all:"{t}"' if len(t.split()) < 4 else "all:" + " AND all:".join(t.split()[:4]) for t in cfg.topics]
    for t in cfg.topics:  # OpenAlex topic search complements arXiv (journals, conferences)
        try:
            got = openalex_search(t, cfg.max_per_query)
            for r in got:
                r["seed"] = "topic"
            recs += got
            report["found"][f"openalex:{t}"] = len(got)
        except Exception as e:
            warn(f"OpenAlex search failed ({t}): {e}")
    for q in queries:
        try:
            got = arxiv_search(q, cfg.max_per_query)
            for r in got:
                r["seed"] = "author" if q.startswith("au:") else "topic"
            recs += got
            report["found"][f"arxiv:{q}"] = len(got)
            log(f"arxiv {q!r}: {len(got)}")
        except Exception as e:
            warn(f"arXiv query failed ({q}): {e}")
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
            report["found"]["reference_expansion"] = len(exp)
            log(f"reference expansion: {len(exp)}")
        except Exception as e:
            warn(f"reference expansion failed: {e}")
    report["raw"] = len(recs)
    recs = dedupe(recs)
    report["after_dedupe"] = len(recs)
    sims = topic_similarity(recs, cfg.topics)
    vocab = cfg.vocab()
    last = author.split()[-1] if author else None
    kept = []
    for r, sim in zip(recs, sims):
        r["relevance"] = {"keyword_hits": relevance(r, vocab), "topic_similarity": round(sim, 3)}
        by_author = bool(last) and r.get("seed") == "author" and last in " ".join(r.get("authors", []))
        if by_author or r["relevance"]["keyword_hits"] >= cfg.min_relevance or sim >= cfg.min_similarity:
            r["kept_because"] = "author" if by_author else "topic"
            kept.append(r)
    kept.sort(key=lambda r: (r["kept_because"] != "author", -r["relevance"]["topic_similarity"]))
    kept = kept[:cfg.max_papers]
    report["after_relevance_filter"] = len(kept)
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
                    warn(f"PDF download failed for {r['id']}: {e} (keeping abstract only)")
                    continue
            r["pdf"] = str(fn.relative_to(corpus_dir))
    report["with_pdf"] = sum(1 for r in kept if r.get("pdf"))
    report["abstract_only"] = len(kept) - report["with_pdf"]
    for r in kept:
        r.pop("references", None)
    # incremental: merge with papers collected in earlier runs
    path = corpus_dir / "collected.jsonl"
    old = [json.loads(l) for l in open(path)] if path.exists() else []
    merged = {r["id"]: r for r in old}
    merged.update({r["id"]: r for r in kept})
    report["new_papers"] = len(set(r["id"] for r in kept) - set(r["id"] for r in old))
    report["total_in_corpus"] = len(merged)
    with open(path, "w") as f:
        for r in merged.values():
            f.write(json.dumps(r) + "\n")
    (corpus_dir / "collect_report.json").write_text(json.dumps(report, indent=1))
    return kept
