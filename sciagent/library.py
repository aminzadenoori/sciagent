"""Add a single paper to the corpus: a local PDF, an arXiv id / URL, or a direct PDF link.

    python -m sciagent add path/to/paper.pdf
    python -m sciagent add 2508.16983
    python -m sciagent add https://arxiv.org/abs/2508.16983
    python -m sciagent add https://example.org/paper.pdf --title "My title"

The paper is appended to data/corpus/added.jsonl, its PDF copied to
data/corpus/pdfs/, and the corpus is re-ingested and re-indexed so it is
searchable immediately (with its tables and figures).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf

from .collect import _get, parse_arxiv_atom

ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/)?(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$", re.I)


def arxiv_id(source: str) -> str | None:
    m = ARXIV_RE.search(source.strip())
    return m.group(1) if m and not Path(source).exists() else None


def pdf_title(path: Path) -> tuple[str, list[str], int | None]:
    """Best-effort title / authors / year from a PDF: metadata first, else largest text on page 1."""
    doc = pymupdf.open(path)
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip()
    authors = [a.strip() for a in re.split(r"[;,]| and ", meta.get("author") or "") if a.strip()]
    year = None
    m = re.search(r"(19|20)\d{2}", meta.get("creationDate") or "")
    if m:
        year = int(m.group(0))
    if len(title) < 8 or title.lower().startswith(("microsoft word", "untitled")):
        best, size = "", 0.0
        for b in doc[0].get_text("dict")["blocks"] if doc.page_count else []:
            for ln in b.get("lines", []):
                t = "".join(s["text"] for s in ln["spans"]).strip()
                s = max((sp["size"] for sp in ln["spans"]), default=0)
                if len(t) > 8 and s > size + 0.5:
                    best, size = t, s
                elif len(t) > 3 and abs(s - size) < 0.5 and best and len(best) < 160:
                    best += " " + t  # title spanning two lines
        title = best or path.stem
    return re.sub(r"\s+", " ", title)[:250], authors[:12], year


def add_paper(source: str, corpus_dir: Path, index_dir: Path, title: str | None = None,
              fetch=_get, rebuild: bool = True) -> dict:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "pdfs").mkdir(exist_ok=True)
    aid = arxiv_id(source)
    rec: dict
    if aid:
        try:
            recs = parse_arxiv_atom(fetch(f"https://export.arxiv.org/api/query?id_list={aid}"))
            rec = recs[0] if recs else {}
        except Exception:
            rec = {}
        rec = {"id": f"arxiv:{aid}", "arxiv_id": aid, "url": f"https://arxiv.org/abs/{aid}", **rec}
        pdf_bytes = fetch(f"https://arxiv.org/pdf/{aid}")
    elif re.match(r"https?://", source):
        pdf_bytes = fetch(source)
        rec = {"id": "url:" + hashlib.sha1(source.encode()).hexdigest()[:10], "url": source}
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"{source} is not a file, an arXiv id or a URL")
        pdf_bytes = p.read_bytes()
        rec = {"id": "local:" + hashlib.sha1(pdf_bytes).hexdigest()[:10]}
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("the downloaded file is not a PDF")
    fn = re.sub(r"[^\w.-]", "_", rec["id"]) + ".pdf"
    (corpus_dir / "pdfs" / fn).write_bytes(pdf_bytes)
    t, authors, year = pdf_title(corpus_dir / "pdfs" / fn)
    rec.setdefault("title", t)
    rec.setdefault("authors", authors)
    rec.setdefault("year", year)
    if title:
        rec["title"] = title
    rec.update({"pdf": f"pdfs/{fn}", "source": "added"})
    rec.pop("pdf_url", None)

    path = corpus_dir / "added.jsonl"
    rows = [json.loads(l) for l in open(path)] if path.exists() else []
    rows = [r for r in rows if r["id"] != rec["id"]] + [rec]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    out = {"paper_id": rec["id"], "title": rec["title"]}
    if rebuild:
        from .index import HybridIndex
        from .ingest import ingest_corpus
        report = ingest_corpus(corpus_dir, index_dir)
        idx = HybridIndex.build(index_dir)
        kinds = {}
        for c in idx.chunks:
            if c["paper_id"] == rec["id"]:
                kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        out.update({"chunks": kinds, "report": report.get(rec["id"], {}), "index": idx})
    return out


def copy_upload(data: bytes, filename: str, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / (re.sub(r"[^\w.-]", "_", Path(filename).name) or "upload.pdf")
    p.write_bytes(data)
    return p
