"""PDF -> structured records: section-aware text chunks, tables, figures.

Design notes
* Headings are found from font statistics (larger/bold than body text) plus a
  numbered-heading regex, so chunks never straddle sections and every chunk
  carries "title > section" context (cheap "contextual retrieval").
* Tables are extracted with PyMuPDF's table finder and stored as Markdown with
  their caption, so table-only facts are retrievable as text.
* Figures are extracted as PNGs and linked to their caption. The caption is
  indexed as text; the image itself is available to the vision tool.
* Text is NFKC-normalised (PDF ligatures like "fi" break exact matching).
* PDFs with almost no extractable text are flagged `needs_ocr` instead of being
  silently indexed as empty.
"""
from __future__ import annotations

import json
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)

HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?\s+[A-Z][\w ,:;()\-/]{2,80}|Abstract|References|Acknowledg(e)?ments?)$")
CAPTION_RE = re.compile(r"^(Fig(ure)?\.?\s*\d+|Table\s*\d+)", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"-\n(?=[a-z])", "", s)  # de-hyphenate line breaks
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    kind: str  # text | table | figure
    section: str
    text: str
    page: int
    meta: dict = field(default_factory=dict)


def _lines(page):
    """Yield (text, size, bold, bbox) per line."""
    d = page.get_text("dict")
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans)
            size = max(s["size"] for s in spans)
            bold = all(("Bold" in s["font"]) or (s["flags"] & 16) for s in spans)
            yield text, size, bold, ln["bbox"]


def chunk_words(words: list[str], size: int = 140, overlap: int = 30):
    if len(words) <= size:
        yield " ".join(words)
        return
    step = size - overlap
    for i in range(0, max(1, len(words) - overlap), step):
        yield " ".join(words[i:i + size])


def parse_pdf(pdf_path: Path, paper_id: str, title: str, fig_dir: Path) -> tuple[list[Chunk], dict]:
    doc = pymupdf.open(pdf_path)
    tables = {}
    for pno, page in enumerate(doc):
        try:
            tables[pno] = page.find_tables().tables
        except Exception:
            tables[pno] = []

    def in_table(pno, bbox):
        r = pymupdf.Rect(bbox)
        return any(pymupdf.Rect(t.bbox).contains(r) or pymupdf.Rect(t.bbox).intersects(r) and
                   (pymupdf.Rect(t.bbox) & r).get_area() > 0.6 * r.get_area() for t in tables[pno])

    all_lines = [(pno, *ln) for pno, page in enumerate(doc) for ln in _lines(page) if not in_table(pno, ln[3])]
    stats = {"pages": doc.page_count, "needs_ocr": False, "tables": 0, "figures": 0}
    if sum(len(l[1]) for l in all_lines) < 200 * max(1, doc.page_count) * 0.1:
        stats["needs_ocr"] = True
        return [], stats
    body_size = statistics.median(l[2] for l in all_lines)

    # ---- section segmentation
    sections: list[tuple[str, int, list[str]]] = []  # (name, page, lines)
    cur_name, cur_page, buf = "Front matter", 0, []
    captions: list[tuple[int, str, tuple]] = []
    caption_mode = None
    for pno, text, size, bold, bbox in all_lines:
        t = norm(text)
        is_heading = (size > body_size + 0.8 or bold) and len(t) < 90 and HEADING_RE.match(t)
        if is_heading:
            sections.append((cur_name, cur_page, buf))
            cur_name, cur_page, buf = re.sub(r"^\d+(\.\d+)*\.?\s+", "", t), pno, []
            caption_mode = None
            continue
        if CAPTION_RE.match(t):
            captions.append((pno, t, bbox))
            caption_mode = len(captions) - 1
            continue
        if caption_mode is not None and size < body_size - 0.3:  # caption continues in small font
            p, c, bb = captions[caption_mode]
            captions[caption_mode] = (p, c + " " + t, bb)
            continue
        caption_mode = None
        buf.append(t)
    sections.append((cur_name, cur_page, buf))

    chunks: list[Chunk] = []
    n = 0
    for name, pno, lines in sections:
        if name.lower().startswith("reference") or not lines:
            continue
        text = norm(" ".join(lines))
        for piece in chunk_words(text.split()):
            chunks.append(Chunk(f"{paper_id}#t{n}", paper_id, "text", name,
                                f"[{title} > {name}] {piece}", pno))
            n += 1

    # ---- tables
    for pno, page in enumerate(doc):
        for ti, tab in enumerate(tables[pno]):
            # Cell-wise text: tab.extract() scrambles decimals in some PDFs ("0.27" -> "027 .").
            rows = [[norm(page.get_textbox(pymupdf.Rect(c))) if c else "" for c in r.cells] for r in tab.rows]
            if not rows or len(rows) < 2:
                continue
            md = "| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n"
            md += "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
            cap = _nearest_caption(captions, pno, tab.bbox, "table")
            stats["tables"] += 1
            chunks.append(Chunk(f"{paper_id}#tab{pno}_{ti}", paper_id, "table", "Table",
                                f"[{title} > {cap or 'Table'}]\n{md}", pno, {"caption": cap, "rows": rows}))

    # ---- figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    for pno, page in enumerate(doc):
        for ii, info in enumerate(page.get_image_info(xrefs=True)):
            x0, y0, x1, y1 = info["bbox"]
            if (x1 - x0) < 80 or (y1 - y0) < 60 or not info.get("xref"):
                continue  # logos, icons
            pix = pymupdf.Pixmap(doc, info["xref"])
            if pix.n - pix.alpha >= 4:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            out = fig_dir / f"{paper_id}_p{pno}_{ii}.png"
            pix.save(out)
            cap = _nearest_caption(captions, pno, info["bbox"], "figure") or "(no caption found)"
            stats["figures"] += 1
            chunks.append(Chunk(f"{paper_id}#fig{pno}_{ii}", paper_id, "figure", "Figure",
                                f"[{title} > Figure] {cap}", pno, {"caption": cap, "image_path": str(out)}))
    return chunks, stats


def _nearest_caption(captions, pno, bbox, kind):
    want = "fig" if kind == "figure" else "table"
    best, bd = None, 1e9
    for p, c, bb in captions:
        if p != pno or not c.lower().startswith(want):
            continue
        d = min(abs(bb[1] - bbox[3]), abs(bb[3] - bbox[1]))  # below figure / above table
        if d < bd:
            best, bd = c, d
    return best


def ingest_corpus(corpus_dir: Path, out_dir: Path) -> dict:
    """corpus_dir has *.jsonl metadata (+ pdfs/, optional supp/*.csv). Writes chunks.jsonl, papers.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    papers, chunks, report = [], [], {}
    metas = []
    for fn in sorted(corpus_dir.glob("*.jsonl")):
        metas += [json.loads(l) for l in open(fn) if l.strip()]
    # Same paper from several sources (seed card + collected PDF): keep the richest record.
    rank = lambda m: (2 if m.get("pdf") and (corpus_dir / m["pdf"]).exists() else 1 if m.get("sections") else 0)
    best = {}
    for m in metas:
        if m["id"] not in best or rank(m) > rank(best[m["id"]]):
            best[m["id"]] = m
    for m in best.values():
        pdf = corpus_dir / m["pdf"] if m.get("pdf") else None
        if m.get("sections"):  # pre-structured record (seed "paper cards")
            cs = [Chunk(f"{m['id']}#t{i}", m["id"], "text", name, f"[{m['title']} > {name}] {norm(body)}", 0)
                  for i, (name, body) in enumerate(m["sections"])]
            st = {"source": "card"}
        elif pdf and pdf.exists():
            try:
                cs, st = parse_pdf(pdf, m["id"], m["title"], out_dir / "figures")
            except Exception as e:  # corrupt PDF: fall back to abstract only
                cs, st = [], {"error": repr(e)}
        else:
            cs, st = [], {"missing_pdf": True}
        if not cs and m.get("abstract"):  # graceful degradation: abstract-only paper
            cs = [Chunk(f"{m['id']}#abs", m["id"], "text", "Abstract", f"[{m['title']} > Abstract] {m['abstract']}", 0)]
            st["abstract_only"] = True
        supp = sorted((corpus_dir / "supp").glob(f"{m['id']}_*.csv")) if (corpus_dir / "supp").exists() else []
        m["data_files"] = [str(p) for p in supp]
        papers.append(m)
        chunks.extend(cs)
        report[m["id"]] = st
    with open(out_dir / "papers.jsonl", "w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    with open(out_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    json.dump(report, open(out_dir / "ingest_report.json", "w"), indent=1)
    return report
