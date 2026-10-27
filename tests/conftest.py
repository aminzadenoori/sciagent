"""Shared fixtures: a tiny corpus built on the fly (no network, no API key).

It deliberately contains the awkward cases ingestion must survive:
a normal PDF with a figure and a table (with decimals and a ligature),
a corrupt PDF, an image-only "scanned" PDF, a record whose PDF is missing,
and a structured card that duplicates the normal PDF's id.
"""
from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _png() -> bytes:
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
    ax.bar(["A", "B"], [90, 10])
    ax.set_ylabel("score (%)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _story_pdf(path: Path, html: str, images: dict[str, bytes] | None = None):
    arch = pymupdf.Archive()
    for n, b in (images or {}).items():
        arch.add(b, n)
    css = "body {font-size: 10px;} h1 {font-size: 15px;} h2 {font-size: 12px;} td, th {border: 1px solid black; padding: 3px;} " \
          "table {border-collapse: collapse;} .caption {font-size: 8px;}"
    story = pymupdf.Story(html=html, user_css=css, archive=arch)
    w = pymupdf.DocumentWriter(str(path))
    rect = pymupdf.paper_rect("letter")
    more = True
    while more:
        dev = w.begin_page(rect)
        more, _ = story.place(rect + (50, 50, -50, -50))
        story.draw(dev)
        w.end_page()
    w.close()


GOOD_HTML = """<h1>Toy embeddings for knowledge graphs</h1>
<h2>Abstract</h2><p>We study a toy model called ToyE on the ToyKG benchmark. It reaches MRR 0.427 with an efficient
configuration.</p>
<h2>1. Method</h2><p>ToyE uses rotations and reflections. The margin is 9.5 and the dimension is 64.</p>
<p class="caption">Table 1. Results on ToyKG.</p>
<table><tr><th>Model</th><th>MRR</th><th>Hits@10</th></tr><tr><td>ToyE</td><td>0.427</td><td>0.512</td></tr>
<tr><td>Baseline</td><td>0.301</td><td>0.455</td></tr></table>
<h2>2. Results</h2><p>ToyE beats the baseline by 0.126 MRR. Figure 1 compares the two systems.</p>
<img src="fig.png" width="250"/>
<p class="caption">Figure 1. Score of system A (90 percent) versus system B (10 percent).</p>
<h2>3. Conclusion</h2><p>Rotations help on ToyKG.</p>"""


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("corpus")
    (d / "pdfs").mkdir()
    _story_pdf(d / "pdfs" / "good.pdf", GOOD_HTML, {"fig.png": _png()})
    (d / "pdfs" / "corrupt.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    scan = pymupdf.open()
    page = scan.new_page()
    page.insert_image(pymupdf.Rect(50, 50, 500, 400), stream=_png())
    scan.save(d / "pdfs" / "scan.pdf")
    metas = [
        {"id": "toy:good", "title": "Toy embeddings for knowledge graphs", "authors": ["Ada Lovelace"], "year": 2024, "pdf": "pdfs/good.pdf",
         "abstract": "short abstract"},
        {"id": "toy:good", "title": "Toy embeddings (card)", "authors": ["Ada Lovelace"], "year": 2024,
         "sections": [["Summary", "card text that should lose to the PDF"]]},
        {"id": "toy:corrupt", "title": "Corrupt file", "authors": ["B. Broken"], "year": 2020, "pdf": "pdfs/corrupt.pdf",
         "abstract": "Fallback abstract about negative sampling for embeddings."},
        {"id": "toy:scan", "title": "Scanned paper", "authors": ["C. Scanner"], "year": 2019, "pdf": "pdfs/scan.pdf",
         "abstract": "Abstract of a scanned paper on link prediction."},
        {"id": "toy:missing", "title": "Missing PDF", "authors": ["D. Gone"], "year": 2021, "pdf": "pdfs/nope.pdf",
         "abstract": "Abstract about knowledge graph question answering with constrained decoding."},
        {"id": "toy:card", "title": "Card paper", "authors": ["Ada Lovelace", "E. Other"], "year": 2025,
         "sections": [["Method", "CardNet aligns graph embeddings with a language model using a linear projection."],
                      ["Results", "CardNet improves exact match from 0.20 to 0.61 on CardQA."]]},
    ]
    with open(d / "meta.jsonl", "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    return d


@pytest.fixture(scope="session")
def built(corpus, tmp_path_factory):
    from sciagent.index import HybridIndex
    from sciagent.ingest import ingest_corpus
    out = tmp_path_factory.mktemp("index")
    report = ingest_corpus(corpus, out)
    return SimpleNamespace(report=report, index=HybridIndex.build(out), dir=out)


# ------------------------------------------------------------------ fake Claude
def tool_use(name, input, id=None):
    return SimpleNamespace(type="tool_use", id=id or f"tu_{name}_{abs(hash(json.dumps(input, sort_keys=True))) % 10**6}",
                           name=name, input=input)


def text(t):
    return SimpleNamespace(type="text", text=t)


def response(*blocks):
    return SimpleNamespace(content=list(blocks), stop_reason="tool_use",
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))


class FakeClient:
    """Scripted stand-in for anthropic.Anthropic(). `script` items are responses or callables(kwargs)->response."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(copy.deepcopy(kw))  # the agent mutates its message list later
        if not self.script:
            raise AssertionError("FakeClient script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(kw) if callable(item) else item


@pytest.fixture
def fake():
    return SimpleNamespace(Client=FakeClient, tool_use=tool_use, text=text, response=response)


ATOM_ONE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><entry>
  <id>http://arxiv.org/abs/2508.16983v2</id><published>2025-08-23T10:00:00Z</published>
  <title>ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation</title>
  <summary>Abstract.</summary><author><name>Riccardo Pozzi</name></author><author><name>Sahar Vahdati</name></author>
</entry></feed>"""
