"""End-to-end collection with simulated arXiv / OpenAlex responses (no network).

Checks the whole automated path: author seed + topic search + reference expansion ->
dedupe across sources -> relevance filter (keywords + TF-IDF topic similarity) ->
PDF download with abstract-only fallback -> incremental merge -> report -> ingestion.
Also covers topic-only collection and the agent's fetch_paper tool.
"""
import json
import urllib.parse

import pytest

import sciagent.collect as C
from sciagent.collect import CollectConfig, collect


def work(wid, title, abstract, authors, arxiv=None, refs=(), year=2024):
    inv = {}
    for i, w in enumerate(abstract.split()):
        inv.setdefault(w, []).append(i)
    return {"id": f"https://openalex.org/{wid}", "display_name": title, "publication_year": year,
            "abstract_inverted_index": inv,
            "authorships": [{"author": {"display_name": a}} for a in authors],
            "locations": [{"landing_page_url": f"https://arxiv.org/abs/{arxiv}"}] if arxiv else [],
            "ids": {}, "referenced_works": list(refs), "cited_by_count": 3}


AUTHOR_WORKS = [
    work("W1", "SMART relation-aware geometric representations for knowledge graphs",
         "knowledge graph embedding models geometric transformations link prediction", ["Sahar Vahdati"], "2507.13001",
         refs=["https://openalex.org/W9", "https://openalex.org/W8"]),
    work("W2", "Are researchers being replaced by artificial intelligence",
         "perspective essay on research practice", ["Angelo Salatino", "Sahar Vahdati"], "2605.16294"),
]
REFS = [work("W9", "RotatE knowledge graph embedding by relational rotation",
             "knowledge graph embedding rotation complex space link prediction", ["Zhiqing Sun"], "1902.10197"),
        work("W8", "Protein folding with deep networks", "protein structure prediction biology", ["Someone Else"])]
TOPIC = [work("W5", "Hallucination in large language models grounded with knowledge graphs",
              "large language model hallucination factual accuracy knowledge graph grounding", ["A Researcher"]),
         work("W6", "Cooking recipes dataset", "food recipes ingredients", ["B Cook"])]
ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2507.13001v1</id><published>2025-07-17T00:00:00Z</published>
<title>SMART: Relation-Aware Learning of Geometric Representations for Knowledge Graphs</title>
<summary>knowledge graph embedding link prediction</summary><author><name>Kossi Amouzouvi</name></author><author><name>Sahar Vahdati</name></author></entry>
<entry><id>http://arxiv.org/abs/2601.00001v1</id><published>2026-01-01T00:00:00Z</published>
<title>Unrelated galaxy survey</title><summary>astronomy galaxies redshift</summary><author><name>C Astro</name></author></entry>
</feed>"""


@pytest.fixture
def web(monkeypatch, corpus):
    pdf = (corpus / "pdfs" / "good.pdf").read_bytes()
    calls = []

    def fake_get(url, retries=3, timeout=30):
        calls.append(url)
        u = urllib.parse.unquote(url)
        if "api.openalex.org/authors" in u:
            return json.dumps({"results": [{"id": "https://openalex.org/A1"}]}).encode()
        if "api.openalex.org/works" in u and "author.id" in u:
            return json.dumps({"results": AUTHOR_WORKS}).encode()
        if "api.openalex.org/works" in u and "openalex_id" in u:
            return json.dumps({"results": REFS}).encode()
        if "api.openalex.org/works" in u and "search" in u:
            return json.dumps({"results": TOPIC}).encode()
        if "export.arxiv.org" in u:
            return ATOM
        if "1902.10197" in u:
            raise RuntimeError("HTTP 503")  # one PDF download fails -> abstract-only fallback
        if "/pdf/" in u:
            return pdf
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(C, "_get", fake_get)
    return calls


def cfg(**kw):
    base = dict(topics=["knowledge graph embedding link prediction", "large language model hallucination knowledge graph"],
                sleep=0, max_per_query=10)
    base.update(kw)
    return CollectConfig(**base)


def test_full_pipeline(web, tmp_path):
    out = tmp_path / "corpus"
    kept = collect(cfg(author="Sahar Vahdati"), out, log=lambda *a: None)
    ids = {r["id"] for r in kept}
    # author papers kept, the cited RotatE paper pulled in by reference expansion, the relevant topic paper kept
    assert {"arxiv:2507.13001", "arxiv:2605.16294", "arxiv:1902.10197", "openalex:W5"} <= ids
    # off-topic material from every source is filtered out
    assert not ids & {"openalex:W8", "openalex:W6", "arxiv:2601.00001"}
    # SMART came from OpenAlex and arXiv: merged into one record
    assert sum(1 for r in kept if "SMART" in r["title"] or "smart" in r["title"].lower()) == 1
    by = {r["id"]: r for r in kept}
    assert by["arxiv:2605.16294"]["kept_because"] == "author"   # off-vocabulary but authored by the seed author
    assert by["arxiv:1902.10197"]["relevance"]["topic_similarity"] > 0.1
    # PDFs downloaded; the failing one degrades to abstract-only instead of aborting
    assert by["arxiv:2507.13001"]["pdf"].endswith(".pdf") and (out / by["arxiv:2507.13001"]["pdf"]).exists()
    assert "pdf" not in by["arxiv:1902.10197"]
    rep = json.loads((out / "collect_report.json").read_text())
    assert rep["after_dedupe"] < rep["raw"] and rep["with_pdf"] >= 2 and rep["abstract_only"] >= 1
    assert any("1902.10197" in w for w in rep["warnings"])


def test_collected_papers_are_ingested(web, tmp_path):
    from sciagent.index import HybridIndex
    from sciagent.ingest import ingest_corpus
    out = tmp_path / "corpus"
    collect(cfg(author="Sahar Vahdati"), out, log=lambda *a: None)
    report = ingest_corpus(out, tmp_path / "index")
    idx = HybridIndex.build(tmp_path / "index")
    assert report["arxiv:2507.13001"].get("figures") == 1          # full PDF: figure extracted
    assert report["arxiv:1902.10197"].get("abstract_only")          # failed download: abstract still searchable
    assert idx.search("rotation complex space", k=1)[0].paper_id == "arxiv:1902.10197"


def test_topic_only_collection(web, tmp_path):
    kept = collect(cfg(author=None), tmp_path / "c", log=lambda *a: None)
    ids = {r["id"] for r in kept}
    assert "openalex:W5" in ids and not any("openalex.org/authors" in u or "au%3A" in u for u in web)
    assert all(r["kept_because"] == "topic" for r in kept)


def test_collection_is_incremental(web, tmp_path):
    out = tmp_path / "c"
    collect(cfg(author="Sahar Vahdati"), out, log=lambda *a: None)
    n1 = len(open(out / "collected.jsonl").readlines())
    collect(cfg(author=None, topics=["large language model hallucination knowledge graph"]), out, log=lambda *a: None)
    assert len(open(out / "collected.jsonl").readlines()) == n1  # nothing lost, nothing duplicated
    assert json.loads((out / "collect_report.json").read_text())["new_papers"] == 0


def test_all_sources_down_degrades_gracefully(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "_get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    kept = collect(cfg(author="Sahar Vahdati"), tmp_path / "c", log=lambda *a: None)
    rep = json.loads((tmp_path / "c" / "collect_report.json").read_text())
    assert kept == [] and len(rep["warnings"]) >= 3


def test_agent_can_fetch_a_missing_paper(web, built, tmp_path, corpus):
    import shutil
    from sciagent.tools import ToolContext, call_tool
    data = tmp_path / "data"
    shutil.copytree(corpus, data / "corpus")
    ctx = ToolContext(built.index, data, allow_network=True)
    out = json.loads(call_tool(ctx, "fetch_paper", {"source": "2507.13001"}))
    assert out["paper_id"] == "arxiv:2507.13001" and out["indexed_parts"]["figure"] == 1
    assert "arxiv:2507.13001" in ctx.index.papers and ctx.added_papers == ["arxiv:2507.13001"]
    hits = json.loads(call_tool(ctx, "search_papers", {"query": "ToyE", "paper_ids": ["arxiv:2507.13001"]}))
    assert hits["n"] >= 1
    offline = ToolContext(built.index, data, allow_network=False)
    assert "network disabled" in json.loads(call_tool(offline, "fetch_paper", {"source": "2507.13001"}))["error"]
