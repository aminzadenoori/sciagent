"""Adding a single paper (PDF / arXiv / URL) and asking about one paper only."""
import json
import shutil
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from conftest import ATOM_ONE
from sciagent.library import add_paper, arxiv_id, pdf_title
from sciagent.tools import ToolContext, call_tool


@pytest.mark.parametrize("src,expected", [
    ("2508.16983", "2508.16983"), ("2508.16983v2", "2508.16983"), ("https://arxiv.org/abs/2508.16983v1", "2508.16983"),
    ("https://arxiv.org/pdf/2508.16983.pdf", "2508.16983"), ("paper.pdf", None), ("https://example.org/x.pdf", None)])
def test_arxiv_id_parsing(src, expected):
    assert arxiv_id(src) == expected


def test_pdf_title_falls_back_to_largest_text(corpus):
    title, _, _ = pdf_title(corpus / "pdfs" / "good.pdf")
    assert title == "Toy embeddings for knowledge graphs"


@pytest.fixture
def fresh(tmp_path, corpus):
    c = tmp_path / "corpus"
    (c / "pdfs").mkdir(parents=True)
    shutil.copy(corpus / "meta.jsonl", c / "meta.jsonl")
    for f in (corpus / "pdfs").iterdir():
        shutil.copy(f, c / "pdfs" / f.name)
    return c, tmp_path / "index"


def test_add_local_pdf_is_indexed_with_tables_and_figures(fresh, corpus):
    c, idx = fresh
    out = add_paper(str(corpus / "pdfs" / "good.pdf"), c, idx)
    assert out["paper_id"].startswith("local:") and out["title"] == "Toy embeddings for knowledge graphs"
    assert out["chunks"]["text"] >= 3 and out["chunks"]["table"] == 1 and out["chunks"]["figure"] == 1
    assert out["paper_id"] in out["index"].papers
    hits = out["index"].search("ToyE rotations", paper_ids=[out["paper_id"]])
    assert hits and all(h.paper_id == out["paper_id"] for h in hits)
    again = add_paper(str(corpus / "pdfs" / "good.pdf"), c, idx, title="Renamed")  # same file -> replaced, not duplicated
    rows = [json.loads(l) for l in open(c / "added.jsonl")]
    assert len(rows) == 1 and rows[0]["title"] == "Renamed" and again["paper_id"] == out["paper_id"]


def test_add_arxiv_id_uses_api_metadata(fresh, corpus):
    c, idx = fresh
    pdf = (corpus / "pdfs" / "good.pdf").read_bytes()
    calls = []

    def fetch(url, **kw):
        calls.append(url)
        return ATOM_ONE if "export.arxiv.org" in url else pdf

    out = add_paper("https://arxiv.org/abs/2508.16983v2", c, idx, fetch=fetch)
    assert out["paper_id"] == "arxiv:2508.16983" and out["title"].startswith("ReFactX")
    assert any("id_list=2508.16983" in u for u in calls) and any(u.endswith("/pdf/2508.16983") for u in calls)


def test_add_rejects_non_pdf_and_missing_file(fresh):
    c, idx = fresh
    with pytest.raises(ValueError):
        add_paper("https://example.org/page.html", c, idx, fetch=lambda u, **k: b"<html>")
    with pytest.raises(FileNotFoundError):
        add_paper("no_such_file.pdf", c, idx)


def test_focus_restricts_search(built):
    ctx = ToolContext(built.index, built.dir, allow_network=False)
    ctx.focus_paper = "toy:card"
    out = json.loads(call_tool(ctx, "search_papers", {"query": "ToyE rotations MRR"}))
    assert out["n"] >= 1 and {r["paper_id"] for r in out["results"]} == {"toy:card"}


def test_agent_focus_prompt_and_unknown_paper(built, fake):
    from sciagent.agent import ResearchAgent
    cid = "toy:card#t1"
    client = fake.Client([fake.response(fake.tool_use("search_papers", {"query": "results"})),
                          fake.response(fake.tool_use("submit_answer", {"answer": f"0.61 [{cid}]", "citations": [cid],
                                                                        "evidence_status": "supported", "confidence": 0.9}))])
    r = ResearchAgent(built.index, built.dir, client=client, allow_network=False).run("Main result?", focus_paper="toy:card")
    assert r.citations == [cid]
    first = client.calls[0]["messages"][0]["content"]
    assert "Card paper" in first and "toy:card" in first and first.endswith("Question: Main result?")
    bad = ResearchAgent(built.index, built.dir, client=fake.Client([]), allow_network=False).run("x", focus_paper="nope")
    assert bad.error and "not in the corpus" in bad.answer


def test_web_upload_and_focused_search(fresh, corpus, monkeypatch):
    from sciagent.index import HybridIndex
    from sciagent.ingest import ingest_corpus
    from sciagent.web import server as web
    c, idx = fresh
    ingest_corpus(c, idx)
    app = web.App.__new__(web.App)
    app.data_dir, app.model, app.index, app.lock = c.parent, "m", HybridIndex.build(idx), threading.Lock()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/api/upload", data=(corpus / "pdfs" / "good.pdf").read_bytes(),
                                     headers={"X-Filename": "my paper.pdf", "Content-Type": "application/pdf"})
        out = json.loads(urllib.request.urlopen(req).read())
        assert out["title"] == "Toy embeddings for knowledge graphs" and out["chunks"]["figure"] == 1
        papers = json.loads(urllib.request.urlopen(base + "/api/papers").read())
        assert papers[0]["id"] == out["paper_id"] and papers[0]["added"] is True  # newest added first
        hits = json.loads(urllib.request.urlopen(base + "/api/search?q=ToyE&paper=" + out["paper_id"]).read())
        assert hits and {h["paper_id"] for h in hits} == {out["paper_id"]}
        bad = urllib.request.Request(base + "/api/upload", data=b"not a pdf", headers={"X-Filename": "x.pdf"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
    finally:
        srv.shutdown()
