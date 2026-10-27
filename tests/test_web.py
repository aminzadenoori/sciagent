"""Web interface: endpoints and the live event stream, with a scripted fake Claude."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from sciagent.web import server as web


@pytest.fixture
def site(built, fake, monkeypatch, tmp_path):
    (tmp_path / "index").mkdir()
    app = web.App.__new__(web.App)
    app.data_dir, app.model, app.index, app.lock = built.dir, "m", built.index, threading.Lock()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", app
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.headers.get("Content-Type"), r.read()


def test_page_status_papers_search(site, monkeypatch):
    base, _ = site
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ct, body = get(base + "/")
    assert ct.startswith("text/html") and b"sciagent" in body
    st = json.loads(get(base + "/api/status")[1])
    assert st["has_key"] is False and st["papers"] >= 5
    hits = json.loads(get(base + "/api/search?q=CardNet")[1])
    assert hits[0]["paper_id"] == "toy:card" and "text" in hits[0]


def test_source_and_figure(site, built):
    base, _ = site
    fig = built.index.search("system", kinds=["figure"])[0].chunk_id
    src = json.loads(get(base + "/api/source?id=" + urllib.request.quote(fig))[1])
    assert src["kind"] == "figure" and src["has_image"]
    ct, png = get(base + "/api/figure?id=" + urllib.request.quote(fig))
    assert ct == "image/png" and png[:4] == b"\x89PNG"


def test_ask_without_key_explains(site, monkeypatch):
    base, _ = site
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ev = json.loads(get(base + "/api/ask?q=hello")[1].decode().split("data: ")[1])
    assert ev["type"] == "error" and "ANTHROPIC_API_KEY" in ev["message"]


def test_ask_streams_tool_events_and_final(site, fake, built, monkeypatch):
    base, _ = site
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cid = built.index.search("CardNet", k=1)[0].chunk_id
    script = [fake.response(fake.tool_use("search_papers", {"query": "CardNet"})),
              fake.response(fake.tool_use("submit_answer", {"answer": f"Yes [{cid}].", "citations": [cid],
                                                            "evidence_status": "supported", "confidence": 0.9}))]
    import sciagent.agent as A
    orig = A.ResearchAgent.__init__
    monkeypatch.setattr(A.ResearchAgent, "__init__",
                        lambda self, idx, d, client=None, **kw: orig(self, idx, d, client=fake.Client(script), **kw))
    body = get(base + "/api/ask?q=" + urllib.request.quote("Does CardNet help?"))[1].decode()
    events = [json.loads(x) for x in body.split("data: ")[1:]]
    assert [e["type"] for e in events] == ["start", "tool_call", "tool_result", "final"]
    final = events[-1]["result"]
    assert final["evidence_status"] == "supported" and final["sources"][cid]["paper_id"] == "toy:card"
