"""Questions about a paper that is NOT in the corpus: the agent must search outside, fetch it, and answer from it."""
import json
import shutil
import urllib.parse

import pytest

import sciagent.collect as C
from sciagent.agent import ResearchAgent
from sciagent.index import HybridIndex
from sciagent.ingest import ingest_corpus
from sciagent.tools import ToolContext, call_tool

TARGET = "SMART: Relation-Aware Learning of Geometric Representations for Knowledge Graphs"
ATOM = f"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2507.13001v1</id><published>2025-07-17T00:00:00Z</published>
<title>{TARGET}</title><summary>geometric transformations for knowledge graph embeddings</summary>
<author><name>Kossi Amouzouvi</name></author><author><name>Sahar Vahdati</name></author></entry>
<entry><id>http://arxiv.org/abs/2401.00002v1</id><published>2024-01-01T00:00:00Z</published>
<title>Smart contracts on blockchains</title><summary>blockchain</summary><author><name>X Y</name></author></entry>
</feed>""".encode()
OPENALEX = {"results": [{"id": "https://openalex.org/W7", "display_name": "A survey of geometric KG embeddings",
                         "publication_year": 2023, "abstract_inverted_index": {"survey": [0]}, "authorships": [],
                         "locations": [], "ids": {}, "referenced_works": []}]}


@pytest.fixture
def net(monkeypatch, corpus):
    pdf = (corpus / "pdfs" / "good.pdf").read_bytes()

    def fake_get(url, retries=3, timeout=30):
        u = urllib.parse.unquote(url)
        if "export.arxiv.org" in u:
            return ATOM
        if "api.openalex.org" in u:
            return json.dumps(OPENALEX).encode()
        if "arxiv.org/pdf/2507.13001" in u:
            return pdf
        raise AssertionError(url)

    monkeypatch.setattr(C, "_get", fake_get)


@pytest.fixture
def data(tmp_path, corpus):
    d = tmp_path / "data"
    shutil.copytree(corpus, d / "corpus")
    ingest_corpus(d / "corpus", d / "index")
    return d, HybridIndex.build(d / "index")


def test_external_search_ranks_exact_title_first(net, data):
    d, idx = data
    out = json.loads(call_tool(ToolContext(idx, d), "search_external_literature", {"query": "SMART relation-aware geometric representations knowledge graphs"}))
    top = out["results"][0]
    assert top["title"] == TARGET and top["fetchable"] == "2507.13001" and top["already_in_corpus"] is False
    assert {r["id"] for r in out["results"]} >= {"arxiv:2507.13001", "openalex:W7"}  # both sources merged


def _first_chunk_from_last_result(kw):
    for m in reversed(kw["messages"]):
        if m["role"] == "user" and isinstance(m["content"], list):
            res = json.loads(m["content"][0]["content"])
            if res.get("results"):
                return res["results"][0]["chunk_id"]
    raise AssertionError("no search results seen")


def test_agent_searches_outside_fetches_and_answers(net, data, fake):
    d, idx = data
    sub = lambda status, cites, text: fake.tool_use("submit_answer", {"answer": text, "citations": cites,
                                                                     "evidence_status": status, "confidence": 0.8})
    script = [
        fake.response(fake.tool_use("search_papers", {"query": "SMART geometric transformations"}, id="t1")),
        fake.response(sub("insufficient", [], "SMART is not in the corpus.")),   # gives up too early -> rejected
        fake.response(fake.tool_use("search_external_literature", {"query": "SMART relation-aware geometric representations"}, id="t2")),
        fake.response(fake.tool_use("fetch_paper", {"source": "2507.13001"}, id="t3")),
        fake.response(fake.tool_use("search_papers", {"query": "ToyE rotations", "paper_ids": ["arxiv:2507.13001"]}, id="t4")),
        lambda kw: fake.response(sub("supported", [c := _first_chunk_from_last_result(kw)],
                                     f"Added from outside the corpus. It uses rotations [{c}].")),
    ]
    agent = ResearchAgent(idx, d, client=fake.Client(script), allow_network=True)
    r = agent.run("What transformations does the SMART paper use?")
    assert any("external check failed" in g for g in r.guardrail_events)
    assert [s.tool for s in r.steps] == ["search_papers", "search_external_literature", "fetch_paper", "search_papers"]
    assert r.evidence_status == "supported" and r.citations[0].startswith("arxiv:2507.13001#")
    assert r.added_papers == ["arxiv:2507.13001"] and "arxiv:2507.13001" in agent.index.papers
    assert not r.invalid_citations


def test_abstaining_after_searching_outside_is_accepted(net, data, fake):
    d, idx = data
    script = [fake.response(fake.tool_use("search_papers", {"query": "quantum"}, id="a")),
              fake.response(fake.tool_use("search_external_literature", {"query": "quantum gravity loops"}, id="b")),
              fake.response(fake.tool_use("submit_answer", {"answer": "No matching paper found; closest: ...", "citations": [],
                                                            "evidence_status": "insufficient", "confidence": 0.6}))]
    r = ResearchAgent(idx, d, client=fake.Client(script), allow_network=True).run("What does the loop quantum gravity paper by X claim?")
    assert r.evidence_status == "insufficient" and not any("external" in g for g in r.guardrail_events)


def test_no_external_requirement_when_offline_or_focused(data, fake):
    d, idx = data
    abstain = lambda: [fake.response(fake.tool_use("search_papers", {"query": "x"}, id="a")),
                       fake.response(fake.tool_use("submit_answer", {"answer": "Not in this paper.", "citations": [],
                                                                     "evidence_status": "insufficient", "confidence": 0.5}))]
    r1 = ResearchAgent(idx, d, client=fake.Client(abstain()), allow_network=False).run("q")
    r2 = ResearchAgent(idx, d, client=fake.Client(abstain()), allow_network=True).run("q", focus_paper="toy:card")
    assert r1.evidence_status == r2.evidence_status == "insufficient"
    assert not r1.guardrail_events and not r2.guardrail_events
