"""Literature-review sub-agent started by the main agent through research_topic.

Simulated arXiv with three FICTIONAL papers on requirements classification whose PDFs
contain F1 result tables. Checks: the main agent can start the sub-agent; the sub-agent
searches, downloads (within its cap), reads tables and returns findings; findings that
cite unread chunks are dropped; the main agent can cite the sub-agent's chunks.
"""
import json
import shutil
import urllib.parse

import pytest

import sciagent.collect as C
from conftest import _story_pdf
from sciagent.agent import ResearchAgent
from sciagent.index import HybridIndex
from sciagent.ingest import ingest_corpus
from sciagent.subagents import LiteratureAgent
from sciagent.tools import ToolContext, call_tool, tool_specs

PAPERS = {
    "2501.00001": ("Zero-shot requirements classification with large language models",
                   [["GPT-4 zero-shot", "PROMISE, binary FR/NFR", "0.81"], ["GPT-4 few-shot", "PROMISE, binary FR/NFR", "0.86"]]),
    "2501.00002": ("Fine-tuned transformers for non-functional requirements classification",
                   [["BERT fine-tuned", "PROMISE_exp, 11-class NFR", "0.92"], ["RoBERTa fine-tuned", "PROMISE_exp, 11-class NFR", "0.90"]]),
    "2501.00003": ("Prompting strategies for software requirements classification",
                   [["Llama few-shot", "PROMISE, binary FR/NFR", "0.79"]]),
}


def entry(aid, title):
    return (f"<entry><id>http://arxiv.org/abs/{aid}v1</id><published>2025-01-01T00:00:00Z</published><title>{title}</title>"
            f"<summary>Requirements classification on the PROMISE dataset, reported with F1.</summary>"
            f"<author><name>Fictional Author</name></author></entry>")


def feed(entries):
    return ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">' + "".join(entries) + "</feed>").encode()


@pytest.fixture
def lit(monkeypatch, tmp_path, corpus):
    pdfs = {}
    for aid, (title, rows) in PAPERS.items():
        body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
        html = (f"<h1>{title}</h1><h2>Abstract</h2><p>We study requirements classification on PROMISE.</p>"
                f"<h2>1. Results</h2><p>Table 1 reports F1 scores.</p><p class='caption'>Table 1. F1 on PROMISE.</p>"
                f"<table><tr><th>Model</th><th>Setting</th><th>F1</th></tr>{body}</table>")
        _story_pdf(tmp_path / f"{aid}.pdf", html)
        pdfs[aid] = (tmp_path / f"{aid}.pdf").read_bytes()

    def fake_get(url, retries=3, timeout=30):
        u = urllib.parse.unquote(url)
        if "id_list=" in u:
            aid = u.split("id_list=")[1][:10]
            return feed([entry(aid, PAPERS[aid][0])])
        if "export.arxiv.org" in u:
            return feed([entry(a, t) for a, (t, _) in PAPERS.items()])
        if "api.openalex.org" in u:
            return json.dumps({"results": []}).encode()
        for aid in PAPERS:
            if f"/pdf/{aid}" in u:
                return pdfs[aid]
        raise AssertionError(url)

    monkeypatch.setattr(C, "_get", fake_get)
    d = tmp_path / "data"
    shutil.copytree(corpus, d / "corpus")
    ingest_corpus(d / "corpus", d / "index")
    return d, HybridIndex.build(d / "index")


def last_results(kw):
    for m in reversed(kw["messages"]):
        if m["role"] == "user" and isinstance(m["content"], list):
            for c in m["content"]:
                body = json.loads(c["content"]) if c.get("content", "").startswith("{") else {}
                if body.get("results") or body.get("findings"):
                    return body
    raise AssertionError("no results in messages")


def test_research_topic_is_offered_to_the_main_agent():
    spec = {t["name"]: t for t in tool_specs()}["research_topic"]
    assert "literature-review agent" in spec["description"] and spec["input_schema"]["required"] == ["question"]


def test_main_agent_starts_subagent_and_cites_its_findings(lit, fake):
    d, idx = lit
    q = "What is the highest F1 score for requirements classification with LLMs on the PROMISE dataset?"
    tu = fake.tool_use

    def findings(kw):
        res = last_results(kw)["results"]
        tables = [r["chunk_id"] for r in res if r["kind"] == "table"]
        return fake.response(tu("submit_findings", {
            "summary": "Searched arXiv; downloaded 2 papers; both report F1 on PROMISE variants.",
            "findings": [
                {"paper_id": "arxiv:2501.00001", "finding": "GPT-4 few-shot reaches F1 0.86", "metric": "F1",
                 "value": "0.86", "dataset": "PROMISE", "setting": "binary FR/NFR, few-shot",
                 "chunk_ids": [c for c in tables if c.startswith("arxiv:2501.00001")]},
                {"paper_id": "arxiv:2501.00002", "finding": "BERT fine-tuned reaches F1 0.92", "metric": "F1",
                 "value": "0.92", "dataset": "PROMISE_exp", "setting": "11-class NFR, fine-tuned",
                 "chunk_ids": [c for c in tables if c.startswith("arxiv:2501.00002")] + ["arxiv:2501.00002#t99"]},
                {"paper_id": "arxiv:2501.00003", "finding": "not downloaded", "chunk_ids": ["arxiv:2501.00003#t0"]}],
            "papers_rejected": [{"title": PAPERS["2501.00003"][0], "reason": "download budget used"}],
            "gaps": "Settings differ across papers."}, id="sf"))

    def answer(kw):
        f = last_results(kw)["findings"]
        a, b = f[0]["chunk_ids"][0], f[1]["chunk_ids"][0]
        return fake.response(tu("submit_answer", {
            "answer": f"Among the papers found, the highest F1 is 0.92 (fine-tuned BERT, PROMISE_exp 11-class) [{b}]; "
                      f"for LLM prompting the best is 0.86 (GPT-4 few-shot, binary PROMISE) [{a}]. These settings "
                      f"are not directly comparable.",
            "citations": [a, b], "evidence_status": "partially_supported", "confidence": 0.6,
            "conflicts": [f"0.92 is fine-tuned on PROMISE_exp [{b}] vs 0.86 few-shot LLM on PROMISE [{a}]"]}, id="sa"))

    script = [
        fake.response(tu("search_papers", {"query": "requirements classification PROMISE F1"}, id="m1")),
        fake.response(tu("research_topic", {"question": q, "max_papers": 2}, id="m2")),
        fake.response(tu("search_external_literature", {"query": "requirements classification PROMISE LLM"}, id="s1")),
        fake.response(tu("fetch_paper", {"source": "2501.00001"}, id="s2"), tu("fetch_paper", {"source": "2501.00002"}, id="s3")),
        fake.response(tu("fetch_paper", {"source": "2501.00003"}, id="s4")),       # over the cap of 2
        fake.response(tu("search_papers", {"query": "F1 PROMISE", "kinds": ["table"], "k": 6,
                                           "paper_ids": ["arxiv:2501.00001", "arxiv:2501.00002"]}, id="s5")),
        findings,
        answer,
    ]
    events = []
    agent = ResearchAgent(idx, d, client=fake.Client(script), allow_network=True)
    r = agent.run(q, on_event=events.append)

    assert [s.tool for s in r.steps] == ["search_papers", "research_topic"]
    sub = [e for e in events if e.get("depth") == 1 and e["type"] == "tool_call"]
    assert [e["tool"] for e in sub] == ["search_external_literature", "fetch_paper", "fetch_paper", "search_papers"]
    assert r.added_papers == ["arxiv:2501.00001", "arxiv:2501.00002"]           # cap respected, third not fetched
    assert "arxiv:2501.00003" not in agent.index.papers
    assert r.evidence_status == "partially_supported" and len(r.citations) == 2 and not r.invalid_citations
    assert all(c.startswith(("arxiv:2501.00001#tab", "arxiv:2501.00002#tab")) for c in r.citations)
    assert not any("external check" in g for g in r.guardrail_events)


def test_subagent_drops_unread_citations_and_reports_budget(lit, fake):
    d, idx = lit
    ctx = ToolContext(idx, d, client=None, model="m", allow_network=True)
    tu = fake.tool_use
    script = [
        fake.response(tu("search_external_literature", {"query": "requirements classification"}, id="a")),
        fake.response(tu("fetch_paper", {"source": "2501.00001"}, id="b")),
        fake.response(tu("fetch_paper", {"source": "2501.00002"}, id="c")),
        lambda kw: fake.response(tu("submit_findings", {"summary": "s", "findings": [
            {"paper_id": "arxiv:2501.00001", "finding": "made up", "chunk_ids": ["arxiv:2501.00001#t0"]}]}, id="d")),
    ]
    ctx.client = fake.Client(script)
    out = LiteratureAgent(ctx, max_papers=1).run("q")
    assert out["papers_fetched"] == ["arxiv:2501.00001"]
    assert out["findings"] == [] and out["dropped_citations"] == ["arxiv:2501.00001#t0"]   # never read
    budget = json.loads(ctx.client.calls[3]["messages"][-1]["content"][0]["content"])
    assert "download budget" in budget["error"]


def test_subagent_must_search_outside_first(lit, fake):
    d, idx = lit
    tu = fake.tool_use
    ctx = ToolContext(idx, d, model="m", allow_network=True)
    ctx.client = fake.Client([
        fake.response(tu("submit_findings", {"summary": "nothing", "findings": []}, id="a")),
        fake.response(tu("search_external_literature", {"query": "requirements"}, id="b")),
        fake.response(tu("submit_findings", {"summary": "searched", "findings": []}, id="c"))])
    out = LiteratureAgent(ctx).run("q")
    assert out["summary"] == "searched" and "rejected: no outside search" in out["events"]


def test_research_topic_offline_or_without_llm(built):
    assert "network disabled" in json.loads(call_tool(ToolContext(built.index, built.dir, client=object(), allow_network=False),
                                                      "research_topic", {"question": "q"}))["error"]
    assert "unavailable" in json.loads(call_tool(ToolContext(built.index, built.dir), "research_topic", {"question": "q"}))["error"]
