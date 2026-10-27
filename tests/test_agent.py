"""Agent loop with a scripted fake Claude: tool routing, guardrails, failure handling.

These test the *harness* deterministically; the model's own judgement is
tested by the eval harness (evals/run_eval.py) against the real API.
"""
import json

import pytest

from sciagent.agent import ResearchAgent


def make_agent(built, client, **kw):
    return ResearchAgent(built.index, built.dir, client=client, allow_network=False, **kw)


def first_hit_id(built, q):
    return built.index.search(q, k=1)[0].chunk_id


def submit(fake, answer, citations, status="supported", conf=0.8, conflicts=None):
    return fake.tool_use("submit_answer", {"answer": answer, "citations": citations, "evidence_status": status,
                                           "confidence": conf, "conflicts": conflicts or []})


def test_happy_path_search_then_cited_answer(built, fake):
    cid = first_hit_id(built, "CardNet exact match")
    client = fake.Client([
        fake.response(fake.text("Let me search."), fake.tool_use("search_papers", {"query": "CardNet exact match"})),
        fake.response(submit(fake, f"CardNet improves EM from 0.20 to 0.61 [{cid}].", [cid])),
    ])
    r = make_agent(built, client).run("How much does CardNet improve exact match?")
    assert r.evidence_status == "supported" and r.citations == [cid] and not r.invalid_citations
    assert [s.tool for s in r.steps] == ["search_papers"]
    assert r.usage["input_tokens"] == 200
    # the tool result was fed back to the model with the matching tool_use_id
    last_user = client.calls[1]["messages"][-1]
    assert last_user["role"] == "user" and last_user["content"][0]["type"] == "tool_result"
    # prompt caching is requested on system + tools
    assert client.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_model_chooses_tools_not_fixed_pipeline(built, fake):
    """Different questions -> different tool sequences, driven by the model."""
    fig = built.index.search("score of system", kinds=["figure"])[0].chunk_id
    client = fake.Client([
        fake.response(fake.tool_use("list_papers", {"author": "Ada"})),
        fake.response(fake.tool_use("search_papers", {"query": "system score", "kinds": ["figure"]})),
        fake.response(fake.tool_use("inspect_figure", {"chunk_id": fig, "question": "values?"})),
        fake.response(fake.text("Bar A reads about 90.")),  # consumed by the vision call inside the tool
        fake.response(submit(fake, f"A is 90% [{fig}].", [fig])),
    ])
    r = make_agent(built, client).run("q")
    assert [s.tool for s in r.steps] == ["list_papers", "search_papers", "inspect_figure"]
    assert "Bar A reads about 90" in r.steps[-1].output_preview
    assert r.evidence_status == "supported"


def test_fabricated_citation_rejected_then_fixed(built, fake):
    cid = first_hit_id(built, "CardNet")
    client = fake.Client([
        fake.response(fake.tool_use("search_papers", {"query": "CardNet"})),
        fake.response(submit(fake, "Claim [toy:card#t99].", ["toy:card#t99"])),  # never returned
        fake.response(submit(fake, f"Claim [{cid}].", [cid])),
    ])
    r = make_agent(built, client).run("q")
    assert r.citations == [cid] and not r.invalid_citations
    assert any("citation check failed" in g for g in r.guardrail_events)
    rejected = client.calls[2]["messages"][-1]["content"][-1]
    assert rejected["is_error"] is True and "toy:card#t99" in rejected["content"]


def test_persistent_fabricated_citation_is_stripped_and_downgraded(built, fake):
    client = fake.Client([
        fake.response(fake.tool_use("search_papers", {"query": "CardNet"})),
        fake.response(submit(fake, "Claim [fake:paper#t1].", ["fake:paper#t1"], conf=0.95)),
        fake.response(submit(fake, "Claim [fake:paper#t1].", ["fake:paper#t1"], conf=0.95)),
    ])
    r = make_agent(built, client).run("q")
    assert r.invalid_citations == ["fake:paper#t1"] and "fake:paper#t1" not in r.answer
    assert "[citation removed]" in r.answer and r.confidence <= 0.3
    assert "invalid_citations_stripped" in r.guardrail_events


def test_citation_seen_only_in_other_session_is_invalid(built, fake):
    """seen-chunk bookkeeping is per run: a chunk retrieved in an earlier question can't be cited now."""
    cid = first_hit_id(built, "CardNet")
    a = make_agent(built, fake.Client([
        fake.response(fake.tool_use("search_papers", {"query": "CardNet"})),
        fake.response(submit(fake, f"x [{cid}]", [cid]))]))
    assert a.run("q1").citations == [cid]
    b = make_agent(built, fake.Client([
        fake.response(fake.tool_use("list_papers", {})),
        fake.response(submit(fake, f"x [{cid}]", [cid])),
        fake.response(submit(fake, f"x [{cid}]", [cid]))]))
    assert b.run("q2").invalid_citations == [cid]


def test_abstaining_without_looking_is_rejected(built, fake):
    client = fake.Client([
        fake.response(submit(fake, "Not in corpus.", [], status="insufficient")),
        fake.response(fake.tool_use("search_papers", {"query": "YAGO3-10 SMART MRR"})),
        fake.response(submit(fake, "The corpus does not report this.", [], status="insufficient", conf=0.9)),
    ])
    r = make_agent(built, client).run("What MRR does SMART get on YAGO3-10?")
    assert r.evidence_status == "insufficient"
    assert any("abstain check failed" in g for g in r.guardrail_events)
    assert [s.tool for s in r.steps] == ["search_papers"]


def test_supported_answer_without_citations_rejected(built, fake):
    cid = first_hit_id(built, "CardNet")
    client = fake.Client([
        fake.response(fake.tool_use("search_papers", {"query": "CardNet"})),
        fake.response(submit(fake, "Yes it improves.", [])),
        fake.response(submit(fake, f"Yes [{cid}].", [cid])),
    ])
    r = make_agent(built, client).run("q")
    assert r.citations == [cid]


def test_conflicting_status_and_inline_citations_in_conflicts(built, fake):
    a, b = first_hit_id(built, "CardNet"), first_hit_id(built, "ToyE rotations")
    client = fake.Client([
        fake.response(fake.tool_use("search_papers", {"query": "CardNet"}),
                      fake.tool_use("search_papers", {"query": "ToyE rotations"}, id="tu2")),
        fake.response(submit(fake, "Papers disagree.", [], status="conflicting",
                             conflicts=[f"Paper A says X [{a}] vs paper B says Y [{b}]"])),
    ])
    r = make_agent(built, client).run("q")
    assert r.evidence_status == "conflicting" and set(r.citations) == {a, b}
    assert len(r.steps) == 2  # two parallel tool calls in one turn


def test_step_budget_forces_final_answer(built, fake):
    seen = {}

    def final(kw):
        seen["tool_choice"] = kw.get("tool_choice")
        return fake.response(submit(fake, "Partial answer.", [], status="insufficient", conf=0.2))

    loop = [fake.response(fake.tool_use("search_papers", {"query": f"q{i}"}, id=f"t{i}")) for i in range(3)]
    r = make_agent(built, fake.Client(loop + [final]), max_steps=3).run("q")
    assert seen["tool_choice"] == {"type": "tool", "name": "submit_answer"}
    assert "step_budget_exhausted" in r.guardrail_events and r.answer == "Partial answer."


def test_plain_text_answer_is_nudged_to_structured_output(built, fake):
    cid = first_hit_id(built, "CardNet")
    client = fake.Client([
        fake.response(fake.tool_use("search_papers", {"query": "CardNet"})),
        fake.response(fake.text("CardNet is great.")),
        fake.response(submit(fake, f"CardNet [{cid}]", [cid])),
    ])
    r = make_agent(built, client).run("q")
    assert "plain_text_answer" in r.guardrail_events and r.citations == [cid]


def test_tool_errors_are_returned_to_model_not_raised(built, fake):
    client = fake.Client([
        fake.response(fake.tool_use("kg_leakage_analysis", {"dataset": "YAGO3-10"}),
                      fake.tool_use("get_paper_outline", {"paper_id": "nope"}, id="t2"),
                      fake.tool_use("search_papers", {"wrong_arg": 1}, id="t3"),
                      fake.tool_use("no_such_tool", {}, id="t4")),
        fake.response(submit(fake, "YAGO3-10 is not supported by the analysis tool.", [], status="insufficient")),
    ])
    r = make_agent(built, client).run("Run leakage analysis on YAGO3-10")
    outs = [json.loads(s.output_preview) if s.output_preview.endswith("}") else s.output_preview for s in r.steps]
    assert all("error" in str(o) for o in outs)
    assert r.error is None and r.evidence_status == "insufficient"


def test_api_failure_is_captured(built, fake):
    r = make_agent(built, fake.Client([RuntimeError("401 invalid x-api-key")])).run("q")
    assert r.error and "401" in r.error and r.answer


def test_rate_limit_is_retried(built, fake, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    cid = first_hit_id(built, "CardNet")
    client = fake.Client([RuntimeError("rate_limit_error"), fake.response(fake.tool_use("search_papers", {"query": "CardNet"})),
                          fake.response(submit(fake, f"x [{cid}]", [cid]))])
    assert make_agent(built, client).run("q").citations == [cid]
