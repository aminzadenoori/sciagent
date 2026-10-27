"""End-to-end checks against the real Claude API. Skipped unless ANTHROPIC_API_KEY is set.

    ANTHROPIC_API_KEY=... pytest -m live
"""
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")]
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def agent():
    from sciagent.agent import ResearchAgent
    from sciagent.index import HybridIndex
    idx = HybridIndex.load(ROOT / "data" / "index")
    return ResearchAgent(idx, ROOT / "data", max_steps=10)


def test_live_supported_answer_is_cited(agent):
    r = agent.run("Which elementary geometric transformations does SMART combine?")
    assert r.error is None and r.evidence_status == "supported" and r.citations and not r.invalid_citations
    assert "reflection" in r.answer.lower()


def test_live_conflict_is_surfaced(agent):
    r = agent.run("Can TransE model symmetric relations?")
    assert r.evidence_status in ("conflicting", "partially_supported") and r.conflicts


def test_live_abstains_when_corpus_lacks_answer(agent):
    r = agent.run("What MRR does SMART achieve on YAGO3-10?")
    assert r.evidence_status == "insufficient"


def test_live_sciqa_three_shot_generation():
    from sciagent.sciqa import ClaudeSparql, ExampleRetriever, load_split, normalise_query
    root = ROOT / "data" / "sciqa"
    gen = ClaudeSparql(ExampleRetriever(load_split(root, "train")), k=3)
    test = [i for i in load_split(root, "test") if not i.handcrafted][:3]
    hits = sum(normalise_query(gen(i.question)) == normalise_query(i.sparql) for i in test)
    assert hits >= 2
