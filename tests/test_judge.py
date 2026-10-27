"""LLM judge against golden answers: parsing, prompt, rubric scores, calibration set, judge check, golden answers present."""
import json
import runpy
import sys
from pathlib import Path

import pytest

from sciagent.judge import VERDICTS, judge, parse_verdict, strip_citations, summarize

ROOT = Path(__file__).resolve().parent.parent


def test_parse_verdict_variants():
    assert parse_verdict('{"verdict": "correct", "missing_points": []}')["score"] == 1.0
    assert parse_verdict('Sure:\n```json\n{"verdict": "Partially Correct"}\n```')["verdict"] == "partially_correct"
    assert parse_verdict('{"verdict": "hallucinated"}')["score"] == 0.0
    assert parse_verdict("no json here")["verdict"] == "unparseable"
    assert parse_verdict('{"verdict": "great"}')["score"] is None


def test_judge_prompt_and_settings(fake):
    client = fake.Client([fake.response(fake.text('{"verdict": "correct", "matched_points": ["0.221 -> 0.605"]}'))])
    j = judge(client, "judge-m", "Q?", "From 0.221 to 0.605.", "It goes from 0.221 to 0.605 [arxiv:2507.13411#t3].",
              "fact", "supported")
    call = client.calls[0]
    assert j["verdict"] == "correct" and j["score"] == 1.0
    assert call["model"] == "judge-m" and call["temperature"] == 0.0
    user = call["messages"][0]["content"]
    assert "Golden answer:\nFrom 0.221 to 0.605." in user and "#t3" not in user   # citations stripped
    assert "answer_type: fact" in user and "supported" in user


def test_strip_citations_and_summary():
    assert strip_citations("A [p#t1], B [x:1#tab0_0].") == "A, B."
    s = summarize([{"verdict": "correct", "score": 1.0}, {"verdict": "partially_correct", "score": 0.5},
                   {"verdict": "unparseable", "score": None}])
    assert s == {"n": 3, "mean_score": 0.75, "verdicts": {"correct": 1, "partially_correct": 1, "unparseable": 1}}


def test_every_eval_question_has_a_golden_answer():
    qs = [json.loads(l) for l in open(ROOT / "evals" / "questions.jsonl")]
    assert all(q.get("reference_answer") and q.get("answer_type") in ("fact", "abstain", "open") for q in qs)
    assert {q["id"] for q in qs if q["answer_type"] == "abstain"} >= {"q15", "q16", "q17", "q26"}


def test_calibration_set_covers_every_verdict():
    cases = [json.loads(l) for l in open(ROOT / "evals" / "judge_calibration.jsonl")]
    assert len(cases) >= 12 and all(c["expected"] in VERDICTS for c in cases)
    assert {c["expected"] for c in cases} == set(VERDICTS)


def test_check_judge_reports_agreement(monkeypatch, tmp_path, fake):
    import anthropic
    cases = [json.loads(l) for l in open(ROOT / "evals" / "judge_calibration.jsonl")]
    replies = [c["expected"] for c in cases]
    replies[1] = "correct"          # one deliberate judge mistake: a wrong number graded as correct
    script = [fake.response(fake.text(json.dumps({"verdict": v}))) for v in replies]
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: fake.Client(script))
    mod = runpy.run_path(str(ROOT / "evals" / "check_judge.py"))
    summary = mod["main"](["--judge-model", "m"])
    n = len(cases)
    assert summary["exact_agreement"] == round((n - 1) / n, 3)
    assert summary["pass_fail_agreement"] == round((n - 1) / n, 3)
    assert [d["id"] for d in summary["disagreements"]] == [cases[1]["id"]]
