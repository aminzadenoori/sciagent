"""The agent evaluation harness (evals/run_eval.py) runs end to end and does not reward a naive answerer.

A stand-in for Claude searches once, cites the first hit and always says "supported". A sound harness
must fail it on conflict and unanswerable questions and still produce the report files.
"""
import json
import runpy
import sys
from pathlib import Path

from conftest import response, text, tool_use

ROOT = Path(__file__).resolve().parent.parent


class NaiveModel:
    def __init__(self, *a, **k):
        self.messages = self

    def create(self, **kw):
        msgs = kw["messages"]
        if "tools" not in kw:  # LLM-as-judge call
            return response(text('{"claims": 2, "supported": 1, "unsupported_examples": []}'))
        if len(msgs) == 1:
            return response(tool_use("search_papers", {"query": msgs[0]["content"][:200]}, id="s"))
        res = json.loads(msgs[-1]["content"][0]["content"])
        cid = res["results"][0]["chunk_id"]
        return response(tool_use("submit_answer", {"answer": f"Yes [{cid}].", "citations": [cid],
                                                   "evidence_status": "supported", "confidence": 0.9}, id="a"))


def test_run_eval_end_to_end_with_naive_model(built, tmp_path, monkeypatch):
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", NaiveModel)
    monkeypatch.setenv("SCIAGENT_DATA", str(built.dir.parent))
    if not (built.dir.parent / "index").exists():  # run_eval.py reads $SCIAGENT_DATA/index
        (built.dir.parent / "index").symlink_to(built.dir)
    qs = tmp_path / "q.jsonl"
    rows = [
        {"id": "c1", "category": "conflict", "question": "Does ToyE beat the baseline?",
         "gold": [{"paper_id": "toy:good", "contains": "0.427"}], "expected_status": ["conflicting"], "must_include": [["0.427"]]},
        {"id": "u1", "category": "unanswerable", "question": "What MRR does CardNet get on YAGO3-10?",
         "gold": [], "expected_status": ["insufficient"], "must_include": []},
    ]
    qs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--questions", str(qs), "--judge", "--out", str(out)])
    runpy.run_path(str(ROOT / "evals" / "run_eval.py"), run_name="__main__")
    for f in ("retrieval.json", "agent_runs.jsonl", "scores.json", "report.md"):
        assert (out / f).exists()
    agg = json.loads((out / "scores.json").read_text())["aggregate"]
    assert agg["n"] == 2 and agg["errors"] == 0
    assert agg["pass_rate"] == 0.0                      # naive "always supported" earns nothing here
    assert agg["abstention_accuracy"] == 0.0 and agg["conflict_detection"] == 0.0
    assert agg["judge_groundedness"] == 0.5
