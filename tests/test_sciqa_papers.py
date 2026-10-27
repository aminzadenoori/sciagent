"""SciQA answered from papers, compared with ORKG: selection, matching, scoring, runner, ORKG tool disabled."""
import json
import runpy
import sys
from pathlib import Path

import pytest

from sciagent.sciqa_papers import (aggregate, gold_answer, mentions, number_matches, review_sheet, score, select)

ROOT = Path(__file__).resolve().parent.parent


def binding(**kv):
    return {k: ({"type": "uri", "value": v} if v.startswith("http") else {"type": "literal", "value": v}) for k, v in kv.items()}


def item(id, t, q, **gold):
    return {"id": id, "template": t, "question": q, "sparql": "", "gold": gold}


ITEMS = [
    item("A1", "T05", "Which model has the highest Accuracy on the Story Cloze Test?", model_lbl=["GPT-3 175B (Few-Shot)"]),
    item("A2", "T04", "What is the top benchmark score and its metric on Birdsnap?", metric_lbl=["Accuracy"], score=["90.07%"]),
    item("A3", "T03", "Metrics used on BUCC Russian-to-English?", metric_lbl=["F1 score"]),
    item("A4", "T01", "Models evaluated on SciTLDR?", model_lbl=["CATTS-XSUM", "CATTS"]),
    item("A5", "T02", "Papers with a benchmark on HoC?", paper_lbl=["Some paper"]),           # ORKG-internal: excluded
    item("A6", "T05", "Best model on an empty dataset?"),                                       # empty gold: excluded
    item("A7", "T05", "Which model is best on CommonsenseQA?", model_lbl=["X"]),               # duplicated in train
]


def test_gold_answer_keeps_literals_only():
    ans = {"results": {"bindings": [binding(model="http://orkg.org/orkg/resource/R1", model_lbl="GPT-3 175B (Few-Shot)")]}}
    assert gold_answer(ans) == {"model_lbl": ["GPT-3 175B (Few-Shot)"]}
    assert gold_answer(None) == {} and gold_answer({"results": {"bindings": [{}]}}) == {}


TRAIN = {"which model is best on commonsenseqa?"}


def test_selection_is_stratified_and_skips_unanswerable():
    sel = select(ITEMS, TRAIN, limit=10)
    assert [i["id"] for i in sel] == ["A1", "A2", "A3", "A4"]
    assert {i["kind"] for i in sel} == {"best_model", "top_score", "metric_list", "model_list"}
    assert len(select(ITEMS, set(), limit=2)) == 2


@pytest.mark.parametrize("label,text,ok", [
    ("GPT-3 175B (Few-Shot)", "The best is GPT-3 175B in the few-shot setting (87.7%).", False),
    ("GPT-3 175B (Few-Shot)", "GPT-3 175B (few-shot) reaches 87.7% [p#t1].", True),
    ("GPT-3 175B (Few-Shot)", "GPT-3 175B Few Shot is best.", True),
    ("Massively Multilingual Sentence Embeddings", "LASER (massively multilingual sentence embedding) is best", True),
    ("CATTS", "The CATTS model and CATTS-XSUM were evaluated.", True),
    ("CATTS", "We found CATTLE only.", False),
    ("F1 score", "Models are evaluated with F1.", False),
    ("F1 score", "Evaluation uses the F1-score.", True),
    ("spl", "Success weighted by path length (SPL) is used.", True),
    ("spl", "splendid results", False),
])
def test_fuzzy_mentions(label, text, ok):
    assert mentions(label, text) is ok


@pytest.mark.parametrize("gold,text,ok", [
    ("90.07%", "Accuracy reaches 90.07 on Birdsnap.", True),
    ("90.07%", "Accuracy is 0.9007.", True),
    ("9646.0", "a score of 9,646 points", True),
    ("36.51", "ROUGE-L 36.5", True),     # rounded to one decimal in the paper: still a match
    ("36.51", "ROUGE-L 35.9", False),
    ("0.92", "F1 of 92%", True),
    ("0.92", "F1 of 0.86", False),
])
def test_number_matching(gold, text, ok):
    assert number_matches(gold, text) is ok


def test_scoring_by_kind():
    sel = {i["id"]: i for i in select(ITEMS, TRAIN, limit=10)}
    r = lambda text, status="supported": {"answer": text, "evidence_status": status, "citations": ["p#t1"], "steps": []}
    assert score(sel["A1"], r("GPT-3 175B (Few-Shot) is best [p#t1]."))["outcome"] == "agrees"
    assert score(sel["A1"], r("PaLM is best [p#t1]."))["outcome"] == "disagrees"
    assert score(sel["A1"], r("Not reported.", "insufficient"))["outcome"] == "abstained"
    s = score(sel["A2"], r("Best is 90.07 but measured as top-1 error."))
    assert s["score"] == 0.5 and s["score_found"] and not s["metric_found"]
    assert score(sel["A2"], r("Accuracy of 90.07%"))["score"] == 1.0
    s = score(sel["A4"], r("Models: CATTS."))
    assert s["score"] == 0.5 and s["found"] == ["CATTS"]


def test_aggregate_and_review_sheet():
    sel = select(ITEMS, TRAIN, limit=10)
    results = [{"answer": "GPT-3 175B (Few-Shot)", "evidence_status": "supported", "steps": [{"tool": "research_topic"}]},
               {"answer": "no idea", "evidence_status": "insufficient", "steps": []},
               {"answer": "F1 score", "evidence_status": "supported", "steps": []},
               {"answer": "BART", "evidence_status": "supported", "steps": []}]
    scores = [score(i, r) for i, r in zip(sel, results)]
    agg = aggregate(scores)
    assert agg["all"]["n"] == 4 and agg["all"]["agrees"] == 0.5 and agg["all"]["used_literature_agent"] == 0.25
    sheet = review_sheet(sel, results, scores)
    assert "A2" in sheet and "A4" in sheet and "| A1 |" not in sheet and "Accuracy = 90.07%" in sheet


def test_orkg_tool_is_disabled_in_agent(built, fake):
    from sciagent.agent import ResearchAgent
    client = fake.Client([fake.response(fake.tool_use("query_scholarly_kg", {"question": "best model?"}, id="q")),
                          fake.response(fake.tool_use("submit_answer", {"answer": "x", "citations": [],
                                        "evidence_status": "insufficient", "confidence": 0.1}, id="s"))])
    r = ResearchAgent(built.index, built.dir, client=client, allow_network=False,
                      disabled_tools={"query_scholarly_kg"}).run("q")
    assert "query_scholarly_kg" not in [t["name"] for t in client.calls[0]["tools"]]
    assert "disabled" in r.steps[0].output_preview


def test_runner_end_to_end_with_stub_agent(tmp_path, monkeypatch, built):
    import sciagent.agent as A
    import sciagent.sciqa as SQ
    root = tmp_path / "data" / "sciqa"
    for split, qs in (("train", [("T1", "T05", "Which model is best on HoC?", "GPT-2")]),
                      ("test", [("Q1", "T05", "Which model has the highest Accuracy on the Story Cloze Test?", "GPT-3 175B (Few-Shot)"),
                                ("Q2", "T03", "Metrics used on BUCC?", "F1 score")])):
        (root / split).mkdir(parents=True)
        var = "model_lbl" if qs[0][1] == "T05" else "metric_lbl"
        (root / split / "questions.json").write_text(json.dumps({"questions": [
            {"id": i, "question": {"string": q}, "query": {"sparql": "SELECT"}, "template_id": t, "auto_generated": True,
             "query_type": "Factoid"} for i, t, q, _ in qs]}))
        (root / split / "answers.json").write_text(json.dumps({"answers": [
            {"id": i, "answer": {"results": {"bindings": [binding(**{("model_lbl" if t == "T05" else "metric_lbl"): g})]}}}
            for i, t, q, g in qs]}))
    import shutil
    shutil.copytree(built.dir, tmp_path / "data" / "index")
    monkeypatch.setenv("SCIAGENT_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    seen = {}

    def fake_run(self, question, **kw):
        seen.setdefault("disabled", self.disabled_tools)
        res = A.AgentResult(question=question)
        res.answer = "GPT-3 175B (Few-Shot) reaches the best accuracy [p#t1]." if "Story Cloze" in question else "BLEU."
        res.evidence_status, res.citations = "supported", ["p#t1"]
        return res

    monkeypatch.setattr(A.ResearchAgent, "__init__", lambda self, *a, **k: setattr(self, "disabled_tools", k.get("disabled_tools")))
    monkeypatch.setattr(A.ResearchAgent, "run", fake_run)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_sciqa_papers.py", "--limit", "5", "--out", str(out)])
    runpy.run_path(str(ROOT / "evals" / "run_sciqa_papers.py"), run_name="__main__")
    agg = json.loads((out / "scores.json").read_text())["aggregate"]
    assert agg["all"]["n"] == 2 and agg["all"]["agrees"] == 0.5
    assert seen["disabled"] == {"query_scholarly_kg"}
    assert "Q2" in (out / "review.md").read_text() and (out / "report.md").exists()
    assert (out / "workspace" / "index").exists()   # ran on a copy of the corpus
