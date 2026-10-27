"""Collection parsing (no network) and the evaluation harness's own correctness."""
import json
from pathlib import Path

from sciagent.collect import _abstract_from_inverted, dedupe, openalex_work_to_record, parse_arxiv_atom, relevance
from sciagent.evaluation import aggregate, gold_coverage, load_questions, retrieval_metrics, score_answer

ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <entry>
  <id>http://arxiv.org/abs/2508.16983v2</id>
  <published>2025-08-23T10:00:00Z</published>
  <title>ReFactX: Scalable Reasoning with
   Reliable Facts via Constrained Generation</title>
  <summary>  Knowledge gaps and hallucinations ...  </summary>
  <author><name>Riccardo Pozzi</name></author>
  <author><name>Sahar Vahdati</name></author>
  <link title="pdf" href="http://arxiv.org/pdf/2508.16983v2" rel="related" type="application/pdf"/>
 </entry>
</feed>"""


def test_parse_arxiv_atom():
    (r,) = parse_arxiv_atom(ATOM)
    assert r["id"] == "arxiv:2508.16983" and r["year"] == 2025
    assert r["title"] == "ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation"
    assert r["authors"][-1] == "Sahar Vahdati" and r["pdf_url"].endswith("2508.16983v2")


def test_openalex_record_and_abstract_reconstruction():
    w = {"id": "https://openalex.org/W1", "display_name": "SMART", "publication_year": 2025,
         "abstract_inverted_index": {"geometric": [1], "Relation-aware": [0], "learning": [2]},
         "authorships": [{"author": {"display_name": "Sahar Vahdati"}}],
         "locations": [{"landing_page_url": "https://arxiv.org/abs/2507.13001"}], "ids": {"doi": None},
         "referenced_works": ["https://openalex.org/W2"]}
    r = openalex_work_to_record(w)
    assert r["id"] == "arxiv:2507.13001" and r["abstract"] == "Relation-aware geometric learning"
    assert r["pdf_url"] == "https://arxiv.org/pdf/2507.13001"
    assert _abstract_from_inverted(None) == ""


def test_dedupe_merges_sources():
    a = {"id": "arxiv:1", "arxiv_id": "1", "title": "A Paper", "abstract": "", "seed": "topic"}
    b = {"id": "openalex:W9", "arxiv_id": "1", "title": "A paper", "abstract": "text", "doi": "10.1/x", "seed": "author"}
    (m,) = dedupe([a, b])
    assert m["abstract"] == "text" and m["doi"] == "10.1/x" and m["seed"] == "topic"


def test_relevance_filter():
    vocab = ["knowledge graph", "embedding", "link prediction"]
    assert relevance({"title": "Knowledge graph embedding", "abstract": "for link prediction"}, vocab) == 3
    assert relevance({"title": "Protein folding", "abstract": ""}, vocab) == 0


# ------------------------------------------------------------------ eval harness
Q = {"id": "x", "category": "conflict", "question": "Can TransE model symmetric relations?",
     "gold": [{"paper_id": "p1", "contains": "symmetric"}], "expected_status": ["conflicting"],
     "must_include": [["rotate"], ["loss"]], "must_not_include": ["transe can never"]}


def test_score_answer_pass():
    r = {"answer": "RotatE says no, but the loss-aware paper says yes [p1#t0].", "citations": ["p1#t0"],
         "evidence_status": "conflicting", "steps": [{"tool": "search_papers"}], "invalid_citations": []}
    s = score_answer(Q, r)
    assert s["pass"] and s["fact_recall"] == 1.0 and s["citation_precision"] == 1.0


def test_score_answer_fails_on_wrong_status_trap_or_fabrication():
    base = {"answer": "RotatE and loss.", "citations": ["p1#t0"], "evidence_status": "conflicting", "steps": [], "invalid_citations": []}
    assert not score_answer(Q, {**base, "evidence_status": "supported"})["pass"]
    assert not score_answer(Q, {**base, "answer": "RotatE, loss: TransE can never do it"})["pass"]
    assert not score_answer(Q, {**base, "invalid_citations": ["p9#t1"]})["pass"]
    assert score_answer(Q, {**base, "citations": ["p2#t0"]})["citation_precision"] == 0.0


def test_aggregate_reports_robustness_metrics():
    rows = [score_answer({**Q, "category": "unanswerable", "expected_status": ["insufficient"], "must_include": []},
                         {"answer": "", "citations": [], "evidence_status": "insufficient", "steps": []}),
            score_answer(Q, {"answer": "rotate loss", "citations": ["p1#t0"], "evidence_status": "supported", "steps": []})]
    agg = aggregate(rows)
    assert agg["abstention_accuracy"] == 1.0 and agg["conflict_detection"] == 0.0 and agg["pass_rate"] == 0.5


def test_retrieval_metrics_on_toy_index(built):
    qs = [{"id": "t", "question": "CardNet exact match", "gold": [{"paper_id": "toy:card", "contains": "0.61"}]}]
    m = retrieval_metrics(built.index, qs, modes=("bm25",))
    assert m["bm25"]["recall@1"] == 1.0 and m["bm25"]["MRR"] == 1.0


def test_gold_coverage_detects_bad_gold(built):
    qs = [{"id": "bad", "question": "?", "gold": [{"paper_id": "toy:card", "contains": "phrase that does not exist"}]}]
    assert gold_coverage(built.index, qs)


def test_eval_set_schema():
    qs = load_questions(Path(__file__).resolve().parent.parent / "evals" / "questions.jsonl")
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids)) >= 25
    statuses = {"supported", "partially_supported", "conflicting", "insufficient"}
    for q in qs:
        assert set(q["expected_status"]) <= statuses and q["question"].strip()
        assert all(isinstance(alts, list) and alts for alts in q.get("must_include", []))
    cats = {q["category"] for q in qs}
    assert {"conflict", "unanswerable", "figure", "table", "analysis", "false_premise"} <= cats
