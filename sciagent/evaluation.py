"""Evaluation: retrieval metrics (no LLM needed) and answer metrics for agent runs.

Gold evidence is specified as (paper_id, phrase) instead of chunk ids, so the
gold set survives re-chunking and re-ingestion: a retrieved chunk is relevant
for a gold item if it comes from that paper and contains that phrase.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def is_relevant(chunk: dict, gold: dict) -> bool:
    return chunk["paper_id"] == gold["paper_id"] and gold["contains"].lower() in chunk["text"].lower()


def gold_coverage(index, questions) -> list[str]:
    """Sanity check of the eval set itself: every gold item must exist in the corpus."""
    problems = []
    for q in questions:
        for g in q.get("gold", []):
            if not any(is_relevant(c, g) for c in index.chunks):
                problems.append(f"{q['id']}: no chunk matches {g}")
    return problems


def retrieval_metrics(index, questions, modes=("bm25", "dense", "hybrid"), ks=(1, 3, 5, 10)) -> dict:
    out = {}
    qs = [q for q in questions if q.get("gold")]
    for mode in modes:
        rec = {k: [] for k in ks}
        allk = {k: [] for k in ks}
        rr, ndcg = [], []
        per_q = {}
        for q in qs:
            hits = index.search(q["question"], k=max(ks), mode=mode, per_paper=max(ks))
            chunks = [index.by_id[h.chunk_id] for h in hits]
            rel = [any(is_relevant(c, g) for g in q["gold"]) for c in chunks]
            first = next((i for i, r in enumerate(rel) if r), None)
            rr.append(0.0 if first is None else 1.0 / (first + 1))
            for k in ks:
                found = [any(is_relevant(c, g) for c in chunks[:k]) for g in q["gold"]]
                rec[k].append(sum(found) / len(found))
                allk[k].append(float(all(found)))
            dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rel[:10]) if r)
            ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(q["gold"]))))
            ndcg.append(dcg / ideal if ideal else 0.0)
            per_q[q["id"]] = {"first_relevant_rank": None if first is None else first + 1}
        out[mode] = {**{f"recall@{k}": round(sum(v) / len(v), 3) for k, v in rec.items()},
                     **{f"all_gold@{k}": round(sum(v) / len(v), 3) for k, v in allk.items()},
                     "MRR": round(sum(rr) / len(rr), 3), "nDCG@10": round(sum(ndcg) / len(ndcg), 3),
                     "n_questions": len(qs), "per_question": per_q}
    return out


# ------------------------------------------------------------------ answers
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("–", "-").replace("_", " ")).strip()


def score_answer(q: dict, r: dict, index=None) -> dict:
    """Score one agent result (dict from AgentResult.to_dict())."""
    text = _norm(r.get("answer", "") + " " + " ".join(r.get("conflicts", [])))
    facts = q.get("must_include", [])
    fact_hits = [any(_norm(alt).replace("_", " ") in text for alt in alts) for alts in facts]
    traps = [t for t in q.get("must_not_include", []) if _norm(t) in text]
    status_ok = r.get("evidence_status") in q["expected_status"]
    cited = [c for c in r.get("citations", []) if not c.endswith("#meta")]
    gold_papers = {g["paper_id"] for g in q.get("gold", [])}
    cit_precision = cit_recall = None
    if gold_papers and cited:
        cited_papers = {c.split("#")[0] for c in cited}
        # precision: cited chunks that are relevant to *some* gold item OR from a gold paper
        cit_precision = round(sum(c.split("#")[0] in gold_papers for c in cited) / len(cited), 3)
        cit_recall = round(len(cited_papers & gold_papers) / len(gold_papers), 3)
    elif gold_papers:
        cit_precision, cit_recall = 0.0, 0.0
    tools_used = [s["tool"] for s in r.get("steps", [])]
    exp_tools = q.get("expected_tools")
    tool_ok = None if not exp_tools else any(t in tools_used for t in exp_tools)
    return {
        "id": q["id"], "category": q["category"],
        "status": r.get("evidence_status"), "status_ok": status_ok,
        "fact_recall": round(sum(fact_hits) / len(fact_hits), 3) if facts else None,
        "trap_hits": traps,
        "citation_precision": cit_precision, "citation_recall": cit_recall,
        "invalid_citations": len(r.get("invalid_citations", [])),
        "expected_tool_used": tool_ok, "n_tool_calls": len(tools_used), "tools": tools_used,
        "guardrails": r.get("guardrail_events", []), "error": r.get("error"),
        "seconds": r.get("seconds"), "tokens": r.get("usage", {}),
        # strict pass: right status, all required facts, no trap, no fabricated citation
        "pass": bool(status_ok and all(fact_hits) and not traps and not r.get("invalid_citations") and not r.get("error")),
    }


def aggregate(scores: list[dict]) -> dict:
    def mean(key, rows=scores):
        v = [s[key] for s in rows if s[key] is not None]
        return round(sum(map(float, v)) / len(v), 3) if v else None
    by_cat = {}
    for s in scores:
        by_cat.setdefault(s["category"], []).append(s)
    return {
        "n": len(scores), "pass_rate": mean("pass"), "status_accuracy": mean("status_ok"),
        "fact_recall": mean("fact_recall"), "citation_precision": mean("citation_precision"),
        "citation_recall": mean("citation_recall"),
        "abstention_accuracy": mean("status_ok", [s for s in scores if s["category"] in ("unanswerable", "tool_failure")]),
        "conflict_detection": mean("status_ok", [s for s in scores if s["category"] == "conflict"]),
        "trap_rate": round(sum(bool(s["trap_hits"]) for s in scores) / len(scores), 3),
        "invalid_citation_total": sum(s["invalid_citations"] for s in scores),
        "expected_tool_rate": mean("expected_tool_used"),
        "mean_tool_calls": round(sum(s["n_tool_calls"] for s in scores) / len(scores), 2),
        "errors": sum(bool(s["error"]) for s in scores),
        "by_category": {c: {"n": len(v), "pass_rate": mean("pass", v)} for c, v in sorted(by_cat.items())},
    }


JUDGE_PROMPT = """You check whether an answer is grounded in its cited evidence.
Question: {q}
Answer: {a}
Cited evidence:
{ev}
For each factual claim in the answer, decide if the cited evidence supports it. Reply with JSON only:
{{"claims": <int>, "supported": <int>, "unsupported_examples": [<short strings>]}}"""


def judge_groundedness(client, model, q: dict, r: dict, index) -> dict | None:
    """Optional LLM-as-judge: fraction of answer claims supported by the cited chunks."""
    ev = "\n".join(f"[{c}] {index.by_id[c]['text'][:1200]}" for c in r.get("citations", []) if c in index.by_id)
    if not ev:
        return None
    resp = client.messages.create(model=model, max_tokens=400, messages=[
        {"role": "user", "content": JUDGE_PROMPT.format(q=q["question"], a=r["answer"], ev=ev)}])
    txt = "".join(getattr(b, "text", "") for b in resp.content)
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        d = json.loads(m.group(0))
        d["groundedness"] = round(d["supported"] / max(1, d["claims"]), 3)
        return d
    except Exception:
        return None
