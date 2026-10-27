"""Answer SciQA questions from the PAPERS and compare with ORKG's answers (SciQA gold).

This evaluates the main system (retrieval, literature agent, reading result tables,
comparing numbers) against ground truth made independently by other people.

* Question selection: SciQA test questions of types that papers can answer:
    T05  best model on a dataset for a metric      -> did the answer name ORKG's model?
    T04  top score and its metric on a dataset     -> right metric and score (tolerance)?
    T03  metrics used on a dataset                 -> share of ORKG's metrics mentioned
    T01  models evaluated on a dataset             -> share of ORKG's models mentioned
  Excluded: lists of ORKG papers (T02), code links (T07), ORKG field taxonomy (T06, T08),
  hand-written questions (mostly about ORKG comparisons), and items with empty gold answers
  or duplicated in the train split.
* The ORKG tool (query_scholarly_kg) is DISABLED during this evaluation: otherwise the
  system could read the answer key.
* Matching is fuzzy (paper wording differs from ORKG labels) and deterministic.
* A disagreement is not automatically the system's fault: ORKG may be incomplete or
  outdated. Disagreements are written to a review sheet with the system's citations.
"""
from __future__ import annotations

import difflib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

KINDS = {"T05": "best_model", "T04": "top_score", "T03": "metric_list", "T01": "model_list"}
FRAMING = ("Answer this question from the scientific literature (papers), citing them. If papers disagree or "
           "report different settings, say so.\n\nQuestion: ")


# ------------------------------------------------------------------ gold answers
def gold_answer(answer_json: dict | None) -> dict:
    """Literal values from SciQA's gold answer, grouped by variable (URIs are dropped)."""
    out = defaultdict(list)
    for b in ((answer_json or {}).get("results") or {}).get("bindings", []):
        for var, v in b.items():
            if v.get("type") == "literal" and v.get("value", "").strip():
                out[var].append(v["value"].strip())
    return dict(out)


def load_items(root: Path, split: str = "test") -> list[dict]:
    qs = json.load(open(root / split / "questions.json"))["questions"]
    ans = {a["id"]: a.get("answer") for a in json.load(open(root / split / "answers.json"))["answers"]}
    return [{"id": q["id"], "question": q["question"]["string"], "template": q.get("template_id"),
             "sparql": q["query"]["sparql"], "gold": gold_answer(ans.get(q["id"]))} for q in qs]


def _labels(item: dict) -> list[str]:
    g = item["gold"]
    for key in ("model_lbl", "metric_lbl"):
        if g.get(key):
            return g[key]
    return []


def select(items: list[dict], train_questions: set[str], limit: int = 30, seed: int = 0,
           templates=tuple(KINDS)) -> list[dict]:
    """Stratified (round-robin over templates), skipping empty gold and items duplicated in train."""
    pools = defaultdict(list)
    for it in items:
        t = it["template"]
        if t not in templates or it["question"].strip().lower() in train_questions:
            continue
        if t == "T04" and not (it["gold"].get("metric_lbl") and it["gold"].get("score")):
            continue
        if t != "T04" and not _labels(it):
            continue
        pools[t].append({**it, "kind": KINDS[t]})
    rng = random.Random(seed)
    for p in pools.values():
        rng.shuffle(p)
    out, order = [], [t for t in templates if pools.get(t)]
    while len(out) < limit and any(pools[t] for t in order):
        for t in order:
            if pools[t] and len(out) < limit:
                out.append(pools[t].pop())
    return out


# ------------------------------------------------------------------ matching
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.+%]+", " ", s.lower())).strip()


def mentions(label: str, text: str, threshold: float = 0.85) -> bool:
    """Is `label` (an ORKG name) mentioned in `text`, allowing small wording differences?"""
    lab, txt = norm(label), norm(text)
    if not lab:
        return False
    if re.search(r"(^| )" + re.escape(lab) + r"($| )", txt):
        return True
    lt, tt = lab.split(), txt.split()
    if len(lab) <= 3:  # very short names ("F1", "BLEU"-like) must match a whole token exactly
        return lab in tt
    for n in {max(1, len(lt) - 1), len(lt), len(lt) + 1}:
        for i in range(0, max(1, len(tt) - n + 1)):
            if difflib.SequenceMatcher(None, lab, " ".join(tt[i:i + n])).ratio() >= threshold:
                return True
    return False


def numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text.replace(",", ""))]


def number_matches(gold: str, text: str, rel: float = 0.01, absolute: float = 0.051) -> bool:
    try:
        g = float(gold.strip().rstrip("%").replace(",", ""))
    except ValueError:
        return mentions(gold, text)
    for x in numbers(text):
        for cand in (x, x / 100.0, x * 100.0):  # 0.951 vs 95.1 (%)
            if abs(cand - g) <= max(absolute if abs(g) >= 1 else absolute / 100, rel * abs(g)):
                return True
    return False


# ------------------------------------------------------------------ scoring
def score(item: dict, result: dict) -> dict:
    text = (result.get("answer") or "") + " " + " ".join(result.get("conflicts") or [])
    abstained = result.get("evidence_status") == "insufficient"
    kind, g = item["kind"], item["gold"]
    detail = {}
    if kind == "best_model":
        hit = any(mentions(l, text) for l in g["model_lbl"])
        s = 1.0 if hit else 0.0
    elif kind == "top_score":
        m = any(mentions(l, text) for l in g["metric_lbl"])
        v = any(number_matches(x, text) for x in g["score"])
        detail = {"metric_found": m, "score_found": v}
        s = 1.0 if (m and v) else 0.5 if v else 0.0
    else:
        labels = _labels(item)
        found = [l for l in labels if mentions(l, text)]
        detail = {"gold_n": len(labels), "found": found}
        s = len(found) / len(labels)
    outcome = "abstained" if abstained and s == 0 else "agrees" if s >= 0.5 else "disagrees"
    return {"id": item["id"], "template": item["template"], "kind": kind, "score": round(s, 3), "outcome": outcome,
            "status": result.get("evidence_status"), "n_citations": len(result.get("citations") or []),
            "tools": [st.get("tool") for st in result.get("steps", [])], "error": result.get("error"), **detail}


def aggregate(scores: list[dict]) -> dict:
    def block(rows):
        if not rows:
            return None
        n = len(rows)
        return {"n": n, "mean_score": round(sum(r["score"] for r in rows) / n, 3),
                "agrees": round(sum(r["outcome"] == "agrees" for r in rows) / n, 3),
                "disagrees": round(sum(r["outcome"] == "disagrees" for r in rows) / n, 3),
                "abstained": round(sum(r["outcome"] == "abstained" for r in rows) / n, 3),
                "used_literature_agent": round(sum("research_topic" in r["tools"] for r in rows) / n, 3),
                "errors": sum(bool(r["error"]) for r in rows)}
    by = defaultdict(list)
    for r in scores:
        by[r["kind"]].append(r)
    return {"all": block(scores), **{k: block(v) for k, v in sorted(by.items())}}


def gold_text(item: dict) -> str:
    g = item["gold"]
    if item["kind"] == "top_score":
        return f"{', '.join(g['metric_lbl'])} = {', '.join(g['score'])}"
    labels = _labels(item)
    return ", ".join(labels[:8]) + (f" (+{len(labels) - 8} more)" if len(labels) > 8 else "")


def review_sheet(items: list[dict], results: list[dict], scores: list[dict]) -> str:
    """Markdown sheet of disagreements for a human to judge: system error, or ORKG incomplete/outdated?"""
    rows = ["# Disagreements to review\n",
            "ORKG is not complete or always current. For each row decide: **system wrong**, **ORKG outdated/"
            "incomplete**, or **both defensible** (different setting).\n",
            "| id | question | ORKG says | system says (first 300 chars) | citations | your verdict |",
            "|---|---|---|---|---|---|"]
    for it, r, s in zip(items, results, scores):
        if s["outcome"] == "agrees":
            continue
        ans = re.sub(r"\s+", " ", (r.get("answer") or ""))[:300].replace("|", "/")
        rows.append(f"| {it['id']} | {it['question'].replace('|', '/')} | {gold_text(it).replace('|', '/')} | {ans} | "
                    f"{len(r.get('citations') or [])} | |")
    return "\n".join(rows) + "\n"
