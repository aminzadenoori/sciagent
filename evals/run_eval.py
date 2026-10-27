"""Evaluation harness.

  python evals/run_eval.py --retrieval-only          # no API key needed
  python evals/run_eval.py                            # full agent eval (needs ANTHROPIC_API_KEY)
  python evals/run_eval.py --judge                    # + primary score: LLM judge vs golden answers
  python evals/run_eval.py --ids q08 q15 --judge      # a subset

Writes evals/results/<timestamp>/{retrieval.json, agent_runs.jsonl, scores.json, report.md}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sciagent.evaluation import (aggregate, gold_coverage, judge_groundedness, load_questions,  # noqa: E402
                                 retrieval_metrics, score_answer)
from sciagent.index import HybridIndex  # noqa: E402

DATA = Path(os.environ.get("SCIAGENT_DATA", ROOT / "data"))


def fmt_table(rows: dict, cols: list[str]) -> str:
    s = "| mode | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for m, r in rows.items():
        s += f"| {m} | " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--model", default=os.environ.get("SCIAGENT_MODEL", "claude-sonnet-5-5"))
    ap.add_argument("--judge", action="store_true",
                    help="LLM judge: compare each answer with its golden answer (primary score) and check that "
                         "cited passages support the answer (groundedness)")
    ap.add_argument("--judge-model", default=os.environ.get("SCIAGENT_JUDGE_MODEL"),
                    help="model for the judge (default: same as --model); a different model reduces self-preference")
    ap.add_argument("--out", default=str(ROOT / "evals" / "results" / time.strftime("%Y%m%d-%H%M%S")))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qs = load_questions(Path(a.questions))
    if a.ids:
        qs = [q for q in qs if q["id"] in a.ids]
    index = HybridIndex.load(DATA / "index")

    problems = gold_coverage(index, qs)
    if problems:
        print("EVAL SET PROBLEMS (gold evidence not in corpus):\n  " + "\n  ".join(problems))
    ret = retrieval_metrics(index, qs)
    (out / "retrieval.json").write_text(json.dumps(ret, indent=1))
    cols = ["recall@1", "recall@3", "recall@5", "all_gold@5", "MRR", "nDCG@10"]
    report = [f"# Evaluation report\n\nCorpus: {len(index.papers)} papers, {len(index.chunks)} chunks. "
              f"Questions: {len(qs)} ({sum(bool(q.get('gold')) for q in qs)} with gold evidence).\n",
              "## Retrieval\n", fmt_table({m: r for m, r in ret.items()}, cols)]
    print(report[-1])

    if not a.retrieval_only:
        from sciagent.agent import ResearchAgent
        import anthropic
        client = anthropic.Anthropic()
        from sciagent.judge import judge as judge_reference, summarize
        agent = ResearchAgent(index, DATA, client=client, model=a.model)
        judge_model = a.judge_model or a.model
        scores = []
        with open(out / "agent_runs.jsonl", "w") as f:
            for q in qs:
                print(f"[{q['id']}] {q['question']}")
                r = agent.run(q["question"]).to_dict()
                s = score_answer(q, r, index)
                if a.judge:
                    if q.get("reference_answer"):
                        s["reference_judge"] = judge_reference(client, judge_model, q["question"], q["reference_answer"],
                                                               r["answer"], q.get("answer_type", "fact"),
                                                               r.get("evidence_status"))
                    s["judge"] = judge_groundedness(client, judge_model, q, r, index)
                scores.append(s)
                f.write(json.dumps({"question": q, "result": r, "score": s}) + "\n")
                v = (s.get("reference_judge") or {}).get("verdict", "-")
                print(f"     verdict={v} status={s['status']} rule-pass={s['pass']} tools={s['tools']}")
        agg = aggregate(scores)
        if a.judge:
            g = [s["judge"]["groundedness"] for s in scores if s.get("judge")]
            agg["judge_groundedness"] = round(sum(g) / len(g), 3) if g else None
            js = [s["reference_judge"] for s in scores if s.get("reference_judge")]
            agg["golden_answer_judge"] = summarize(js)
            by_cat = {}
            for s in scores:
                if s.get("reference_judge"):
                    by_cat.setdefault(s["category"], []).append(s["reference_judge"])
            agg["golden_answer_judge"]["by_category"] = {c: summarize(v)["mean_score"] for c, v in sorted(by_cat.items())}
        (out / "scores.json").write_text(json.dumps({"aggregate": agg, "per_question": scores}, indent=1))
        report += ["\n## Agent\n", f"Answer model: `{a.model}`. Judge model: `{judge_model}`.\n"]
        if agg.get("golden_answer_judge"):
            g = agg["golden_answer_judge"]
            report += [f"\n### Primary score: LLM judge vs golden answers\n\nMean score **{g['mean_score']}** over {g['n']} "
                       f"questions (correct / correct abstention = 1, partially correct = 0.5, otherwise 0).\n\n"
                       f"Verdicts: " + ", ".join(f"{k} {v}" for k, v in g["verdicts"].items()) + "\n\n"
                       "| id | category | verdict | missing | contradictions | judge's explanation |\n|---|---|---|---|---|---|\n" +
                       "".join(f"| {s['id']} | {s['category']} | {s['reference_judge'].get('verdict')} | "
                               f"{'; '.join(s['reference_judge'].get('missing_points') or [])[:160]} | "
                               f"{'; '.join(s['reference_judge'].get('contradictions') or [])[:160]} | "
                               f"{(s['reference_judge'].get('explanation') or '')[:200]} |\n"
                               for s in scores if s.get("reference_judge"))]
        report += ["\n### Secondary checks (rule-based)\n", "```json\n" + json.dumps(
                       {k: v for k, v in agg.items() if k != "golden_answer_judge"}, indent=1) + "\n```\n",
                   "\n| id | category | status | rule pass | fact recall | tools |\n|---|---|---|---|---|---|\n" +
                   "".join(f"| {s['id']} | {s['category']} | {s['status']} | {s['pass']} | {s['fact_recall']} | "
                           f"{', '.join(s['tools'])} |\n" for s in scores)]
        print(json.dumps(agg, indent=1))
    (out / "report.md").write_text("\n".join(report))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
