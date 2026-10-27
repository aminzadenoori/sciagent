"""Who judges the judge? Measure the LLM judge against hand-labelled verdicts (needs ANTHROPIC_API_KEY).

  python evals/check_judge.py                         # judge model = $SCIAGENT_JUDGE_MODEL or $SCIAGENT_MODEL
  python evals/check_judge.py --judge-model claude-opus-5-5

evals/judge_calibration.jsonl holds 12 answers with known verdicts: paraphrases and rounding that must count as
correct, a wrong number, a half answer, a one-sided answer to a real disagreement, an accepted false premise,
a wrong abstention, a correct abstention and two invented results. Run this before trusting judge scores; if
agreement is low, fix the judge prompt before looking at system results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sciagent.judge import VERDICTS, judge  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "evals" / "judge_calibration.jsonl"))
    ap.add_argument("--judge-model", default=os.environ.get("SCIAGENT_JUDGE_MODEL", os.environ.get("SCIAGENT_MODEL", "claude-sonnet-5-5")))
    a = ap.parse_args(argv)
    import anthropic
    client = anthropic.Anthropic()
    cases = [json.loads(l) for l in open(a.cases) if l.strip()]
    rows, confusion = [], Counter()
    for c in cases:
        j = judge(client, a.judge_model, c["question"], c["reference"], c["candidate"], c["answer_type"])
        ok = j["verdict"] == c["expected"]
        # a softer view: does the judge at least put the answer on the right side of pass/fail?
        same_side = (VERDICTS.get(j["verdict"], -1) >= 0.5) == (VERDICTS[c["expected"]] >= 0.5)
        rows.append({"id": c["id"], "expected": c["expected"], "got": j["verdict"], "exact": ok, "same_side": same_side,
                     "note": c.get("note"), "explanation": j.get("explanation")})
        confusion[(c["expected"], j["verdict"])] += 1
        print(f"{c['id']}  expected {c['expected']:<20} got {j['verdict']:<20} {'OK' if ok else 'DIFF'}  ({c.get('note')})")
    n = len(rows)
    summary = {"judge_model": a.judge_model, "n": n,
               "exact_agreement": round(sum(r["exact"] for r in rows) / n, 3),
               "pass_fail_agreement": round(sum(r["same_side"] for r in rows) / n, 3),
               "disagreements": [r for r in rows if not r["exact"]]}
    out = ROOT / "evals" / "results" / "judge_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "disagreements"}, indent=1))
    print(f"wrote {out}")
    return summary


if __name__ == "__main__":
    main()
