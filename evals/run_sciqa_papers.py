"""Answer SciQA questions from the papers and compare with ORKG's answers (needs ANTHROPIC_API_KEY + internet).

  python evals/run_sciqa_papers.py --limit 30
  python evals/run_sciqa_papers.py --limit 8 --templates T05 T04      # cheaper first look
  python evals/run_sciqa_papers.py --dry-run                            # show the selected questions only

What happens for each question:
  1. the main agent answers it as a normal user question, from papers (corpus + literature agent);
     the ORKG tool is disabled, so the answer key cannot be looked up;
  2. the answer is compared with SciQA's gold answer (which comes from ORKG) by fuzzy matching.

Each run works on a COPY of the corpus, so papers the agents download do not pollute data/corpus.
Writes evals/results/sciqa-papers-<time>/{selected.json, runs.jsonl, scores.json, report.md, review.md}.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sciagent.sciqa import download, load_split  # noqa: E402
from sciagent.sciqa_papers import (FRAMING, KINDS, aggregate, gold_text, load_items, review_sheet, score,  # noqa: E402
                                   select)

DATA = Path(os.environ.get("SCIAGENT_DATA", ROOT / "data"))


def make_workspace(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for d in ("corpus", "index"):
        if (src / d).exists():
            shutil.copytree(src / d, dst / d, dirs_exist_ok=True)
    if (src / "kg").exists() and not (dst / "kg").exists():
        try:
            (dst / "kg").symlink_to((src / "kg").resolve(), target_is_directory=True)
        except OSError:  # Windows without symlink rights: the leakage tool will download again if used
            pass
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--templates", nargs="*", default=list(KINDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("SCIAGENT_MODEL", "claude-sonnet-5-5"))
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "evals" / "results" / ("sciqa-papers-" + time.strftime("%Y%m%d-%H%M%S"))))
    a = ap.parse_args()

    root = download(DATA / "sciqa")
    train_q = {t.question.strip().lower() for t in load_split(root, "train")}
    items = select(load_items(root, "test"), train_q, limit=a.limit, seed=a.seed, templates=tuple(a.templates))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected.json").write_text(json.dumps(items, indent=1))
    print(f"selected {len(items)} questions: " + ", ".join(f"{t}={sum(i['template'] == t for i in items)}" for t in a.templates))
    if a.dry_run:
        for it in items:
            print(f"  [{it['template']}] {it['question']}\n        ORKG: {gold_text(it)}")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")

    from sciagent.agent import ResearchAgent
    from sciagent.index import HybridIndex
    ws = make_workspace(DATA, out / "workspace")
    agent = ResearchAgent(HybridIndex.load(ws / "index"), ws, model=a.model, max_steps=a.max_steps,
                          disabled_tools={"query_scholarly_kg"})
    results, scores = [], []
    with open(out / "runs.jsonl", "w") as f:
        for i, it in enumerate(items, 1):
            print(f"[{i}/{len(items)}] {it['template']} {it['question']}")
            r = agent.run(FRAMING + it["question"]).to_dict()
            s = score(it, r)
            results.append(r)
            scores.append(s)
            f.write(json.dumps({"item": it, "result": r, "score": s}) + "\n")
            f.flush()
            print(f"      ORKG: {gold_text(it)}\n      -> {s['outcome']} (score {s['score']}, status {r['evidence_status']})")
    agg = aggregate(scores)
    (out / "scores.json").write_text(json.dumps({"aggregate": agg, "per_question": scores}, indent=1))
    (out / "review.md").write_text(review_sheet(items, results, scores))
    lines = ["# SciQA answered from papers vs ORKG\n", f"Model `{a.model}`, {len(items)} questions, ORKG tool disabled.\n",
             "| type | n | mean score | agrees | disagrees | abstained | used literature agent |", "|---|---|---|---|---|---|---|"]
    for k, b in agg.items():
        if b:
            lines.append(f"| {k} | {b['n']} | {b['mean_score']} | {b['agrees']} | {b['disagrees']} | {b['abstained']} | "
                         f"{b['used_literature_agent']} |")
    lines.append("\nDisagreements are listed in review.md for a human verdict (ORKG may be incomplete or outdated).")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(agg, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
