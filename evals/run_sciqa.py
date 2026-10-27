"""SciQA NL->SPARQL benchmark (no training, no fine-tuning).

  python evals/run_sciqa.py --generator knn                     # offline baseline, no API key
  python evals/run_sciqa.py --generator claude --shots 0 1 3 5  # Claude, k retrieved train examples
  python evals/run_sciqa.py --generator claude --shots 3 --execute --limit 100   # + answer F1 on ORKG

Protocol
* train split = few-shot example pool only; test split = evaluation (valid via --split valid).
* every metric is reported overall and separately for handcrafted (HQ) vs template-generated (AQ)
  questions, with a 95% bootstrap confidence interval on the headline metric.
* leakage audit: test items whose question text or normalised gold query also occurs in train
  are reported and excluded by default (--keep-overlap to keep them).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sciagent.sciqa import (ClaudeSparql, ExampleRetriever, download, evaluate, execute, knn_template,  # noqa: E402
                            load_split, normalise_query, template_hit_at_k)


def bootstrap_ci(values, n=1000, seed=0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [round(100 * float(np.percentile(means, 2.5)), 2), round(100 * float(np.percentile(means, 97.5)), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "sciqa"))
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--generator", default="knn", choices=["knn", "claude"])
    ap.add_argument("--shots", type=int, nargs="*", default=[3])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--execute", action="store_true", help="run queries on the ORKG endpoint for answer F1")
    ap.add_argument("--model")
    ap.add_argument("--keep-overlap", action="store_true", help="keep test items duplicated in train")
    ap.add_argument("--out", default=str(ROOT / "evals" / "results" / ("sciqa-" + time.strftime("%Y%m%d-%H%M%S"))))
    a = ap.parse_args()

    root = download(Path(a.data))
    train, items = load_split(root, "train"), load_split(root, a.split)
    # Leakage audit. The official split has a few test items whose question or gold query also occurs in
    # train; a retrieval-based few-shot system would simply copy them. They are excluded by default.
    assert not {t.id for t in train} & {i.id for i in items}, "id overlap between train and eval split"
    tq = {t.question.strip().lower() for t in train}
    tsq = {normalise_query(t.sparql) for t in train}
    leaked = [i.id for i in items if i.question.strip().lower() in tq or normalise_query(i.sparql) in tsq]
    print(f"leakage audit: {len(leaked)} {a.split} items duplicate a train question or query: {leaked}")
    if not a.keep_overlap:
        items = [i for i in items if i.id not in leaked]
    if a.limit:  # stratified: keep all handcrafted questions, fill with generated ones
        hq = [i for i in items if i.handcrafted]
        aq = [i for i in items if not i.handcrafted]
        items = hq + aq[:max(0, a.limit - len(hq))]
    retr = ExampleRetriever(train)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"split": a.split, "n": len(items), "leaked_excluded": [] if a.keep_overlap else leaked, "example_retrieval": template_hit_at_k(items, retr), "runs": {}}
    print("example retrieval:", report["example_retrieval"])

    settings = [("knn-template", None)] if a.generator == "knn" else [(f"claude-{k}shot", k) for k in a.shots]
    for name, k in settings:
        gen = (lambda q: knn_template(q, retr)) if k is None else ClaudeSparql(retr, k=k, model=a.model)
        preds, executed = {}, ({} if a.execute else None)
        t0 = time.time()
        for n, it in enumerate(items):
            try:
                preds[it.id] = gen(it.question)
            except Exception as e:
                preds[it.id] = ""
                print(f"  [{it.id}] generation failed: {e}")
            if a.execute:
                executed[it.id] = execute(preds[it.id])
            if (n + 1) % 50 == 0:
                print(f"  {name}: {n + 1}/{len(items)}")
        m = evaluate(items, preds, executed)
        em = [normalise_query(i.sparql) == normalise_query(preds[i.id]) for i in items]
        m["all"]["exact_match_95CI"] = bootstrap_ci(em)
        m["seconds"] = round(time.time() - t0, 1)
        report["runs"][name] = m
        with open(out / f"predictions_{name}.jsonl", "w") as f:
            for it in items:
                f.write(json.dumps({"id": it.id, "question": it.question, "gold": it.sparql, "pred": preds[it.id],
                                    "handcrafted": it.handcrafted, "template": it.template_id,
                                    **({"executed": executed.get(it.id)} if executed is not None else {})}) + "\n")
        print(name, json.dumps(m, indent=1))
    (out / "sciqa_report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
