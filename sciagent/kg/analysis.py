"""Test-leakage analysis and the inverse-rule baseline for link prediction.

Reproduces, on real benchmark files, the argument of Dettmers et al. (2018):
on WN18 most test triples are trivially recoverable from an inverse (or
equivalent) relation in the training set, so a one-line rule model scores
near state-of-the-art; on WN18RR it collapses.

All ranking uses the *filtered* setting and the RANDOM tie-breaking protocol
recommended by Sun et al. (2020): the true entity is placed uniformly at random
among candidates with the same score, and we report the exact expectation.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

INV = "^-1"


def augment(triples):
    """Add reciprocal triples (t, r^-1, h) so head prediction == tail prediction."""
    out = list(triples)
    out += [(t, r + INV, h) for h, r, t in triples]
    return out


def _pairs_by_rel(triples):
    d = defaultdict(set)
    for h, r, t in triples:
        d[r].add((h, t))
    return d


def mine_implications(train, min_conf: float = 0.9, min_support: int = 10):
    """Rules q' => q with confidence |pairs(q') & pairs(q)| / |pairs(q')| on the augmented train set.

    With reciprocal triples this captures inverse pairs (a^-1 => b), symmetric
    relations (r^-1 => r) and duplicate/equivalent relations (a => b).
    """
    by_rel = _pairs_by_rel(augment(train))
    rules = defaultdict(list)  # q -> [(q', conf)]
    rels = list(by_rel)
    for qp in rels:
        P = by_rel[qp]
        if len(P) < min_support:
            continue
        for q in rels:
            if q == qp:
                continue
            conf = len(P & by_rel[q]) / len(P)
            if conf >= min_conf:
                rules[q].append((qp, round(conf, 3)))
    return dict(rules)


def leakage_stats(split: dict, min_conf: float = 0.9) -> dict:
    train, test = split["train"], split["test"]
    aug = augment(train)
    pair_rels = defaultdict(set)
    for h, r, t in aug:
        pair_rels[(h, t)].add(r)
    rules = mine_implications(train, min_conf)
    any_link = rule_hit = 0
    for h, r, t in test:
        rs = pair_rels.get((h, t), set()) | {x + INV if not x.endswith(INV) else x[:-len(INV)] for x in pair_rels.get((t, h), set())}
        rs.discard(r)
        if rs:
            any_link += 1
        if any(qp in pair_rels.get((h, t), ()) for qp, _ in rules.get(r, [])):
            rule_hit += 1
    n = len(test)
    return {
        "n_train": len(train), "n_test": n,
        "n_entities": len({e for h, _, t in train for e in (h, t)}),
        "n_relations": len({r for _, r, _ in train}),
        "test_pairs_linked_in_train_pct": round(100 * any_link / n, 2),
        "test_recoverable_by_rule_pct": round(100 * rule_hit / n, 2),
        "n_rules": sum(len(v) for v in rules.values()),
        "example_rules": [f"{qp} => {q} (conf {c})" for q, lst in list(rules.items())[:8] for qp, c in lst[:1]],
    }


def _expected_rr_and_hits(higher: int, ties: int, k: int = 10):
    """True entity rank uniform on [higher+1, higher+1+ties]."""
    lo, hi = higher + 1, higher + 1 + ties
    n = ties + 1
    rr = (_harmonic(hi) - _harmonic(lo - 1)) / n
    hits = max(0, min(k, hi) - lo + 1) / n
    return rr, hits, (lo + hi) / 2


def _harmonic(n: int) -> float:
    if n < 64:
        return float(sum(1.0 / i for i in range(1, n + 1)))
    return float(np.log(n) + 0.5772156649015329 + 1 / (2 * n) - 1 / (12 * n * n))


def inverse_rule_eval(split: dict, min_conf: float = 0.9, max_test: int | None = None, seed: int = 0) -> dict:
    """Filtered MRR / Hits@10 of the rule model (score 1 if a rule fires, else 0)."""
    train = split["train"]
    rules = mine_implications(train, min_conf)
    aug_train = augment(train)
    out_by = defaultdict(set)  # (h, q) -> tails in augmented train
    for h, r, t in aug_train:
        out_by[(h, r)].add(t)
    known = defaultdict(set)  # all true tails, all splits (for filtering)
    for s in ("train", "valid", "test"):
        for h, r, t in augment(split[s]):
            known[(h, r)].add(t)
    entities = {e for s in split.values() for h, _, t in s for e in (h, t)}
    N = len(entities)
    test = augment(split["test"])
    if max_test and len(test) > max_test:
        idx = np.random.default_rng(seed).choice(len(test), max_test, replace=False)
        test = [test[i] for i in idx]
    rr, h10, mr = [], [], []
    for h, q, t in test:
        cand = set()
        for qp, _ in rules.get(q, []):
            cand |= out_by.get((h, qp), set())
        filt = known[(h, q)] - {t}
        cand -= filt
        if t in cand:
            higher, ties = 0, len(cand) - 1
        else:
            higher = len(cand)
            ties = N - len(cand | filt | {t})
        a, b, c = _expected_rr_and_hits(higher, ties)
        rr.append(a), h10.append(b), mr.append(c)
    return {"model": "inverse-rule", "MRR": round(float(np.mean(rr)), 4), "Hits@10": round(float(np.mean(h10)), 4),
            "MR": round(float(np.mean(mr)), 1), "n_queries": len(test), "n_rules": sum(len(v) for v in rules.values())}
