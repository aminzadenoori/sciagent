"""KG leakage analysis (data-analysis component) and the evaluation metrics it relies on."""
import math

import numpy as np
import pytest

from sciagent.kg.analysis import _expected_rr_and_hits, augment, inverse_rule_eval, leakage_stats, mine_implications


def toy_split(with_inverse=True, n=100):
    """5% of hypernym triples held out for test; their inverse (hyponym) stays in train, as in WN18."""
    train, test = [], []
    for i in range(n):
        a, b = f"e{i}", f"e{i + 100}"
        if i % 20 == 0:  # test triple
            test.append((a, "hypernym", b))
            if with_inverse:
                train.append((b, "hyponym", a))
        else:
            train.append((a, "hypernym", b))
            if with_inverse:
                train.append((b, "hyponym", a))
    return {"train": train, "valid": [], "test": test}


def test_augment_adds_reciprocals():
    assert augment([("a", "r", "b")]) == [("a", "r", "b"), ("b", "r^-1", "a")]


def test_inverse_rules_are_mined():
    rules = mine_implications(toy_split()["train"], min_conf=0.9, min_support=5)
    assert "hyponym^-1" in [q for q, c in rules["hypernym"]]  # conf = 95/100


def test_leakage_full_when_inverses_present():
    s = leakage_stats(toy_split())
    assert s["test_pairs_linked_in_train_pct"] == 100.0 and s["test_recoverable_by_rule_pct"] == 100.0


def test_no_leakage_without_inverses():
    s = leakage_stats(toy_split(with_inverse=False))
    assert s["test_pairs_linked_in_train_pct"] == 0.0 and s["test_recoverable_by_rule_pct"] == 0.0


def test_rule_model_perfect_on_leaky_split():
    r = inverse_rule_eval(toy_split())
    assert r["MRR"] == pytest.approx(1.0) and r["Hits@10"] == pytest.approx(1.0)


def test_rule_not_mined_when_too_many_exceptions():
    split = toy_split()
    # 30 extra hyponym facts with no hypernym counterpart: conf(hyponym^-1 => hypernym) = 95/130 < 0.9
    split["train"] += [(f"x{i}", "hyponym", f"y{i}") for i in range(30)]
    assert "hyponym^-1" not in [q for q, _ in mine_implications(split["train"], min_conf=0.9).get("hypernym", [])]


def test_rule_model_is_chance_without_leakage():
    r = inverse_rule_eval(toy_split(with_inverse=False))
    assert r["MRR"] < 0.2


@pytest.mark.parametrize("higher,ties", [(0, 0), (0, 4), (3, 10), (7, 200), (0, 5000), (120, 40000)])
def test_random_tie_breaking_expectation_matches_brute_force(higher, ties):
    rr, hits, mean_rank = _expected_rr_and_hits(higher, ties, k=10)
    ranks = np.arange(higher + 1, higher + ties + 2)
    assert rr == pytest.approx(np.mean(1 / ranks), rel=1e-4)
    assert hits == pytest.approx(np.mean(ranks <= 10))
    assert mean_rank == pytest.approx(ranks.mean())
