"""SciQA NL->SPARQL: metrics, example retrieval (no training), baseline, Claude generator, split hygiene."""
from pathlib import Path

import pytest

from sciagent import sciqa
from sciagent.sciqa import (ClaudeSparql, ExampleRetriever, Item, answer_values, bleu4, clean_sparql, evaluate,
                            knn_template, literal_f1, normalise_query, rouge, set_prf, sparql_tokens)

Q1 = 'SELECT DISTINCT ?paper ?paper_lbl WHERE { ?dataset a orkgc:Dataset; rdfs:label ?dataset_lbl. FILTER (str(?dataset_lbl) = "HoC") ?benchmark orkgp:HAS_DATASET ?dataset. ?cont orkgp:HAS_BENCHMARK ?benchmark. ?paper orkgp:P31 ?cont; rdfs:label ?paper_lbl. }'
Q2 = 'SELECT DISTINCT ?model ?model_lbl WHERE { ?dataset a orkgc:Dataset; rdfs:label ?dataset_lbl. FILTER (str(?dataset_lbl) = "CommonsenseQA") ?benchmark orkgp:HAS_DATASET ?dataset. ?cont orkgp:HAS_BENCHMARK ?benchmark; orkgp:HAS_MODEL ?model. ?model rdfs:label ?model_lbl. }'

TRAIN = [
    Item("A1", "What are the titles and IDs of research papers that include a benchmark for the HoC dataset?", Q1, "T02", False, "Factoid"),
    Item("A2", "Can you list the models that have been evaluated on the CommonsenseQA dataset?", Q2, "T01", False, "Factoid"),
    Item("A3", "What are the titles and IDs of research papers that include a benchmark for the WikiText-2 dataset?",
         Q1.replace("HoC", "WikiText-2"), "T02", False, "Factoid"),
]


# ------------------------------------------------------------------ metrics
def test_bleu_identity_and_disjoint():
    t = sparql_tokens(Q1)
    assert bleu4([t], [t]) == pytest.approx(1.0)
    assert bleu4([t], [["foo", "bar", "baz", "qux"]]) == 0.0
    assert bleu4([t], [[]]) == 0.0


def test_bleu_brevity_penalty():
    t = sparql_tokens(Q1)
    assert bleu4([t], [t[: len(t) // 2]]) < 0.7


def test_rouge_identity():
    t = sparql_tokens(Q2)
    assert rouge(t, t) == {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}


def test_exact_match_ignores_whitespace_and_variable_names():
    renamed = Q1.replace("?paper_lbl", "?title").replace("?paper", "?p").replace("  ", " ")
    assert normalise_query(renamed) == normalise_query(Q1.replace(" WHERE", "\nWHERE"))
    assert normalise_query(Q1) != normalise_query(Q1.replace("HoC", "SST-2"))  # literals matter


def test_literal_f1():
    assert literal_f1(Q1, Q1) == 1.0
    assert literal_f1(Q1, Q1.replace("HoC", "SST-2")) == 0.0


def test_answer_set_prf():
    assert set_prf(["a", "b"], ["a", "b"]) == (1.0, 1.0, 1.0)
    p, r, f = set_prf(["a", "b"], ["a", "c", "d"])
    assert (round(p, 3), r, round(f, 3)) == (0.333, 0.5, 0.4)
    assert set_prf([], []) == (1.0, 1.0, 1.0) and set_prf(["a"], None) == (0.0, 0.0, 0.0)


def test_answer_values_normalise_numbers_and_ask():
    j = {"results": {"bindings": [{"v": {"type": "literal", "value": "2857.3636363636365e0"}}]}}
    assert answer_values(j) == [f"{2857.3636363636365:.6g}"]
    assert answer_values({"boolean": True}) == ["true"]


def test_clean_sparql_strips_fences_and_prefixes():
    raw = "```sparql\nPREFIX orkgp: <http://orkg.org/orkg/predicate/>\nSELECT ?x WHERE { ?x orkgp:P31 ?y }\n```"
    assert clean_sparql(raw) == "SELECT ?x WHERE { ?x orkgp:P31 ?y }"


# ------------------------------------------------------------------ retrieval + baseline (no training)
def test_example_retriever_returns_same_template():
    r = ExampleRetriever(TRAIN)
    ex, score = r.top_k("What are the titles and IDs of research papers that include a benchmark for the SST-2 dataset?", 1)[0]
    assert ex.template_id == "T02" and score > 0
    assert r.top_k("anything", 0) == []


def test_knn_template_substitutes_aligned_entity():
    r = ExampleRetriever(TRAIN)
    q = knn_template("What are the titles and IDs of research papers that include a benchmark for the SciERC dataset?", r)
    assert '"SciERC"' in q and "HoC" not in q and "WikiText-2" not in q
    q = knn_template("Can you list the models that have been evaluated on the Story Cloze Test dataset?", r)
    assert '"Story Cloze Test"' in q


def test_evaluate_splits_handcrafted_and_generated():
    items = TRAIN[:2] + [Item("H1", "hand question", Q2, None, True, "Factoid")]
    preds = {"A1": Q1, "A2": Q1, "H1": ""}
    m = evaluate(items, preds)
    assert m["template_generated"]["n"] == 2 and m["handcrafted"]["n"] == 1
    assert m["template_generated"]["exact_match"] == 50.0 and m["handcrafted"]["exact_match"] == 0.0


def test_claude_generator_uses_retrieved_train_examples(fake):
    client = fake.Client([fake.response(fake.text("```sparql\n" + Q1.replace("HoC", "SciERC") + "\n```"))])
    gen = ClaudeSparql(ExampleRetriever(TRAIN), client=client, model="m", k=2)
    out = gen("What are the titles and IDs of research papers that include a benchmark for the SciERC dataset?")
    assert '"SciERC"' in out and not out.startswith("```")
    call = client.calls[0]
    prompt = call["messages"][0]["content"]
    assert prompt.count("SPARQL:") == 3 and "HoC" in prompt  # 2 shots + the query slot
    assert call["temperature"] == 0.0


def test_zero_shot_prompt_has_no_examples():
    gen = ClaudeSparql(ExampleRetriever(TRAIN), client=object(), model="m", k=0)
    assert "Examples" not in gen.prompt("q?")


# ------------------------------------------------------------------ real split hygiene (skips if data absent)
DATA = Path(__file__).resolve().parent.parent / "data" / "sciqa"


@pytest.mark.skipif(not (DATA / "test" / "questions.json").exists(), reason="SciQA not downloaded")
def test_official_split_sizes_and_disjoint_ids():
    tr, va, te = (sciqa.load_split(DATA, s) for s in ("train", "valid", "test"))
    assert (len(tr), len(va), len(te)) == (1795, 257, 513)
    assert not ({i.id for i in tr} & {i.id for i in te}) and not ({i.id for i in tr} & {i.id for i in va})
    assert all(i.answers is not None for i in te)
