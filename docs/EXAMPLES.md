# Example questions and outputs

Tool outputs below are **real**, captured from the built index in the build sandbox. Final agent answers need the
Claude API; generate them with `python scripts/run_examples.py` (writes `docs/examples_output.md`) — I have not
pasted model answers here that were not actually produced.

## 1. Conflicting evidence — "Can TransE model symmetric relations?"

Expected behaviour: `evidence_status="conflicting"`; RotatE's relation-pattern table says TransE cannot model symmetry,
while *Loss-Aware Pattern Inference* (Nayyeri, …, Vahdati, Lehmann) argues that proof assumes a zero residual that
the training losses never enforce, so with a suitable margin TransE can. The disagreement is driven by an assumption
about the loss, which the answer should name.

First retrieval step (`search_papers`, k=4):

```
pakdd2021:loss-aware-pattern-inference#t0   Problem   "Several papers claim that TransE and its successor..."
report:kg-leakage#t4                        Conclusion
report:kg-leakage#t3                        Results
pakdd2021:loss-aware-pattern-inference#t3   Results   "On FB15k-237, TransE with gamma1 = 0 reaches MRR 27.4..."
```

RotatE's side is not in the top 4 for this phrasing, so a second query (e.g. "RotatE relation patterns symmetry
TransE") is needed before the agent can state the conflict with citations to both sides — the reason a fixed
single-retrieval pipeline would under-report this conflict.

## 2. Multimodal — "In Figure 1 of the leakage report, which dataset has the lowest bars?"

```
search_papers(query="leakage report figure", kinds=["figure","table"])
  report:kg-leakage#fig0_0  figure  "Figure 1. Percentage of test triples recoverable by a mined rule (blue) and filtered MRR x100..."
  report:kg-leakage#tab0_0  table   "Table 1. Leakage statistics and rule-model performance ..."
inspect_figure(chunk_id="report:kg-leakage#fig0_0", question=...)   -> Claude vision reads the PNG
```

Ground truth (from the generated figure): FB15k-237 has the lowest bars (0.0 recoverable, MRR×100 ≈ 0.1); WN18 ≈ 93
for both. The table holds the same numbers to 3 decimals, so a vision reading can be checked against the table.

## 3. Data analysis — "Compute the inverse-relation test leakage of WN18RR"

`kg_leakage_analysis(dataset="WN18RR")` (real output, abridged):

```json
{"leakage": {"n_train": 86835, "n_test": 3134, "test_pairs_linked_in_train_pct": 34.97,
             "test_recoverable_by_rule_pct": 33.57, "n_rules": 6,
             "example_rules": ["_derivationally_related_form => _derivationally_related_form^-1 (conf 0.932)",
                               "_verb_group => _verb_group^-1 (conf 0.931)", "_similar_to => _similar_to^-1 (conf 0.925)"]},
 "rule_model": {"MRR": 0.3359, "Hits@10": 0.3358, "MR": 13593.2, "n_queries": 6268},
 "note": "computed now from the benchmark files; cite as tool output, not as a paper"}
```

## 4. Missing evidence — "What MRR does SMART achieve on YAGO3-10?"

The SMART card reports WN18RR, FB15k-237, YouTube and COKG only. Expected: search, then
`evidence_status="insufficient"`, no number. The harness rejects an `insufficient` answer given without searching
(tested in `test_abstaining_without_looking_is_rejected`).

## 5. Corpus question — "Which papers in the corpus have Sahar Vahdati as a co-author?"

`list_papers(author="Vahdati")` → 7 papers (ARC survey 2026, Are Researchers Being Replaced 2026, ReFactX 2025,
Aligning KGs and LMs 2025, SMART 2025, SCI-IDEA 2025, Loss-Aware Pattern Inference 2021), citable as `<id>#meta`.

## 6. Structured scholarly KG — "Which model has the highest Accuracy on the Story Cloze Test benchmark (ORKG)?"

`query_scholarly_kg` retrieves the 3 most similar SciQA *train* questions as examples, Claude writes SPARQL, and it is
executed on ORKG. The non-LLM baseline produces this query for the same question (it matches the gold query):

```sparql
SELECT DISTINCT ?model ?model_lbl WHERE {
  ?metric a orkgc:Metric; rdfs:label ?metric_lbl. FILTER (str(?metric_lbl) = "Accuracy")
  { SELECT ?model ?model_lbl WHERE {
      ?dataset a orkgc:Dataset; rdfs:label ?dataset_lbl. FILTER (str(?dataset_lbl) = "Story Cloze Test")
      ?benchmark orkgp:HAS_DATASET ?dataset; orkgp:HAS_EVALUATION ?eval.
      ?eval orkgp:HAS_VALUE ?value; orkgp:HAS_METRIC ?metric.
      ?cont orkgp:HAS_BENCHMARK ?benchmark; orkgp:HAS_MODEL ?model.
      ?model rdfs:label ?model_lbl. }
    ORDER BY DESC(?value) LIMIT 1 } }
```

A typical baseline failure (wrong template, right entity) — what the LLM is expected to fix:

```
Q:    List the title and ID of research papers that contain a benchmark over the Penn Treebank (Word Level) dataset?
gold: SELECT DISTINCT ?paper ?paper_lbl WHERE { ... ?paper orkgp:P31 ?cont; rdfs:label ?paper_lbl. }
pred: SELECT DISTINCT ?model ?model_lbl WHERE { ... OPTIONAL {?paper orkgp:HAS_MODEL ?model ...} }
```
