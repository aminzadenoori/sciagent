# Evaluation and testing

Three layers, from cheapest to most expensive:

1. **Unit / integration tests** (`pytest`, 111 offline + 4 live) — deterministic, no API key.
2. **Retrieval and data-analysis evaluation** — offline, real numbers below.
3. **Agent and LLM evaluation** — needs `ANTHROPIC_API_KEY`; harness, metrics and question set are ready
   (`make eval`, `make sciqa-claude`). **These were not run in the build sandbox (no API key, no ORKG access), so no
   agent/LLM scores are reported here.** Run them and the results land in `evals/results/<timestamp>/report.md`.

## 1. Test suite design

The agent is tested with a **scripted fake Claude client** (`tests/conftest.py`), so the harness behaviour is checked
deterministically; the model's judgement is what the eval harness measures. Tests build a tiny corpus on the fly that
contains the awkward cases on purpose: a PDF with a table (decimals) and a figure, a ligature, a corrupt PDF, an
image-only "scanned" PDF, a missing PDF, and a duplicate record (card vs PDF).

| File | What it checks |
|---|---|
| `test_ingest_index.py` (18) | section detection, contextual chunk headers, chunk overlap loses nothing, ligatures, **table cells keep exact decimals**, table text not duplicated into body, **figure + caption + PNG extracted**, corrupt / scanned / missing PDF fallbacks, dedupe prefers full text, BM25 exact-token hits, dense paraphrase hit, kind/paper filters, per-paper cap, empty query, index round-trip |
| `test_agent.py` (13) | search→cited answer; tool results fed back with matching ids; prompt caching requested; **model-chosen tool sequences** incl. vision call; fabricated citation **rejected then fixed**; persistent fabrication **stripped + confidence capped**; citations don't leak across runs; **abstaining without searching rejected**; supported-without-citation rejected; conflicting answers with citations inside `conflicts`; parallel tool calls; **step budget forces `submit_answer`**; plain-text answer nudged; tool errors returned not raised; API failure captured; rate limits retried |
| `test_tools.py` (11) | schemas valid; seen-chunk registration; empty results say so; missing ids; author filter; outline lists figures/tables; **vision call receives the PNG**; non-figure rejected; python stdout / timeout / exception; offline tools fail gracefully |
| `test_analysis.py` (13) | reciprocal augmentation; inverse rules mined; rules *not* mined with too many exceptions; leakage 100% / 0% on constructed splits; rule model perfect on leaky split, chance without; **random tie-breaking expectation = brute force** (6 cases) |
| `test_sciqa.py` (14) | BLEU identity / disjoint / brevity penalty; ROUGE identity; exact match invariant to whitespace and variable names but not literals; literal F1; answer-set P/R/F1; number normalisation; fence/prefix cleaning; example retrieval returns same template; **kNN baseline substitutes the aligned entity**; HQ/AQ split reporting; **Claude prompt contains k retrieved train examples, temperature 0**; zero-shot has no examples; official split sizes and disjoint ids |
| `test_collect_eval.py` (10) | arXiv Atom parsing; OpenAlex abstract reconstruction + arXiv id extraction; dedupe merges sources; relevance filter; answer scoring pass / wrong status / trap / fabricated citation; aggregate robustness metrics; retrieval metrics; **gold-coverage check catches bad gold labels**; eval-set schema and category coverage |
| `test_web.py` (4) | page, status, search, source and figure endpoints; no-key message; **live event stream** start → tool_call → tool_result → final with resolved sources |
| `test_library.py` (13) | arXiv id parsing (6 forms); PDF title extraction; **adding a local PDF indexes its text, table and figure**; re-adding replaces instead of duplicating; arXiv add uses API metadata; non-PDF and missing file rejected; **focus restricts search to one paper**; focused agent prompt; unknown paper; web upload + focused search; non-PDF upload rejected |
| `test_collect_pipeline.py` (6) | **whole collection pipeline on simulated arXiv/OpenAlex**: author seed + topic search + reference expansion, cross-source merge, off-topic papers dropped from every source, author papers kept even off-vocabulary, failed PDF → abstract-only, report counts, collected papers ingested with figures; topic-only mode; incremental re-runs add nothing twice; all sources down → empty result + warnings, no crash; **agent `fetch_paper` adds a missing paper mid-run** |
| `test_outside_corpus.py` (4) | external search merges arXiv + OpenAlex and ranks the exact title first; **agent that gives up on a paper missing from the corpus is sent back, searches outside, fetches the paper, searches it and answers with valid citations**; abstaining after an outside search is accepted; no outside search required offline or when the user pinned one paper |
| `test_subagent.py` (5) | `research_topic` offered to the main agent; **main agent starts the literature agent, which searches outside, downloads papers within its cap, reads their result tables and returns findings the main agent cites**; findings citing unread chunks are dropped; download budget enforced; literature agent must search outside before submitting; offline / no-LLM errors |
| `test_live.py` (4, `-m live`) | real API: cited supported answer, conflict surfaced, abstention, SciQA 3-shot exact match |

Result in the build sandbox: **111 passed, 4 skipped (live)** in ~16 s.

Two real bugs were caught by tests while building: whole-table extraction scrambling decimals, and a test fixture
whose leakage rate (33% held out) was unrealistic so the 0.9-confidence rule miner correctly refused to mine the rule —
the fix was to the fixture (5% held out, like WN18), and a negative test was added.

## 2. Agent evaluation set (`evals/questions.jsonl`, 28 questions)

Designed to cover each robustness behaviour, not just accuracy:

| Category | ids | Expected behaviour |
|---|---|---|
| factual / numeric | q01 q02 q04–q07 q14 q19 q23 q25 | exact numbers with citations |
| mixed evidence | q03 | ReFactX helps on 2WikiMH but *hurts* on Mintaka/WebQSP |
| conflict | q08 q09 q20 | RotatE vs loss-aware paper; reported vs RANDOM-protocol MRR; LLM vs human judges |
| cross-paper | q24 | WN18RR numbers from different papers are not directly comparable |
| table / figure | q10 / q11 | values from the report's table; reading the bar chart (vision) |
| analysis | q12 | compute WN18RR leakage with the tool |
| multi-hop | q13 | ConvE motivation + computed numbers |
| unanswerable | q15–q17 | `insufficient`, no invented numbers |
| false premise | q18 | "AI beats humans on ARC-AGI-2" is false (68.8% vs near-perfect humans) |
| evidence quality | q22 | perspective essay with informal evidence |
| corpus meta | q21 | which papers are Vahdati's (list_papers) |
| tool failure | q26 | YAGO3-10 not supported by the analysis tool |
| structured KG | q27 | ORKG via NL→SPARQL |
| literature review | q28 | best F1 for LLM requirements classification on PROMISE: starts the literature agent, compares only comparable settings |

Gold evidence is `(paper_id, phrase)`, so it survives re-chunking; `gold_coverage()` verifies every gold item exists in
the corpus before each run. **Answer metrics:** status accuracy, required-fact recall (alternatives allowed), trap
phrases (hallucination markers), citation precision/recall against gold papers, invalid-citation count, expected-tool
use, tool calls, tokens, latency; a strict `pass` = right status ∧ all facts ∧ no trap ∧ no fabricated citation.
Aggregates include **abstention accuracy** and **conflict detection** rates; `--judge` adds LLM-as-judge groundedness
of each answer against the text of the chunks it cites.

## 3. Results that were computed

### 3.1 Retrieval (21 questions with gold evidence, 11-paper corpus, 45 chunks)

| retriever | recall@1 | recall@3 | recall@5 | all-gold@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| BM25 | **0.667** | **0.881** | **0.952** | **0.905** | **0.813** | **0.940** |
| dense (TF-IDF+SVD) | 0.452 | 0.762 | 0.929 | 0.857 | 0.672 | 0.833 |
| hybrid (RRF) | 0.548 | 0.810 | 0.929 | 0.857 | 0.734 | 0.886 |

Reading: on a tiny corpus of exact scientific terms, lexical matching wins and the LSA encoder drags fusion down at
rank 1; by k=5 all three find ≥93% of gold evidence, and the agent typically reads 6 results per query. Misses at
rank 1 are mostly numeric questions where several chunks of the same paper compete (q05, q10, q20, q24). The conflict
question q08 ("Can TransE model symmetric relations?") retrieves the loss-aware paper first but not RotatE's
relation-pattern chunk in the top 4 — the agent has to issue a second query, which is exactly why it is agentic
rather than single-shot. Next step: neural encoder (`SCIAGENT_EMBED_MODEL`) and re-measure before changing the default.

### 3.2 SciQA NL→SPARQL — non-LLM baseline (no training)

Official split; 7 test / 3 valid items that duplicate a train question or query were **excluded** (leakage audit).

| split | subset | n | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | exact match | literal F1 |
|---|---|---|---|---|---|---|---|---|
| test | all | 506 | 89.06 | 91.86 | 88.67 | 91.38 | 71.74 (95% CI 67.4–75.9) | 91.83 |
| test | template-generated | 485 | 91.15 | 93.97 | 91.64 | 93.63 | 74.85 | 95.60 |
| test | **handcrafted** | 21 | 14.23 | 43.07 | 19.98 | 39.47 | **0.00** | 4.76 |
| valid | all | 254 | 87.12 | 92.03 | 89.15 | 91.58 | 71.65 (95% CI 66.5–76.8) | 90.35 |
| valid | handcrafted | 13 | 26.90 | 51.07 | 29.35 | 46.33 | 0.00 | 5.13 |

Few-shot example retrieval quality (share of test questions with a same-template train question in the top-k):
**78.4% @1, 94.2% @3, 99.2% @5**. With overlap kept (513 items) exact match is 71.65%, i.e. the leaked items don't
move the aggregate much here, but they would for a system that copies retrieved queries.

Takeaways that shape the LLM experiments:
- Copying the nearest train query and swapping entity strings already gets ~75% exact match on template questions,
  so high SciQA scores mostly measure example retrieval; **handcrafted questions are the real test** and must be
  reported separately.
- The baseline's errors are template confusions where TF-IDF matches the entity rather than the intent (e.g. "list
  papers with a benchmark on X" retrieves a "list models evaluated on X" example). That is what the LLM must fix.
- 3 shots cover the right template 94% of the time; 5 shots add little coverage but more distractors — the
  0/1/3/5-shot sweep (`make sciqa-claude`) tests this directly.

### 3.3 Benchmark leakage analysis (real data, no training)

| dataset | train | test | test triples recoverable by a mined rule | rule-only filtered MRR | Hits@10 |
|---|---|---|---|---|---|
| WN18 | 141,442 | 5,000 | **93.16%** | **0.931** | 0.932 |
| WN18RR | 86,835 | 3,134 | 33.57% | 0.336 | 0.336 |
| FB15k-237 | 272,115 | 20,466 | 0.00% | 0.0007 | 0.0007 |

A rule-only model with no embeddings reaches 0.93 MRR on WN18, matching the argument of Dettmers et al. that WN18 mostly
measures inverse-relation lookup; WN18RR's remaining 34% come from symmetric relations (derivationally related form,
verb group, similar to).

## 4. How to run the LLM evaluations

```bash
export ANTHROPIC_API_KEY=...
make eval                                                     # 28 agent questions + judge
python evals/run_sciqa.py --generator claude --shots 0 1 3 5  # SciQA query metrics, no training
python evals/run_sciqa.py --generator claude --shots 3 --execute --limit 120   # + answer F1 on ORKG
pytest -m live
```
