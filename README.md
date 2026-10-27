# sciagent — an agentic research assistant for knowledge-graph & LLM research

A small agentic AI system that collects scientific papers, indexes their text, tables and figures, and answers
research questions with **Claude** (Anthropic API, tool use), choosing its own tools: literature retrieval, figure
inspection (vision), Python, a benchmark-leakage analysis, and NL→SPARQL querying of the Open Research Knowledge
Graph (ORKG). Every claim must cite evidence the agent actually retrieved; conflicting or missing evidence is
reported as such rather than smoothed over.

**Showcase topic:** the research area of Sahar Vahdati — knowledge-graph embeddings / link prediction and
knowledge graphs + LLMs for factual reliability — including a real published disagreement (RotatE vs.
*Loss-Aware Pattern Inference* on whether TransE can model symmetric relations).

**How correctness is measured:** a set of questions with **golden answers written in advance**. The system answers
each question, and an **LLM judge** compares its answer with the golden answer. See
[How we check that the answers are correct](#how-we-check-that-the-answers-are-correct). No model is trained or
fine-tuned anywhere in this project.

| Doc | What's in it |
|---|---|
| [How we check that the answers are correct](#how-we-check-that-the-answers-are-correct) | The evaluation: golden answers, LLM judge, worked example, how to run it |
| [ORKG and SciQA (optional)](#orkg-and-sciqa-optional) | A structured knowledge graph and an independent question bank, used as extras |
| [docs/DESIGN.md](docs/DESIGN.md) | Architecture, design choices, how each review criterion is addressed |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Test suite, eval harness, results (retrieval, SciQA, leakage analysis) |
| [docs/EXAMPLES.md](docs/EXAMPLES.md) | Example questions, tool traces and outputs |
| [docs/AI_USAGE.md](docs/AI_USAGE.md) | How a coding agent (Claude) was used to build this, and what was checked by hand |

## How we check that the answers are correct

### The idea in one picture

```
 28 questions ──► agentic system ──► system answer ──┐
                                                     ├──► LLM judge ──► verdict + score
 28 golden answers (written in advance) ─────────────┘        │
                                                              └──► what matched, what is missing, what contradicts
```

1. **Questions with golden answers.** `evals/questions.jsonl` holds 28 questions. Each has a **golden answer**, the
   answer a careful expert would give, written in advance from the paper texts.
2. **The system answers.** Each question goes to the agentic system exactly as a user's question would. It searches,
   reads and uses tools, then answers with citations.
3. **An LLM judge compares.** A separate Claude call sees the question, the golden answer and the system's answer, and
   gives a verdict from a fixed rubric.
4. **Scores are added up** into one number per category and overall, with every verdict explained.

### What the judge can decide

| Verdict | Score | When |
|---|---|---|
| correct | 1 | the answer contains all key points of the golden answer and contradicts none |
| partially correct | 0.5 | some key points are present and none is contradicted (e.g. only one side of a real disagreement) |
| incorrect | 0 | the answer contradicts the golden answer (e.g. a different number) or misses its main point |
| correct abstention | 1 | the answer is not in the papers, and the system says so |
| wrong abstention | 0 | the system says "not available" although the answer is in the papers |
| hallucinated | 0 | the answer is not in the papers, but the system states specific values anyway |

Each golden answer has a type, which tells the judge what "correct" means:

| Type | Questions | Correct means |
|---|---|---|
| `fact` | 23 | the answer contains the golden answer's key points (numbers may be rounded or written as percentages) |
| `abstain` | 4 | the answer says the information is not available and invents nothing |
| `open` | 1 | the answer meets listed criteria; used for the literature-search question, whose best value depends on the papers found |

### A worked example

*This is an illustration of the format, not a real run.*

**Question (q08):** Can TransE model symmetric relations?

**Golden answer (written in advance):** The papers disagree, so the answer should be "conflicting". RotatE's
comparison says TransE cannot model symmetric relations (h + r = t and t + r = h together force r = 0). The
loss-aware pattern inference paper argues that this proof assumes an exact zero residual that the training losses
never enforce; under margin-based losses TransE can encode symmetric relations when the margin is chosen
appropriately. The disagreement comes from the assumption about the loss.

**System answer A:** "No. TransE cannot model symmetric relations, because h + r = t and t + r = h force r = 0
[arxiv:1902.10197#t1]."

**Judge's output for A:**

```json
{"verdict": "partially_correct",
 "matched_points": ["RotatE's argument that h + r = t and t + r = h force r = 0"],
 "missing_points": ["the loss-aware paper's counter-argument", "that the papers disagree", "the role of the loss"],
 "contradictions": [],
 "explanation": "States one side of a documented disagreement as settled."}
```

The score is 0.5. A system answer that presents both sides and names the loss assumption would get **correct** (1.0).

### Can we trust the judge?

The judge is itself an LLM, so it is checked before its scores are used.

- **Calibration set.** `evals/judge_calibration.jsonl` holds 12 answers whose correct verdict was decided by hand. They
  cover every verdict and the usual traps:
  - paraphrases and rounding that must still count as correct;
  - a wrong number and a half answer;
  - a one-sided answer to a real disagreement;
  - an accepted false premise;
  - a wrong abstention and a correct abstention;
  - invented results.
- **Agreement check.** `python evals/check_judge.py` runs the judge on them and reports:
  - **exact agreement:** the same verdict as the label;
  - **pass/fail agreement:** the same side of pass/fail.

  If agreement is low, fix the judge before looking at system scores.
- **Bias controls:**
  - temperature 0 and a written rubric;
  - citation markers removed before judging;
  - every verdict lists matched, missing and contradicted points, so a person can audit it;
  - `--judge-model` lets a different model judge than the one answering, which reduces self-preference.

### How to run it and read the results

```bash
python evals/check_judge.py              # step 1: is the judge reliable?  -> evals/results/judge_check.json
python evals/run_eval.py --judge         # step 2: score the system       -> evals/results/<date-time>/report.md
python evals/run_eval.py --judge --judge-model claude-opus-5-5   # optional: a different model as judge
```

`report.md` starts with the **primary score**, then gives:
1. the verdict counts;
2. a per-category score (facts, figures, conflicts, unanswerable…);
3. a table with one row per question: verdict, missing points, contradictions and the judge's explanation;
4. the rule-based secondary checks.

The full answers and tool traces are in `agent_runs.jsonl`. Both commands need `ANTHROPIC_API_KEY`.

### Everything that is evaluated, and against what

| Evaluation | What it tells you | Compared against | Command | Status |
|---|---|---|---|---|
| **Golden answers + LLM judge (primary)** | Are the system's answers correct and honest? | 28 golden answers | `run_eval.py --judge` | ready; needs API key |
| Judge calibration | Can the judge be trusted? | 12 hand-labelled verdicts | `check_judge.py` | ready; needs API key |
| Retrieval | Does search find the passage that holds the answer? | gold passages for 21 questions | `make eval-retrieval` | **run**: recall@5 0.95 (BM25) |
| Rule-based checks (secondary) | Right status, required facts, no forbidden claims, no made-up citations | the same 28 questions | part of `run_eval.py` | ready; needs API key |
| Independent golden answers (optional) | The same, with answers written by other people | ORKG answers to SciQA questions | `run_sciqa_papers.py --judge` | ready; needs API key + internet |
| SciQA query writing (optional) | Can Claude write correct ORKG queries? | SciQA's correct queries | `run_sciqa.py` | baseline run; Claude run needs API key |
| Software tests | Does every part of the code behave as specified? | expected behaviour | `make test` | **run**: 141 pass |

### Limits of this evaluation

- **The golden answers were written by the same process that built the system**, from summaries of the papers. Check a
  few against the original PDFs. For answers written by others, use the optional SciQA-from-papers run.
- **28 questions show behaviour, not precise statistics.** A difference of one or two questions is noise.
- **Don't tune on the test.** If prompts are changed after seeing which questions fail, the score stops being a fair
  test. Then write new held-out questions, or split the set into a development half and a test half.
- **The judge can make mistakes.** That is why the calibration check exists, and why every verdict is explained.

## The seven review criteria: what was done and where to look

**Status in one line:** everything below is implemented and covered by 141 offline tests. Retrieval, the SciQA
baseline and the benchmark-leakage analysis were run on real data. The main evaluation (golden answers + LLM judge) is
built and tested but **was not run here**, because the build sandbox had no API key. `python evals/check_judge.py` and
`python evals/run_eval.py --judge` produce it.

### 1. How the agentic system is structured

- **Two agents, each with a clear job.**
  - `ResearchAgent` (`sciagent/agent.py`) answers the question and judges the evidence. It has 12 tools: paper search,
    passage reading, paper list and outline, figure reading, Python, the leakage analysis, ORKG querying, outside
    search, paper download, `research_topic`, and `submit_answer`.
  - `LiteratureAgent` (`sciagent/subagents.py`) is started through `research_topic` when a topic isn't in the corpus.
    It searches, screens, downloads and reads papers, then returns cited findings.
- **The model plans; the code enforces.** Claude picks the tools and their order; there is no fixed pipeline. Code
  enforces what a model shouldn't police itself:
  - a step budget, ending in a forced final answer;
  - a structured final answer (`submit_answer`);
  - cited passages must have been returned by a tool in this run;
  - no "insufficient evidence" before searching the corpus, then outside it;
  - the literature agent has a download cap and its findings must cite passages it read.
- **Tools fail soft and everything is traced.** Errors go back to the model as messages, and rate limits are
  retried. Every run records its tool calls, tokens and time, and the web page shows the steps live.
- **No agent framework.** The loop is written directly on the Anthropic SDK, so each rule is visible and testable
  with a scripted fake Claude. → [docs/DESIGN.md §1](docs/DESIGN.md), `tests/test_agent.py`, `tests/test_subagent.py`

### 2. How coding/AI agents were used while building it

The repository was built with Claude as a coding agent in a cloud sandbox, steered by short instructions. The topic,
scope (no training), the SciQA-style testing, the UI, the paper-upload flow and the literature agent were all changes
of direction given during the session.

The agent probed the environment first (which APIs were reachable), wrote tests before claiming results, and checked
the UI from screenshots. Tests and screenshots caught real bugs:
- whole-table PDF extraction scrambling decimals ("0.27" → "027 .");
- a test stand-in for the downloader being silently ignored;
- a CSS rule that showed a hidden bar.

Numbers that could not be computed are not reported. → [docs/AI_USAGE.md](docs/AI_USAGE.md)

### 3. Retrieval and RAG quality

- **Section-aware chunking.** Each chunk carries a `[paper title > section]` header.
- **Hybrid retrieval.** BM25 (exact tokens such as dataset names and numbers) and dense retrieval (paraphrases),
  combined with reciprocal rank fusion, with a cap of 3 chunks per paper so evidence from several papers surfaces.
- **Measured on 21 questions with gold evidence:**

  | | Recall@5 | MRR |
  |---|---|---|
  | BM25 | 0.95 | 0.81 |
  | Hybrid | 0.93 | 0.73 |

  On this small corpus, keyword search beats hybrid. That is reported rather than tuned away; the next step is a
  neural encoder.
- **Re-chunking-proof gold labels.** Gold evidence is stored as (paper, phrase), so it survives re-chunking, and a
  check verifies every gold item exists before each run.

→ [docs/EVALUATION.md §4.1](docs/EVALUATION.md), `sciagent/index.py`, `sciagent/ingest.py`

### 4. Multimodal handling

- **Tables** are extracted cell by cell, so numbers stay exact, and indexed as Markdown with their caption.
- **Figures** are saved as images and linked to their caption, and `inspect_figure` sends the image to Claude vision.
- **Known ground truth.** The leakage report is a generated PDF with a known chart and table, so figure and table
  answers can be checked.
- **Damaged or missing PDFs.** Corrupt, scanned (image-only) and missing PDFs fall back to the abstract and are
  flagged, never silently lost.
- **Web viewer.** The source panel shows the actual figure image or table behind a citation.

Limits: there is no OCR for scanned PDFs, and real-paper figures only arrive after `make collect`.
→ `sciagent/ingest.py`, `tests/test_ingest_index.py`, `tests/test_tools.py`

### 5. Machine-learning fundamentals (no training, by design)

- **Train/test leakage, measured on real benchmarks.** A rule-only model with no learning recovers 93.2% of WN18 test
  triples and reaches filtered MRR **0.931**. That falls to 0.336 on WN18RR and 0.0007 on FB15k-237. Ranking uses the
  filtered setting with random tie-breaking, and the expected-rank formula is checked against brute force.
- **SciQA query writing (optional experiment), done properly:**
  - **Clean split:** the train split is only a pool of few-shot examples, and its vectoriser never sees test data.
  - **Leakage audit:** it found **7 test items duplicated in train** in the official split; they are excluded.
  - **No-LLM baseline:** it gets 74.9% exact match on template questions but **0%** on hand-written ones, so results
    are always reported per subset.
  - **Confidence intervals:** bootstrap 95% intervals on the headline metric.
- **Comparability.** The literature agent records dataset version, task variant, metric averaging and training
  regime, so incomparable scores are not ranked against each other.

→ `sciagent/kg/analysis.py`, `sciagent/sciqa.py`, [docs/EVALUATION.md §4.2–4.3](docs/EVALUATION.md)

### 6. Evaluation and testing

- **Primary:** 28 questions with golden answers, judged by an LLM against a six-verdict rubric. The judge is checked on
  12 hand-labelled answers. Full explanation, worked example and limits:
  [How we check that the answers are correct](#how-we-check-that-the-answers-are-correct).
- **Secondary, without an LLM:**
  - right status (supported / conflicting / insufficient);
  - required facts present;
  - forbidden claims absent;
  - no made-up citations;
  - citation precision and recall;
  - abstention and conflict-detection rates.
- **Retrieval, measured:** search finds the passage that holds the answer in its top 5 results for 95% of questions.
- **Software tests (141 offline + 4 live).**
  - **The agent** is tested with a scripted fake Claude, so every rule in criterion 1 has a test.
  - **The rest of the pipeline:**
    - ingestion edge cases;
    - retrieval;
    - every tool;
    - all metrics and the judge;
    - the collection pipeline against simulated arXiv/OpenAlex;
    - the literature agent;
    - the web endpoints.
  - **The evaluation harness itself** is run end to end with a naive stand-in model, which it correctly scores 0%.

→ [docs/EVALUATION.md](docs/EVALUATION.md)

### 7. Robustness to missing or conflicting evidence

- **Disagreement.** Every answer carries a status: supported, partially supported, conflicting or insufficient.
  Disagreements are listed side by side with citations and the reason for the disagreement. Example: RotatE vs
  *Loss-Aware Pattern Inference* on TransE and symmetric relations.
- **Missing evidence.** The agent must search the corpus, then search outside it and fetch a matching paper, before
  it may say "insufficient", and then it lists the closest candidates instead of answering from memory.
- **Made-up evidence.** Fabricated citations are rejected, then stripped, and confidence is capped. False premises are
  corrected (eval q18).
- **Incomparable numbers.** These are flagged (eval q24, q28), and "highest" is stated as "among the papers found".
- **Degraded inputs.** Failed downloads, sources being offline, and damaged PDFs all degrade gracefully with
  warnings.

→ [docs/DESIGN.md §6](docs/DESIGN.md), `tests/test_agent.py`, `tests/test_outside_corpus.py`

## ORKG and SciQA (optional)

**You don't need this part to understand or evaluate the system.** The main evaluation is golden answers + LLM judge
([above](#how-we-check-that-the-answers-are-correct)). ORKG and SciQA are extras with two uses:

1. **A structured evidence source.** ORKG is available to the agent through the `query_scholarly_kg` tool.
2. **Independent golden answers.** SciQA provides questions and answers written by other people.

The two are easy to mix up, because they play different roles.

| | ORKG | SciQA |
|---|---|---|
| What it is | The **Open Research Knowledge Graph**: a database of research results stored as structured facts (paper → contribution → benchmark → dataset → metric → score → model) | A **question bank built on top of ORKG**: 2,565 questions (1,795 train / 257 valid / 513 test) |
| What it contains | Facts. No questions. | For each question: the English question, the **correct SPARQL query**, and the **correct answer** (the result of that query on ORKG in 2023) |
| Role here | The **source of answers**: queries are run against it | The **ground truth**: what generated queries and answers are scored against |
| Analogy | The library catalogue | An exam about the catalogue, with its answer key |

**Use 1 — independent golden answers for the main system (`run_sciqa_papers.py --judge`)**

1. Take SciQA test questions that papers can answer:
   - the best model on a dataset;
   - the top score and its metric;
   - the metrics used on a dataset;
   - the models evaluated on a dataset.

   Questions about ORKG's own records are excluded.
2. The system answers them **from the papers**, with the ORKG tool **switched off**, so it cannot look up the answer
   key.
3. The LLM judge compares each answer with ORKG's answer, and fuzzy matching gives a second, rule-based score.
4. Disagreements are written to `review.md` for a person to decide. ORKG can be incomplete or outdated, so a
   disagreement is not automatically the system's fault.

**Use 2 — the query-writing experiment (`run_sciqa.py`), step by step**

1. A SciQA test question is taken, e.g. *"Which model has the highest Accuracy on the Story Cloze Test?"*
2. The 3 most similar SciQA **train** questions are retrieved by text similarity, with their correct queries.
   Nothing is trained: they are only shown to Claude as examples.
3. Claude writes a SPARQL query for the test question.
4. **Query-level score:** the query is compared with SciQA's correct query (BLEU, ROUGE, exact match). This needs no
   ORKG access.
5. **Answer-level score (optional, `--execute`):** the query is run on ORKG, and its results are compared with
   SciQA's correct answer (precision / recall / F1). This is the only step that contacts ORKG.

In the agent the same mechanism is the `query_scholarly_kg` tool. Claude uses it for structured questions (best score
on a benchmark, models evaluated on a dataset), and the ORKG result is returned as citable evidence.

**Where queries are sent: `SCIQA_ENDPOINT`**

`SCIQA_ENDPOINT` is the web address of the ORKG SPARQL endpoint, used only in steps that *run* a query (step 5 and
the agent tool). It defaults to `https://orkg.org/triplestore`. Point it elsewhere to use a stable snapshot or a local
copy; the 2023 challenge used `https://ltdemos.informatik.uni-hamburg.de/orkg/sparql`. Neither address could be
verified from the build sandbox, so test the endpoint first:

```bash
python -c "from sciagent.sciqa import execute; print(execute('SELECT ?p WHERE { ?p a orkgc:Paper } LIMIT 3'))"
# a list of values = endpoint works;  None = unreachable or not a SPARQL endpoint
```

**Caveats**

- **Snapshot drift.** SciQA's correct answers were taken from ORKG in 2023, and ORKG has changed since. Answer-level
  scores therefore partly measure that drift, so query-level scores are the more stable comparison.
- **Coverage.** ORKG only knows what people entered. An empty result can mean a wrong query *or* missing data, and the
  agent says so. ORKG is ground truth for "what ORKG records", not for "what the literature says".
- **Template questions are easy.** A no-LLM baseline that copies the nearest train query already gets 74.9% exact
  match on template-generated questions, and 0% on the 21 hand-written ones. The hand-written questions are the real
  test.
- **New ground truth.** You can make more, the way SciQA's template questions were made: fill question patterns with
  datasets and metrics from ORKG and record ORKG's answers. That is useful, e.g., as reference values for the
  literature agent's "best F1 on dataset X" answers when ORKG covers that dataset.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...          # only needed for the agent / LLM evals
export SCIAGENT_MODEL=claude-sonnet-5-5      # optional; any Claude model with tool use + vision

make build        # download WN18/WN18RR/FB15k-237, run leakage analysis -> report PDF, ingest + index corpus
make test         # 141 offline tests (fake Claude client), 4 live tests skipped without an API key

python -m sciagent serve                     # web interface on http://localhost:8000
python -m sciagent ask "Can TransE model symmetric relations?" -v
python -m sciagent ask "In Figure 1 of the leakage report, which dataset has the lowest bars?" --json
python -m sciagent search "ConvKB CapsE random protocol"        # retrieval only, no LLM
```

### Web interface

`python -m sciagent serve` (or `make serve`) opens a local page at http://localhost:8000. Type a question and watch the
agent's tool calls appear on the right as it works; the answer shows a verdict (supported / the papers disagree /
not enough evidence), numbered citations, and a "where the papers disagree" block. Clicking a citation opens the exact
passage, table or figure image behind it. Without an API key the page runs in search-only mode and shows the
retrieved passages. Standard library only, no extra install.

![Answer view](docs/img/web_answer.png)

The screenshot uses a scripted demo run (no API key was available while building); with a key the same page shows
Claude's real answers. Fonts load from Google Fonts when online.

### Ask about a research topic the corpus does not cover

```bash
python -m sciagent ask "What is the highest F1 score for requirements classification with LLMs on the PROMISE dataset?" -v
python -m sciagent review "requirements classification with LLMs on PROMISE, F1" --max-papers 5   # literature agent only
```

The main agent starts a literature-review agent that searches arXiv and OpenAlex, screens the results, downloads the
best papers, reads their result tables and returns cited findings; the main agent then compares them, flagging numbers
that come from different dataset versions, task variants or training regimes. In the web page the review agent's steps
appear indented under "Started the literature-review agent".

### Ask about a specific paper (including one not in the corpus)

```bash
python -m sciagent add path/to/paper.pdf          # or an arXiv id: 2508.16983, or a PDF link
python -m sciagent ask "What are the main results?" --paper <paper_id printed by add>
```

In the web page: drop a PDF on "Add a paper" (or paste an arXiv id / PDF link), or click any paper in the list. Questions
are then answered only from that paper, with paper-specific starter questions; "Use all papers" switches back.

### Collect a fresh corpus automatically (needs internet)

```bash
make collect      # = python -m sciagent collect --author "Sahar Vahdati" && python -m sciagent build
python -m sciagent collect --author "" --topic "graph neural networks drug discovery" --max-papers 50   # topic only
python -m sciagent build
```

Author works + arXiv/OpenAlex topic search + one-hop reference expansion, cross-source dedupe, relevance filtering
(topic phrases + TF-IDF similarity, no training), open-access PDF download with abstract fallback, incremental merge,
and a `data/corpus/collect_report.json` audit trail. The agent can also pull in a missing paper by itself mid-question
(`fetch_paper`).

The repository ships a small **seed corpus** (`data/seed_corpus/cards.jsonl`: structured summaries of 10 papers,
captured while building, because the build sandbox could not reach arXiv). `make collect` replaces those cards
with full-text PDFs automatically (the ingester keeps the richest record per paper id).

### Evaluate

```bash
# main evaluation (needs ANTHROPIC_API_KEY)
python evals/check_judge.py              # 1. check the LLM judge against 12 hand-labelled answers
python evals/run_eval.py --judge         # 2. 28 questions: LLM judge vs golden answers (primary) + rule-based checks
python evals/run_eval.py --judge --ids q08 q09   # a subset

# no API key needed
make eval-retrieval                      # retrieval metrics (BM25 / dense / hybrid)
make test                                # 141 software tests

# optional extras
python evals/run_sciqa_papers.py --dry-run           # show SciQA questions + ORKG answers that would be used
python evals/run_sciqa_papers.py --limit 30 --judge  # answered from papers, judged against ORKG's answers
make sciqa-baseline                                  # SciQA query writing: non-LLM baseline (offline)
python evals/run_sciqa.py --generator claude --shots 0 1 3 5     # SciQA query writing with Claude
pytest -m live                                       # end-to-end checks against the real API
```

## Repository layout

```
sciagent/
  collect.py      arXiv + OpenAlex collection, expansion, relevance filter, dedupe, PDF download
  ingest.py       PDF -> section-aware chunks, tables (Markdown), figures (PNG + caption); degraded-input handling
  index.py        hybrid retrieval: BM25 + dense (LSA or sentence-transformers) + reciprocal rank fusion
  tools.py        agent tools (schemas + implementations) and the submit_answer contract
  agent.py        main Claude tool-use agent with citation / abstention guardrails and tracing
  subagents.py    literature-review agent (search, screen, download, read, extract findings), started via research_topic
  sciqa.py        SciQA NL->SPARQL: example retrieval, kNN baseline, Claude generator, metrics, ORKG execution
  judge.py        LLM judge: system answer vs golden answer, six-verdict rubric
  evaluation.py   retrieval metrics, rule-based answer checks, groundedness judge
  sciqa_papers.py SciQA questions answered from papers, compared with ORKG (optional)
  kg/             benchmark download, leakage analysis + rule baseline, PDF report generation
  web/            local web interface (stdlib HTTP server + one HTML page, live tool trace via Server-Sent Events)
  cli.py          python -m sciagent {collect,analyze,build,search,ask,review,add,serve}
evals/
  questions.jsonl         28 questions with golden answers (reference_answer, answer_type) and gold passages
  run_eval.py             main evaluation: agent answers -> LLM judge vs golden answers + rule-based checks
  judge_calibration.jsonl 12 hand-labelled answers;  check_judge.py measures the judge against them
  run_sciqa_papers.py     optional: SciQA questions answered from papers, judged against ORKG
  run_sciqa.py            optional: SciQA query-writing experiment
  results/                outputs
tests/            pytest suite (fake Claude client; live tests marked `live`)
data/seed_corpus/ seed paper cards
scripts/          run_examples.py
docs/             design, evaluation, examples, AI usage
```

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Claude API |
| `SCIAGENT_MODEL` | `claude-sonnet-5-5` | model for the agents, vision and SPARQL generation |
| `SCIAGENT_JUDGE_MODEL` | same as `SCIAGENT_MODEL` | model for the LLM judge; a different model reduces self-preference |
| `SCIAGENT_DATA` | `./data` | data directory |
| `SCIAGENT_EMBED_MODEL` | unset (LSA) | e.g. `sentence-transformers/all-MiniLM-L6-v2` for neural dense retrieval |
| `SCIQA_ENDPOINT` | `https://orkg.org/triplestore` | ORKG SPARQL endpoint where generated queries are run (agent ORKG tool and `run_sciqa.py --execute`); see [ORKG and SciQA](#orkg-and-sciqa-optional) |

## Known limitations

- The main evaluation (golden answers + LLM judge) was **not run in the build sandbox**, which had no API key and no
  access to arXiv or ORKG. It is built, unit-tested against a scripted fake client, and run end to end with a stand-in
  model. Numbers in docs/EVALUATION.md are only those actually computed.
- The golden answers come from summaries of the papers; check them against the original PDFs.
- The seed corpus is summaries, not full text. Multimodal handling is exercised on the generated report PDF and test
  PDFs; with `make collect` it runs on real papers.
- `run_python` runs code in a subprocess, not a security sandbox — fine for a local prototype only.
- ORKG content changes over time, so execution-based SciQA scores can drift from the 2023 gold answers.
