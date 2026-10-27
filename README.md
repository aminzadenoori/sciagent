# sciagent — an agentic research assistant for knowledge-graph & LLM research

A small agentic AI system that collects scientific papers, indexes their text, tables and figures, and answers
research questions with **Claude** (Anthropic API, tool use), choosing its own tools: literature retrieval, figure
inspection (vision), Python, a benchmark-leakage analysis, and NL→SPARQL querying of the Open Research Knowledge
Graph (ORKG). Every claim must cite evidence the agent actually retrieved; conflicting or missing evidence is
reported as such rather than smoothed over.

**Showcase topic:** the research area of Sahar Vahdati — knowledge-graph embeddings / link prediction and
knowledge graphs + LLMs for factual reliability — including a real published disagreement (RotatE vs.
*Loss-Aware Pattern Inference* on whether TransE can model symmetric relations).

**Benchmark testing** follows LLM-based NL→SPARQL scientific QA on **SciQA** (official train/valid/test split),
**without any training or fine-tuning**: the train split is only a pool of retrieved few-shot examples.

| Doc | What's in it |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | Architecture, design choices, how each review criterion is addressed |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Test suite, eval harness, results (retrieval, SciQA, leakage analysis) |
| [docs/EXAMPLES.md](docs/EXAMPLES.md) | Example questions, tool traces and outputs |
| [docs/AI_USAGE.md](docs/AI_USAGE.md) | How a coding agent (Claude) was used to build this, and what was checked by hand |

## The seven review criteria: what was done and where to look

**Status in one line:** everything below is implemented and covered by 111 offline tests. Retrieval, the SciQA
baseline and the benchmark-leakage analysis were run on real data. The LLM-based evaluations (agent answers, Claude
SciQA) are built and wired up but **were not run here** (no API key in the build sandbox); `make eval` and
`make sciqa-claude` produce them.

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

→ [docs/EVALUATION.md §3.1](docs/EVALUATION.md), `sciagent/index.py`, `sciagent/ingest.py`

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
- **SciQA (NL→SPARQL), done properly:**
  - **Clean split:** the train split is only a pool of few-shot examples, and its vectoriser never sees test data.
  - **Leakage audit:** it found **7 test items duplicated in train** in the official split; they are excluded.
  - **No-LLM baseline:** it gets 74.9% exact match on template questions but **0%** on hand-written ones, so results
    are always reported per subset.
  - **Confidence intervals:** bootstrap 95% intervals on the headline metric.
- **Comparability.** The literature agent records dataset version, task variant, metric averaging and training
  regime, so incomparable scores are not ranked against each other.

→ `sciagent/kg/analysis.py`, `sciagent/sciqa.py`, [docs/EVALUATION.md §3.2–3.3](docs/EVALUATION.md)

### 6. Evaluation and testing

- **Offline tests (111):** the agent is tested with a scripted fake Claude, so every rule above has a test. Also
  covered:
  - ingestion edge cases;
  - retrieval;
  - each tool;
  - the metrics;
  - the full collection pipeline against simulated arXiv/OpenAlex;
  - the literature agent;
  - the web endpoints.
- **Live tests (4):** these call the real API (`pytest -m live`).
- **Agent evaluation set (28 questions)**, covering every kind of difficulty:
  - factual and numeric questions;
  - table, figure and analysis questions;
  - multi-hop questions;
  - real conflicts;
  - unanswerable questions and a false premise;
  - a tool failure, a structured KG query and a literature review.

  Metrics: status accuracy, required-fact recall, trap phrases, citation precision and recall, fabricated citations,
  abstention and conflict-detection rates, and optional LLM-as-judge groundedness.
- **SciQA runner:** 0/1/3/5-shot, scored by query metrics and by executing answers on ORKG.

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

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...          # only needed for the agent / LLM evals
export SCIAGENT_MODEL=claude-sonnet-5-5      # optional; any Claude model with tool use + vision

make build        # download WN18/WN18RR/FB15k-237, run leakage analysis -> report PDF, ingest + index corpus
make test         # 111 offline tests (fake Claude client), 4 live tests skipped without an API key

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
make eval-retrieval                      # retrieval metrics, no API key (BM25 / dense / hybrid)
make eval                                # full agent eval on 28 questions + LLM-as-judge groundedness
python evals/run_eval.py --ids q08 q09   # a subset

make sciqa-baseline                      # SciQA test, non-LLM nearest-neighbour baseline (offline)
python evals/run_sciqa.py --generator claude --shots 0 1 3 5            # query-level metrics
python evals/run_sciqa.py --generator claude --shots 3 --execute         # + answer P/R/F1 on ORKG endpoint
pytest -m live                           # end-to-end checks against the real API
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
  evaluation.py   retrieval + answer metrics, LLM-as-judge
  kg/             benchmark download, leakage analysis + rule baseline, PDF report generation
  web/            local web interface (stdlib HTTP server + one HTML page, live tool trace via Server-Sent Events)
  cli.py          python -m sciagent {collect,analyze,build,search,ask,serve}
evals/            questions.jsonl (28 designed questions), run_eval.py, run_sciqa.py, results/
tests/            pytest suite (fake Claude client; live tests marked `live`)
data/seed_corpus/ seed paper cards
scripts/          run_examples.py
docs/             design, evaluation, examples, AI usage
```

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Claude API |
| `SCIAGENT_MODEL` | `claude-sonnet-5-5` | model for agent, vision, SPARQL generation, judge |
| `SCIAGENT_DATA` | `./data` | data directory |
| `SCIAGENT_EMBED_MODEL` | unset (LSA) | e.g. `sentence-transformers/all-MiniLM-L6-v2` for neural dense retrieval |
| `SCIQA_ENDPOINT` | `https://orkg.org/triplestore` | ORKG SPARQL endpoint for execution-based metrics |

## Known limitations

- The agent and LLM evals were **not run in the build sandbox** (no API key there, no access to arXiv/ORKG); they are
  wired, unit-tested against a scripted fake client, and covered by live tests you can run (`pytest -m live`).
  Numbers in docs/EVALUATION.md are only those actually computed.
- The seed corpus is summaries, not full text. Multimodal handling is exercised on the generated report PDF and test
  PDFs; with `make collect` it runs on real papers.
- `run_python` runs code in a subprocess, not a security sandbox — fine for a local prototype only.
- ORKG content changes over time, so execution-based SciQA scores can drift from the 2023 gold answers.
