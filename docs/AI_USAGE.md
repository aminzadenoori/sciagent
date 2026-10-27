# How AI coding agents were used

This repository was built in one session with Claude (Opus 5.5) acting as a coding agent in a cloud Linux
workspace, steered by me through short instructions. This note records what the agent did, where I redirected it, and
what was verified rather than trusted.

## Workflow

1. **Environment probe first.** Before writing code the agent checked what the sandbox could reach: arXiv, OpenAlex,
   Semantic Scholar, Hugging Face, Zenodo and ORKG were blocked; PyPI and raw GitHub were open; there was no Anthropic
   API key. That shaped the design: real collectors for my machine, offline-testable everything else, and a scripted
   fake Claude client for deterministic tests.
2. **Topic choice, redirected twice.** The agent first proposed the LK-99 superconductivity debate (good
   conflicting-evidence case). I redirected it to my interviewer's research area (Sahar Vahdati: KG embeddings,
   KGs + LLMs). It then found a genuine published disagreement in that area (RotatE vs *Loss-Aware Pattern
   Inference*) to use as the conflict test case, and real benchmark data (WN18/WN18RR/FB15k-237) on GitHub.
3. **Scope corrections.** The agent started training a NumPy KGE model; I stopped it — no training was wanted. It
   removed that code and kept the training-free leakage analysis. I then asked for testing to follow the NL→SPARQL
   SciQA line of work without training; it located the official SciQA split in the Scholarly-QALD-23 repository and
   built the few-shot-retrieval generator, metrics and a non-LLM baseline.
4. **Product direction from the user, implementation by the agent.** Later instructions added, in order: a web
   interface; asking about a paper that is not in the corpus (upload / arXiv id, then automatic outside search and
   download); stronger automated collection (topic-only mode, relevance ranking, incremental runs, audit report);
   and a separate literature-review agent the main agent can start for topic questions such as "highest F1 for
   requirements classification with LLMs on PROMISE". Finally the user questioned whether SciQA was the right way to
   assess correctness and asked for golden answers judged by an LLM instead; that became the primary measure, with
   SciQA kept as an optional source of independent golden answers. Each change came with tests and was checked in a browser via
   screenshots before delivery.
5. **Seed corpus.** Because PDFs could not be downloaded in the sandbox, the agent read paper pages through its web
   tool and wrote short structured summaries ("cards") in its own words. These are labelled as such, used only as a
   seed, and replaced by full text when `make collect` runs. Paper metadata whose identifier it could not confirm was
   given a neutral id (`pakdd2021:loss-aware-pattern-inference`) rather than a guessed arXiv number.

## What was verified, not trusted

- **Tests before claims.** 141 offline tests now. Real bugs they or screenshot checks surfaced: table decimals
  scrambled by whole-table extraction ("0.27" → "027 ."); an unrealistic test fixture for rule mining; the PDF
  downloader bound at import time so a test stand-in was silently ignored; a CSS rule that displayed a hidden
  "answering only from" bar; "1 papers"-style labels.
- **Metrics checked against brute force** (random tie-breaking expectation) and against known results: the rule-only
  WN18 MRR of 0.931 and WN18RR 0.336 agree with the published motivation for WN18RR.
- **Data hygiene found a real issue:** the SciQA leakage audit found 7 test items that duplicate a train question or
  query in the official split; they are excluded by default and reported.
- **Honest reporting.** No LLM-based numbers are reported because none could be run in the sandbox; the docs say so
  and give the commands. Retrieval results that contradict the default design (BM25 > hybrid on this corpus) are
  reported as measured rather than tuned away.

## What I would review most carefully as the human owner

- The seed cards' numbers against the original papers (they were written from web-tool readings).
- Prompt wording in `agent.py` and `sciqa.py` after the first live eval run.
- `run_python` isolation before using the system with untrusted inputs.
