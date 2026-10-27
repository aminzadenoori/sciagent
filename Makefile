PY ?= python

.PHONY: install data build serve test eval-retrieval judge-check eval sciqa-baseline sciqa-claude examples all

install:
	$(PY) -m pip install -r requirements.txt

# offline-capable pipeline: benchmarks -> leakage report PDF -> ingest + index (seed cards + report)
build:
	$(PY) -m sciagent analyze
	$(PY) -m sciagent build

# automated literature collection (needs internet: arXiv + OpenAlex), then rebuild
collect:
	$(PY) -m sciagent collect --author "Sahar Vahdati"
	$(PY) -m sciagent build

serve:
	$(PY) -m sciagent serve

test:
	$(PY) -m pytest -q

eval-retrieval:
	$(PY) evals/run_eval.py --retrieval-only

judge-check:
	$(PY) evals/check_judge.py

eval:
	$(PY) evals/run_eval.py --judge

sciqa-baseline:
	$(PY) evals/run_sciqa.py --generator knn

sciqa-claude:
	$(PY) evals/run_sciqa.py --generator claude --shots 0 1 3 5 --execute

examples:
	$(PY) scripts/run_examples.py

all: install build test eval-retrieval sciqa-baseline
