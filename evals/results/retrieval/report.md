# Evaluation report

Corpus: 11 papers, 45 chunks. Questions: 27 (21 with gold evidence).

## Retrieval

| mode | recall@1 | recall@3 | recall@5 | all_gold@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.667 | 0.881 | 0.952 | 0.905 | 0.813 | 0.94 |
| dense | 0.452 | 0.762 | 0.929 | 0.857 | 0.672 | 0.833 |
| hybrid | 0.548 | 0.81 | 0.929 | 0.857 | 0.734 | 0.886 |
