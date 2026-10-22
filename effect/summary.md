Web pages (FineWeb-2 Greek): 736 of 5000 have a word-final capital Σ.

| Model | Web pages that differ | Differ where the model reads | Median drift, capitals | 90th percentile |
|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 788 | 329 | 0.0120 | 0.0296 |
| BAAI/bge-small-en-v1.5 | 788 | 427 | 0.0042 | 0.0136 |
| google-bert/bert-base-multilingual-uncased | 788 | 544 | 0.0298 | 0.0778 |

Texts in capitals whose ids differ, and which runtime's ids equal those of the same text as written:

- sentence-transformers/all-MiniLM-L6-v2: 1159; Python 0, transformers.js 1157
- BAAI/bge-small-en-v1.5: 1159; Python 0, transformers.js 1157
- google-bert/bert-base-multilingual-uncased: 1159; Python 0, transformers.js 1157

Median score gap between Python's first and second passage (questions in capitals, passages as written), against the median drift of those questions:

- sentence-transformers/all-MiniLM-L6-v2: gap 0.0164, drift 0.0131
- BAAI/bge-small-en-v1.5: gap 0.0038, drift 0.0044
- google-bert/bert-base-multilingual-uncased: gap 0.0038, drift 0.0409

| Model | Setting | Questions that differ | R@1 Python | R@1 JS | R@1 mixed | Top 1 changed (JS, mixed) | Top 10 changed (JS, mixed) | Gold rank better / worse (mixed) |
|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | questions as written, passages as written | 1 of 900 | 0.022 | 0.022 | 0.022 | 0.0%, 0.0% | 0.1%, 0.1% | 0 / 1 |
| all-MiniLM-L6-v2 | questions capitals, passages as written | 671 of 900 | 0.021 | 0.022 | 0.022 | 12.1%, 12.1% | 48.1%, 48.1% | 353 / 270 |
| all-MiniLM-L6-v2 | questions capitals, passages capitals | 671 of 900 | 0.023 | 0.022 | 0.026 | 21.9%, 11.6% | 86.0%, 46.2% | 304 / 316 |
| bge-small-en-v1.5 | questions as written, passages as written | 1 of 900 | 0.094 | 0.094 | 0.094 | 0.0%, 0.0% | 0.1%, 0.1% | 0 / 1 |
| bge-small-en-v1.5 | questions capitals, passages as written | 671 of 900 | 0.090 | 0.094 | 0.094 | 17.8%, 17.8% | 60.9%, 60.9% | 276 / 296 |
| bge-small-en-v1.5 | questions capitals, passages capitals | 671 of 900 | 0.091 | 0.094 | 0.092 | 32.1%, 13.3% | 97.8%, 56.0% | 271 / 293 |
| bert-base-multilingual-uncased | questions as written, passages as written | 1 of 900 | 0.319 | 0.319 | 0.319 | 0.1%, 0.1% | 0.2%, 0.1% | 0 / 1 |
| bert-base-multilingual-uncased | questions capitals, passages as written | 671 of 900 | 0.233 | 0.319 | 0.319 | 43.8%, 43.8% | 73.7%, 73.7% | 416 / 105 |
| bert-base-multilingual-uncased | questions capitals, passages capitals | 671 of 900 | 0.261 | 0.319 | 0.267 | 51.0%, 47.6% | 99.8%, 74.4% | 264 / 247 |
