llama.cpp@0.7.0 against `tokenizers` 0.23.3 (`transformers` 5.23.0); google/gemma-4-E2B-it, with the vocabulary of ggml-org/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_0.gguf; the first 200 Belebele questions of each language.

### Chat template: Python's ids start with one `<bos>`, llama.cpp's with two

| Language | Ids differ only by the `<bos>` | Accuracy, Python | Accuracy, llama.cpp | p (McNemar) | Answer changed | Bits per character, Python → llama.cpp | More surprising with llama.cpp's ids |
|---|---|---|---|---|---|---|---|
| English | 200 of 200 | 80.5% | 71.5% | 0.00028 | 13.0% | 1.499 → 1.428 | 0 of 200 |
| German | 200 of 200 | 70.5% | 68.5% | 0.39 | 7.5% | 1.549 → 1.502 | 5 of 200 |
| Hindi | 200 of 200 | 63.5% | 58.5% | 0.021 | 12.0% | 1.806 → 1.747 | 0 of 200 |
| Chinese | 200 of 200 | 69.5% | 61.5% | 0.00014 | 12.5% | 3.939 → 3.797 | 1 of 200 |
| All | 800 of 800 | 71.0% | 65.0% | 4.5e-09 | 11.2% | 1.886 → 1.817 | 6 of 800 |

### Plain text: Python's ids have no `<bos>`, llama.cpp's have one

| Language | Ids differ only by the `<bos>` | Accuracy, Python | Accuracy, llama.cpp | p (McNemar) | Answer changed | Bits per character, Python → llama.cpp | More surprising with llama.cpp's ids |
|---|---|---|---|---|---|---|---|
| English | 200 of 200 | 37.0% | 75.5% | 5.3e-15 | 64.5% | 2.582 → 1.355 | 0 of 200 |
| German | 200 of 200 | 39.5% | 65.5% | 8.5e-09 | 58.5% | 3.259 → 1.452 | 0 of 200 |
| Hindi | 200 of 200 | 32.0% | 57.5% | 2.8e-07 | 70.0% | 3.465 → 1.717 | 0 of 200 |
| Chinese | 200 of 200 | 33.5% | 68.5% | 3e-12 | 68.0% | 7.557 → 3.870 | 0 of 200 |
| All | 800 of 800 | 35.5% | 66.8% | 7.7e-39 | 65.2% | 3.597 → 1.767 | 0 of 800 |

### Which llama-server endpoints send two `<bos>`

llama-server b11669-620d2225c, the first English question. Python's ids for it are 187 tokens, with one `<bos>`; llama-server renders the same prompt.

| Endpoint | Prompt tokens | `<bos>` tokens |
|---|---|---|
| `/v1/chat/completions`, which renders the template itself (thinking off) | 187 | 1 |
| `/completion`, given the prompt the template renders in Python | 188 | 2 |
| `/v1/completions`, the same | 188 | 2 |
