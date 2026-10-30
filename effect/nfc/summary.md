llama.cpp@0.7.0 against `tokenizers` 0.23.3 (`transformers` 5.22.0); the first 200 Belebele questions of each language.

### As written

| Model | Language | Prompts that differ | Tokens, llama.cpp / Python | Accuracy, Python | Accuracy, llama.cpp | p (McNemar) | Answer changed | Bits per character, Python → llama.cpp | More surprising with llama.cpp's ids |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | ben_Beng | 200 of 200 | 0.98 | 46.5% | 44.0% | 0.4 | 23.5% | 1.645 → 2.073 | 200 of 200 |
| Qwen2.5-1.5B-Instruct | pan_Guru | 191 of 200 | 0.99 | 34.5% | 32.5% | 0.39 | 9.0% | 2.161 → 2.237 | 190 of 191 |
| Qwen2.5-1.5B-Instruct | ory_Orya | 143 of 200 | 1.00 | 34.5% | 34.5% | 1 | 3.5% | 2.215 → 2.251 | 143 of 143 |
| Qwen2.5-1.5B-Instruct | hin_Deva | 58 of 200 | 1.00 | 45.5% | 45.5% | 1 | 1.0% | 1.505 → 1.519 | 58 of 58 |
| Qwen2.5-1.5B-Instruct | yor_Latn | 141 of 200 | 1.01 | 30.5% | 31.5% | 0.62 | 2.5% | 2.578 → 2.675 | 130 of 141 |
| Qwen2.5-1.5B-Instruct | eng_Latn | 0 of 200 | 1.00 | 84.5% | 84.5% | 1 | 0.0% | 0.860 → 0.860 | 0 of 0 |
| Qwen3-1.7B | ben_Beng | 200 of 200 | 0.98 | 44.5% | 36.5% | 0.0052 | 23.0% | 1.801 → 2.305 | 200 of 200 |
| Qwen3-1.7B | pan_Guru | 191 of 200 | 0.99 | 36.0% | 37.0% | 0.8 | 13.5% | 1.959 → 2.096 | 191 of 191 |
| Qwen3-1.7B | ory_Orya | 143 of 200 | 1.00 | 38.0% | 37.0% | 0.73 | 6.5% | 1.997 → 2.045 | 143 of 143 |
| Qwen3-1.7B | hin_Deva | 58 of 200 | 1.00 | 47.5% | 47.5% | 1 | 1.5% | 1.603 → 1.623 | 58 of 58 |
| Qwen3-1.7B | yor_Latn | 141 of 200 | 1.01 | 26.0% | 25.5% | 1 | 5.0% | 3.211 → 3.295 | 136 of 141 |
| Qwen3-1.7B | eng_Latn | 0 of 200 | 1.00 | 81.0% | 81.0% | 1 | 0.0% | 0.952 → 0.952 | 0 of 0 |

### Decomposed

| Model | Language | Prompts that differ | Tokens, llama.cpp / Python | Accuracy, Python | Accuracy, llama.cpp | p (McNemar) | Answer changed | Bits per character, Python → llama.cpp | More surprising with llama.cpp's ids |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | vie_Latn | 200 of 200 | 2.77 | 71.0% | 40.0% | 4.3e-10 | 65.5% | 1.093 → 2.950 | 200 of 200 |
| Qwen2.5-1.5B-Instruct | kor_Hang | 200 of 200 | 2.74 | 69.0% | 25.0% | 4.8e-18 | 67.5% | 1.913 → 5.412 | 200 of 200 |
| Qwen2.5-1.5B-Instruct | fra_Latn | 200 of 200 | 1.13 | 80.0% | 78.0% | 0.48 | 10.5% | 0.891 → 1.186 | 200 of 200 |
| Qwen2.5-1.5B-Instruct | deu_Latn | 200 of 200 | 1.10 | 68.5% | 67.5% | 0.69 | 6.5% | 1.041 → 1.198 | 200 of 200 |
| Qwen3-1.7B | vie_Latn | 200 of 200 | 2.90 | 64.0% | 34.0% | 1.4e-12 | 47.5% | 1.214 → 3.692 | 200 of 200 |
| Qwen3-1.7B | kor_Hang | 200 of 200 | 2.86 | 54.0% | 24.5% | 3.8e-13 | 46.0% | 2.277 → 7.381 | 200 of 200 |
| Qwen3-1.7B | fra_Latn | 200 of 200 | 1.14 | 68.0% | 68.0% | 1 | 17.0% | 0.973 → 1.346 | 200 of 200 |
| Qwen3-1.7B | deu_Latn | 200 of 200 | 1.11 | 62.0% | 63.0% | 0.83 | 19.0% | 1.128 → 1.312 | 200 of 200 |

### The fix

Normalizing the text to NFC before llama.cpp tokenizes it gives Python's exact ids for 4,000 of 4,000 prompts (every model, language and setting above).
