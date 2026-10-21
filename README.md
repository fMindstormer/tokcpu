# Tokenizer compatibility, 2026-10-19

`transformers` 5.20.2 and `tokenizers` 0.23.3 in Python against `transformers.js@3.7.6`, `transformers.js@4.3.1`, `@huggingface/tokenizers@0.2.0` on Node v22.23.3. Of the top 1000 models, 500 have a tokenizer.json (220 distinct files); 500 were checked on 42 texts.

| Runtime | Same | Differs | Cannot load | Same, by downloads |
|---|---|---|---|---|
| transformers.js@3.7.6 | 362 | 137 | 1 | 46.8% |
| transformers.js@4.3.1 | 375 | 124 | 1 | 47.6% |
| @huggingface/tokenizers@0.2.0 | 375 | 124 | 1 | 47.6% |

Models affected, by cause:

| Cause | tj3 | tj4 | tokenizers_js |
|---|---|---|---|
| byte fallback ignored | 3 | 3 | 3 |
| cannot load | 1 | 1 | 1 |
| final sigma (toLowerCase) | 84 | 84 | 84 |
| no leading ▁ | 11 | 0 | 0 |
| other | 16 | 8 | 8 |
| throws on decode of no ids | 3 | 3 | 3 |
| whitespace (U+FEFF, U+0085) | 55 | 44 | 44 |
