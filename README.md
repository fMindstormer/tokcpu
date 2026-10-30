# Tokenizer compatibility, 2026-10-30

`transformers` 5.22.1 and `tokenizers` 0.23.3 in Python against `transformers.js@3.7.6`, `transformers.js@4.3.2`, `@huggingface/tokenizers@0.2.0` on Node v22.23.4. Of the top 1000 models, 267 have a tokenizer.json (115 distinct files); 267 were checked on 42 texts.

| Runtime | Same | Differs | Cannot load | Same, by downloads |
|---|---|---|---|---|
| transformers.js@3.7.6 | 189 | 77 | 1 | 47.2% |
| transformers.js@4.3.2 | 194 | 72 | 1 | 47.7% |
| @huggingface/tokenizers@0.2.0 | 194 | 72 | 1 | 47.7% |

Models affected, by cause:

| Cause | tj3 | tj4 | tokenizers_js |
|---|---|---|---|
| byte fallback ignored | 2 | 2 | 2 |
| cannot load | 1 | 1 | 1 |
| final sigma (toLowerCase) | 47 | 47 | 47 |
| no leading ▁ | 5 | 0 | 0 |
| other | 4 | 1 | 1 |
| throws on decode of no ids | 1 | 1 | 1 |
| whitespace (U+FEFF, U+0085) | 34 | 31 | 31 |

## Since 2026-10-29

Changed: `transformers` 5.21.1 → 5.22.1. 266 models were checked both times.

**transformers.js@3.7.6**: no change.

**transformers.js@4.3.2**: no change.

**@huggingface/tokenizers@0.2.0**: no change.
