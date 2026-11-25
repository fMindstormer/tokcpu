# Tokenizer compatibility, 2026-11-25

`transformers` 5.24.1 and `tokenizers` 0.23.3 in Python against `transformers.js@3.8.1`, `transformers.js@4.3.2`, `@huggingface/tokenizers@0.2.0` on Node v22.23.5. Of the top 1000 models, 238 have a tokenizer.json (113 distinct files); 238 were checked on 42 texts.

| Runtime | Same | Differs | Cannot load | Same, by downloads |
|---|---|---|---|---|
| transformers.js@3.8.1 | 168 | 69 | 1 | 49.2% |
| transformers.js@4.3.2 | 169 | 68 | 1 | 49.3% |
| @huggingface/tokenizers@0.2.0 | 169 | 68 | 1 | 49.3% |

Models affected, by cause:

| Cause | tj3 | tj4 | tokenizers_js |
|---|---|---|---|
| byte fallback ignored | 1 | 1 | 1 |
| cannot load | 1 | 1 | 1 |
| final sigma (toLowerCase) | 45 | 45 | 45 |
| other | 4 | 3 | 3 |
| throws on decode of no ids | 1 | 1 | 1 |
| whitespace (U+FEFF, U+0085) | 27 | 26 | 26 |

## Since 2026-11-21

Changed: `transformers` 5.24.0 → 5.24.1. 231 models were checked both times.

**transformers.js@3.8.1**: no change.

**transformers.js@4.3.2**: no change.

**@huggingface/tokenizers@0.2.0**: no change.
