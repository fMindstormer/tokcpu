# Tokenizer compatibility, 2026-11-21

`transformers` 5.24.0 and `tokenizers` 0.23.3 in Python against `transformers.js@3.8.1`, `transformers.js@4.3.2`, `@huggingface/tokenizers@0.2.0` on Node v22.23.5. Of the top 1000 models, 250 have a tokenizer.json (120 distinct files); 250 were checked on 42 texts.

| Runtime | Same | Differs | Cannot load | Same, by downloads |
|---|---|---|---|---|
| transformers.js@3.8.1 | 180 | 69 | 1 | 49.9% |
| transformers.js@4.3.2 | 182 | 67 | 1 | 50.1% |
| @huggingface/tokenizers@0.2.0 | 182 | 67 | 1 | 50.1% |

Models affected, by cause:

| Cause | tj3 | tj4 | tokenizers_js |
|---|---|---|---|
| byte fallback ignored | 1 | 1 | 1 |
| cannot load | 1 | 1 | 1 |
| final sigma (toLowerCase) | 44 | 44 | 44 |
| other | 4 | 2 | 2 |
| throws on decode of no ids | 1 | 1 | 1 |
| whitespace (U+FEFF, U+0085) | 28 | 27 | 27 |

## Since 2026-11-20

Changed: `transformers` 5.23.0 → 5.24.0; `node` v22.23.4 → v22.23.5. 246 models were checked both times.

**transformers.js@3.8.1**: no change.

**transformers.js@4.3.2**: no change.

**@huggingface/tokenizers@0.2.0**: no change.
