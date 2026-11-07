# Tokenizer compatibility, 2026-11-07

`transformers` 5.22.2 and `tokenizers` 0.23.3 in Python against `transformers.js@3.7.6`, `transformers.js@4.3.2`, `@huggingface/tokenizers@0.2.0` on Node v22.23.4. Of the top 1000 models, 254 have a tokenizer.json (115 distinct files); 254 were checked on 42 texts.

| Runtime | Same | Differs | Cannot load | Same, by downloads |
|---|---|---|---|---|
| transformers.js@3.7.6 | 182 | 71 | 1 | 48.5% |
| transformers.js@4.3.2 | 186 | 67 | 1 | 49.0% |
| @huggingface/tokenizers@0.2.0 | 186 | 67 | 1 | 49.0% |

Models affected, by cause:

| Cause | tj3 | tj4 | tokenizers_js |
|---|---|---|---|
| byte fallback ignored | 1 | 1 | 1 |
| cannot load | 1 | 1 | 1 |
| final sigma (toLowerCase) | 45 | 45 | 45 |
| no leading ▁ | 4 | 0 | 0 |
| other | 3 | 1 | 1 |
| throws on decode of no ids | 1 | 1 | 1 |
| whitespace (U+FEFF, U+0085) | 31 | 28 | 28 |

## Since 2026-10-30

Changed: `transformers` 5.22.1 → 5.22.2. 248 models were checked both times.

**transformers.js@3.7.6**: no change.

**transformers.js@4.3.2**: no change.

**@huggingface/tokenizers@0.2.0**: no change.
