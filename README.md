# Tokenizer compatibility, 2026-10-29

`transformers` 5.21.1 and `tokenizers` 0.23.3 in Python against `transformers.js@3.7.6`, `transformers.js@4.3.2`, `@huggingface/tokenizers@0.2.0` on Node v22.23.4. Of the top 1000 models, 269 have a tokenizer.json (118 distinct files); 269 were checked on 42 texts.

| Runtime | Same | Differs | Cannot load | Same, by downloads |
|---|---|---|---|---|
| transformers.js@3.7.6 | 190 | 78 | 1 | 46.7% |
| transformers.js@4.3.2 | 195 | 73 | 1 | 47.3% |
| @huggingface/tokenizers@0.2.0 | 195 | 73 | 1 | 47.3% |

Models affected, by cause:

| Cause | tj3 | tj4 | tokenizers_js |
|---|---|---|---|
| byte fallback ignored | 2 | 2 | 2 |
| cannot load | 1 | 1 | 1 |
| final sigma (toLowerCase) | 47 | 47 | 47 |
| no leading ▁ | 6 | 0 | 0 |
| other | 4 | 2 | 2 |
| throws on decode of no ids | 2 | 2 | 2 |
| whitespace (U+FEFF, U+0085) | 35 | 31 | 31 |

## Since 2026-10-28

Changed: `transformers` 5.21.0 → 5.21.1; `node` v22.23.3 → v22.23.4. 266 models were checked both times.

**transformers.js@3.7.6**: no change.

**transformers.js@4.3.2**: no change.

**@huggingface/tokenizers@0.2.0**: no change.
