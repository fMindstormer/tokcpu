# Tokenizer compatibility, 2026-10-28

`transformers` 5.21.0 and `tokenizers` 0.23.3 in Python against `transformers.js@3.7.6`, `transformers.js@4.3.2`, `@huggingface/tokenizers@0.2.0` on Node v22.23.3. Of the top 1000 models, 267 have a tokenizer.json (116 distinct files); 267 were checked on 42 texts.

| Runtime | Same | Differs | Cannot load | Same, by downloads |
|---|---|---|---|---|
| transformers.js@3.7.6 | 188 | 78 | 1 | 46.2% |
| transformers.js@4.3.2 | 192 | 74 | 1 | 46.7% |
| @huggingface/tokenizers@0.2.0 | 192 | 74 | 1 | 46.7% |

Models affected, by cause:

| Cause | tj3 | tj4 | tokenizers_js |
|---|---|---|---|
| byte fallback ignored | 2 | 2 | 2 |
| cannot load | 1 | 1 | 1 |
| final sigma (toLowerCase) | 48 | 48 | 48 |
| no leading ▁ | 6 | 0 | 0 |
| other | 3 | 2 | 2 |
| throws on decode of no ids | 2 | 2 | 2 |
| whitespace (U+FEFF, U+0085) | 36 | 32 | 32 |

## Since 2026-10-27

Changed: `node` v22.23.4 → v22.23.3; `transformers.js@4.3.1` → `transformers.js@4.3.2`. 263 models were checked both times.

**transformers.js@3.7.6**: no change.

**transformers.js@4.3.2**: no change.

**@huggingface/tokenizers@0.2.0**: no change.
