# Tokenizer compatibility, 2026-11-19

`transformers` 5.23.0 and `tokenizers` 0.23.3 in Python against `transformers.js@3.7.6`, `transformers.js@4.3.2`, `@huggingface/tokenizers@0.2.0` on Node v22.23.4. Of the top 1000 models, 252 have a tokenizer.json (119 distinct files); 252 were checked on 42 texts.

| Runtime | Same | Differs | Cannot load | Same, by downloads |
|---|---|---|---|---|
| transformers.js@3.7.6 | 181 | 70 | 1 | 49.7% |
| transformers.js@4.3.2 | 184 | 67 | 1 | 50.1% |
| @huggingface/tokenizers@0.2.0 | 184 | 67 | 1 | 50.1% |

Models affected, by cause:

| Cause | tj3 | tj4 | tokenizers_js |
|---|---|---|---|
| byte fallback ignored | 1 | 1 | 1 |
| cannot load | 1 | 1 | 1 |
| final sigma (toLowerCase) | 44 | 44 | 44 |
| no leading ▁ | 5 | 0 | 0 |
| other | 3 | 2 | 2 |
| throws on decode of no ids | 1 | 1 | 1 |
| whitespace (U+FEFF, U+0085) | 31 | 27 | 27 |

## Since 2026-11-14

The same versions as before. 201 models were checked both times.

**transformers.js@3.7.6**: no change.

**transformers.js@4.3.2**: no change.

**@huggingface/tokenizers@0.2.0**: no change.
