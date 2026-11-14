# Tokenizer compatibility, 2026-11-14

`transformers` 5.23.0 and `tokenizers` 0.23.3 in Python against `transformers.js@3.7.6`, `transformers.js@4.3.2`, `@huggingface/tokenizers@0.2.0` on Node v22.23.4. Of the top 1000 models, 201 have a tokenizer.json (95 distinct files); 201 were checked on 42 texts.

| Runtime | Same | Differs | Cannot load | Same, by downloads |
|---|---|---|---|---|
| transformers.js@3.7.6 | 141 | 59 | 1 | 48.1% |
| transformers.js@4.3.2 | 143 | 57 | 1 | 48.4% |
| @huggingface/tokenizers@0.2.0 | 143 | 57 | 1 | 48.4% |

Models affected, by cause:

| Cause | tj3 | tj4 | tokenizers_js |
|---|---|---|---|
| cannot load | 1 | 1 | 1 |
| final sigma (toLowerCase) | 38 | 38 | 38 |
| no leading ▁ | 4 | 0 | 0 |
| other | 2 | 2 | 2 |
| throws on decode of no ids | 1 | 1 | 1 |
| whitespace (U+FEFF, U+0085) | 27 | 24 | 24 |

## Since 2026-11-07

Changed: `transformers` 5.22.2 → 5.23.0. 196 models were checked both times.

**transformers.js@3.7.6**: no change.

**transformers.js@4.3.2**: no change.

**@huggingface/tokenizers@0.2.0**: no change.
