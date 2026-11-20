# Tokenizer compatibility, 2026-11-20

`transformers` 5.23.0 and `tokenizers` 0.23.3 in Python against `transformers.js@3.8.1`, `transformers.js@4.3.2`, `@huggingface/tokenizers@0.2.0` on Node v22.23.4. Of the top 1000 models, 252 have a tokenizer.json (119 distinct files); 252 were checked on 42 texts.

| Runtime | Same | Differs | Cannot load | Same, by downloads |
|---|---|---|---|---|
| transformers.js@3.8.1 | 183 | 68 | 1 | 50.0% |
| transformers.js@4.3.2 | 184 | 67 | 1 | 50.1% |
| @huggingface/tokenizers@0.2.0 | 184 | 67 | 1 | 50.1% |

Models affected, by cause:

| Cause | tj3 | tj4 | tokenizers_js |
|---|---|---|---|
| byte fallback ignored | 1 | 1 | 1 |
| cannot load | 1 | 1 | 1 |
| final sigma (toLowerCase) | 44 | 44 | 44 |
| other | 3 | 2 | 2 |
| throws on decode of no ids | 1 | 1 | 1 |
| whitespace (U+FEFF, U+0085) | 28 | 27 | 27 |

## Since 2026-11-19

Changed: `transformers.js@3.7.6` → `transformers.js@3.8.1`. 252 models were checked both times.

**transformers.js@3.8.1**: 2 now match; 3 differ for other reasons.

2 now match:

- mistralai/Mistral-7B-Instruct-v0.3 (was: no leading ▁; whitespace (U+FEFF, U+0085))
- mistralai/Mistral-7B-Instruct-v0.2 (was: no leading ▁; whitespace (U+FEFF, U+0085))

3 differ for other reasons:

- hindsight-ai/hs-rerank-1 (no leading ▁; other; whitespace (U+FEFF, U+0085) → other)
- jinaai/jina-reranker-v2-base-multilingual (no leading ▁; whitespace (U+FEFF, U+0085) → whitespace (U+FEFF, U+0085))
- google/siglip-base-patch16-224 (byte fallback ignored; no leading ▁; whitespace (U+FEFF, U+0085) → byte fallback ignored; whitespace (U+FEFF, U+0085))

**transformers.js@4.3.2**: no change.

**@huggingface/tokenizers@0.2.0**: no change.
