"""The smallest final-sigma disagreement, on the Python side: the Greek word ΟΔΟΣ in mBERT's tokenizer.
It downloads only the tokenizer (1.7 MB, no model weights) into mbert/, where sigma.mjs reads the same
files in JavaScript.

    pip install -r requirements.txt
    python sigma.py
    npm ci && node sigma.mjs
"""

import sys
from pathlib import Path

import tokenizers
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "google-bert/bert-base-multilingual-uncased"
REVISION = "7cbf9a625e29989f6b9c6c2fa68234c304f7e38f"
WORD = "ΟΔΟΣ"
EXPECTED = [295, 37187, 62591]     # Python's ids; the word typed in lowercase gives [295, 42501]
HERE = Path(__file__).resolve().parent / "mbert"

for name in ("tokenizer.json", "tokenizer_config.json"):
    hf_hub_download(REPO, name, revision=REVISION, local_dir=HERE)

tok = Tokenizer.from_file(str(HERE / "tokenizer.json"))
tok.no_padding()
tok.no_truncation()
enc = tok.encode(WORD, add_special_tokens=False)
low = tok.encode(WORD.lower(), add_special_tokens=False)
show = lambda label, value: print(f"{label:40s}{value}")
show("Python's own str.lower():", WORD.lower())
show(f"tokenizers {tokenizers.__version__} lowercasing (Rust):", tok.normalizer.normalize_str(WORD))
show(f"tokenizers {tokenizers.__version__}, {WORD}:", f"{' '.join(enc.tokens):12s} {enc.ids}")
show(f"tokenizers {tokenizers.__version__}, {WORD.lower()} (lowercase):", f"{' '.join(low.tokens):12s} {low.ids}")
if enc.ids != EXPECTED:
    sys.exit(f"expected {EXPECTED}; this version of tokenizers gives {enc.ids}")
