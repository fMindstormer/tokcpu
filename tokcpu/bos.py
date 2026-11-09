"""Does llama.cpp's extra <bos> change what Gemma 4 does?

`gguf` found 26 GGUF repos where llama.cpp adds a <bos> token that tokenizer.json does not, 24 of them Gemma 4.
Gemma 4's chat template writes <bos> itself, so a prompt the template renders, tokenized by llama.cpp, starts
with two. This measures what that does in google/gemma-4-E2B-it, in two settings:

- chat template: the prompt as the template renders it. Python's ids start with one <bos>, llama.cpp's with
  two. This is what reaches the model when a program renders the template itself and hands llama.cpp the
  text, as with llama-server's completion endpoint (`server` checks which endpoints do);
- plain text: the same question with no template, as a completion ending in "Answer:". Python's ids have no
  <bos>; llama.cpp's have one.

Each prompt is a Belebele multiple-choice question, tokenized by Python (`tokenizers`, from the model's
tokenizer.json) and by llama.cpp (the GGUF's vocabulary only, through fuzz/llama_tok.c). Both id sequences go
through the same model, weights and code (PyTorch), and both are scored on the same tokens, every one of
Python's ids but the first, so the <bos> is the only thing that differs. For each: the answer the model gives
(the most likely of A-D) and how surprised it is by the prompt (bits per character).

    python -m tokcpu bos       writes effect/bos/summary.md and results.json (needs torch and llama.cpp;
                               downloads about 13 GB; about an hour on an Apple M4's GPU)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .nfc import ASK, QUESTIONS, Reader, load, mc, questions, stats

MODEL = ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7")
GGUF = ("ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_0.gguf", "6938c58e698502037401a5f19f6905f2fb2f7dee")
SETTINGS = ("chat template", "plain text")
LANGS = ["eng_Latn", "deu_Latn", "hin_Deva", "zho_Hans"]
NAMES = {"eng_Latn": "English", "deu_Latn": "German", "hin_Deva": "Hindi", "zho_Hans": "Chinese"}


def prompt(tok, r: dict, setting: str) -> str:
    text = mc(r) + "\n" + ASK
    if setting == "plain text":
        return text + "\n\nAnswer:"
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def plain_letters(tok) -> list[int]:
    """The ids of " A" to " D" after "Answer:", the plain-text prompt's last word."""
    head = tok.encode("Answer:", add_special_tokens=False)
    ids = [tok.encode(f"Answer: {c}", add_special_tokens=False) for c in "ABCD"]
    assert all(i[:-1] == head for i in ids), "each letter should be one token after 'Answer:'"
    return [i[-1] for i in ids]


def measure(out: Path, log=print) -> dict:
    """Every setting and language, saved to out/results.json after each; a rerun keeps what is there."""
    import tempfile
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    from .gguf import Llama
    repo, rev = MODEL
    path = out / "results.json"
    llama = Llama()
    results = json.loads(path.read_text()) if path.exists() else {
        "llama.cpp": llama.name, "tokenizers": tokenizers.__version__, "transformers": transformers.__version__,
        "model": repo, "gguf": f"{GGUF[0]}/{GGUF[1]}", "questions per language": QUESTIONS, "settings": {}}
    todo = [(s, lang) for s in SETTINGS for lang in LANGS if lang not in results["settings"].get(s, {})]
    try:
        if todo:
            tok = AutoTokenizer.from_pretrained(repo, revision=rev)
            reader = Reader(repo, tok, rev, dtype="bfloat16")   # Gemma's activations overflow float16
            letters = {"chat template": None, "plain text": plain_letters(tok)}
            with tempfile.TemporaryDirectory() as tmp:
                load(llama, *GGUF, tmp)
                for setting, lang in todo:
                    t0 = time.time()
                    rows = []
                    for r in questions(lang):
                        text = prompt(tok, r, setting)
                        py = tok.encode(text)   # tokenizer.json adds no <bos>: add_special_tokens changes nothing
                        ll = llama._encode(text)
                        a, na = reader(py, letters[setting], len(py) - 1)
                        b, nb = reader(ll, letters[setting], len(py) - 1)
                        rows.append({"gold": int(r["correct_answer_num"]) - 1, "differs": ll != py,
                                     "only the bos": ll[0] == tok.bos_token_id and ll[1:] == py,
                                     "python": a, "llama.cpp": b, "nll python": na, "nll llama.cpp": nb,
                                     "tokens python": len(py), "tokens llama.cpp": len(ll), "chars": len(text)})
                    results["settings"].setdefault(setting, {})[lang] = rows
                    path.write_text(json.dumps(results))
                    s = stats(rows)
                    log(f"{setting}, {lang}: accuracy {s['accuracy python']:.3f} -> {s['accuracy llama.cpp']:.3f}, "
                        f"answer changed {s['answer changed']:.1%}, bits/char {s['bits python']:.3f} -> {s['bits llama.cpp']:.3f}, "
                        f"{time.time() - t0:.0f} s")
    finally:
        llama.close()
    return results


def server(results: dict, log=print) -> None:
    """Which llama-server endpoints send two <bos>? Serves the whole GGUF (2.8 GB) and counts the prompt tokens
    of the first English question on each, against Python's ids for the same prompt, which hold one."""
    import socket
    import subprocess
    import urllib.request
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL[0], revision=MODEL[1])
    r = questions("eng_Latn")[0]
    text = prompt(tok, r, "chat template")
    py = tok.encode(text)
    messages = [{"role": "user", "content": mc(r) + "\n" + ASK}]
    off = {"enable_thinking": False}   # llama-server turns Gemma 4's thinking on; transformers' template leaves it off
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    gguf = hf_hub_download(GGUF[0], GGUF[1], revision=GGUF[2])
    proc = subprocess.Popen(["llama-server", "-m", gguf, "--port", str(port), "-c", "4096"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def call(path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", None if body is None else json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req))

    try:
        for _ in range(300):
            try:
                if call("/health").get("status") == "ok":
                    break
            except OSError:
                pass
            time.sleep(1)
        rendered = call("/apply-template", {"messages": messages, "chat_template_kwargs": off})["prompt"]
        counts = {
            "/v1/chat/completions": call("/v1/chat/completions", {"messages": messages, "max_tokens": 1,
                                                                  "chat_template_kwargs": off})["usage"]["prompt_tokens"],
            "/completion": call("/completion", {"prompt": text, "n_predict": 1, "cache_prompt": False})["tokens_evaluated"],
            "/v1/completions": call("/v1/completions", {"prompt": text, "max_tokens": 1})["usage"]["prompt_tokens"]}
        results["server"] = {"llama-server": call("/props")["build_info"], "python tokens": len(py),
                             "same prompt": tok.bos_token + rendered == text, "prompt tokens": counts}
        log(f"llama-server: {counts}; Python: {len(py)}")
    finally:
        proc.terminate()
        proc.wait()


ENDPOINTS = {"/v1/chat/completions": "`/v1/chat/completions`, which renders the template itself (thinking off)",
             "/completion": "`/completion`, given the prompt the template renders in Python",
             "/v1/completions": "`/v1/completions`, the same"}

HEADS = {"chat template": "Chat template: Python's ids start with one <bos>, llama.cpp's with two",
         "plain text": "Plain text: Python's ids have no <bos>, llama.cpp's have one"}


def summary(results: dict) -> str:
    lines = [f"{results['llama.cpp']} against `tokenizers` {results['tokenizers']} (`transformers` {results['transformers']}); "
             f"{results['model']}, with the vocabulary of {results['gguf']}; the first {results['questions per language']} "
             "Belebele questions of each language.", ""]
    for setting, langs in results["settings"].items():
        lines += [f"### {HEADS[setting]}", "",
                  "| Language | Ids differ only by the <bos> | Accuracy, Python | Accuracy, llama.cpp | p (McNemar) | Answer changed | Bits per character, Python → llama.cpp | More surprising with llama.cpp's ids |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, rows in [(NAMES.get(lang, lang), rows) for lang, rows in langs.items()] + [("All", [r for rows in langs.values() for r in rows])]:
            s = stats(rows)
            lines.append(f"| {name} | {sum(r['only the bos'] for r in rows)} of {len(rows)} | {s['accuracy python']:.1%} | "
                         f"{s['accuracy llama.cpp']:.1%} | {s['p']:.2g} | {s['answer changed']:.1%} | "
                         f"{s['bits python']:.3f} → {s['bits llama.cpp']:.3f} | {s['more surprising']} of {s['differ']} |")
        lines.append("")
    if "server" in results:
        sv = results["server"]
        lines += ["### Which llama-server endpoints send two <bos>", "",
                  f"llama-server {sv['llama-server']}, the first English question. Python's ids for it are {sv['python tokens']} "
                  "tokens, with one <bos>" + ("; llama-server renders the same prompt." if sv["same prompt"] else
                                             "; llama-server renders a different prompt, so its counts are not comparable."), "",
                  "| Endpoint | Prompt tokens | <bos> tokens |", "|---|---|---|"]
        for path, n in sv["prompt tokens"].items():
            lines.append(f"| {ENDPOINTS[path]} | {n} | {n - sv['python tokens'] + 1} |")
        lines.append("")
    return "\n".join(lines)


def bos(out: Path, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results = measure(out, log)
    server(results, log)
    (out / "results.json").write_text(json.dumps(results))
    (out / "summary.md").write_text(summary(results))
    return results
