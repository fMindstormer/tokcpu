"""Does llama.cpp's missing NFC normalization change what a model does?

`gguf` found that llama.cpp skips the NFC normalizer that Qwen 2 and 3 ask for in their tokenizer.json:
130 of the 192 GGUF repos it checked differ from Python this way. NFC rewrites some characters before
tokenizing, so where the text is not already in NFC, llama.cpp feeds the model ids that Python never would.
This measures what that does downstream, in two settings:

- as written: languages whose ordinary text NFC changes. Bengali, Punjabi, Odia and Hindi use letters that
  NFC splits in two (য় U+09DF becomes য + ়, a "composition exclusion"); Yoruba text often holds marks in
  a form NFC rewrites. English is the control;
- decomposed: text that arrives in NFD, as macOS file names and some copied text do (é as e + U+0301),
  in languages with accents or composed syllables. Python composes it back; llama.cpp does not.

Each prompt is a Belebele multiple-choice question in the model's chat template, tokenized twice: by
Python (`tokenizers`, from the model's tokenizer.json) and by llama.cpp (the GGUF's vocabulary only, through
fuzz/llama_tok.c). Both id sequences go through the same model, weights and code (PyTorch), so the
tokenizer is the only thing that differs. For each: the answer the model gives (the most likely of A-D
after the prompt) and how surprised it is by the prompt (bits per character).

    python -m tokcpu nfc       writes effect/nfc/summary.md, results.json and figure.svg (needs torch,
                               matplotlib and llama.cpp; downloads about 7 GB of weights; 2 hours 45 minutes on an Apple M4's GPU)
"""

from __future__ import annotations

import json
import math
import time
import unicodedata
from pathlib import Path

MODELS = {   # base model -> (its revision, the official GGUF repo, a GGUF file in it, that repo's revision)
    "Qwen/Qwen2.5-1.5B-Instruct": ("989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                                   "qwen2.5-1.5b-instruct-q2_k.gguf", "91cad51170dc346986eccefdc2dd33a9da36ead9"),
    "Qwen/Qwen3-1.7B": ("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "Qwen/Qwen3-1.7B-GGUF",
                        "Qwen3-1.7B-Q8_0.gguf", "90862c4b9d2787eaed51d12237eafdfe7c5f6077"),
}
BELEBELE = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
SETTINGS = {
    "as written": ["ben_Beng", "pan_Guru", "ory_Orya", "hin_Deva", "yor_Latn", "eng_Latn"],   # English: a control, NFC changes nothing
    "decomposed": ["vie_Latn", "kor_Hang", "fra_Latn", "deu_Latn"],
}
QUESTIONS = 200   # the first this many of each language's 900 (a 1.5B model reads about 1,000 tokens a second on an M4)
ASK = "Answer with the letter of the correct option only."


def questions(lang: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("facebook/belebele", f"data/{lang}.jsonl", repo_type="dataset", revision=BELEBELE)
    return [json.loads(line) for line in open(path, encoding="utf-8")][:QUESTIONS]


def prompt(tok, r: dict, setting: str) -> str:
    body = (f"{r['flores_passage']}\n\nQuestion: {r['question']}\n"
            + "".join(f"{c}. {r[f'mc_answer{i}']}\n" for i, c in zip(range(1, 5), "ABCD")))
    if setting == "decomposed":
        body = unicodedata.normalize("NFD", body)
    extra = {"enable_thinking": False} if "Qwen3" in tok.name_or_path else {}
    return tok.apply_chat_template([{"role": "user", "content": body + "\n" + ASK}], tokenize=False,
                                   add_generation_prompt=True, **extra)


class Reader:
    """A causal LM that answers a prompt's question and says how surprised it was by the prompt."""

    def __init__(self, repo: str, tok):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(repo, revision=MODELS[repo][0], dtype=dtype).to(self.device).eval()
        self.letters = [tok.convert_tokens_to_ids(c) for c in "ABCD"]

    def __call__(self, ids: list[int]) -> tuple[int, float]:
        """(the option it picks, 0-3; the prompt's negative log-likelihood in nats)."""
        torch = self.torch
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            h = self.model.model(input_ids=x).last_hidden_state[0]
            head = self.model.lm_head
            answer = int(head(h[-1:])[0, self.letters].float().argmax())
            nll = 0.0
            for s in range(0, len(ids) - 1, 512):   # the vocabulary is 151,936 wide: score in slices
                logits = head(h[s:min(s + 512, len(ids) - 1)]).float()
                nll += float(torch.nn.functional.cross_entropy(logits, x[0, s + 1:s + 1 + len(logits)], reduction="sum"))
        return answer, nll


def measure(out: Path, log=print) -> dict:
    """Every model and language, saved to out/results.json after each language; a rerun keeps what is there."""
    import tempfile
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    from .gguf import Llama
    llama = Llama()
    path = out / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {
        "llama.cpp": llama.name, "tokenizers": tokenizers.__version__, "transformers": transformers.__version__,
        "questions per language": QUESTIONS, "models": {}}
    try:
        for repo, (rev, gguf_repo, gguf_file, gguf_rev) in MODELS.items():
            m = results["models"].setdefault(repo, {"gguf": f"{gguf_repo}/{gguf_file}", "languages": {}})
            if all(lang in m["languages"] for langs in SETTINGS.values() for lang in langs):
                continue
            tok = AutoTokenizer.from_pretrained(repo, revision=rev)
            reader = Reader(repo, tok)
            with tempfile.TemporaryDirectory() as tmp:
                from .gguf import fetch_vocab
                fetch_vocab(gguf_repo, gguf_file, Path(tmp), revision=gguf_rev)
                loaded = llama.ask("load", dir=tmp)
                if loaded["errors"]:
                    raise RuntimeError(f"llama.cpp cannot load {gguf_repo}: {loaded['errors']}")
                for setting, langs in SETTINGS.items():
                    for lang in langs:
                        if lang in m["languages"]:
                            continue
                        t0 = time.time()
                        rows = []
                        for r in questions(lang):
                            text = prompt(tok, r, setting)
                            py = tok.encode(text, add_special_tokens=False)
                            ll = llama._encode(text)
                            a, na = reader(py)
                            b, nb = (a, na) if ll == py else reader(ll)
                            rows.append({"gold": int(r["correct_answer_num"]) - 1, "differs": ll != py,
                                         "python": a, "llama.cpp": b, "nll python": na, "nll llama.cpp": nb,
                                         "tokens python": len(py), "tokens llama.cpp": len(ll),
                                         "chars": len(unicodedata.normalize("NFC", text))})
                        m["languages"][lang] = {"setting": setting, "questions": rows}
                        path.write_text(json.dumps(results))
                        log(f"{repo} {lang} ({setting}): {line(rows)}, {time.time() - t0:.0f} s")
    finally:
        llama.close()
    return results


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: of the b + c questions only one side gets right, is a b-to-c split
    likely by chance?"""
    n = b + c
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def stats(rows: list[dict]) -> dict:
    n = len(rows)
    only_py = sum(r["python"] == r["gold"] != r["llama.cpp"] for r in rows)
    only_ll = sum(r["llama.cpp"] == r["gold"] != r["python"] for r in rows)
    bits = lambda k: sum(r[f"nll {k}"] for r in rows) / sum(r["chars"] for r in rows) / math.log(2)
    return {"questions": n, "differ": sum(r["differs"] for r in rows),
            "accuracy python": sum(r["python"] == r["gold"] for r in rows) / n,
            "accuracy llama.cpp": sum(r["llama.cpp"] == r["gold"] for r in rows) / n,
            "answer changed": sum(r["python"] != r["llama.cpp"] for r in rows) / n,
            "tokens": sum(r["tokens llama.cpp"] for r in rows) / sum(r["tokens python"] for r in rows),
            "bits python": bits("python"), "bits llama.cpp": bits("llama.cpp"),
            "more surprising": sum(r["nll llama.cpp"] > r["nll python"] for r in rows if r["differs"]),
            "only python right": only_py, "only llama.cpp right": only_ll, "p": mcnemar(only_py, only_ll)}


def line(rows: list[dict]) -> str:
    s = stats(rows)
    return (f"{s['differ']}/{s['questions']} differ, accuracy {s['accuracy python']:.3f} -> {s['accuracy llama.cpp']:.3f}, "
            f"answer changed {s['answer changed']:.1%}, tokens x{s['tokens']:.2f}, bits/char {s['bits python']:.3f} -> {s['bits llama.cpp']:.3f}")


def summary(results: dict) -> str:
    lines = [f"{results['llama.cpp']} against `tokenizers` {results['tokenizers']} (`transformers` {results['transformers']}); "
             f"the first {results['questions per language']} Belebele questions of each language.", ""]
    for setting in SETTINGS:
        lines += [f"### {setting.capitalize()}", "",
                  "| Model | Language | Prompts that differ | Tokens, llama.cpp / Python | Accuracy, Python | Accuracy, llama.cpp | p (McNemar) | Answer changed | Bits per character, Python → llama.cpp | More surprising with llama.cpp's ids |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for repo, m in results["models"].items():
            for lang, d in m["languages"].items():
                if d["setting"] != setting:
                    continue
                s = stats(d["questions"])
                lines.append(f"| {repo.split('/')[1]} | {lang} | {s['differ']} of {s['questions']} | {s['tokens']:.2f} | "
                             f"{s['accuracy python']:.1%} | {s['accuracy llama.cpp']:.1%} | {s['p']:.2g} | {s['answer changed']:.1%} | "
                             f"{s['bits python']:.3f} → {s['bits llama.cpp']:.3f} | {s['more surprising']} of {s['differ']} |")
        lines.append("")
    return "\n".join(lines)


NAMES = {"ben_Beng": "Bengali", "pan_Guru": "Punjabi", "ory_Orya": "Odia", "hin_Deva": "Hindi", "yor_Latn": "Yoruba",
         "eng_Latn": "English (control)", "vie_Latn": "Vietnamese", "kor_Hang": "Korean", "fra_Latn": "French", "deu_Latn": "German"}


def figure(results: dict, out: Path) -> None:
    """Accuracy with Python's ids and with llama.cpp's, per language and model: a dumbbell each, and the
    share of answers that change beside it. Needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .figure import GRID, INK, MUTED, SURFACE
    PY, LL = "#8f8e88", "#eb6834"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "svg.fonttype": "none"})
    rows = [(setting, lang) for setting, langs in SETTINGS.items() for lang in langs]
    y = {}
    for i, (setting, lang) in enumerate(rows):   # a gap between the two settings
        y[lang] = i + (setting == "decomposed") * 0.8
    models = list(results["models"])
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.8), sharey=True, facecolor=SURFACE)
    for ax, repo in zip(axes, models):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        for setting, lang in rows:
            d = results["models"][repo]["languages"].get(lang)
            if not d:
                continue
            st = stats(d["questions"])
            a, b = 100 * st["accuracy python"], 100 * st["accuracy llama.cpp"]
            ax.plot([a, b], [y[lang]] * 2, color=GRID, lw=3, solid_capstyle="round", zorder=1)
            ax.scatter([a], [y[lang]], s=42, color=PY, zorder=2, linewidths=1.5, edgecolors=SURFACE)
            ax.scatter([b], [y[lang]], s=42, color=LL, zorder=3, linewidths=1.5, edgecolors=SURFACE)
            ax.text(101, y[lang], f"{100 * st['answer changed']:.0f}%", va="center", ha="left", color=INK, fontsize=8.5)
        ax.axvline(25, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.text(25.5, -0.9, "chance", color=MUTED, fontsize=7.5, va="bottom")
        ax.text(101, -0.9, "answers\nchanged", color=MUTED, fontsize=7.5, va="bottom", ha="left")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Belebele accuracy (%)")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
        ax.set_title(repo.split("/")[1], loc="left", fontsize=10, fontweight="bold", color=INK, pad=22)
    axes[0].set_yticks([y[lang] for _, lang in rows], [NAMES[lang] for _, lang in rows])
    axes[0].set_ylim(max(y.values()) + 0.7, -1.2)
    for setting, label in (("as written", "As written"), ("decomposed", "Arriving decomposed (NFD)")):
        first = y[SETTINGS[setting][0]]
        axes[0].annotate(label, (0, first - 0.62), xycoords=("axes fraction", "data"), xytext=(-118, 0), textcoords="offset points",
                         fontsize=8, fontweight="bold", color=MUTED, va="center")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7, label=l)
               for c, l in ((PY, "Python's ids (NFC applied)"), (LL, "llama.cpp's ids (NFC skipped)"))]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, labelcolor=INK, fontsize=9)
    fig.text(0.01, 0.01, f"The first {results['questions per language']} Belebele questions per language, in the chat template. "
             "The same model, weights and code read both id sequences, so the tokenizer is the only difference.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.94), w_pad=4)
    fig.savefig(out, facecolor=SURFACE, dpi=160)
    fig.savefig(out.with_suffix(".png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def nfc(out: Path, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    results = measure(out, log)
    (out / "summary.md").write_text(summary(results))
    figure(results, out / "figure.svg")
    return results
