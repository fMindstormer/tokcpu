"""Does a Hub tokenizer give the same ids in JavaScript as in Python?

`check` runs a model's tokenizer through the JavaScript runtimes and through two Python references, on a
fixed battery of texts, and reports every place they differ: the ids from a default encode (special
tokens added), and the text from a decode of the reference ids (special tokens skipped, no clean-up).

There are two references because neither alone is the truth. The raw `tokenizers` library is what the
tokenizer.json says; `transformers`' AutoTokenizer is what Python users get, and adds model-specific
behaviour on top (CLIP's decode turns its "</w>" markers into spaces), as transformers.js does. Each
difference is marked by which references it differs from: from both (or from the only one that can load
the model) is a real disagreement between JavaScript and Python; from one only means the two Python
references disagree with each other, and the JavaScript runtimes side with the other one.

`compat` does that for the most-downloaded models on the Hub, testing each distinct tokenizer.json once
(by the Hub's file hash), deleting each download once it is checked, and writes one JSON of results. It
saves as it goes and, run again with the same versions, picks up where it stopped (the model list is kept
next to the results). `report` summarises a results file in Markdown and, given an earlier one, says which
models changed between them: .github/workflows/compat.yml runs both whenever a new release comes out.

    python -m tokcpu check openai/clip-vit-base-patch32 BAAI/bge-m3
    python -m tokcpu check path/to/a/folder/with/tokenizer.json
    python -m tokcpu compat --top 1000 --out build/compat/results.json
    python -m tokcpu report build/compat/results.json --previous old/results.json
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEXTS = {   # name -> (group, text)
    # everyday text
    "english": ("everyday", "The quick brown fox jumps over the lazy dog. It's 3:45pm, time for tea!"),
    "prose": ("everyday", "“Well,” she said — pausing… — “we’ll see.” Then, after a long while, she smiled and left the room; "
                          "nobody followed her (not even the dog)."),
    "numbers": ("everyday", "In 2024, revenue grew 12.5% to $3,400,000 (≈ €3.1M); call +1 (555) 010-9999 by 31/12."),
    "code": ("everyday", "def f(x):\n    return x**2  # square\n\nprint(f(3))\n"),
    "javascript": ("everyday", "const a = [1, 2, 3].map((x) => x * 2); // {}\nif (a !== null) { console.log(`${a}`); }"),
    "url": ("everyday", "Visit https://example.com/path?q=1&r=2#frag or mail me@example.org."),
    "markdown": ("everyday", "# Title\n\n- item **bold** _it_ `code`\n\n| a | b |\n|---|---|\n| 1 | 2 |"),
    "json": ("everyday", '{"name": "Ünïcødé", "values": [1, 2.5, null, true], "nested": {"k": "v"}}'),
    "whitespace": ("everyday", "a  b   c\t\td\n\n\ne  "),
    "windows-lines": ("everyday", "Line one\r\n\tLine two\r\nLine three"),
    "special-tokens": ("everyday", "<s> </s> <pad> <unk> <mask> [CLS] [SEP] [MASK] <|endoftext|> <|im_start|>user\nHi<|im_end|>"),
    # languages
    "german": ("languages", "Grüße aus München! Straße, Maß, Fußgängerübergänge."),
    "french": ("languages", "L'été dernier, nous sommes allés à Noël… « Bonjour », a dit Chloé."),
    "spanish-portuguese": ("languages", "¿Qué tal? ¡Muy bien! Ação, coração, pão e maçã."),
    "russian": ("languages", "Привет, мир! Съешь же ещё этих мягких французских булок."),
    "chinese": ("languages", "东京是日本的首都。今天天气很好！"),
    "japanese": ("languages", "東京は晴れ。カタカナとひらがな、漢字を混ぜます。"),
    "korean": ("languages", "안녕하세요, 세계! 한국어 문장입니다."),
    "arabic": ("languages", "مرحبا بالعالم! هذه جملة عربية مع تشكيل: كَتَبَ"),
    "hebrew": ("languages", "שלום עולם! זה משפט בעברית."),
    "hindi": ("languages", "नमस्ते दुनिया! यह एक हिंदी वाक्य है।"),
    "thai": ("languages", "สวัสดีชาวโลก ภาษาไทยไม่มีช่องว่างระหว่างคำ"),
    "vietnamese": ("languages", "Xin chào thế giới! Tiếng Việt có dấu."),
    "greek": ("languages", "Γειά σου Κόσμε! ΣΟΦΟΣ σοφός."),
    "turkish": ("languages", "İstanbul'da ılık bir gün; IŞIK ışık."),
    "mixed-scripts": ("languages", "ABC абв αβγ אבג 123 ١٢٣ ๑๒๓"),
    # emoji and Unicode
    "emoji": ("unicode", "Great job! 👍🏽🎉 😂😂 ❤️"),
    "emoji-sequences": ("unicode", "👩‍💻 deploys 🚀 to 🇺🇦 and 👨‍👩‍👧"),
    "decomposed": ("unicode", "été au café"),
    "fullwidth": ("unicode", "ＦＵＬＬ ｗｉｄｔｈ １２３"),
    "math-letters": ("unicode", "𝔸𝕓𝕔 𝟙𝟚𝟛 and 𝒳"),
    "zero-width": ("unicode", "zero​width‌non‍joiner⁠word"),
    "no-break-spaces": ("unicode", "Price: 100 €, distance 5 km"),
    "byte-order-mark": ("unicode", "﻿The quick brown fox."),
    "byte-order-mark-inside": ("unicode", "end of the first file.﻿Start of the second file."),
    "next-line": ("unicode", "first line\u0085second line"),
    "controls": ("unicode", "bell\x07 tab\t null\x00 escape\x1b[0m del\x7f"),
    "private-use": ("unicode", "privateuse \U000f0000"),
    # stress
    "long-word": ("stress", "pneumonoultramicroscopicsilicovolcanoconiosis" * 7),
    "long-run": ("stress", "a" * 300),
    "repeats": ("stress", "!!!!!!!!!!" * 6 + " " + "\n" * 12 + "ha" * 40),
    "single-space": ("stress", " "),
}
class Js:
    """fuzz/compat.mjs, the JavaScript runtimes, one process each (on Linux, two transformers.js versions'
    onnxruntime builds cannot share a process), answering as one."""

    def __init__(self, node: str = "node"):
        self.parts = [Node(node, which) for which in ("tj3", "tj4", "tokenizers")]

    def ask(self, **req):
        for p in self.parts:
            p.send(req)
        answers = [p.receive() for p in self.parts]
        if req["op"] == "versions":
            return answers[0] | {"jsonParseBug": any(a["jsonParseBug"] for a in answers),
                                 "runtimes": [rt for a in answers for rt in a["runtimes"]]}
        if failed := next((a for a in answers if "error" in a), None):
            return failed
        if req["op"] == "load":
            return {"runtimes": [rt for a in answers for rt in a["runtimes"]], "errors": {k: v for a in answers for k, v in a["errors"].items()}}
        return {k: v for a in answers for k, v in a.items()}

    def close(self):
        for p in self.parts:
            p.close()


class Node:
    """One fuzz/compat.mjs process."""

    def __init__(self, node: str, *runtimes: str):
        self.err = tempfile.TemporaryFile()
        self.p = subprocess.Popen([node, str(ROOT / "fuzz" / "compat.mjs"), *runtimes], cwd=ROOT / "fuzz", text=True,
                                  encoding="utf-8", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err)

    def send(self, req: dict):
        try:
            self.p.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.p.stdin.flush()
        except BrokenPipeError:   # receive says why
            pass

    def receive(self) -> dict:
        line = self.p.stdout.readline()
        if not line:
            self.p.wait(10)
            self.err.seek(0)
            raise RuntimeError(f"fuzz/compat.mjs exited ({self.p.returncode}): {self.err.read().decode(errors='replace')[-2000:]}")
        return json.loads(line)

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(10)
        except Exception:   # it died, or hangs
            self.p.kill()
        self.err.close()


def first_difference(a: list, b: list) -> int:
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))


def cause(text_name: str, ref_tokens: list[str], js_tokens: list[str], at: int = 0) -> str:
    """A best guess at which known difference this is, from what the two sides produced (`at` is the
    first difference, as an index into both token lists)."""
    if any(t.startswith("<0x") for t in ref_tokens) and not any(t.startswith("<0x") for t in js_tokens):
        return "byte fallback ignored"
    if text_name in ("byte-order-mark", "byte-order-mark-inside", "next-line"):
        return "whitespace (U+FEFF, U+0085)"
    if at < len(ref_tokens) and at < len(js_tokens) and ref_tokens[at].startswith("▁") \
            and not js_tokens[at].startswith("▁") and all(t.startswith("<") for t in ref_tokens[:at]):
        return "no leading ▁"
    # a final ς where Python has σ: as itself, or as its last byte in ByteLevel's alphabet (0x82 is Ĥ)
    if text_name == "greek" and any("ς" in t or "Ĥ" in t for t in js_tokens[at:at + 1]):
        return "final sigma (toLowerCase)"
    return "other"


VERDICTS = {   # which references a difference is against -> what it says
    "both": "JavaScript differs from Python",
    "transformers": "the Python references disagree; JavaScript sides with tokenizers",
    "tokenizers": "the Python references disagree; JavaScript sides with transformers",
}


def references(d: Path) -> tuple[dict, object]:
    """The two Python references: the raw `tokenizers` library, and `transformers`' AutoTokenizer, which
    adds model-specific behaviour on top. Each maps to (encode, decode) functions, or to an error."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, logging
    logging.set_verbosity_error()
    refs = {}
    raw = None
    try:
        raw = Tokenizer.from_file(str(d / "tokenizer.json"))
        raw.no_padding()
        raw.no_truncation()
        refs["tokenizers"] = (lambda t: raw.encode(t).ids, lambda ids: raw.decode(ids, skip_special_tokens=True))
    except Exception as e:
        refs["tokenizers"] = f"cannot load it: {str(e)[:200]}"
    try:
        try:
            tf = AutoTokenizer.from_pretrained(str(d), trust_remote_code=False)   # never run a repo's code
        except Exception:   # a model config it cannot read (timm's, say): the tokenizer config alone may do
            if not (d / "config.json").exists():
                raise
            with tempfile.TemporaryDirectory() as bare:
                for f in d.iterdir():
                    if f.is_file() and f.name != "config.json":
                        shutil.copy(f, bare)
                tf = AutoTokenizer.from_pretrained(bare, trust_remote_code=False)
        refs["transformers"] = (tf.encode, lambda ids: tf.decode(ids, skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False))
    except Exception as e:
        refs["transformers"] = f"cannot load it: {type(e).__name__}: {str(e)[:200]}"
    return refs, raw


def check_dir(d: Path, js: Js) -> dict:
    """Compare one tokenizer folder in every JavaScript runtime against both Python references. Decoding
    skips special tokens and turns clean-up off on both sides: the clean-up default has changed between
    versions, and it would hide what the decoder itself does."""
    refs, raw = references(d)
    ok = {k: v for k, v in refs.items() if not isinstance(v, str)}
    out = {"model": _kind(json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))),
           "references": {k: (v if isinstance(v, str) else "ok") for k, v in refs.items()}, "runtimes": {}}
    if not ok:
        return out | {"error": "neither Python reference can load it"}
    names = list(TEXTS)
    texts = [TEXTS[n][1] for n in names]
    want = {k: [enc(t) for t in texts] for k, (enc, _) in ok.items()}
    want_text = {k: [ok[k][1](ids) for ids in want[k]] for k in ok}
    token = lambda i: raw.id_to_token(i) if raw is not None and isinstance(i, int) else str(i)
    load = js.ask(op="load", dir=str(d.resolve()))
    if "errors" not in load:
        return out | {"error": f"node could not read it: {load.get('error')}"}
    enc = js.ask(op="encode", texts=texts)
    dec = {k: js.ask(op="decode", ids=want[k]) for k in ok}
    for rt in load["runtimes"]:
        if rt in load["errors"]:
            out["runtimes"][rt] = {"load_error": load["errors"][rt]}
            continue
        diffs = {}
        for i, name in enumerate(names):
            got = enc[rt][i]
            per_ref = {}
            for k in ok:
                if isinstance(got, dict):
                    per_ref[k] = {"kind": "encode", "error": got["error"]}
                elif got != want[k][i]:
                    at = first_difference(want[k][i], got)
                    a = [token(x) for x in want[k][i][max(0, at - 2):at + 4]]
                    b = [token(x) for x in got[max(0, at - 2):at + 4]]
                    per_ref[k] = {"kind": "encode", "at": at, "python": a, "js": b,
                                  "cause": cause(name, a, b, min(at, 2))}
                else:
                    g = dec[k][rt][i]
                    if isinstance(g, dict):
                        per_ref[k] = {"kind": "decode", "error": g["error"]}
                    elif g != want_text[k][i]:
                        per_ref[k] = {"kind": "decode", "python": want_text[k][i][:200], "js": g[:200],
                                      "cause": cause(name, [], [])}
            if per_ref:
                first = per_ref.get("transformers") or per_ref["tokenizers"]
                diffs[name] = first | {"against": sorted(per_ref),
                                       "verdict": "both" if len(per_ref) == len(ok) else next(iter(per_ref))}
        out["runtimes"][rt] = {"differences": diffs}
    return out


def _types(x: dict | None) -> str | None:
    """A component's type, with a Sequence's parts in brackets: "Sequence(NFC,Lowercase)"."""
    if x is None:
        return None
    parts = x.get("normalizers") or x.get("pretokenizers") or x.get("decoders")
    return x["type"] + (f"({','.join(filter(None, map(_types, parts)))})" if parts else "")


def _kind(spec: dict) -> dict:
    """The parts of a tokenizer that say what kind it is, for the report."""
    m = spec["model"]
    kind = m.get("type") or ("BPE" if "merges" in m else "Unigram" if isinstance(m.get("vocab"), list)
                             else "WordPiece" if "max_input_chars_per_word" in m else "WordLevel")
    return {"model": kind, "normalizer": _types(spec.get("normalizer")),
            "pre_tokenizer": _types(spec.get("pre_tokenizer")), "decoder": _types(spec.get("decoder")),
            "byte_fallback": bool(m.get("byte_fallback"))}


def download(repo: str, to: Path) -> Path | str:
    """tokenizer.json, and the small files AutoTokenizer reads next to it."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, GatedRepoError
    try:
        hf_hub_download(repo, "tokenizer.json", local_dir=to)
    except GatedRepoError:
        return "gated"
    except EntryNotFoundError:
        return "no tokenizer.json"
    except Exception as e:   # rate limits that outlast the retries, network errors: try again next run
        return f"download failed: {type(e).__name__}"
    for name in ("tokenizer_config.json", "special_tokens_map.json", "config.json"):
        try:
            hf_hub_download(repo, name, local_dir=to)
        except Exception:
            pass
    if not (to / "tokenizer_config.json").exists():
        (to / "tokenizer_config.json").write_text("{}")
    return to


def versions(js: Js) -> dict:
    import tokenizers
    import transformers
    v = js.ask(op="versions")
    return {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            "node": v["node"], "v8": v["v8"],
            "json_parse_bug": v["jsonParseBug"], "runtimes": v["runtimes"]}


def warn_v8(v: dict, log=print):
    if v["json_parse_bug"]:
        log(f"warning: this Node ({v['node']}) has V8's JSON.parse bug (nodejs/node#63785), which can corrupt a\n"
            "tokenizer loaded after another one; differences may be spurious. Use --node with Node 22.")


def check(repos: list[str], node: str = "node", log=print) -> int:
    js = Js(node)
    warn_v8(versions(js), log)
    bad = 0
    for repo in repos:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(repo) if (Path(repo) / "tokenizer.json").exists() else download(repo, Path(tmp))
            if isinstance(d, str):
                log(f"{repo}: skipped ({d})")
                continue
            r = check_dir(d, js)
        if "error" in r:
            log(f"{repo}: {r['error']}")
            continue
        for k, v in r["references"].items():
            if v != "ok":
                log(f"{repo}: Python {k} {v}")
        log(f"\n{repo}  ({r['model']['model']}, normalizer {r['model']['normalizer']}, pre-tokenizer {r['model']['pre_tokenizer']})")
        for rt, res in r["runtimes"].items():
            if "load_error" in res:
                bad += 1
                log(f"  {rt}: cannot load it: {res['load_error']}")
                continue
            diffs = res["differences"]
            real = sum(d_["verdict"] == "both" for d_ in diffs.values())
            bad += bool(real)
            log(f"  {rt}: " + ("same on all texts" if not diffs else
                               f"differs from Python on {real} of {len(TEXTS)} texts"
                               + (f"; the references disagree on {len(diffs) - real} more" if len(diffs) > real else "")))
            for name, d_ in sorted(diffs.items(), key=lambda x: x[1]["verdict"] != "both"):
                vs = "vs " + " and ".join(d_["against"])
                if "error" in d_:
                    log(f"    {name:24s} {d_['kind']} error ({vs}): {d_['error'][:100]}")
                elif d_["kind"] == "encode":
                    log(f"    {name:24s} ids differ at token {d_['at']} ({vs}): python {d_['python']} js {d_['js']}  [{d_['cause']}]")
                else:
                    log(f"    {name:24s} decode ({vs}): python {d_['python'][:60]!r} js {d_['js'][:60]!r}")
    js.close()
    return 1 if bad else 0


def log_now(*a):
    print(*a, flush=True)


def hub_models(top: int, cache: Path, log=log_now) -> list[dict]:
    """The `top` most-downloaded models, each with its tokenizer.json's hash (or why it has none)."""
    if cache.exists():
        models = json.loads(cache.read_text())
        if len(models) >= top:
            return models[:top]
    from huggingface_hub import HfApi
    api = HfApi()
    models = [{"repo": m.id, "downloads": m.downloads, "likes": m.likes, "pipeline": m.pipeline_tag,
               "library": m.library_name} for m in api.list_models(sort="downloads", limit=top)]

    def blob(m):
        try:
            info = {p.path: p for p in api.get_paths_info(m["repo"], ["tokenizer.json"])}
            p = info.get("tokenizer.json")
            if p is None:
                return m | {"skip": "no tokenizer.json"}
            return m | {"tokenizer": (p.lfs.sha256 if p.lfs else p.blob_id), "size": p.size}
        except Exception as e:
            return m | {"skip": "gated" if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e) else str(e)[:80]}
    with ThreadPoolExecutor(8) as ex:
        models = list(ex.map(blob, models))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(models))
    return models


def compat(top: int, out: Path, node: str = "node", log=log_now) -> dict:
    """Check the `top` most-downloaded models; one test per distinct tokenizer.json."""
    models = hub_models(top, out.with_name(out.stem + "-models.json"), log)
    distinct = {}
    for m in models:
        if "tokenizer" in m:
            distinct.setdefault(m["tokenizer"], m["repo"])
    js = Js(node)
    v = versions(js)
    warn_v8(v, log)
    old = json.loads(out.read_text()) if out.exists() else None
    results = {}
    if old and same_versions(old["versions"]) == same_versions(v):
        results = {h: r for h, r in old["tokenizers"].items() if not str(r.get("error", "")).startswith("download failed")}
    elif old:
        log(f"{out} was made with other versions; starting again")
    log(f"{len(models)} models; {sum('tokenizer' in m for m in models)} have a tokenizer.json; "
        f"{len(distinct)} distinct files; {len(results)} already checked")

    def save():
        data = {"generated": time.strftime("%Y-%m-%d"), "versions": v,
                "texts": {k: {"group": g, "text": t} for k, (g, t) in TEXTS.items()},
                "models": models, "tokenizers": results}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(out)
        return data
    todo = [(h, repo) for h, repo in distinct.items() if h not in results]
    for n, (h, repo) in enumerate(todo):
        with tempfile.TemporaryDirectory() as tmp:
            d = download(repo, Path(tmp))
            try:
                results[h] = {"error": d} if isinstance(d, str) else check_dir(d, js)
            except Exception as e:   # one odd tokenizer must not end the run
                results[h] = {"error": f"check failed: {type(e).__name__}: {str(e)[:200]}"}
                js.close()
                js = Js(node)
        if (n + 1) % 10 == 0:
            save()
            log(f"  {n + 1} / {len(todo)} checked")
    js.close()
    return save()


def same_versions(v: dict) -> tuple:
    """What has to match for one run to continue another."""
    return v["transformers"], v["tokenizers"], v["node"], tuple(v["runtimes"])


def short(rt: str) -> str:
    """A runtime's column in the tables, the same across its minor releases: "transformers.js@4.3.1" -> "tj4"."""
    name, version = rt.rsplit("@", 1)
    if name == "transformers.js":
        return f"tj{version.split('.')[0]}"
    return {"@huggingface/tokenizers": "tokenizers_js", "llama.cpp": "llama_cpp"}.get(name, name)


def _cause(text_name: str, diff: dict, js: bool = True) -> str:
    if not js:   # gguf's causes, tested when it ran
        return diff.get("cause") or "other"
    if "error" in diff:
        return "throws on " + diff["kind"] + ("" if "empty" not in diff["error"] else " of no ids")
    if diff["kind"] == "encode":   # again, so results saved by an older version get today's rules
        return cause(text_name, diff["python"], diff["js"], min(diff["at"], 2))
    return diff.get("cause") or "other"


def tables(data: dict) -> dict[str, list[dict]]:
    """The results as flat tables for the dashboard: one row per model, and one per difference."""
    texts = data["texts"]
    js = data.get("kind") != "gguf"
    models, diffs = [], []
    by_hash = {}
    for m in data["models"]:
        if "tokenizer" in m:
            by_hash.setdefault(m["tokenizer"], []).append(m)
    for m in data["models"]:
        if "tokenizer" not in m:
            continue
        r = data["tokenizers"].get(m["tokenizer"])
        if r is None:
            continue
        row = {"repo": m["repo"], "downloads": m["downloads"], "pipeline": m["pipeline"] or "", "library": m["library"] or "",
               "tokenizer": m["tokenizer"][:12], "shared_with": len(by_hash[m["tokenizer"]]) - 1}
        if "error" in r and "runtimes" not in r:
            models.append(row | {"checked": False, "note": r["error"]})
            continue
        refs = r["references"]
        row |= {"checked": True, "note": "", "model": r["model"]["model"], "normalizer": r["model"]["normalizer"] or "",
                "pre_tokenizer": r["model"]["pre_tokenizer"] or "", "decoder": r["model"]["decoder"] or "",
                "transformers_loads": refs.get("transformers") == "ok"}
        causes = set()
        for rt in data["versions"]["runtimes"]:
            col = short(rt)
            res = r["runtimes"].get(rt, {})
            if "load_error" in res:
                row |= {col: "cannot load", f"{col}_texts": None, f"{col}_disputed": None}
                causes.add("cannot load")
                continue
            d = res.get("differences", {})
            real = {k: v for k, v in d.items() if v["verdict"] == "both"}
            row |= {col: "differs" if real else "same", f"{col}_texts": len(real), f"{col}_disputed": len(d) - len(real),
                    f"{col}_causes": sorted({_cause(k, v, js) for k, v in real.items()})}
            causes |= {_cause(k, v, js) for k, v in real.items()}
            if m["repo"] == by_hash[m["tokenizer"]][0]["repo"]:   # each tokenizer's differences once
                for name, v in d.items():
                    diffs.append({"repo": m["repo"], "downloads": sum(x["downloads"] for x in by_hash[m["tokenizer"]]),
                                  "models": len(by_hash[m["tokenizer"]]), "runtime": col, "text": name,
                                  "group": texts[name]["group"], "kind": v["kind"],
                                  "verdict": v["verdict"], "cause": _cause(name, v, js),
                                  "python": json.dumps(v.get("python", v.get("error", "")), ensure_ascii=False)[:160],
                                  "js": json.dumps(v.get("js", ""), ensure_ascii=False)[:160]})
        row["causes"] = ", ".join(sorted(causes))
        models.append(row)
    return {"models": models, "differences": diffs,
            "texts": [{"text": k, "group": v["group"], "content": v["text"]} for k, v in texts.items()]}


def summary(data: dict, previous: dict | None = None, most: int = 15) -> str:
    """The results in Markdown: per runtime, how many models match Python and why the others don't; and,
    given an earlier results file, which models changed between the two (among models both checked)."""
    v, rows = data["versions"], [r for r in tables(data)["models"] if r["checked"]]
    cols = {short(rt): rt for rt in v["runtimes"]}
    listed = data["models"]
    which = (f"Of the top {len(listed)} GGUF repos, {sum('tokenizer' in m for m in listed)} name a base model"
             if data.get("kind") == "gguf" else
             f"Of the top {len(listed)} models, {sum('tokenizer' in m for m in listed)} have a tokenizer.json "
             f"({len(data['tokenizers'])} distinct files)")
    out = [f"# Tokenizer compatibility, {data['generated']}", "",
           f"`transformers` {v['transformers']} and `tokenizers` {v['tokenizers']} in Python against "
           + ", ".join(f"`{rt}`" for rt in v["runtimes"]) + (f" on Node {v['node']}" if v.get("node") else "")
           + f". {which}; {len(rows)} were checked on {len(data['texts'])} texts.", ""]
    if v["json_parse_bug"]:
        out += ["**This Node has V8's JSON.parse bug (nodejs/node#63785); differences may be spurious.**", ""]
    out += ["| Runtime | Same | Differs | Cannot load | Same, by downloads |", "|---|---|---|---|---|"]
    total = sum(r["downloads"] for r in rows) or 1
    for col, rt in cols.items():
        n = lambda s: sum(r.get(col) == s for r in rows)
        same = sum(r["downloads"] for r in rows if r.get(col) == "same") / total
        out.append(f"| {rt} | {n('same')} | {n('differs')} | {n('cannot load')} | {same:.1%} |")
    causes = sorted({c for r in rows for col in cols for c in r.get(f"{col}_causes") or []}
                    | ({"cannot load"} if any(r.get(col) == "cannot load" for r in rows for col in cols) else set()))
    out += ["", "Models affected, by cause:", "", "| Cause | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for c in causes:
        out.append(f"| {c} | " + " | ".join(str(sum(_has(r, col, c) for r in rows)) for col in cols) + " |")
    if previous is None:
        return "\n".join(out) + "\n"

    pv = previous["versions"]
    was = {r["repo"]: r for r in tables(previous)["models"] if r["checked"]}
    changed = [f"`{k}` {pv[k]} → {v[k]}" for k in ("transformers", "tokenizers", "node") if pv[k] != v[k]]
    changed += [f"`{rt}` → `{cols[short(rt)]}`" if short(rt) in cols else f"`{rt}` dropped" for rt in pv["runtimes"]
                if rt not in v["runtimes"]]
    out += ["", f"## Since {previous['generated']}", "",
            ("Changed: " + "; ".join(changed) + "." if changed else "The same versions as before.")
            + f" {sum(r['repo'] in was for r in rows)} models were checked both times.", ""]
    for col in cols:
        moves = {"now match": [], "now differ": [], "differ for other reasons": []}
        for r in sorted(rows, key=lambda r: -r["downloads"]):
            w = was.get(r["repo"])
            if w is None or col not in w:
                continue
            now, then = r.get(col), w.get(col)
            if now == "same" and then != "same":
                moves["now match"].append(f"{r['repo']} (was: {_list(w.get(f'{col}_causes')) or then})")
            elif now != "same" and then == "same":
                moves["now differ"].append(f"{r['repo']} ({_list(r.get(f'{col}_causes')) or now})")
            elif now == then == "differs" and r[f"{col}_causes"] != w[f"{col}_causes"]:
                moves["differ for other reasons"].append(f"{r['repo']} ({_list(w[f'{col}_causes'])} → {_list(r[f'{col}_causes'])})")
        out.append(f"**{cols[col]}**: " + ("; ".join(f"{len(m)} {what}" for what, m in moves.items() if m) or "no change") + ".")
        for what, m in moves.items():
            if m:
                out += ["", f"{len(m)} {what}:", ""] + [f"- {x}" for x in m[:most]] + ([f"- and {len(m) - most} more"] if len(m) > most else [])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _list(causes: list[str] | None) -> str:
    return "; ".join(causes or [])


def _has(row: dict, col: str, cause: str) -> bool:
    return row.get(col) == "cannot load" if cause == "cannot load" else cause in (row.get(f"{col}_causes") or [])
