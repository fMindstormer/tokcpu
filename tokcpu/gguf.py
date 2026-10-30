"""Does a GGUF on the Hub tokenize like the model it was made from?

llama.cpp, and everything built on it (Ollama, LM Studio, llama-cpp-python), does not read tokenizer.json:
a converter copies the vocabulary into the GGUF file once, with a guess at which pre-tokenizer it needs, and
llama.cpp's own code does the rest. `gguf` checks that on the most-downloaded GGUF repos. Each one is
compared with its base model (the repo its `base_model:quantized:` tag names) on compat's 42 texts and
both Python references, exactly as `compat` compares the JavaScript runtimes.

Only the start of each GGUF is downloaded: the header holds every key and value, the vocabulary included,
before the first tensor. It is saved as a GGUF with no tensors, which llama.cpp loads with vocab_only.
llama.cpp itself runs in fuzz/llama_tok.c, built here against an installed llama.cpp (`brew install
llama.cpp`, or LLAMA_CPP=its prefix).

    python -m tokcpu gguf --top 100 --out build/gguf/results.json
    python -m tokcpu gguf bartowski/Llama-3.2-3B-Instruct-GGUF     one repo, printed
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

from .compat import ROOT, TEXTS, check_dir, download, log_now

SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}   # GGUF type -> struct
STRING, ARRAY = 8, 9


def llama_prefix() -> Path:
    for p in (os.environ.get("LLAMA_CPP"), shutil.which("llama-cli") and Path(shutil.which("llama-cli")).resolve().parent.parent):
        if p and (Path(p) / "include" / "llama.h").exists():
            return Path(p)
    try:
        return Path(subprocess.run(["brew", "--prefix", "llama.cpp"], capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        raise SystemExit("llama.cpp not found: `brew install llama.cpp`, or set LLAMA_CPP to where it is installed")


def build_helper() -> tuple[Path, str]:
    """fuzz/llama_tok.c, compiled against the installed llama.cpp; and that llama.cpp's version."""
    p = llama_prefix()
    pc = (p / "lib" / "pkgconfig" / "llama.pc").read_text() if (p / "lib" / "pkgconfig" / "llama.pc").exists() else ""
    version = next((line.split(":", 1)[1].strip() for line in pc.splitlines() if line.startswith("Version:")), "unknown")
    out = ROOT / "build" / f"llama_tok-{version}"
    if not out.exists():
        incs, libs = [p / "include"], [p / "lib"]
        brew = Path(os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew"))
        for ggml in (p, p.parent / "ggml", brew / "opt" / "ggml", Path("/usr/local/opt/ggml")):   # Homebrew keeps ggml apart
            if (ggml / "include" / "ggml.h").exists():
                if ggml != p:
                    incs.append(ggml / "include")
                    libs.append(ggml / "lib")
                break
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cc", "-O2", str(ROOT / "fuzz" / "llama_tok.c"), *(f"-I{i}" for i in incs), *(f"-L{x}" for x in libs),
                        "-lllama", *(f"-Wl,-rpath,{x}" for x in libs), "-o", str(out)], check=True)
    return out, version


class Llama:
    """fuzz/llama_tok.c, answering the requests compat's check_dir sends to the JavaScript runtimes, for the
    vocab.gguf in the folder it is given."""

    def __init__(self):
        exe, version = build_helper()
        self.name = f"llama.cpp@{version}"
        self.p = subprocess.Popen([str(exe)], text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.n_vocab = 0
        self.model = ""   # the loaded GGUF's tokenizer.ggml.model, set by check_gguf

    def _ask(self, line: str) -> str:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        answer = self.p.stdout.readline()
        if not answer:
            raise RuntimeError(f"llama_tok exited ({self.p.wait(10)})")
        return answer.rstrip("\n")

    def ask(self, op: str, **req):
        if op == "load":
            a = self._ask(f"load {Path(req['dir']) / 'vocab.gguf'}")
            self.n_vocab = int(a.split()[1]) if a.startswith("ok") else 0
            return {"runtimes": [self.name], "errors": {} if a.startswith("ok") else {self.name: a[6:]}}
        if op == "encode":
            return {self.name: [self._encode(t) for t in req["texts"]]}
        if op == "decode":
            return {self.name: [self._decode(ids) for ids in req["ids"]]}
        raise ValueError(op)

    def _encode(self, text: str):
        a = self._ask(f"encode {text.encode().hex()}")
        return [int(x) for x in a.split()[1:]] if a.startswith("ids") else {"error": a[6:]}

    def _decode(self, ids: list[int]):
        if any(not 0 <= i < self.n_vocab for i in ids):   # llama.cpp would abort
            return {"error": f"an id past the GGUF's {self.n_vocab} tokens"}
        a = self._ask("decode " + " ".join(map(str, ids)))
        return bytes.fromhex(a[5:]).decode(errors="replace") if a.startswith("text") else {"error": a[6:]}

    def close(self):
        self.p.stdin.close()
        self.p.wait(10)


def read_header(read) -> tuple[bytes, dict]:
    """A GGUF's header, cut before the tensor infos with its tensor count set to 0 and without the split.*
    keys of a model in several files, and its keys and values (arrays of more than 16 items summarised by
    their length). `read(start, end)` gives the file's bytes."""
    buf = bytearray()

    def need(end):
        while len(buf) < end:
            more = read(len(buf), max(end, 2 * len(buf), 1 << 22))
            if not more:
                raise ValueError("the file ends inside its header")
            buf.extend(more)

    def take(at, n):
        need(at + n)
        return bytes(buf[at:at + n]), at + n

    def value(at, t):
        if t == STRING:
            n, at = take(at, 8)
            s, at = take(at, struct.unpack("<Q", n)[0])
            return s.decode(errors="replace"), at
        if t == ARRAY:
            h, at = take(at, 12)
            it, n = struct.unpack("<IQ", h)
            items = []
            if it in SCALARS and n > 16:   # skip fixed-size items in one go
                size = struct.calcsize(SCALARS[it])
                need(at + n * size)
                return f"[{n} items]", at + n * size
            for _ in range(n):
                x, at = value(at, it)
                if len(items) < 16:
                    items.append(x)
            return (items if n <= 16 else f"[{n} items]"), at
        raw, at = take(at, struct.calcsize(SCALARS[t]))
        return struct.unpack("<" + SCALARS[t], raw)[0], at

    head, at = take(0, 24)
    magic, version, _tensors, kvs = struct.unpack("<4sIQQ", head)
    if magic != b"GGUF" or version < 2:
        raise ValueError(f"not a GGUF v2+ file: {magic!r} v{version}")
    meta, kept = {}, []
    for _ in range(kvs):
        start = at
        n, at = take(at, 8)
        key, at = take(at, struct.unpack("<Q", n)[0])
        t, at = take(at, 4)
        meta[key.decode()], at = value(at, struct.unpack("<I", t)[0])
        if not key.startswith(b"split."):
            kept.append(bytes(buf[start:at]))
    out = bytearray(struct.pack("<4sIQQ", magic, version, 0, len(kept)) + b"".join(kept))
    align = int(meta.get("general.alignment", 32))
    out.extend(b"\0" * (-len(out) % align))
    return bytes(out), meta


def gguf_files(api, repo: str) -> list[str]:
    """The repo's GGUFs, smallest first, leaving out vision projectors, importance matrices and later shards."""
    files = [f for f in api.list_repo_tree(repo, recursive=True) if f.path.endswith(".gguf") and getattr(f, "size", None)]
    files = [f for f in files if not any(w in f.path.lower() for w in ("mmproj", "imatrix"))
             and ("-of-0" not in f.path or "-00001-of-" in f.path)]
    return [f.path for f in sorted(files, key=lambda f: f.size)]


def fetch_vocab(repo: str, file: str, to: Path, revision: str | None = None) -> dict:
    """Save the GGUF's header as `to`/vocab.gguf; its keys and values."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    with fs.open(f"{repo}@{revision}/{file}" if revision else f"{repo}/{file}", "rb", block_size=0) as f:
        def read(start, end):
            f.seek(start)
            return f.read(end - start)
        header, meta = read_header(read)
    (to / "vocab.gguf").write_bytes(header)
    return meta


def base_of(tags: list[str]) -> str | None:
    """The model a GGUF repo was made from, by the Hub's tags: "base_model:quantized:org/name"."""
    for prefix in ("base_model:quantized:", "base_model:"):
        for t in tags:
            rest = t[len(prefix):]
            if t.startswith(prefix) and "/" in rest and ":" not in rest:
                return rest
    return None


GGUF_KEYS = ("tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
             "tokenizer.ggml.add_space_prefix", "general.architecture", "general.name")


def check_gguf(repo: str, base: str, llama: Llama, api) -> dict:
    """One GGUF repo against its base model's tokenizer."""
    from huggingface_hub.utils import GatedRepoError
    files = gguf_files(api, repo)
    if not files:
        return {"error": "no GGUF file"}
    with tempfile.TemporaryDirectory() as tmp:
        for file in files[:4]:   # the smallest may be a draft model, or something else with no vocabulary
            try:
                meta = fetch_vocab(repo, file, Path(tmp))
            except GatedRepoError:
                return {"error": "gated"}
            if meta.get("tokenizer.ggml.model", "none") != "none":
                break
        info = {"file": file, "gguf": {k: meta.get(k) for k in GGUF_KEYS}}
        llama.model = meta.get("tokenizer.ggml.model", "")
        if meta.get("tokenizer.ggml.model", "none") == "none":   # llama.cpp aborts on a vocabulary of type none
            return info | {"error": "no vocabulary in the GGUF (not a language model)"}
        d = download(base, Path(tmp))
        if isinstance(d, str):
            return info | {"error": f"base model: {d}"}
        r = check_dir(d, llama) | info
        for name, diff in r["runtimes"].get(llama.name, {}).get("differences", {}).items():
            diff["cause"] = explain(name, diff, d, llama)
    error = r["runtimes"][llama.name].get("load_error", "")
    if error and "vocabulary" not in error:   # an architecture, or hyperparameters, this llama.cpp cannot run at all
        return info | {"error": f"llama.cpp does not support it: {error.split('error loading model: ')[-1]}"}
    return r


def explain(text_name: str, diff: dict, d: Path, llama: Llama) -> str:
    """Which known difference this is, tested where it can be: is llama.cpp's encoding Python's with the
    BOS or EOS token added or left out, or Python's without the tokenizer's normalizer?"""
    if "error" in diff:
        return f"throws on {diff['kind']}"
    if diff["kind"] == "decode":
        return label(text_name, diff, "")
    from tokenizers import Tokenizer
    text = TEXTS[text_name][1]
    got = llama._encode(text)
    raw = Tokenizer.from_file(str(d / "tokenizer.json"))
    raw.no_padding()
    raw.no_truncation()
    want = raw.encode(text).ids
    if isinstance(got, list):
        if got in (want[1:], want[:-1]) or want in (got[1:], got[:-1]):
            return "BOS/EOS added differently"
        if raw.normalizer is not None:
            raw.normalizer = None
            if raw.encode(text).ids == got:
                return "normalizer not applied"
    if text_name == "special-tokens":
        return "special tokens in text"
    return label(text_name, diff, llama.model)


CLEAN_UP = [(" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"),
            (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")]   # transformers' clean_up_tokenization


def label(text_name: str, diff: dict, model: str) -> str:
    """The causes that show in a difference itself (`model` is the GGUF's tokenizer.ggml.model)."""
    if diff["kind"] == "decode":
        py, got = diff["python"], diff["js"]
        if got == " " + py:
            return "decode adds a leading space"
        for a, b in CLEAN_UP:
            py = py.replace(a, b)
        return "decode cleans up spaces" if py == got else "decode"
    if model == "gpt2" and text_name in ("hindi", "thai"):   # a virama or vowel sign put with the other letter
        return "pre-tokenizer: Hindi and Thai marks"
    if model == "gpt2" and text_name == "windows-lines":
        return "pre-tokenizer: \\r\\n"
    return "other"


def gguf_repos(top: int, cache: Path) -> list[dict]:
    """The `top` most-downloaded GGUF repos, each with its base model (or why it has none)."""
    if cache.exists():
        repos = json.loads(cache.read_text())
        if len(repos) >= top:
            return repos[:top]
    from huggingface_hub import HfApi
    repos = []
    for m in HfApi().list_models(filter="gguf", sort="downloads", limit=top):
        base = base_of(m.tags or [])
        row = {"repo": m.id, "downloads": m.downloads, "likes": m.likes, "pipeline": m.pipeline_tag, "library": "gguf"}
        repos.append(row | ({"tokenizer": m.id, "base": base} if base else {"skip": "no base model"}))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(repos))
    return repos


def campaign(top: int, out: Path, log=log_now) -> dict:
    """Check the `top` most-downloaded GGUF repos; results shaped like compat's, one "tokenizer" per repo, so
    `report` reads them too."""
    import tokenizers
    import transformers
    from huggingface_hub import HfApi
    api = HfApi()
    repos = gguf_repos(top, out.with_name(out.stem + "-repos.json"))
    llama = Llama()
    v = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "node": None,
         "json_parse_bug": False, "runtimes": [llama.name]}
    old = json.loads(out.read_text()) if out.exists() else None
    results = {}
    if old and old["versions"] == v:
        results = {k: r for k, r in old["tokenizers"].items() if not str(r.get("error", "")).startswith(("download failed", "check failed"))}
    log(f"{len(repos)} GGUF repos; {sum('base' in r for r in repos)} name a base model; {len(results)} already checked")

    def save():
        data = {"kind": "gguf", "generated": time.strftime("%Y-%m-%d"), "versions": v,
                "texts": {k: {"group": g, "text": t} for k, (g, t) in TEXTS.items()}, "models": repos, "tokenizers": results}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(out)
        return data
    todo = [r for r in repos if "base" in r and r["repo"] not in results]
    for n, r in enumerate(todo):
        try:
            results[r["repo"]] = check_gguf(r["repo"], r["base"], llama, api)
        except Exception as e:   # one odd repo must not end the run
            results[r["repo"]] = {"error": f"check failed: {type(e).__name__}: {str(e)[:200]}"}
            llama.close()
            llama = Llama()
        if (n + 1) % 10 == 0:
            save()
            log(f"  {n + 1} / {len(todo)} checked")
    llama.close()
    return save()


def check(repos: list[str], log=print) -> int:
    """`gguf REPO...`: print each GGUF repo's differences from its base model."""
    from huggingface_hub import HfApi
    api = HfApi()
    llama = Llama()
    bad = 0
    for repo in repos:
        base = base_of(api.model_info(repo).tags or [])
        if base is None:
            log(f"{repo}: no base_model tag")
            continue
        r = check_gguf(repo, base, llama, api)
        if "error" in r:
            log(f"{repo}: {r['error']}")
            continue
        g = r["gguf"]
        log(f"\n{repo} ({r['file']}, {g['tokenizer.ggml.model']}/{g['tokenizer.ggml.pre']}) against {base}")
        for rt, res in r["runtimes"].items():
            if "load_error" in res:
                bad += 1
                log(f"  {rt}: cannot load it: {res['load_error']}")
                continue
            diffs = res["differences"]
            bad += any(d["verdict"] == "both" for d in diffs.values())
            log(f"  {rt}: " + ("same on all texts" if not diffs else f"differs on {len(diffs)} of {len(TEXTS)} texts"))
            for name, d in diffs.items():
                vs = "vs " + " and ".join(d["against"])
                if "error" in d:
                    log(f"    {name:24s} {d['kind']} error ({vs}): {d['error'][:100]}")
                elif d["kind"] == "encode":
                    log(f"    {name:24s} ids differ at token {d['at']} ({vs}): python {d['python']} llama.cpp {d['js']}  [{d['cause']}]")
                else:
                    log(f"    {name:24s} decode ({vs}): python {d['python'][:60]!r} llama.cpp {d['js'][:60]!r}")
    llama.close()
    return 1 if bad else 0
