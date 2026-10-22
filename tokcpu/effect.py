"""Does a tokenizer disagreement change what the model does?

`compat` found that transformers.js lowercases a capital Σ at the end of a word to ς, because JavaScript's
toLowerCase follows Unicode's final-sigma rule, while the Rust library, and so Python, lowercases each
character alone and gives σ. 69 BERT-style WordPiece models are affected, all-MiniLM-L6-v2 and the BGE and E5
families among them. This measures what that does downstream.

Each text is tokenized twice, by Python (`tokenizers`) and by transformers.js 4 (fuzz/compat.mjs), and both
id sequences go through the same model, weights and code (PyTorch, in Python), so the tokenizer is the only
thing that differs. The texts are Greek:

- Belebele's Greek passages and questions (edited prose), as written and in capitals (a headline, a sign,
  caps lock), for embedding drift and for retrieval: each question should find its own passage;
- FineWeb-2's Greek test split (web pages as found), for how often the ids differ in ordinary text.

    python -m tokcpu effect     writes effect/summary.md, effect/figure.svg and effect/results.json (and the pairs, if missing)

The pairs are built once (they need Node 22 and fuzz/'s npm install) and kept in the repo, so the
experiment itself runs from them alone. Each row holds the text and, per model, Python's ids and
transformers.js's ids, the latter only where they differ, both cut to the first 512 (the most these models
read); "differs later" marks a text whose ids differ only after that. Web pages keep their FineWeb-2 id,
not their text.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

from .compat import ROOT, Node

MODELS = {   # repo -> pooling, as each model's sentence-transformers config says (mean for plain BERT)
    "sentence-transformers/all-MiniLM-L6-v2": "mean",
    "BAAI/bge-small-en-v1.5": "cls",
    "google-bert/bert-base-multilingual-uncased": "mean",
}
LONGEST = {"sentence-transformers/all-MiniLM-L6-v2": 256}   # sentence-transformers' max_seq_length; else 512
WEB_PAGES = 5000
# A capital sigma that JavaScript lowercases to ς: after a cased letter, and not before one (Unicode's
# Final_Sigma, near enough for counting; the ids, not this, decide whether a text differs).
FINAL = re.compile(r"(?<=[^\W\d_])Σ(?![^\W\d_])")


def caps(text: str) -> str:
    return text.upper()


def texts() -> list[dict]:
    """Belebele's Greek passages and questions, as written and in capitals, and FineWeb-2's Greek web pages."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rows = []
    belebele = [json.loads(l) for l in open(hf_hub_download("facebook/belebele", "data/ell_Grek.jsonl", repo_type="dataset"), encoding="utf-8")]
    passages = list(dict.fromkeys(r["flores_passage"] for r in belebele))
    for variant, f in (("as written", str), ("capitals", caps)):
        rows += [{"source": "belebele", "kind": "passage", "id": f"p{i}", "variant": variant, "text": f(p)} for i, p in enumerate(passages)]
        rows += [{"source": "belebele", "kind": "question", "id": f"q{i}", "variant": variant, "text": f(r["question"]),
                  "passage": f"p{passages.index(r['flores_passage'])}"} for i, r in enumerate(belebele)]
    web = pq.read_table(hf_hub_download("HuggingFaceFW/fineweb-2", "data/ell_Grek/test/000_00000.parquet", repo_type="dataset"),
                        columns=["id", "text"]).slice(0, WEB_PAGES).to_pylist()
    rows += [{"source": "fineweb-2", "kind": "web page", "id": w["id"], "variant": "as written", "text": w["text"]} for w in web]
    for r in rows:
        r["finals"] = len(FINAL.findall(r["text"]))
    return rows


def pairs(out: Path, node: str = "node", log=print) -> list[dict]:
    """Each text's ids in Python and in transformers.js 4, for each model. Web pages whose ids agree for
    every model are counted, not kept: they would only repeat the same ids twice."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    rows = texts()
    js = Node(node, "tj4")
    try:
        for repo in MODELS:
            d = Path(hf_hub_download(repo, "tokenizer.json")).parent
            hf_hub_download(repo, "tokenizer_config.json")
            tok = Tokenizer.from_file(str(d / "tokenizer.json"))
            tok.no_truncation()
            tok.no_padding()
            js.send({"op": "load", "dir": str(d)})
            loaded = js.receive()
            runtime = loaded["runtimes"][0]
            if loaded["errors"]:
                raise RuntimeError(f"{runtime} cannot load {repo}: {loaded['errors']}")
            for start in range(0, len(rows), 200):
                chunk = rows[start:start + 200]
                js.send({"op": "encode", "texts": [r["text"] for r in chunk]})
                got = js.receive()[runtime]
                for r, py, j in zip(chunk, tok.encode_batch([r["text"] for r in chunk]), got):
                    a, b = fit(py.ids, 512), fit(j, 512)   # as much as any of these models reads
                    r.setdefault("ids", {})[repo] = {"python": a} | ({} if a == b else {"js": b}) | ({"differs later": True} if j != py.ids and a == b else {})
            log(f"{repo}: {sum('js' in r['ids'][repo] for r in rows)} of {len(rows)} texts differ")
    finally:
        js.close()
    web = [r for r in rows if r["source"] == "fineweb-2"]
    keep = [r for r in rows if r["source"] != "fineweb-2" or any(x.keys() - {"python"} for x in r["ids"].values())]
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"runtime": runtime, "models": list(MODELS), "web pages": len(web),
                            "web pages with a final Σ": sum(r["finals"] > 0 for r in web)}) + "\n")
        for r in keep:   # a web page's text stays in FineWeb-2, under its id
            f.write(json.dumps({k: v for k, v in r.items() if not (k == "text" and r["source"] == "fineweb-2")}, ensure_ascii=False) + "\n")
    return read_pairs(out)


def read_pairs(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        head, *rows = [json.loads(l) for l in f]
    return head, rows


def fit(ids: list[int], longest: int) -> list[int]:
    """Truncated as sentence-transformers truncates: the start of the text, then [SEP]."""
    return ids if len(ids) <= longest else ids[:longest - 1] + ids[-1:]


class Embedder:
    def __init__(self, repo: str):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained(repo).eval()
        self.pooling = MODELS[repo]
        self.longest = LONGEST.get(repo, 512)

    def __call__(self, batch: list[list[int]]):
        torch = self.torch
        batch = [fit(ids, self.longest) for ids in batch]
        n = max(map(len, batch))
        ids = torch.tensor([x + [0] * (n - len(x)) for x in batch])
        mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in batch])
        with torch.no_grad():
            h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        v = h[:, 0] if self.pooling == "cls" else (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return torch.nn.functional.normalize(v, dim=-1)

    def all(self, seqs: list[list[int]], size: int = 32):
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))   # similar lengths batch together
        out = [None] * len(seqs)
        for s in range(0, len(order), size):
            idx = order[s:s + size]
            for i, v in zip(idx, self([seqs[i] for i in idx])):
                out[i] = v
        return self.torch.stack(out)


def retrieval(q, p, gold: list[int]) -> dict:
    """Each question's ranking of the passages: the gold passage's rank (1 = first) and the top 10."""
    s = q @ p.T
    order = s.argsort(dim=1, descending=True)
    rank = [int((order[i] == g).nonzero()) + 1 for i, g in enumerate(gold)]
    top2 = s.topk(2, dim=1).values
    return {"rank": rank, "top10": order[:, :10].tolist(), "gap": (top2[:, 0] - top2[:, 1]).tolist()}


def measure(path: Path, log=print) -> dict:
    head, rows = read_pairs(path)
    results = {"runtime": head["runtime"], "web pages": head["web pages"], "web pages with a final Σ": head["web pages with a final Σ"], "models": {}}
    for repo in head["models"]:
        log(f"{repo} ...")
        emb = Embedder(repo)
        py = emb.all([r["ids"][repo]["python"] for r in rows])
        js = emb.all([r["ids"][repo].get("js", r["ids"][repo]["python"]) for r in rows])
        cos = (py * js).sum(1).tolist()
        texts = [{"source": r["source"], "kind": r["kind"], "variant": r["variant"], "finals": r["finals"],
                  "differs": "js" in r["ids"][repo], "drift": max(0.0, 1 - c),
                  "changed": sum(a != b for a, b in zip(fit(r["ids"][repo]["python"], emb.longest), fit(r["ids"][repo].get("js", r["ids"][repo]["python"]), emb.longest)))}
                 for r, c in zip(rows, cos)]
        web = [r for r in rows if r["source"] == "fineweb-2"]
        m = {"pooling": emb.pooling, "longest": emb.longest, "texts": texts,
             "web pages that differ": sum(bool(r["ids"][repo].keys() - {"python"}) for r in web),
             "web pages that differ where the model reads": sum(t["changed"] > 0 for t, r in zip(texts, rows) if r["source"] == "fineweb-2"),
             "retrieval": {}}
        where = {(r["kind"], r["variant"], r["id"]): i for i, r in enumerate(rows) if r["source"] == "belebele"}
        # Which runtime reads capitals closer to the text as written? An uncased model should see the same ids.
        ids = lambda r, side: r["ids"][repo].get(side, r["ids"][repo]["python"])
        upper = [(rows[where[k[0], "as written", k[2]]], r) for k, i in where.items() if k[1] == "capitals" for r in [rows[i]] if "js" in r["ids"][repo]]
        m["capitals that differ"] = len(upper)
        m["capitals equal as written"] = {side: sum(ids(w, "python") == ids(c, side) for w, c in upper) for side in ("python", "js")}
        for qv, pv in (("as written", "as written"), ("capitals", "as written"), ("capitals", "capitals")):
            qs = [r for r in rows if r["kind"] == "question" and r["variant"] == qv]
            ps = sorted((r for r in rows if r["kind"] == "passage" and r["variant"] == pv), key=lambda r: int(r["id"][1:]))
            qi = [where["question", qv, r["id"]] for r in qs]
            pi = [where["passage", pv, r["id"]] for r in ps]
            gold = [int(r["passage"][1:]) for r in qs]
            runs = {"python": retrieval(py[qi], py[pi], gold),            # everything in Python
                    "javascript": retrieval(js[qi], js[pi], gold),        # everything in the browser
                    "mixed": retrieval(js[qi], py[pi], gold)}             # passages indexed in Python, questions in the browser
            m["retrieval"][f"questions {qv}, passages {pv}"] = {
                "questions": len(qs), "questions that differ": sum(texts[i]["differs"] for i in qi),
                "finals": [r["finals"] for r in qs], **{k: v["rank"] for k, v in runs.items()},
                "gap": runs["python"]["gap"],   # how far ahead Python's first passage is of its second
                "top 1 changed": {k: [a[0] != b[0] for a, b in zip(runs["python"]["top10"], runs[k]["top10"])] for k in ("javascript", "mixed")},
                "top 10 changed": {k: [set(a) != set(b) for a, b in zip(runs["python"]["top10"], runs[k]["top10"])] for k in ("javascript", "mixed")}}
        results["models"][repo] = m
    return results


def summary(results: dict) -> str:
    """The numbers the README quotes, as Markdown."""
    import statistics
    lines = [f"Web pages (FineWeb-2 Greek): {results['web pages with a final Σ']} of {results['web pages']} have a word-final capital Σ.", ""]
    lines += ["| Model | Web pages that differ | Differ where the model reads | Median drift, capitals | 90th percentile |", "|---|---|---|---|---|"]
    for repo, m in results["models"].items():
        d = [t["drift"] for t in m["texts"] if t["source"] == "belebele" and t["variant"] == "capitals" and t["differs"]]
        lines.append(f"| {repo} | {m['web pages that differ']} | {m['web pages that differ where the model reads']} | "
                     f"{statistics.median(d):.4f} | {statistics.quantiles(d, n=10)[-1]:.4f} |")
    lines += ["", "Texts in capitals whose ids differ, and which runtime's ids equal those of the same text as written:", ""]
    lines += [f"- {repo}: {m['capitals that differ']}; Python {m['capitals equal as written']['python']}, transformers.js {m['capitals equal as written']['js']}"
              for repo, m in results["models"].items()]
    lines += ["", "Median score gap between Python's first and second passage (questions in capitals, passages as written), "
              "against the median drift of those questions:", ""]
    for repo, m in results["models"].items():
        r = m["retrieval"]["questions capitals, passages as written"]
        d = [t["drift"] for t in m["texts"] if t["kind"] == "question" and t["variant"] == "capitals" and t["differs"]]
        lines.append(f"- {repo}: gap {statistics.median(r['gap']):.4f}, drift {statistics.median(d):.4f}")
    lines += ["", "| Model | Setting | Questions that differ | R@1 Python | R@1 JS | R@1 mixed | Top 1 changed (JS, mixed) | Top 10 changed (JS, mixed) | Gold rank better / worse (mixed) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for repo, m in results["models"].items():
        for setting, r in m["retrieval"].items():
            at1 = {k: sum(x == 1 for x in r[k]) / r["questions"] for k in ("python", "javascript", "mixed")}
            pct = lambda xs: f"{100 * sum(xs) / len(xs):.1f}%"
            better = sum(b < a for a, b in zip(r["python"], r["mixed"]))
            worse = sum(b > a for a, b in zip(r["python"], r["mixed"]))
            lines.append(f"| {repo.split('/')[1]} | {setting} | {r['questions that differ']} of {r['questions']} | {at1['python']:.3f} | {at1['javascript']:.3f} | "
                         f"{at1['mixed']:.3f} | {pct(r['top 1 changed']['javascript'])}, {pct(r['top 1 changed']['mixed'])} | "
                         f"{pct(r['top 10 changed']['javascript'])}, {pct(r['top 10 changed']['mixed'])} | {better} / {worse} |")
    return "\n".join(lines) + "\n"


def effect(out: Path, node: str = "node", log=print) -> dict:
    path = out / "pairs.jsonl.gz"
    if not path.exists():
        pairs(path, node, log)
    results = measure(path, log)
    (out / "results.json").write_text(json.dumps(results))
    (out / "summary.md").write_text(summary(results))
    from .figure import figure
    figure(results, out / "figure.svg")
    return results
