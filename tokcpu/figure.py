"""The figure for `python -m tokcpu effect`: how far the final-sigma difference moves embeddings, and when
it changes which passage a question retrieves. Needs matplotlib."""

from __future__ import annotations

from pathlib import Path

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e2e1dc", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, in model order
GROUPS = [   # (label, which texts)
    ("Web pages, as found", lambda t: t["source"] == "fineweb-2"),
    ("Questions, in capitals", lambda t: t["kind"] == "question" and t["variant"] == "capitals"),
    ("Passages, in capitals", lambda t: t["kind"] == "passage" and t["variant"] == "capitals"),
]
SETTING = "questions capitals, passages as written"
BINS = ["0", "1", "2", "3", "4+"]


def figure(results: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "svg.fonttype": "none"})
    models = list(results["models"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.25, 1]}, facecolor=SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # A: drift of every text whose ids differ where the model reads, one row per group and model
    for g, (label, keep) in enumerate(GROUPS):
        for k, repo in enumerate(models):
            d = [t["drift"] for t in results["models"][repo]["texts"] if keep(t) and t["changed"]]
            if not d:
                continue
            y = g + (k - 1) * 0.22
            jitter = [y + ((i * 0.6180339) % 1 - 0.5) * 0.14 for i in range(len(d))]
            a.scatter(d, jitter, s=6, color=SERIES[k], alpha=0.35, linewidths=0, rasterized=True)
            med = sorted(d)[len(d) // 2]
            a.plot([med, med], [y - 0.1, y + 0.1], color=INK, lw=2, solid_capstyle="round")
    a.set_xscale("log")
    a.set_yticks(range(len(GROUPS)), [g[0] for g in GROUPS])
    a.invert_yaxis()
    a.tick_params(axis="y", length=0, labelcolor=INK)
    a.grid(axis="x", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_xlabel("Embedding drift, 1 − cosine(Python ids, transformers.js ids)   (black tick: median)")
    a.set_title("A. How far the embedding moves", loc="left", fontsize=10, fontweight="bold", color=INK)

    # B: share of questions whose top passage changes, by how many word-final capital sigmas they hold
    for k, repo in enumerate(models):
        r = results["models"][repo]["retrieval"][SETTING]
        changed = r["top 1 changed"]["mixed"]
        share, ns = [], []
        for i, _ in enumerate(BINS):
            hit = [c for c, f in zip(changed, r["finals"]) if (f >= 4 if i == 4 else f == i)]
            ns.append(len(hit))
            share.append(100 * sum(hit) / len(hit) if hit else float("nan"))
        b.plot(range(len(BINS)), share, color=SERIES[k], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=repo.split("/")[1])
    b.set_xticks(range(len(BINS)), [f"{x}\nn={n}" for x, n in zip(BINS, ns)])
    b.set_xlabel("Word-final capital Σ in the question")
    b.set_ylabel("Questions whose top passage changes (%)")
    b.set_ylim(bottom=0)
    b.grid(axis="y", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.set_title("B. When the retrieved passage changes", loc="left", fontsize=10, fontweight="bold", color=INK)
    fig.legend(*b.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, labelcolor=INK, fontsize=9)
    fig.text(0.01, 0.01, "Greek text. The same model and weights read both id sequences, so the tokenizer is the only difference. "
             "A: texts whose ids differ within what the model reads.\nB: Belebele questions in capitals against its passages as written; "
             "passages embedded from Python's ids, questions from transformers.js's.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out, facecolor=SURFACE, dpi=160)
    fig.savefig(out.with_suffix(".png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
