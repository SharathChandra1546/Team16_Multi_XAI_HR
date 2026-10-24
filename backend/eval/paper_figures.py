"""Figures for the paper from results/paper/numbers.json (run eval/paper_tables.py first).

    python eval/paper_figures.py

Writes results/paper/fig_*.pdf (vector, for LaTeX) and fig_*.png (300 dpi). Print-first light styling:
thin marks, 95% Wilson intervals as whiskers, recessive grid, colour only for identity (finance / health).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
OUT = BACKEND / "results" / "paper"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DOMAIN_COLOR = {"finance": "#2a78d6", "health": "#eb6834"}  # categorical slots 1-2, validated together
POOLED = "#0b0b0b"
LABEL = {"ours:full": "Ours (full)", "direct": "Direct", "b3": "B3 reviewer loop", "explingo": "Explingo-style",
         "critic_rule": "Critic-Rule", "musicco": "Musicco-style", "ours:no_verifier": "– verifier",
         "ours:no_critic": "– critic", "ours:no_repair": "– repair", "ours:no_policy_rules": "– policy rules"}


def style(ax, xlabel):
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel(xlabel, color=INK2, fontsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


def whisker(ax, y, r, color, marker="o", size=5, label=None):
    if not r or r.get("n", 0) == 0 or r.get("rate") is None:
        return
    p, lo, hi = 100 * r["rate"], 100 * r["lo"], 100 * r["hi"]
    ax.plot([lo, hi], [y, y], color=color, linewidth=1.4, solid_capstyle="round", zorder=2)
    ax.plot([p], [y], marker=marker, color=color, markersize=size, markeredgecolor="white", markeredgewidth=0.8,
            zorder=3, label=label, linestyle="none")


def fig_main(num):
    systems = list(num["main"].keys())
    if not systems:
        return
    fig, ax = plt.subplots(figsize=(3.4, 0.42 * len(systems) + 0.9))
    for i, s in enumerate(systems):
        y = len(systems) - 1 - i
        for j, dom in enumerate(("finance", "health")):
            whisker(ax, y + (0.18 if j == 0 else -0.18), num["main"][s][dom]["error_free_llm"], DOMAIN_COLOR[dom],
                    label=dom.capitalize() if i == 0 else None)
        pooled = num["main"][s]["pooled"]["error_free_llm"]
        if pooled and pooled.get("rate") is not None:
            ax.plot([100 * pooled["rate"]], [y], marker="D", color=POOLED, markersize=4, linestyle="none", zorder=4,
                    label="Pooled" if i == 0 else None)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([LABEL.get(s, s) for s in reversed(systems)])
    style(ax, "Error-free explanations (%), 95% CI")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout()
    save(fig, "fig_main_comparison")


def fig_ablation(num):
    abl = num.get("ablations") or {}
    if not abl:
        return
    names = list(abl.keys())
    fig, ax = plt.subplots(figsize=(3.4, 0.36 * len(names) + 0.8))
    for i, s in enumerate(names):
        whisker(ax, len(names) - 1 - i, abl[s]["error_free_llm"], INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABEL.get(s, s) for s in reversed(names)])
    style(ax, "Error-free explanations (%), 95% CI")
    fig.tight_layout()
    save(fig, "fig_ablation")


def fig_faults(num):
    fl = (num.get("faults") or {}).get("llm")
    if not fl:
        return
    items = sorted(fl["per_fault"].items(), key=lambda kv: (kv[1]["rate"] or 0))
    fig, ax = plt.subplots(figsize=(3.4, 0.24 * len(items) + 0.8))
    for i, (f, r) in enumerate(items):
        whisker(ax, i, r, INK, size=4)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f.replace("_", " ") for f, _ in items])
    style(ax, "Injected faults blocked by the verifier (%), 95% CI")
    fig.tight_layout()
    save(fig, "fig_faults_llm")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def main() -> int:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK, "axes.labelcolor": INK2,
                         "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42})
    num = json.loads((OUT / "numbers.json").read_text(encoding="utf-8"))
    fig_main(num)
    fig_ablation(num)
    fig_faults(num)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
