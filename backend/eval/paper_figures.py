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
    """Per variant: error-free rate and template-fallback rate (both % of texts, 95% Wilson intervals)."""
    abl = num.get("ablations") or {}
    if not abl:
        return
    import math

    def wil(k, n, z=1.96):
        if not n:
            return None
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return {"rate": p, "lo": max(0.0, (c - h) / d), "hi": min(1.0, (c + h) / d), "n": n}

    names = list(abl.keys())
    fig, ax = plt.subplots(figsize=(3.4, 0.5 * len(names) + 0.9))
    for i, s in enumerate(names):
        y = len(names) - 1 - i
        st = abl[s]["status"]
        n = sum(st.values())
        whisker(ax, y + 0.15, abl[s]["error_free_llm"], DOMAIN_COLOR["finance"], label="Error-free" if i == 0 else None)
        whisker(ax, y - 0.15, wil(st.get("fallback", 0), n), DOMAIN_COLOR["health"], marker="s", size=4.5,
                label="Template fallback" if i == 0 else None)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABEL.get(s, s) for s in reversed(names)])
    style(ax, "% of texts, 95% CI")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0.1)
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
    style(ax, "Faults blocked (%), 95% CI")
    fig.tight_layout()
    save(fig, "fig_faults_llm")


def fig_coverage(num):
    cov = num.get("coverage_risk") or {}
    if not cov:
        return
    colors = {"ours:full": INK, "direct": "#2a78d6", "b3": "#eb6834", "explingo": "#1baf7a",
              "critic_rule": "#eda100", "musicco": "#e87ba4"}
    markers = {"ours:full": "D", "direct": "o", "b3": "s", "explingo": "^", "critic_rule": "v", "musicco": "P"}
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for s, pts in cov.items():
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: p["coverage"])
        xs = [100 * p["coverage"] for p in pts]
        ys = [100 * p["error_rate"] for p in pts]
        ax.plot(xs, ys, color=colors.get(s, MUTED), linewidth=2 if s == "ours:full" else 1.4,
                marker=markers.get(s, "o"), markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                label=LABEL.get(s, s), zorder=3 if s == "ours:full" else 2)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Coverage: cases explained (%)", color=INK2, fontsize=8)
    ax.set_ylabel("Error rate among explained (%)", color=INK2, fontsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, handletextpad=0.3,
              columnspacing=0.8, borderaxespad=0.1)
    fig.tight_layout()
    save(fig, "fig_coverage_risk")


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
    fig_coverage(num)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
