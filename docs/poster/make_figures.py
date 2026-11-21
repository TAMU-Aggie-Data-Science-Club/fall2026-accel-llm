"""Poster figures, white background, 300 dpi (ADSC template rule). Run: python make_figures.py
fig1 architecture · fig2 HPRC wall-clock (Cyrus, 11/20 workshop) · fig3 local runs · fig4 tokens per round."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.size": 22, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.facecolor": "white"})
BLUE, ORANGE, GRAY, NAVY = "#2a6fdb", "#e8833a", "#8a8f98", "#1f163e"


def tokens_per_round(alpha, k):
    return (1 - alpha ** (k + 1)) / (1 - alpha)


assert abs(tokens_per_round(0.7, 5) - 2.94) < 0.01 and abs(tokens_per_round(0.9, 8) - 6.13) < 0.01

# Figure 1: architecture
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 13); ax.set_ylim(-0.75, 6); ax.axis("off")
boxes = [(0.2, "Laptop (edge)", "Draft model\nproposes k tokens + q(x)"),
         (4.6, "Network link", "WebSocket over\nSSH tunnel + VPN"),
         (9.0, "HPRC GPU node", "FastAPI verifier\ntarget checks all k\nin one pass")]
for x, title, sub in boxes:
    ax.add_patch(FancyBboxPatch((x, 2.6), 3.8, 3.0, boxstyle="round,pad=0.05", fc="#eef3fc", ec=BLUE, lw=3))
    ax.text(x + 1.9, 5.0, title, ha="center", va="center", weight="bold", color=NAVY, fontsize=24)
    ax.text(x + 1.9, 3.6, sub, ha="center", va="center", fontsize=19)
for x0, x1, y in [(4.05, 4.55, 4.4), (8.45, 8.95, 4.4)]:
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=30, color=BLUE, lw=3))
ax.add_patch(FancyArrowPatch((10.9, 2.5), (2.1, 2.5), connectionstyle="arc3,rad=-0.25",
                             arrowstyle="-|>", mutation_scale=30, color=ORANGE, lw=3))
ax.text(6.5, 0.4, "accepted tokens + correction", ha="center", color=ORANGE, fontsize=20)
ax.add_patch(FancyBboxPatch((4.4, -0.7), 4.2, 0.7, boxstyle="round,pad=0.05", fc="white", ec=GRAY, lw=2))
ax.text(6.5, -0.35, "Streamlit dashboard", ha="center", va="center", fontsize=19, color=NAVY)
fig.tight_layout(); fig.savefig("fig1_architecture.png")

# Figure 2: HPRC wall-clock, cloud-only vs distributed (Cyrus's 11/20 benchmark)
cloud, dist = [1.317, 0.803], [3.881, 3.742]
x = np.arange(2); w = 0.36
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w / 2, cloud, w, color=GRAY, label="Cloud-only (HPRC target)")
ax.bar(x + w / 2, dist, w, color=BLUE, label="Distributed edge → cloud")
for i in range(2):
    ax.text(i - w / 2, cloud[i], f"{cloud[i]:.2f} s", ha="center", va="bottom")
    ax.text(i + w / 2, dist[i], f"{dist[i]:.2f} s\n({cloud[i] / dist[i]:.2f}×)", ha="center", va="bottom")
ax.set_xticks(x, ["Test 1", "Test 2"]); ax.set_ylabel("Wall-clock time (s)"); ax.set_ylim(0, 5.6)
ax.legend(frameon=False, loc="upper center", ncol=2); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("fig2_hprc_wallclock.png")

# Figure 3: local runs (Qwen2.5-0.5B draft -> 1.5B target, k = 5, 60 tokens)
prompts = ["Code", "Factual", "Creative"]
accept, speedup = [80, 34, 19], [0.42, 0.29, 0.21]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.bar(prompts, accept, color=BLUE); ax1.set_ylabel("Draft tokens accepted (%)"); ax1.set_ylim(0, 100)
ax2.bar(prompts, speedup, color=ORANGE); ax2.set_ylabel("Speedup vs target alone"); ax2.set_ylim(0, 1.2)
ax2.axhline(1, ls="--", color=GRAY, lw=3); ax2.text(2.45, 1.02, "break-even (1×)", ha="right", va="bottom", color=GRAY)
for ax, vals, fmt in [(ax1, accept, "{:.0f}%"), (ax2, speedup, "{:.2f}×")]:
    for i, v in enumerate(vals):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom")
fig.tight_layout(); fig.savefig("fig3_measured_runs.png")

# Figure 4: expected tokens per round vs acceptance rate
a = np.linspace(0.3, 0.95, 200)
fig, ax = plt.subplots(figsize=(12, 7))
for k, color in [(3, GRAY), (5, BLUE), (8, ORANGE)]:
    ax.plot(a, tokens_per_round(a, k), lw=4, color=color, label=f"k = {k} drafted tokens")
ax.set_xlabel("Acceptance rate α"); ax.set_ylabel("Expected tokens per round")
ax.legend(frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig4_tokens_per_round.png")
print("wrote fig1-fig4")
