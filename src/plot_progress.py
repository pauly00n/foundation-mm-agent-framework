"""
plot_progress.py — Generate an experiment progress graph from results.jsonl.

Reads outputs/results.jsonl and plots validation loss per experiment.
Green step-line tracks the current best (lowest) loss. Breakthroughs
(new best) are green dots with commit-message labels; regressions are
red dots sitting off the line.

Output: outputs/progress_graph.png

Run:
    uv run src/plot_progress.py
"""

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT    = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "outputs" / "results.jsonl"
OUTPUT_PNG   = REPO_ROOT / "outputs" / "progress_graph.png"

LINE_COLOR      = "#22c55e"   # green
GOOD_COLOR      = "#15803d"   # darker green — breakthroughs
BAD_COLOR       = "#dc2626"   # red — regressions
LABEL_SIZE      = 5           # very small text
MAX_LABEL_LEN   = 40          # truncate commit messages


def get_commit_message(commit_hash: str) -> str:
    """Look up the one-line git commit message for a short hash, truncated."""
    try:
        result = subprocess.run(
            ["git", "log", "--format=%s", "-1", commit_hash],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        msg = result.stdout.strip()
        if len(msg) > MAX_LABEL_LEN:
            msg = msg[:MAX_LABEL_LEN] + "…"
        return msg
    except Exception:
        return commit_hash


def main():
    # Read experiment results
    experiments = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                experiments.append(json.loads(line))

    if not experiments:
        print("No experiments found in results.jsonl")
        return

    xs = list(range(1, len(experiments) + 1))
    ys = [r["val_loss"] for r in experiments]
    labels = [get_commit_message(r["experiment_id"]) for r in experiments]

    # Classify each experiment: breakthrough (new best loss) vs worsened
    is_good = []
    best_so_far = float("inf")
    best_line_xs = []   # x coords for the step-line (best-so-far)
    best_line_ys = []   # y coords for the step-line
    for x, y in zip(xs, ys):
        if y <= best_so_far:
            best_so_far = y
            is_good.append(True)
        else:
            is_good.append(False)
        best_line_xs.append(x)
        best_line_ys.append(best_so_far)

    fig, ax = plt.subplots(figsize=(14, 9))

    # Green step-line: current best loss over time
    ax.step(best_line_xs, best_line_ys, where="mid", linewidth=2,
            color=LINE_COLOR, zorder=2, label="Current best")

    # Plot points — green for breakthroughs, red for regressions
    good_xs = [x for x, g in zip(xs, is_good) if g]
    good_ys = [y for y, g in zip(ys, is_good) if g]
    bad_xs  = [x for x, g in zip(xs, is_good) if not g]
    bad_ys  = [y for y, g in zip(ys, is_good) if not g]

    ax.scatter(good_xs, good_ys, s=30, color=GOOD_COLOR, zorder=3, label="Breakthrough")
    ax.scatter(bad_xs,  bad_ys,  s=30, color=BAD_COLOR,  zorder=3, label="Worsened")

    # Annotate only breakthroughs with commit message
    for x, y, label, good in zip(xs, ys, labels, is_good):
        if good:
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=LABEL_SIZE,
                color=GOOD_COLOR,
                rotation=45,
                ha="left",
                va="bottom",
            )

    ax.set_ylim(0, max(ys) + 0.5)
    ax.set_title("AutoML Experiment Progress", fontsize=14, fontweight="bold")
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_xticks(xs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved → {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
