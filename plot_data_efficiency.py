"""Test accuracy against training images per class, one line per model.

Reads every train.py / resnet_baseline.py log in the given folders (default:
the current one), whatever the file names:
- the model comes from the log's args (TDA, small CNN, ResNet-18,
  ResNet-18 pretrained; augmented runs get their own line),
- the number of images per class from --train-per-class (none = all 18),
- the score is the mean test accuracy over the last 10 epochs, as everywhere.

    python plot_data_efficiency.py              # logs in .
    python plot_data_efficiency.py logs/ --out data_efficiency.png

Prints a table (mean ± std over seeds, number of seeds) and saves the plot:
mean as a line, ± one std as a band.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FULL = 18  # training images per class without --train-per-class
# Fixed colour (and marker) per model, so a missing model never repaints the others.
STYLE = {
    "TDA": ("#2a78d6", "o"),
    "Small CNN": ("#eb6834", "s"),
    "ResNet-18": ("#1baf7a", "^"),
    "ResNet-18 pretrained": ("#eda100", "D"),
}


def describe(log):
    """(model, images per class, seed) for one log, or None if it's not a run log."""
    args = log.get("args", {})
    if "test_acc" not in log or len(log["test_acc"]) < 10:
        return None
    if "diagrams" in args:
        if not args["diagrams"].endswith("diagrams_lcc.pkl"):
            return None  # runs on the old diagrams, before the reproduction
        model = "TDA"
    elif "arch" in args or "pretrained" in args:
        if args.get("arch", "resnet18") == "small":
            model = "Small CNN"
        else:
            model = "ResNet-18 pretrained" if args.get("pretrained") else "ResNet-18"
        if args.get("augment"):
            model += " (aug)"
    else:
        return None
    return model, args.get("train_per_class") or FULL, args["seed"]


def collect(folders):
    """{(model, n): {seed: score}}. If two logs are the same run, the newer file wins."""
    runs = {}
    for path in sorted((p for f in folders for p in Path(f).glob("*.json")),
                       key=lambda p: p.stat().st_mtime):
        try:
            log = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        key = describe(log) if isinstance(log, dict) else None
        if key is None:
            continue
        model, n, seed = key
        seeds = runs.setdefault((model, n), {})
        if seed in seeds:
            print(f"note: {path.name} replaces an older log of {model}, n={n}, seed {seed}")
        seeds[seed] = 100 * float(np.mean(log["test_acc"][-10:]))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="*", default=["."])
    parser.add_argument("--out", default="data_efficiency.png")
    args = parser.parse_args()

    runs = collect(args.folders)
    if not runs:
        raise SystemExit("no run logs found")
    models = sorted({m for m, _ in runs}, key=lambda m: (list(STYLE).index(m)
                                                         if m in STYLE else len(STYLE), m))
    sizes = sorted({n for _, n in runs})

    print(f"{'model':22}" + "".join(f"{f'n={n}':>18}" for n in sizes))
    for m in models:
        cells = []
        for n in sizes:
            s = list(runs.get((m, n), {}).values())
            cells.append(f"{np.mean(s):5.1f} ± {np.std(s, ddof=1) if len(s) > 1 else 0:3.1f} ({len(s)})"
                         if s else "-")
        print(f"{m:22}" + "".join(f"{c:>18}" for c in cells))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ends = []  # (label, x, y) at each line's right end
    for m in models:
        color, marker = STYLE.get(m.removesuffix(" (aug)"), ("#888888", "x"))
        dashed = m.endswith("(aug)")
        ns = [n for n in sizes if (m, n) in runs]
        scores = [list(runs[(m, n)].values()) for n in ns]
        mean = np.array([np.mean(s) for s in scores])
        std = np.array([np.std(s, ddof=1) if len(s) > 1 else 0 for s in scores])
        ax.fill_between(ns, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(ns, mean, color=color, marker=marker, markersize=7, linewidth=2,
                linestyle="--" if dashed else "-", label=m,
                markeredgecolor="white", markeredgewidth=1.5)
        ends.append((m, ns[-1], mean[-1]))

    # Label each line at its end. Labels closer than `gap` form a group, which
    # is spaced `gap` apart around the group's mean height.
    lo, hi = ax.get_ylim()
    gap = 0.05 * (hi - lo)

    def spaced(group):
        centre = np.mean([y for _, _, y in group])
        return [centre + (i - (len(group) - 1) / 2) * gap for i in range(len(group))]

    groups = []
    for end in sorted(ends, key=lambda e: e[2]):
        groups.append([end])
        while len(groups) > 1 and spaced(groups[-1])[0] < spaced(groups[-2])[-1] + gap:
            groups[-2:] = [groups[-2] + groups[-1]]
    for group in groups:
        for (m, x, y), y_text in zip(group, spaced(group)):
            ax.annotate(m, (x, y), xytext=(x * 1.08, y_text), va="center",
                        fontsize=9, color="#333333")

    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 1.9)  # room for the line labels
    ax.set_xlabel("training images per class")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("MPEG-7: accuracy vs training data",
                 fontsize=11, loc="left")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#999999")
    ax.tick_params(colors="#555555")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
