"""Plot H0 persistence diagrams for a few shapes, as a visual sanity check.

One row per shape: the mask, then its diagram for each direction in ANGLES.
Run: python plot_diagrams.py [name ...]   (default: apple-1 fork-1 octopus-1 bird-1)
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from diagrams import h0_diagram
from shapes import find_files, load_mask

NAMES = ["apple-1", "fork-1", "octopus-1", "bird-1"]
ANGLES = {"right": 0.0, "up": np.pi / 2, "left": np.pi, "down": -np.pi / 2}
# Essential points never die; they are drawn on this line above the plot area.
ESS_Y = 1.08


def plot_diagram(ax, pairs, essential, title):
    """Birth on x, death on y, with the diagonal and the essential line."""
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8)
    ax.axhline(ESS_Y, color="0.6", lw=0.8, ls="--")
    if len(pairs):
        ax.scatter(pairs[:, 0], pairs[:, 1], s=12, color="C0")
    ax.scatter(essential, np.full(len(essential), ESS_Y), s=30,
               marker="^", color="C3")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, ESS_Y + 0.05)
    ax.set_aspect("equal")
    ax.set_title(f"{title}: {len(pairs)} pairs, {len(essential)} ess.", fontsize=9)
    ax.tick_params(labelsize=7)


def main(names, out="diagrams.png"):
    files = {f.stem: f for f in find_files("MPEG7")}
    ncols = 1 + len(ANGLES)
    fig, axes = plt.subplots(len(names), ncols,
                             figsize=(2.6 * ncols, 2.6 * len(names)))
    axes = np.atleast_2d(axes)
    for row, name in zip(axes, names):
        mask = load_mask(files[name])
        row[0].imshow(mask, cmap="gray")
        row[0].set_title(name, fontsize=9)
        row[0].axis("off")
        for ax, (label, angle) in zip(row[1:], ANGLES.items()):
            pairs, essential = h0_diagram(mask, angle)
            plot_diagram(ax, pairs, essential, label)
    for ax in axes[-1, 1:]:
        ax.set_xlabel("birth", fontsize=8)
    for ax in axes[:, 1]:
        ax.set_ylabel("death", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES)
