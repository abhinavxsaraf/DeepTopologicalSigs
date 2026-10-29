"""Load the MPEG-7 CE-Shape-1 Part B dataset as binary masks.
- Only files named <class>-<number>.gif are used. This excludes
  confusions.gif and shapedata.gif, which are not dataset shapes.
- Pixels are thresholded at 128. If the image border is mostly "on",
  the mask is inverted, so True always means "object pixel".
"""
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PATTERN = re.compile(r"^(.+)-(\d+)$")  # "apple-1" -> ("apple", "1")


def find_files(root):
    """All dataset images under root, in a fixed (sorted) order."""
    return sorted(
        f for f in Path(root).rglob("*")
        if f.suffix.lower() == ".gif" and PATTERN.match(f.stem)
    )


def load_mask(path):
    """Read one image as a boolean mask with True = object."""
    img = np.array(Image.open(path).convert("L"))
    mask = img >= 128
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]) 
    if border.mean() > 0.5:  # object is dark on a light background
        mask = ~mask
    return mask


def largest_component(mask):
    """Keep only the largest 4-connected component of the mask, as the
    authors do (reduce_to_largest_connected_component)."""
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background
    return lab == sizes.argmax()


def load_dataset(root):
    """Return masks, integer labels, class names and file paths."""
    files = find_files(root)
    masks = [load_mask(f) for f in files]
    names = [PATTERN.match(f.stem).group(1) for f in files]
    classes = sorted(set(names))
    labels = np.array([classes.index(n) for n in names])
    return masks, labels, classes, files


def summarize(masks, labels, classes, files):
    """Print sanity checks: counts, image sizes, connected components."""
    counts = np.bincount(labels, minlength=len(classes))
    print(f"{len(masks)} images, {len(classes)} classes")
    bad = [(c, n) for c, n in zip(classes, counts) if n != 20]
    print("classes without 20 images:", bad if bad else "none")

    shapes = np.array([m.shape for m in masks])
    print("height range:", shapes[:, 0].min(), "-", shapes[:, 0].max())
    print("width range: ", shapes[:, 1].min(), "-", shapes[:, 1].max())

    fill = np.array([m.mean() for m in masks])
    print(f"object fraction: min {fill.min():.3f}, max {fill.max():.3f}")

    # Default structure in 2D = 4-connectivity, matching the paper's edges.
    multi = []
    for m, f in zip(masks, files):
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1]
            multi.append((f.name, n, sizes[:5].tolist()))
    print(f"{len(multi)} images with more than one 4-connected component")
    for name, n, sizes in multi[:20]:
        print(f"  {name}: {n} components, largest sizes {sizes}")


def plot_one_per_class(masks, labels, classes, out="classes.png"):
    """Save a grid showing the first image of each class."""
    import matplotlib.pyplot as plt

    cols = 10
    rows = int(np.ceil(len(classes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k < len(classes):
            i = int(np.flatnonzero(labels == k)[0])
            ax.imshow(masks[i], cmap="gray")
            ax.set_title(classes[k], fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print("saved", out)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "MPEG7"
    masks, labels, classes, files = load_dataset(root)
    summarize(masks, labels, classes, files)
    plot_one_per_class(masks, labels, classes)