"""Compute and save the H0 diagrams of every MPEG-7 image in N_DIRECTIONS directions.

Each mask is first reduced to its largest connected component, so every diagram has exactly one essential
point.

Run: python compute_diagrams.py [root] [out]   (default: MPEG7 diagrams_lcc.pkl)

Output is a pickled dict:
  diagrams: list with one entry per image; each entry is a list of
            N_DIRECTIONS (pairs, essential) tuples, as returned by h0_diagram.
  f_max:    (n_images, N_DIRECTIONS) array, the highest height of the object
            in each direction.
  angles:   the N_DIRECTIONS angles, 2*pi*k / N_DIRECTIONS.
  labels, classes, files: as from shapes.load_dataset, in the same order.
"""
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from diagrams import h0_diagram_gudhi, heights
from shapes import PATTERN, find_files, largest_component, load_mask

N_DIRECTIONS = 32
ANGLES = 2 * np.pi * np.arange(N_DIRECTIONS) / N_DIRECTIONS
CHECKPOINT_EVERY = 50


def image_diagrams(path):
    """All N_DIRECTIONS diagrams of one image, and f_max in each direction."""
    mask = largest_component(load_mask(path))
    diagrams = [h0_diagram_gudhi(mask, a) for a in ANGLES]
    f_max = [heights(mask, a)[2].max() for a in ANGLES]
    return diagrams, f_max


def main(root="MPEG7", out="diagrams_lcc.pkl"):
    # Workers get file paths, not masks, so large arrays aren't copied around.
    files = find_files(root)
    names = [PATTERN.match(f.stem).group(1) for f in files]
    classes = sorted(set(names))
    labels = np.array([classes.index(n) for n in names])

    # Diagrams done so far are saved to a checkpoint every CHECKPOINT_EVERY
    # images. If the run is interrupted, running it again continues from there.
    partial = Path(out + ".partial")
    diagrams = []
    if partial.exists():
        with open(partial, "rb") as f:
            diagrams = pickle.load(f)
        print(f"resuming from {partial}: {len(diagrams)} images already done")

    start = time.time()
    todo = files[len(diagrams):]
    with Pool() as pool:
        # imap keeps the order of files, so diagrams[i] belongs to files[i].
        for i, d in enumerate(pool.imap(image_diagrams, todo), len(diagrams) + 1):
            diagrams.append(d)  # (diagrams, f_max) of one image
            print(f"{i}/{len(files)} {files[i - 1].stem}, {time.time() - start:.0f}s",
                  flush=True)
            if i % CHECKPOINT_EVERY == 0:
                save(diagrams, partial)

    f_max = np.array([f for _, f in diagrams])
    diagrams = [d for d, _ in diagrams]
    save({"diagrams": diagrams, "f_max": f_max, "angles": ANGLES, "labels": labels,
          "classes": classes, "files": [str(p) for p in files]}, Path(out))
    partial.unlink(missing_ok=True)
    print("saved", out)


def save(obj, path):
    """Pickle obj to path. Writing to a temporary file first means an
    interruption mid-write can't leave a broken file at path."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main(*sys.argv[1:3])
