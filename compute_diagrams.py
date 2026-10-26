"""Compute and save the H0 diagrams of every MPEG-7 image in N_DIRECTIONS directions.

Uses h0_diagram_gudhi: test_diagrams.py checks it matches h0_diagram, and it is
much faster on large images. Images are processed in parallel, one per worker.

Run: python compute_diagrams.py [root] [out]   (default: MPEG7 diagrams.pkl)
If interrupted, run the same command again: it resumes from <out>.partial.

Output is a pickled dict:
  diagrams: list with one entry per image; each entry is a list of
            N_DIRECTIONS (pairs, essential) tuples, as returned by h0_diagram.
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

from diagrams import h0_diagram_gudhi
from shapes import PATTERN, find_files, load_mask

N_DIRECTIONS = 32
ANGLES = 2 * np.pi * np.arange(N_DIRECTIONS) / N_DIRECTIONS
CHECKPOINT_EVERY = 50


def image_diagrams(path):
    """All N_DIRECTIONS diagrams of one image."""
    mask = load_mask(path)
    return [h0_diagram_gudhi(mask, a) for a in ANGLES]


def main(root="MPEG7", out="diagrams.pkl"):
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
            diagrams.append(d)
            print(f"{i}/{len(files)} {files[i - 1].stem}, {time.time() - start:.0f}s",
                  flush=True)
            if i % CHECKPOINT_EVERY == 0:
                save(diagrams, partial)

    save({"diagrams": diagrams, "angles": ANGLES, "labels": labels,
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
