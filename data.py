"""Load the saved diagrams as padded tensors the network can take directly.

Each diagram is its finite H0 pairs plus the essential point turned into
the finite pair (b, f_max). The diagrams must come from compute_diagrams.py run on the
largest component of each mask, so each has exactly one essential point.
"""
import pickle

import numpy as np
import torch

from layers import pad_diagrams

# drops pairs with death - birth <= 0.01, on the same [0, 1] height scale.
# In our diagrams that removes ~98% of the pairs: the mean per diagram goes
# from 112.9 to 2.6, and the largest diagram from 1151 points to 48.
MIN_PERSISTENCE = 0.01


def load_dataset(path="diagrams_lcc.pkl", min_persistence=MIN_PERSISTENCE):
    """All images, every direction, padded to a common number of points.

    Returns
      points:  (n_images, n_directions, P, 2) float tensor of (birth, death),
               P = largest diagram after the cleanup. Padding rows are 0.
      mask:    (n_images, n_directions, P) bool tensor, True for real points.
      labels:  (n_images,) long tensor, class index of each image.
      classes: list of class names, so classes[labels[i]] names image i.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    n_images = len(data["diagrams"])
    n_directions = len(data["angles"])

    # One flat list in image-major order, so the padded result reshapes
    # straight back to (image, direction).
    diagrams = []
    for image, f_max in zip(data["diagrams"], data["f_max"]):
        for (pairs, essential), top in zip(image, f_max):
            assert len(essential) == 1, "expected one component per mask"
            pairs = np.asarray(pairs).reshape(-1, 2)
            pairs = np.vstack([pairs, [[essential[0], top]]])
            diagrams.append(pairs[pairs[:, 1] - pairs[:, 0] > min_persistence])

    points, mask = pad_diagrams(diagrams)
    P = points.shape[1]
    points = points.reshape(n_images, n_directions, P, 2)
    mask = mask.reshape(n_images, n_directions, P)
    labels = torch.as_tensor(data["labels"], dtype=torch.long)
    return points, mask, labels, data["classes"]
