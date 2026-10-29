# Deep Learning with Topological Signatures

A from-scratch PyTorch reproduction of Hofer et al., [*Deep Learning with Topological Signatures*](https://arxiv.org/abs/1707.04041) (NeurIPS 2017), on the MPEG-7 CE-Shape-1 Part B shape classification task.

## Background

Persistent homology summarises the shape of data as a *persistence diagram*: a set of (birth, death) points, one per topological feature. Diagrams are unordered and vary in size, so they don't fit directly into a neural network. The paper's contribution is an input layer that maps a diagram to a fixed-size vector by evaluating learnable Gaussian-like "structure elements" at every point and summing, with a log transform that makes near-diagonal (low-persistence, noisy) points count for little.

For MPEG-7, each binary shape is filtered by height along 32 directions, giving 32 H0 diagrams per image. One structure-element layer per direction (shared with its two neighbouring directions) feeds a small classifier over 70 classes.

## Files

- `shapes.py`: load MPEG-7 images as binary masks
- `diagrams.py`: H0 diagrams under directional height filtrations (hand-written union-find, plus a gudhi cross-check)
- `compute_diagrams.py`: precompute and cache diagrams for the whole dataset
- `layers.py`: the structure-element input layer
- `model.py`, `data.py`, `train.py`: the network, data loading, and the authors' training protocol
- `plot_diagrams.py`: visual sanity check of a few shapes' diagrams

## Usage

```bash
pip install -r requirements.txt
python compute_diagrams.py MPEG7 diagrams_lcc.pkl   # MPEG-7 images in ./MPEG7 (not included)
python train.py
```

## Status

Work in progress. Best result so far is 85.4% test accuracy (one seed) against the paper's 91.8%.
