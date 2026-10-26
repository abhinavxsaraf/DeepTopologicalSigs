"""Input layers that turn persistence diagrams into fixed-size vectors (Hofer et al. 2017).

- pad_diagrams: a batch of diagrams of different sizes as one padded tensor.
- StructureElementLayer: the paper's layer (Definition 1) for finite
  (birth, death) points.
- EssentialLayer: a 1D version for essential points, which only have a birth.

A diagram is a set: no order and no fixed size. Each layer evaluates N
learnable functions ("structure elements") at every point and sums over the
points, which gives N numbers that don't depend on order or count.
"""
import numpy as np
import torch
from torch import nn


def pad_diagrams(diagrams, dim=2):
    """Stack diagrams with different numbers of points into one tensor.

    diagrams: list of B arrays, each (n_i, dim), or (n_i,) when dim == 1.
    Returns
      points: (B, P, dim) float tensor with P = max n_i (at least 1).
              Rows past n_i are padding, filled with 0.
      mask:   (B, P) bool tensor, True where a row is a real point.
    """
    B = len(diagrams)
    P = max(1, max(len(d) for d in diagrams))
    points = torch.zeros(B, P, dim)
    mask = torch.zeros(B, P, dtype=torch.bool)
    for i, d in enumerate(diagrams):
        n = len(d)
        if n:
            points[i, :n] = torch.from_numpy(
                np.asarray(d, dtype=np.float32).reshape(n, dim))
            mask[i, :n] = True
    return points, mask


class StructureElementLayer(nn.Module):
    """The paper's input layer for finite points.

    A point (b, d) is first rotated by 45 degrees to
    rho(b, d) = (x0, x1) = ((b + d) / sqrt(2), (d - b) / sqrt(2)), so the
    diagonal becomes the line x1 = 0 and x1 is the persistence scaled by
    1/sqrt(2).
    Structure element i, with centre mu_i = (mu0, mu1) and sharpness
    sigma_i = (sigma0, sigma1), both learned, takes the value

      exp(-sigma0^2 (x0 - mu0)^2 - sigma1^2 (x1  - mu1)^2)   if x1 >= nu
      exp(-sigma0^2 (x0 - mu0)^2 - sigma1^2 (x1' - mu1)^2)   if 0 < x1 < nu
      0                                                      if x1 == 0

    with x1' = ln(x1 / nu) * nu + nu. For x1 < nu the log pushes points towards
    -inf as they approach the diagonal, so their value fades to 0. That's what
    makes the layer stable: points near the diagonal (noise) barely count.
    The two formulas agree at x1 = nu, so the value is continuous there.

    The layer's output for a diagram is, for each i, the sum of element i's
    value over all points of the diagram.
    """

    def __init__(self, n_elements, nu=0.1, s=3.0):
        super().__init__()
        self.nu = nu
        # Centres: (birth, death) uniform over the part of [0, 1]^2 above the
        # diagonal, then put into the same coordinates as the points: rotated,
        # and x1 passed through the log below nu.
        b, d = torch.rand(2, n_elements).sort(dim=0).values
        mu0 = (b + d) / 2 ** 0.5
        mu1 = (d - b) / 2 ** 0.5
        mu1 = torch.where(mu1 < nu,
                          torch.log(mu1.clamp(min=1e-8) / nu) * nu + nu, mu1)
        self.mu = nn.Parameter(torch.stack([mu0, mu1], dim=1))       # (N, 2)
        # Their code computes exp(-s (x - mu)^2) with s = 3 (4 for the shape
        # datasets), with no square on s. Under our sigma^2 that's sigma = sqrt(s).
        self.sigma = nn.Parameter(
            torch.full((n_elements, 2), s ** 0.5))                   # (N, 2)

    def forward(self, points, mask):
        """points: (B, P, 2) as (birth, death). mask: (B, P). Returns (B, N)."""
        
        birth = points[:,:,0]
        death = points[:,:,1]
        x0 = (birth + death) / 2 ** 0.5
        x1 = (death - birth) / 2 ** 0.5

        x1_log = torch.where(x1 < self.nu,torch.log(x1.clamp(min=1e-8)/self.nu)*self.nu + self.nu,x1)

        s = torch.exp(-self.sigma[:,0]**2 * (x0[...,None] - self.mu[:,0]) ** 2 - self.sigma[:,1] ** 2 * (x1_log[...,None] - self.mu[:,1]) ** 2)
        keep = (x1 > 0) & mask
        s = s * keep[...,None]
        return s.sum(dim=1)

class EssentialLayer(nn.Module):
    """1D structure elements for essential points.

    An essential point never dies, so it only has a birth x. Element i takes
    the value exp(-sigma_i^2 (x - mu_i)^2), and the output is again the sum
    over points. This is the plain Gaussian part of the 2D formula; check
    Section 5 of the paper for exactly how they treat essential points.
    """

    def __init__(self, n_elements):
        super().__init__()
        self.mu = nn.Parameter(torch.rand(n_elements))                   # (N,)
        self.sigma = nn.Parameter(torch.full((n_elements,), 3.0 ** 0.5))  # (N,)

    def forward(self, points, mask):
        """points: (B, P, 1) births. mask: (B, P). Returns (B, N)."""
        # TODO 5: evaluate exp(-sigma^2 (x - mu)^2) for every point and
        #   element, shape (B, P, N); set padding to 0; sum over the points.
        x = points[:,:,0]
        s = torch.exp(-self.sigma ** 2 * (x[...,None] - self.mu) ** 2)
        s = mask[...,None] * s
        return s.sum(dim=1)
