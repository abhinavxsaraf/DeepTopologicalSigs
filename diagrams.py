"""H0 persistence diagrams of binary shapes under directional height filtrations.

- h0_diagram: union-find written by hand, so the algorithm is visible.
- h0_diagram_gudhi: the same diagram from gudhi, as a cross-check.

"""
import numpy as np
import gudhi

# Pairs with death - birth below this are dropped as floating-point noise:
# cos(pi/2) is 6e-17, not 0, so pixels that should tie can differ slightly.
EPS = 1e-9


def heights(mask, angle):
    """Normalised height f(p) = (<p - b, d> + r) / (2r) of every object pixel p.

    Coordinates: p = (x, y) with x = column (grows to the right) and y = -row
    (grows upwards, since row 0 is the top of the image).
    - d = (cos(angle), sin(angle)): angle = 0 points right, pi/2 points up.
    - b: barycentre of the object, the mean of its pixel coordinates.
    - r: largest distance from b to an object pixel.
    Subtracting b removes position and dividing by r removes size, so
    <p - b, d> / r lies in [-1, 1]; the + r and 2r then move it to [0, 1].
    This matches the authors' code (NormalizedBarycentricHeightFiltration in
    pershombox), which the initial values in layers.py assume.

    Returns rows, cols, h: three 1D arrays with one entry per object pixel.
    """
    
    rows,cols = np.nonzero(mask)
    d = np.array([np.cos(angle),np.sin(angle)])
    pixels = np.column_stack([cols,-rows]).astype(float)
    b = np.mean(pixels,axis=0,keepdims=True)
    r = np.max(np.linalg.norm(pixels-b,axis=1))
    heights = ((pixels-b) @ d + r) / (2 * r)
    return rows,cols,heights



def h0_diagram(mask, angle):
    """H0 diagram of the object, by union-find.

    Pixels are added from lowest to highest height. Each one either starts a
    new component (no added 4-neighbours), joins one, or merges several.
    On a merge the elder rule applies: the component with the smallest birth
    survives, and every other one dies at the current height.

    Returns
      pairs: (n, 2) array of (birth, death) for components that died,
             without pairs where death - birth <= EPS.
      essential: sorted 1D array with the births of components that never died.
    """
    rows, cols, h = heights(mask, angle)
    H, W = mask.shape
    # Pixel (r, c) has the flat index r * W + c.
    # parent[i] == -1 means pixel i has not been added yet.
    parent = np.full(H * W, -1)
    birth = np.zeros(H * W)  # only meaningful at roots

    def find(i):
        """Root of i's component. Path halving keeps the trees shallow."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for k in np.argsort(h, kind="stable"):
        r, c, t = rows[k], cols[k], h[k]
        p = r * W + c
        # Roots of the components that p's already-added neighbours are in.
        roots = set()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and parent[rr * W + cc] != -1:
                roots.add(find(rr * W + cc))
        
        if not roots:
            parent[p] = p
            birth[p] = t
        elif len(roots) == 1:
            parent[p] = next(iter(roots))
        else:
            elder = min(roots,key=lambda q: birth[q])
            for q in roots:
                if q != elder:
                    if t - birth[q] > EPS:
                        pairs.append((birth[q],t))
                    parent[q] = elder
            parent[p] = elder

    alive = {find(r * W + c) for r, c in zip(rows, cols)}
    essential = np.sort([birth[q] for q in alive])

    return np.array(pairs).reshape(-1, 2), essential


def h0_diagram_gudhi(mask, angle):
    """Same output as h0_diagram, computed with gudhi."""
    rows, cols, h = heights(mask, angle)

    # gudhi wants a value for every pixel, not only the object pixels.
    # Background gets a value above every object height, so it enters the
    # filtration last; anything involving it is removed below.
    big = h.max() + 1
    f = np.full(mask.shape, big)
    f[rows, cols] = h

    # vertices=f: each pixel is a vertex with filtration value f. gudhi adds
    # the edges and squares between neighbouring pixels, each entering when
    # its last vertex does. Edges join only 4-neighbours, so H0 matches the
    # 4-connected union-find.
    cc = gudhi.CubicalComplex(vertices=f)
    cc.compute_persistence()
    d = cc.persistence_intervals_in_dimension(0)  # (n, 2) array of (birth, death)

    d = d[d[:, 0] < big]          # drop components born in the background
    # Once the background enters, every object component merges into it, so
    # components that are essential for the object die at `big` (or inf).
    essential = np.sort(d[d[:, 1] >= big, 0])
    pairs = d[d[:, 1] < big]
    pairs = pairs[pairs[:, 1] - pairs[:, 0] > EPS]
    return pairs, essential
