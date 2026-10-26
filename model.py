"""The MPEG-7 network of Hofer et al. 2017 (supplementary figure; their
src/mpeg7/experiments.py, MyModel).

Input: for every image, 32 H0 diagrams, one per filtration direction
(data.load_dataset gives them padded: points (B, 32, P, 2), mask (B, 32, P)).

- Branch: one per direction i. It looks at the diagrams of directions
  i - 1, i and i + 1 (the "3-tuple" in the figure) and turns them into
  25 features.
- Model: the 32 branches side by side, their 32 * 25 features concatenated,
  then a small classifier head with 70 outputs.
"""
import torch
from torch import nn

from layers import StructureElementLayer


class Branch(nn.Module):
    """One direction's branch.

      3 diagrams --input layer--> (3, 75)       
                 --conv 3->16, conv 16->4--> (4, 75)
                 --max over the 4 channels--> (75,)
                 --Linear 75->25, BatchNorm, Linear 25->25, ReLU, Dropout--> (25,)

    The convolutions have kernel size 1: for each structure element j they
    take the 3 values (one per view) and mix them into 16, then 4, numbers,
    with the same weights for every j. They have no bias and no activation
    in between, as in the authors' code.
    """

    def __init__(self, n_elements=75, s=4.0):
        super().__init__()
        # s = 4: their sharpness for the shape datasets (sigma = 2 here).
        self.slayer = StructureElementLayer(n_elements, s=s)
        self.conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=1, bias=False),
            nn.Conv1d(16, 4, kernel_size=1, bias=False),
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_elements, 25),
            nn.BatchNorm1d(25),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, points, mask):
        """points: (B, 3, P, 2). mask: (B, 3, P). Returns (B, 25)."""
        B,_,P,_ = points.shape
        merged_points = points.reshape(-1,P,2)
        merged_mask = mask.reshape(-1,P)

        features = self.slayer(merged_points,merged_mask)
        features = features.reshape(B,3,-1)

        conv_features = self.conv(features)
        conv_features = conv_features.amax(dim=1)

        return self.mlp(conv_features)


class Model(nn.Module):
    """32 branches, concatenated, then the head:
    Linear 800->100, BatchNorm, Dropout, Linear 100->70.

    The output is raw class scores (logits); nn.CrossEntropyLoss applies
    the softmax itself.
    """

    def __init__(self, n_directions=32, n_classes=70, n_elements=75, s=4.0):
        super().__init__()
        self.branches = nn.ModuleList(
            Branch(n_elements, s) for _ in range(n_directions))
        self.head = nn.Sequential(
            nn.Linear(n_directions * 25, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.3),
            nn.Linear(100, n_classes),
        )

    def forward(self, points, mask):
        """points: (B, D, P, 2). mask: (B, D, P). Returns (B, n_classes)."""

        B,D,P,_ = points.shape
        branch_outputs = []
        for i,branch in enumerate(self.branches):
            idx = [(i-1) % D,i % D,(i+1) % D]
            out = branch(points[:,idx],mask[:,idx])
            branch_outputs.append(out)
        
        out = torch.cat(branch_outputs,dim=1)
        return self.head(out)
