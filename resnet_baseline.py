"""Non-TDA baselines: a CNN on the MPEG-7 masks themselves.

--arch picks the network:
- resnet18 (default): torchvision's ResNet-18, 11.2M parameters.
- small: SmallCNN, ~176K parameters, about the size of the TDA model
  (184K). Always trained from scratch.

Same protocol as train.py, so the numbers compare directly:
- the same stratified 90/10 split for a given seed (train.split),
- the score is the mean test accuracy over the last 10 epochs.
The TDA model gets 92.5 ± 3.2% over seeds 0-4.

Input: each mask reduced to its largest component (as for the diagrams),
cropped to its bounding box, padded to a square and resized to size × size.

Two switches:
- --pretrained: start from ImageNet weights instead of random ones.
- --augment: random rotations, flips and scale changes while training.
  Off by default, since the TDA model trains without augmentation.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

from shapes import load_dataset as load_masks, largest_component
from train import split, subsample

# ImageNet statistics, which the pretrained weights expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Rotation by any angle is safe: to_square leaves room for it (see there).
AUGMENT = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=180, scale=(0.85, 1.0)),
])


def to_square(mask, size):
    """One mask as a size × size float image, object = 1, background = 0.

    mask: 2D bool array (any shape). Returns a (size, size) float32 tensor
    with values in [0, 1].

    1. Crop the mask to the bounding box of its object pixels (h × w).
    2. Pad it with background to a side × side square, side = ceil(√(h² + w²)),
       with the box in the centre. The box then fits inside the square in any
       rotation about the centre, so AUGMENT never cuts a shape off.
    3. Resize to size × size. Most images are shrunk a lot (up to ~1000 px
       down to 128), so smooth while shrinking: interpolate with
       mode="bilinear" and antialias=True. That gives grey edge pixels
       rather than a jagged 0/1 outline.

    Translation and scale of the original shape don't matter after this:
    only the shape inside its bounding box is kept.
    """
    rows, cols = np.nonzero(mask)
    box = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = box.shape

    side = math.ceil(math.hypot(h, w))
    square = np.zeros((side, side), dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = box

    x = torch.from_numpy(square)[None, None]
    x = nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                  antialias=True, align_corners=False)
    return x[0, 0].clamp(0, 1)


def load_images(root="MPEG7", size=128, cache=None):
    """All 1400 images as a (1400, 1, size, size) uint8 tensor (0-255),
    plus labels (long tensor) and class names, in the same order as
    shapes.load_dataset and compute_diagrams.py.

    If cache is given and exists it is loaded instead; otherwise it is
    written after computing.
    """
    if cache and Path(cache).exists():
        data = torch.load(cache)
        return data["images"], data["labels"], data["classes"]

    masks, labels, classes, files = load_masks(root)
    images = torch.stack([to_square(largest_component(m), size) for m in masks])
    images = (images * 255).round().to(torch.uint8).unsqueeze(1)
    labels = torch.as_tensor(labels)
    if cache:
        torch.save({"images": images, "labels": labels, "classes": classes}, cache)
    return images, labels, classes


class ImageSet(Dataset):
    """Images as floats in [0, 1], each passed through transform if given."""

    def __init__(self, images, labels, transform=None):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i].float() / 255
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i]


def make_model(arch, n_classes, pretrained=False):
    """The network for --arch: make_resnet or SmallCNN."""
    if arch == "small":
        return SmallCNN(n_classes)
    return make_resnet(n_classes, pretrained)


class SmallCNN(nn.Module):
    """A plain CNN of about the TDA model's size, (B, 1, H, W) -> (B, n_classes).

    Four blocks of conv 3×3 (no bias) -> BatchNorm -> ReLU -> max-pool 2×2,
    with 32, 48, 96 and 128 channels; then a global average pool and a
    Linear 128 -> n_classes. Pooling over the whole map makes the head
    independent of the image size. ~176K parameters for 70 classes.
    """

    def __init__(self, n_classes, channels=(32, 48, 96, 128)):
        super().__init__()
        blocks = []
        c_in = 1
        for c_out in channels:
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.fc = nn.Linear(c_in, n_classes)

    def forward(self, x):
        return self.fc(self.features(x).mean(dim=(2, 3)))


def make_resnet(n_classes, pretrained=False):
    """ResNet-18 that takes (B, 1, H, W) images in [0, 1] and returns
    (B, n_classes) logits.

    - torchvision's resnet18, with ImageNet weights if pretrained, else
      randomly initialised (nothing downloaded).
    - Its ImageNet head (Linear 512 -> 1000) is replaced by a fresh
      Linear 512 -> n_classes.
    - The ResNet expects 3 normalised channels, so the one channel is
      repeated 3 times and normalised with IMAGENET_MEAN / IMAGENET_STD.
      This is done in both cases, so the two runs differ only in the weights.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return nn.Sequential(GreyToRGB(), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD), model)


class GreyToRGB(nn.Module):
    """(B, 1, H, W) -> (B, 3, H, W), the one channel repeated."""

    def forward(self, x):
        return x.repeat(1, 3, 1, 1)


def train_epoch(model, loader, optimizer, device):
    """One pass over loader (which shuffles and augments), one SGD step per
    batch. Returns the mean loss over the samples, as in train.py.
    """
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    total = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += len(y) * loss.item()
        n += len(y)
    return total / n


@torch.inference_mode()
def evaluate(model, loader, device):
    """Fraction of correctly classified samples, a float in [0, 1]."""
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        n += len(y)
    return correct / n


@torch.inference_mode()
def predict(model, loader, device):
    """Predicted class of each sample in loader's order, a long tensor on the CPU."""
    model.eval()
    return torch.cat([model(x.to(device)).argmax(dim=1).cpu() for x, _ in loader])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["resnet18", "small"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true",
                        help="start from ImageNet weights")
    parser.add_argument("--augment", action="store_true",
                        help="random flips, rotations and scaling while training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="default 0.01 if --pretrained, else 0.1")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--size", type=int, default=128, help="image side in pixels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-per-class", type=int, default=None,
                        help="train on only this many images per class (the test set stays the same)")
    parser.add_argument("--root", default="MPEG7")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log", default="resnet_log.json")
    parser.add_argument("--wandb", action="store_true",
                        help="also log to Weights & Biases")
    parser.add_argument("--wandb-project", default="toposig-mpeg7")
    parser.add_argument("--name", default=None, help="wandb run name")
    args = parser.parse_args()
    if args.pretrained and args.arch == "small":
        parser.error("--pretrained needs --arch resnet18: SmallCNN has no pretrained weights")
    if args.lr is None:
        # Fine-tuning pretrained weights needs a smaller step than training from scratch.
        args.lr = 0.01 if args.pretrained else 0.1
    if args.name is None:
        args.name = (f"{args.arch}-{'pretrained' if args.pretrained else 'scratch'}"
                     f"{'-aug' if args.augment else ''}"
                     f"{f'-n{args.train_per_class}' if args.train_per_class else ''}"
                     f"-seed{args.seed}")

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.name, config=vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, labels, classes = load_images(args.root, args.size,
                                          cache=f"images_{args.size}.pt")
    train_idx, test_idx = split(labels, seed=args.seed)
    if args.train_per_class:
        train_idx = subsample(train_idx, labels, args.train_per_class, args.seed)
    train_set = ImageSet(images[train_idx], labels[train_idx],
                         AUGMENT if args.augment else None)
    test_set = ImageSet(images[test_idx], labels[test_idx])
    # The train accuracy is measured without augmentation, like the test accuracy.
    train_eval_set = ImageSet(images[train_idx], labels[train_idx])
    loader = lambda ds, shuffle: DataLoader(ds, args.batch_size, shuffle=shuffle,
                                            num_workers=args.workers)
    print(f"{len(train_idx)} train, {len(test_idx)} test images, "
          f"{len(classes)} classes, device {device}, run {args.name}")

    model = make_model(args.arch, len(classes), args.pretrained).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    history = {"args": vars(args), "loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        loss = train_epoch(model, loader(train_set, True), optimizer, device)
        train_acc = evaluate(model, loader(train_eval_set, False), device)
        test_acc = evaluate(model, loader(test_set, False), device)
        scheduler.step()
        history["loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"epoch {epoch:3d}  lr {lr:.2e}  "
              f"loss {loss:.4f}  train {train_acc:.3f}  test {test_acc:.3f}  "
              f"({time.time() - start:.1f}s)", flush=True)
        with open(args.log, "w") as f:
            json.dump(history, f)
        if args.wandb:
            wandb.log({"loss": loss, "train_acc": train_acc,
                       "test_acc": test_acc, "lr": lr}, step=epoch)

    # The last epoch's test predictions, for compare_errors.py.
    history["test_idx"] = test_idx.tolist()
    history["test_pred"] = predict(model, loader(test_set, False), device).tolist()
    with open(args.log, "w") as f:
        json.dump(history, f)

    last = history["test_acc"][-10:]
    print(f"mean test accuracy over the last {len(last)} epochs: "
          f"{np.mean(last):.4f} (TDA model: 0.925 ± 0.032)")
    if args.wandb:
        wandb.summary["final_test_acc"] = float(np.mean(last))
        wandb.finish()


if __name__ == "__main__":
    main()
