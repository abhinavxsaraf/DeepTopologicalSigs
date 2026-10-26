"""Train the MPEG-7 network with the authors' protocol (Hofer et al. 2017;
their src/mpeg7/experiments.py).

- One random 90/10 split, stratified: 18 train and 2 test images per class.
- CrossEntropy, SGD (lr 0.1, momentum 0.7), batch size 128, 300 epochs,
  lr halved every 20 epochs.
- The score is the mean test accuracy over the last 10 epochs. The paper
  reports 91.8%.

Run: python train.py            (python train.py --help for the options)
The per-epoch loss and accuracies are saved to train_log.json, and also
sent to Weights & Biases with --wandb (run `wandb login` once first).
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from data import load_dataset
from model import Model


def split(labels, test_size=0.1, seed=0):
    """Stratified random split of the images.

    labels: (n,) long tensor. Returns (train_idx, test_idx), two long
    tensors of image indices that together cover 0..n-1 once each.
    """

    train,test = train_test_split(np.arange(len(labels)),test_size=test_size,random_state=seed,stratify=labels.numpy())
    return torch.as_tensor(train),torch.as_tensor(test)


def train_epoch(model, points, mask, labels, optimizer, batch_size, device):
    """One pass over (points, mask, labels) in a random order, one SGD step
    per batch. The tensors are the training set only and stay on the CPU;
    each batch is moved to `device`. Returns the mean loss over the samples.
    """
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    n = points.shape[0]
    order = torch.randperm(n)
    total = 0

    for start in range(0,n,batch_size):
        idx = order[start:start + batch_size]
        x = points[idx].to(device)
        m = mask[idx].to(device)
        y = labels[idx].to(device)
        logits = model(x,m)
        loss = loss_fn(logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += len(idx) * loss.item()
    return total / n

@torch.inference_mode()
def evaluate(model, points, mask, labels, batch_size, device):
    """Fraction of correctly classified samples, a float in [0, 1]."""

    model.eval()
    n = points.shape[0]
    correct = 0

    for start in range(0,n,batch_size):
        x = points[start:start+batch_size].to(device)
        m = mask[start:start+batch_size].to(device)
        y = labels[start:start+batch_size].to(device)
        logits = model(x,m)
        correct += (logits.argmax(dim=1) == y).sum().item()
    return correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.7)
    parser.add_argument("--lr-step", type=int, default=20,
                        help="halve the lr every this many epochs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", default="train_log.json")
    parser.add_argument("--wandb", action="store_true",
                        help="also log to Weights & Biases")
    parser.add_argument("--wandb-project", default="toposig-mpeg7")
    parser.add_argument("--name", default=None, help="wandb run name")
    args = parser.parse_args()

    if args.wandb:
        # Imported here so that runs without --wandb don't need it installed.
        import wandb
        wandb.init(project=args.wandb_project, name=args.name, config=vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    points, mask, labels, classes = load_dataset()
    train_idx, test_idx = split(labels, seed=args.seed)
    train = points[train_idx], mask[train_idx], labels[train_idx]
    test = points[test_idx], mask[test_idx], labels[test_idx]
    print(f"{len(train_idx)} train, {len(test_idx)} test images, "
          f"{len(classes)} classes, device {device}")

    model = Model(n_directions=points.shape[1], n_classes=len(classes)).to(device)

    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.lr_step,gamma=0.5)

    history = {"args": vars(args), "loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]  # the lr this epoch trains with
        loss = train_epoch(model, *train, optimizer, args.batch_size, device)
        train_acc = evaluate(model, *train, args.batch_size, device)
        test_acc = evaluate(model, *test, args.batch_size, device)
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

    last = history["test_acc"][-10:]
    print(f"mean test accuracy over the last {len(last)} epochs: "
          f"{np.mean(last):.4f} (paper: 0.918)")
    if args.wandb:
        wandb.summary["final_test_acc"] = float(np.mean(last))
        wandb.finish()


if __name__ == "__main__":
    main()
