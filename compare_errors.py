"""Which test images each model gets wrong, side by side.

Reads the logs of train.py / resnet_baseline.py runs on the same seed (so the
same test images), which hold the last epoch's test predictions:

    python compare_errors.py tda=train_log_seed3.json small=small_log_seed3.json

Prints every test image that at least one model gets wrong, with each model's
prediction, then how many errors each model has, how many are shared and how
many a majority vote of the models would make.
"""
import json
import sys
from collections import Counter

from shapes import PATTERN, find_files


def load(spec):
    """'name=path' -> (name, {image index: predicted class})."""
    name, path = spec.split("=", 1)
    with open(path) as f:
        log = json.load(f)
    if "test_pred" not in log:
        sys.exit(f"{path} has no test predictions: rerun it with the current script")
    return name, dict(zip(log["test_idx"], log["test_pred"]))


def main(specs, root="MPEG7"):
    files = find_files(root)
    names = [PATTERN.match(f.stem).group(1) for f in files]
    classes = sorted(set(names))
    labels = [classes.index(n) for n in names]

    runs = [load(s) for s in specs]
    test = sorted(runs[0][1])
    if any(sorted(preds) != test for _, preds in runs):
        sys.exit("the runs have different test images: use logs from the same seed")

    wrong = {name: {i for i in test if preds[i] != labels[i]} for name, preds in runs}
    width = max(len(f.stem) for f in files)
    print(f"{'image':{width}}  " + "  ".join(f"{name:>14}" for name, _ in runs))
    for i in sorted(set().union(*wrong.values()), key=lambda i: files[i].stem):
        cells = [classes[preds[i]] if i in wrong[name] else "ok"
                 for name, preds in runs]
        print(f"{files[i].stem:{width}}  " + "  ".join(f"{c:>14}" for c in cells))

    print()
    for name, w in wrong.items():
        print(f"{name}: {len(w)} wrong of {len(test)}")
    print(f"wrong in all models: {len(set.intersection(*wrong.values()))}")
    if len(runs) >= 3:
        # Majority vote; ties go to the first model's prediction.
        vote_wrong = 0
        for i in test:
            votes = Counter(preds[i] for _, preds in runs)
            top = max(votes.values())
            pick = next(preds[i] for _, preds in runs if votes[preds[i]] == top)
            vote_wrong += pick != labels[i]
        print(f"majority vote: {vote_wrong} wrong of {len(test)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
