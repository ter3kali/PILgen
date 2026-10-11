"""One command: your labelled folder -> trained weights (prep + train with sensible settings).

    python retrain.py --images images --tags tags --out my_weights.npz [--epochs 30]
Then label faster:   python label_tool.py --images images --tags tags --weights my_weights.npz
Re-run after each labelling session; the model gets better as labels grow.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--tags", default="tags")
    ap.add_argument("--out", default="my_weights.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--size", type=int, default=64)
    a = ap.parse_args(argv)
    if not os.path.isdir(a.tags):
        sys.exit(f"no labels yet: {a.tags} (run label_tool.py first)")
    n = len([f for f in os.listdir(a.tags) if f.endswith((".txt", ".json"))])
    if n < 20:
        sys.exit(f"only {n} labelled images - label at least ~50 before training (a few hundred for real results)")
    n_val = max(5, n // 10)
    min_count = max(3, min(20, n // 30))
    prepared = os.path.splitext(a.out)[0] + "_prepared.npz"
    py = sys.executable
    print(f"{n} labelled images -> validation {n_val}, min tag count {min_count}")
    steps = [[py, os.path.join(HERE, "prep_dataset.py"), "--images", a.images, "--tags", a.tags, "--size", str(a.size),
              "--top-k", "300", "--min-count", str(min_count), "--out", prepared],
             [py, os.path.join(HERE, "train.py"), "--data", prepared, "--n-val", str(n_val), "--epochs", str(a.epochs),
              "--size", str(a.size), "--out", a.out]]
    for cmd in steps:
        r = subprocess.run(cmd, cwd=None)
        if r.returncode:
            sys.exit(f"step failed: {os.path.basename(cmd[1])}")
    print(f"\nsaved {a.out}. Use it:\n  python ipai.py label --images {a.images} --tags {a.tags} --weights {a.out}\n"
          f"  python ipai.py batch --images {a.images} --out prompts --vit {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
