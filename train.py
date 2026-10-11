"""Train the NumPy ViT tagger on synthetic data (CPU only).

    python train.py --n-train 8000 --epochs 20 --out v11_weights.npz
"""
import argparse
import json
import os
import sys
import time
import numpy as np

import tags as T
import synth
from vit import TinyViT, AdamW, loss_and_grad, sigmoid


def augment(x, rng):
    x = x.astype(np.float32)
    flip = rng.random(len(x)) < .5
    x[flip] = x[flip][:, :, ::-1]
    x *= rng.uniform(.9, 1.1, (len(x), 1, 1, 1)).astype(np.float32)
    x += rng.normal(0, 4, x.shape).astype(np.float32)
    return np.clip(x, 0, 255)


def predict_all(model, X, bs=256):
    out = []
    for i in range(0, len(X), bs):
        out.append(model.forward(X[i:i + bs])[0])
    return np.concatenate(out)


def f1_best_threshold(p, y):
    best_t, best_f = .5, -1
    for t in np.arange(.05, .95, .05):
        pred = p >= t
        tp = (pred & (y > 0)).sum()
        f = 2 * tp / max(1, pred.sum() + (y > 0).sum())
        if f > best_f:
            best_t, best_f = t, f
    return best_t, best_f


def evaluate(model, X, EX, MU, thr=None):
    lg = predict_all(model, X)
    loss, _ = loss_and_grad(lg, EX, MU)
    acc = {}
    for gi, (s, e) in enumerate(T.GROUPS):
        ok = EX[:, gi] >= 0
        acc[T.GROUP_NAMES[gi]] = float((lg[ok, s:e].argmax(1) == EX[ok, gi]).mean()) if ok.any() else float("nan")
    s, e = T.MULTI_SLICE
    p = sigmoid(lg[:, s:e])
    t = np.full(len(T.MULTI), .5) if thr is None else thr
    pred = p >= t
    tp = (pred & (MU > 0)).sum(0)
    f1 = 2 * tp / np.maximum(1, pred.sum(0) + (MU > 0).sum(0))
    return loss, acc, f1, p


def flag_metrics(lg, EX, thr=.5):
    """Moderation view of the rating head: flagged = P(questionable)+P(explicit) >= thr."""
    from vit import softmax
    gi = T.GROUP_NAMES.index("rating")
    gs, ge = T.GROUPS[gi]
    p = softmax(lg[:, gs:ge])
    score = p[:, 2:].sum(1)
    y = EX[:, gi] >= 2
    pred = score >= thr
    tp = (pred & y).sum()
    return dict(precision=float(tp / max(1, pred.sum())), recall=float(tp / max(1, y.sum())),
                support=int(y.sum())), score, y


def tune_flag_threshold(score, y, target_recall=.95):
    """Highest threshold that still reaches target recall (moderation prefers recall)."""
    for t in np.arange(.9, .04, -.05):
        pred = score >= t
        if (pred & y).sum() / max(1, y.sum()) >= target_recall:
            return float(t)
    return .05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="", help="prepared.npz from prep_dataset.py (else synthetic)")
    ap.add_argument("--cache", default="synth_cache.npz")
    ap.add_argument("--out", default="v11_weights.npz")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (benchmarking)")
    a = ap.parse_args()

    if a.data:
        d = np.load(a.data)
        X, EX, MU = d["X"], d["EX"], d["MU"]
        T.load_vocab_json(d["vocab"])
        a.size = X.shape[1]
        n = len(X)
        if n < 10:
            sys.exit(f"only {n} usable samples in {a.data} - label more images first (need ~50 minimum, hundreds for real results)")
        # shuffle once so the validation split is not just the last files alphabetically
        perm0 = np.random.default_rng(a.seed).permutation(n)
        X, EX, MU = X[perm0], EX[perm0], MU[perm0]
        max_val = max(2, int(n * 0.25))
        if a.n_val > max_val:
            print(f"note: --n-val {a.n_val} is too large for {n} samples; using {max_val}")
            a.n_val = max_val
        a.n_train = n - a.n_val
        print(f"loaded {a.data}: {X.shape}, {len(T.MULTI)} tags, groups {T.GROUP_NAMES}")
    elif os.path.exists(a.cache) and not a.max_steps:
        d = np.load(a.cache)
        X, EX, MU = d["X"], d["EX"], d["MU"]
        print("loaded cached data", X.shape)
    else:
        t0 = time.time()
        X, EX, MU = synth.make_dataset(a.n_train + a.n_val, a.size, seed=a.seed + 1)
        print(f"generated {len(X)} images in {time.time() - t0:.1f}s")
        if not a.max_steps:
            np.savez_compressed(a.cache, X=X, EX=EX, MU=MU)
    Xtr, EXtr, MUtr = X[:a.n_train], EX[:a.n_train], MU[:a.n_train]
    Xva, EXva, MUva = X[a.n_train:], EX[a.n_train:], MU[a.n_train:]

    freq = np.clip(MUtr.mean(0), 1e-3, 1)
    pos_w = np.clip((1 - freq) / freq, 1, 4).astype(np.float32)

    model = TinyViT(img=a.size, patch=a.patch, dim=a.dim, depth=a.depth,
                    heads=a.heads, seed=a.seed)
    print(f"params: {model.n_params():,}  tokens: {(a.size // a.patch) ** 2}")
    opt = AdamW(model.P, lr=a.lr)
    rng = np.random.default_rng(a.seed)
    a.batch = max(1, min(a.batch, len(Xtr)))          # small datasets: never a batch bigger than the data
    steps_per_epoch = max(1, len(Xtr) // a.batch)
    total, step = steps_per_epoch * a.epochs, 0
    warm = max(1, min(100, total // 10))
    ckpt, start_ep = a.out + ".ckpt.npz", 0
    if os.path.exists(ckpt):  # resume after an interruption
        c = np.load(ckpt)
        same = all(("p/" + k) in c.files and c["p/" + k].shape == model.P[k].shape for k in model.P)
        if not same or int(c["epoch"]) >= a.epochs:
            print("ignoring old checkpoint (different data/settings or already finished); training from scratch")
            os.remove(ckpt)
            c = None
    if os.path.exists(ckpt):
        for k in model.P:
            model.P[k][...] = c["p/" + k]
            opt.m[k][...] = c["m/" + k]
            opt.v[k][...] = c["v/" + k]
        opt.t, start_ep = int(c["t"]), int(c["epoch"])
        step = start_ep * steps_per_epoch
        rng = np.random.default_rng(a.seed + start_ep)
        print(f"resumed from epoch {start_ep}")

    for ep in range(start_ep, a.epochs):
        perm = rng.permutation(len(Xtr))
        t0, run = time.time(), 0.
        for bi in range(steps_per_epoch):
            idx = perm[bi * a.batch:(bi + 1) * a.batch]
            xb = augment(Xtr[idx], rng)
            lg, cache = model.forward(xb)
            loss, dl = loss_and_grad(lg, EXtr[idx], MUtr[idx], pos_w)
            opt.step(model.backward(dl, cache),
                     lr=a.lr * min(1, (step + 1) / warm) *
                     .5 * (1 + np.cos(np.pi * step / total)))
            run += loss
            step += 1
            if a.max_steps and step >= a.max_steps:
                print(f"{step} steps in {time.time() - t0:.1f}s "
                      f"({(time.time() - t0) / step * 1000:.0f} ms/step)")
                return
        vl, acc, f1, _ = evaluate(model, Xva, EXva, MUva)
        print(f"ep {ep + 1:2d}/{a.epochs} train {run / steps_per_epoch:.3f} val {vl:.3f} "
              f"| " + " ".join(f"{k}={v:.2f}" for k, v in acc.items()) +
              f" | tagF1 {f1.mean():.3f} | {time.time() - t0:.0f}s", flush=True)
        np.savez(ckpt, t=opt.t, epoch=ep + 1,
                 **{"p/" + k: v for k, v in model.P.items()},
                 **{"m/" + k: v for k, v in opt.m.items()},
                 **{"v/" + k: v for k, v in opt.v.items()})

    # tune one threshold per tag on validation data
    _, _, _, p = evaluate(model, Xva, EXva, MUva)
    thr = np.array([f1_best_threshold(p[:, i], MUva[:, i])[0] for i in range(len(T.MULTI))],
                   np.float32)
    vl, acc, f1, _ = evaluate(model, Xva, EXva, MUva, thr)
    print("\nFinal per-tag F1 (tuned thresholds)" +
          (" - 15 weakest:" if len(T.MULTI) > 15 else ":"))
    for t, f, th in sorted(zip(T.MULTI, f1, thr), key=lambda r: r[1])[:15]:
        print(f"  {t:18s} F1 {f:.2f}  thr {th:.2f}")
    extra = {}
    if "rating" in T.GROUP_NAMES:
        lg = predict_all(model, Xva)
        _, score, yflag = flag_metrics(lg, EXva)
        ft = tune_flag_threshold(score, yflag)
        m, _, _ = flag_metrics(lg, EXva, ft)
        print(f"flag head (questionable+explicit): threshold {ft:.2f} -> "
              f"precision {m['precision']:.3f} recall {m['recall']:.3f} (n_flagged_val={m['support']})")
        extra["flag_threshold"] = np.float32(ft)
    print("exclusive accuracy:", {k: round(v, 3) for k, v in acc.items()}, "| mean F1", round(float(f1.mean()), 3))
    model.save(a.out, thresholds=thr, vocab=np.array(T.vocab_json()), **extra)
    print("saved", a.out)
    if os.path.exists(ckpt):
        os.remove(ckpt)   # finished: a leftover checkpoint would hijack the next run


if __name__ == "__main__":
    main()
