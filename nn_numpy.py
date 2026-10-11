"""
A small neural network in plain NumPy (no PyTorch) that learns caption -> shape program.

Idea
    Input : the caption as word + word-pair features (bag of words and bigrams)
    Net   : one hidden layer (ReLU), backprop and Adam written by hand
    Output: a fixed number of shape "slots" (the first K shapes of each program,
            which are the big ones, since the fitter works big to small).
            Each slot predicts: which kind of shape (or none), plus the numbers
            (coordinates and color) for every kind. Plus the background color.
    Loss  : cross-entropy for the kind + squared error for the numbers of the
            true kind only.

Usage
    python nn_numpy.py train data --slots 40 --hidden 256 --epochs 300
    python nn_numpy.py gen model_np.npz "girl with blue hair" --n 4

Needs the dataset folder from build_dataset.py with captions filled in.

v2 changes (why the output was blurry, and what was fixed)
    * strokes are now learned (they used to be silently dropped, shifting every slot)
    * opacity (alpha) is now learned (it used to be thrown away, so semi-transparent
      shapes came back solid)
    * --order spatial (default): the first K shapes are put in a fixed top-to-bottom,
      left-to-right order before training, so "slot 7" means the same *place* in every
      program instead of an arbitrary shape. Decoding paints big shapes first.
      --order draw keeps the original file order (old behaviour).
    * models saved by the old version cannot be loaded; retrain.
Still true: a regression net outputs the *average* of all programs a caption could mean,
so some softness is inherent. Use --temp/--noise for variety, or retrieval for sharpness.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from imgprog import NCOORD, render, similarity

KINDS = ["circle", "rect", "tri", "line", "stroke"]
NK = len(KINDS)
NL = NK + 1            # kind logits per slot (index 0 = empty slot)
PD = 12                # numbers per kind: up to 8 coordinates | R G B at 8:11 | opacity at 11
SLOT = NL + NK * PD    # outputs per slot
COL0, ALPHA = 8, 11
VERSION = 2


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def feats(cap):
    w = words(cap)
    return set(w) | {a + "_" + b for a, b in zip(w, w[1:])}


def featurize(caps, vocab):
    X = np.zeros((len(caps), len(vocab)), np.float32)
    for i, c in enumerate(caps):
        for f in feats(c):
            j = vocab.get(f)
            if j is not None:
                X[i, j] = 1
    return X / np.sqrt(np.maximum(X.sum(1, keepdims=True), 1))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------- program <-> arrays ----------------
def parse(prog):
    size, bg, shapes = 64, (0, 0, 0), []
    for line in prog.strip().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "size":
            size = int(p[1])
        elif p[0] == "bg":
            bg = tuple(int(x) for x in p[1:4])
        elif p[0] == "skip":
            shapes.append(("skip", [], [], 255))
        elif p[0] in KINDS:
            v = [int(x) for x in p[1:]]
            n = NCOORD[p[0]]
            if len(v) in (n + 3, n + 4):               # coordinates, R G B, optional opacity
                shapes.append((p[0], v[:n], v[n:n + 3], v[n + 3] if len(v) > n + 3 else 255))
    return size, bg, shapes


def _is_width(op, i):
    return (op == "line" and i == 4) or (op == "stroke" and i in (6, 7))


def _area(op, v):
    """rough size of a shape (bounding-box area), used to paint big shapes first"""
    if op == "circle":
        return (2 * v[2]) ** 2
    xs, ys = v[0:len(v):2][:3 if op in ("tri", "stroke") else 2], v[1:len(v):2][:3 if op in ("tri", "stroke") else 2]
    return (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)


def _centre(op, v):
    if op == "circle":
        return v[0], v[1]
    n = 3 if op in ("tri", "stroke") else 2
    xs, ys = v[0:2 * n:2], v[1:2 * n:2]
    return sum(xs) / n, sum(ys) / n


def targets(prog, K, order="spatial"):
    size, bg, shapes = parse(prog)
    live = [s for s in shapes if s[0] != "skip"]
    if order == "spatial":
        live = live[:K]                                # the K biggest (fitter works big -> small)
        live.sort(key=lambda s: (int(_centre(s[0], s[1])[1] * 4 / size), _centre(s[0], s[1])[0]))
        chosen = live
    else:
        chosen = shapes[:K]                            # original file order, skips stay as gaps
    kind = np.zeros(K, np.int64)
    par = np.zeros((K, PD), np.float32)
    msk = np.zeros((K, PD), np.float32)
    for k, (op, v, col, al) in enumerate(chosen[:K]):
        if op == "skip":
            continue
        kind[k] = KINDS.index(op) + 1
        for i, x in enumerate(v):
            par[k, i] = min(x / 8, 1.0) if _is_width(op, i) else x / size
            msk[k, i] = 1
        par[k, COL0:COL0 + 3] = np.array(col) / 255
        msk[k, COL0:COL0 + 3] = 1
        par[k, ALPHA] = al / 255
        msk[k, ALPHA] = 1
    return kind, par, msk, np.array(bg, np.float32) / 255


def decode(logits, p, bg, size, temp, noise, rng, order="spatial"):
    lines = [f"size {size}", "bg " + " ".join(str(int(round(float(np.clip(c, 0, 1)) * 255))) for c in bg)]
    out = []
    for k in range(logits.shape[0]):
        if temp > 0:
            z = logits[k] / temp
            z = z - z.max()
            pr = np.exp(z)
            pr /= pr.sum()
            cls = int(rng.choice(NL, p=pr))
        else:
            cls = int(logits[k].argmax())
        if cls == 0:
            continue                                   # empty slot
        op = KINDS[cls - 1]
        vals = p[k, cls - 1] + (rng.normal(0, noise, PD) if noise > 0 else 0)
        coords = []
        for i in range(NCOORD[op]):
            if _is_width(op, i):
                lo = 0 if (op == "stroke" and i == 7) else 1
                coords.append(int(np.clip(round(float(vals[i]) * 8), lo, 8)))
            else:
                x = int(round(float(np.clip(vals[i], 0, 1)) * size))
                coords.append(max(1, x) if (op == "circle" and i == 2) else x)
        col = [int(round(float(np.clip(c, 0, 1)) * 255)) for c in vals[COL0:COL0 + 3]]
        al = int(round(float(np.clip(vals[ALPHA], 0, 1)) * 255))
        tail = col + ([al] if al < 250 else [])
        out.append((_area(op, coords), " ".join([op] + [str(x) for x in coords + tail])))
    if order == "spatial":
        out.sort(key=lambda t: -t[0])                  # big shapes first, details on top
    return "\n".join(lines + [t[1] for t in out])


# ---------------- the network ----------------
class Net:
    def __init__(self, V, H, K, rng=None, params=None):
        self.K = K
        if params is not None:
            self.p = params
            return
        out = K * SLOT + 3
        self.p = {
            "W1": (rng.normal(0, np.sqrt(2 / V), (V, H))).astype(np.float32),
            "b1": np.zeros(H, np.float32),
            "W2": (rng.normal(0, 0.5 / np.sqrt(H), (H, out))).astype(np.float32),
            "b2": np.zeros(out, np.float32),
        }

    def split(self, O):
        B, K = len(O), self.K
        S = O[:, :K * SLOT].reshape(B, K, SLOT)
        return S[:, :, :NL], S[:, :, NL:].reshape(B, K, NK, PD), O[:, K * SLOT:]

    def predict(self, X):
        p = self.p
        O = np.maximum(X @ p["W1"] + p["b1"], 0) @ p["W2"] + p["b2"]
        logits, praw, bgraw = self.split(O)
        return logits, sig(praw), sig(bgraw)

    def loss_grad(self, X, kind, par, msk, bg_t, lam=10.0, lam_bg=5.0):
        p = self.p
        B, K = len(X), self.K
        bi, ki = np.arange(B)[:, None], np.arange(K)[None, :]

        z1 = X @ p["W1"] + p["b1"]
        h = np.maximum(z1, 0)
        O = h @ p["W2"] + p["b2"]
        logits, praw, bgraw = self.split(O)

        # kind: softmax cross-entropy
        e = np.exp(logits - logits.max(-1, keepdims=True))
        sm = e / e.sum(-1, keepdims=True)
        ce = -np.log(sm[bi, ki, kind] + 1e-9).mean()
        dlog = sm.copy()
        dlog[bi, ki, kind] -= 1
        dlog /= B * K

        # numbers: only the true kind's block, only the used dimensions
        pp = sig(praw)
        has = (kind > 0)[..., None]
        ci = np.maximum(kind - 1, 0)
        psel = pp[bi, ki, ci]
        diff = (psel - par) * msk * has
        npar = max(1.0, float((msk * has).sum()))
        mse = float((diff ** 2).sum() / npar)
        dpraw = np.zeros_like(praw)
        dpraw[bi, ki, ci] = lam * 2 * diff / npar * psel * (1 - psel)

        # background color
        bgp = sig(bgraw)
        mse_bg = float(((bgp - bg_t) ** 2).mean())
        dbgraw = lam_bg * 2 * (bgp - bg_t) / (B * 3) * bgp * (1 - bgp)

        dO = np.zeros_like(O)
        dO[:, :K * SLOT] = np.concatenate([dlog, dpraw.reshape(B, K, NK * PD)], -1).reshape(B, -1)
        dO[:, K * SLOT:] = dbgraw

        dh = dO @ p["W2"].T
        dz1 = dh * (z1 > 0)
        grads = {"W2": h.T @ dO, "b2": dO.sum(0), "W1": X.T @ dz1, "b1": dz1.sum(0)}
        return ce + lam * mse + lam_bg * mse_bg, (ce, mse, mse_bg), grads


class Adam:
    def __init__(self, params, lr, wd=1e-4):
        self.lr, self.wd, self.t = lr, wd, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr):
        self.t += 1
        for k in params:
            g = grads[k] + (self.wd * params[k] if k.startswith("W") else 0)
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


# ---------------- train ----------------
def load_data(root):
    items = []
    for p in sorted((Path(root) / "programs").glob("*.ipg")):
        cf = Path(root) / "captions" / f"{p.stem}.txt"
        cap = cf.read_text().strip() if cf.exists() else ""
        img = Path(root) / "images" / f"{p.stem}.png"
        if cap and img.exists():
            items.append((cap, p.read_text(), img))
    return items


def train(a):
    items = load_data(a.data)
    if len(items) < 4:
        raise SystemExit("Need at least a few captioned samples (fill in data/captions/*.txt).")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(items))
    n_val = max(1, len(items) // 10)
    val = [items[i] for i in order[:n_val]]
    tr = [items[i] for i in order[n_val:]]
    size = parse(items[0][1])[0]
    if a.slots == 0:                                   # auto: as many slots as the longest program
        a.slots = min(400, max(len(parse(p)[2]) for _, p, _ in items))
        print(f"using {a.slots} slots")

    vocab = {f: i for i, f in enumerate(sorted({f for c, _, _ in tr for f in feats(c)}))}
    X = featurize([c for c, _, _ in tr], vocab)
    Xv = featurize([c for c, _, _ in val], vocab)
    T = [targets(p, a.slots, a.order) for _, p, _ in tr]
    Tv = [targets(p, a.slots, a.order) for _, p, _ in val]
    nstroke = sum(1 for _, p, _ in tr for s in parse(p)[2] if s[0] == "stroke")
    nall = sum(len(parse(p)[2]) for _, p, _ in tr)
    print(f"learning all {NK} shape kinds + opacity (strokes are {100 * nstroke / max(1, nall):.0f}% of your shapes); order={a.order}")
    stack = lambda ts: tuple(np.stack(z) for z in zip(*ts))
    kind, par, msk, bg = stack(T)
    kind_v, par_v, msk_v, bg_v = stack(Tv)

    net = Net(len(vocab), a.hidden, a.slots, rng)
    opt = Adam(net.p, a.lr)
    print(f"{len(tr)} train / {len(val)} val, {len(vocab)} caption features, "
          f"{sum(v.size for v in net.p.values()) / 1e6:.2f}M weights")

    def val_similarity():
        logits, p, bgp = net.predict(Xv)
        sims = []
        for i, (_, _, img) in enumerate(val):
            prog = decode(logits[i], p[i], bgp[i], size, 0, 0, rng, a.order)
            sims.append(similarity(render(prog), Image.open(img).convert("RGB")))
        return float(np.mean(sims))

    base = []                                   # baseline: a random training program
    for i, (_, _, img) in enumerate(val):
        other = tr[int(rng.integers(len(tr)))][1]
        base.append(similarity(render(other), Image.open(img).convert("RGB")))
    print(f"baseline similarity (random training program): {np.mean(base):.4f}")

    steps = a.epochs * int(np.ceil(len(tr) / a.batch))
    t = 0
    for ep in range(1, a.epochs + 1):
        perm = rng.permutation(len(tr))
        tot = 0.0
        for s in range(0, len(tr), a.batch):
            idx = perm[s:s + a.batch]
            xb = X[idx]
            if a.word_drop > 0:                 # randomly hide some words (regularization)
                xb = xb * (rng.random(xb.shape) > a.word_drop)
            lr = a.lr * 0.5 * (1 + np.cos(np.pi * t / steps))
            loss, _, g = net.loss_grad(xb, kind[idx], par[idx], msk[idx], bg[idx])
            opt.step(net.p, g, lr)
            tot += loss * len(idx)
            t += 1
        if ep % a.log_every == 0 or ep == a.epochs:
            vl, (vce, vmse, _), _ = net.loss_grad(Xv, kind_v, par_v, msk_v, bg_v)
            print(f"epoch {ep}/{a.epochs}  train loss {tot / len(tr):.4f}  "
                  f"val loss {vl:.4f}  val similarity {val_similarity():.4f}")

    meta = json.dumps(dict(vocab=sorted(vocab, key=vocab.get), K=a.slots, size=size, H=a.hidden, v=VERSION, order=a.order))
    np.savez(a.out, meta=np.array(meta), **net.p)
    print(f"saved {a.out}")


# ---------------- generate ----------------
def gen(a):
    d = np.load(a.model, allow_pickle=False)
    meta = json.loads(d["meta"].item())
    if meta.get("v") != VERSION:
        raise SystemExit("this model was trained with the old nn_numpy (no strokes/opacity); retrain it with this version")
    vocab = {f: i for i, f in enumerate(meta["vocab"])}
    net = Net(0, 0, meta["K"], params={k: d[k] for k in ("W1", "b1", "W2", "b2")})
    X = featurize([a.prompt], vocab)
    if not X.any():
        raise SystemExit("None of your prompt's words were in the training captions. "
                         "Use words your captions use.")
    logits, p, bg = net.predict(X)
    rng = np.random.default_rng(a.seed)
    for n in range(1, a.n + 1):
        prog = decode(logits[0], p[0], bg[0], meta["size"], a.temp, a.noise, rng, meta.get("order", "spatial"))
        Path(f"gen_{n}.ipg").write_text(prog)
        render(prog, scale=4, aa=True).save(f"gen_{n}.png")
        print(f"wrote gen_{n}.ipg and gen_{n}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("data")
    t.add_argument("--slots", type=int, default=0, help="0 = auto")
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--word-drop", type=float, default=0.1)
    t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--order", choices=["spatial", "draw"], default="spatial",
                   help="slot order: spatial (fixed top-left to bottom-right, default) or draw (file order)")
    t.add_argument("--out", default="model_np.npz")
    g = sub.add_parser("gen")
    g.add_argument("model")
    g.add_argument("prompt")
    g.add_argument("--n", type=int, default=1)
    g.add_argument("--temp", type=float, default=0.5, help="0 = always the most likely shape kind")
    g.add_argument("--noise", type=float, default=0.01, help="random wobble on the numbers")
    g.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else gen(a)
