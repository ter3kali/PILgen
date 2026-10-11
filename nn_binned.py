"""
nn_binned.py - caption -> shape program, but SHARP: every number is a *choice among bins*,
not an average.

Why: nn_numpy.py regresses each coordinate/colour with squared error, so when a caption could mean
several different programs it outputs their average (a soft blob). Here every number (x, y, radius,
R, G, B, opacity ...) is predicted as a probability over NB bins (default 16). Generating *samples*
a bin, so you get one definite, crisp program per sample (different samples = different plausible
pictures), instead of a muddy mean.

Needs nn_numpy.py (v2) and imgprog.py in the same folder. Same dataset folder as nn_numpy.py.

    python nn_binned.py train data --slots 40 --hidden 128 --epochs 200
    python nn_binned.py gen model_bin.npz "black hair, white background" --n 4 --temp 0.7
    --temp 0   most likely bin for every number (the "mode", sharp but always the same)
    --temp 1   follow the learned probabilities;  higher = wilder

"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import nn_numpy as N
from imgprog import edge_similarity, render, sharpness_ratio, similarity

NK, NL, PD = N.NK, N.NL, N.PD
VERSION = 3


def slot_size(NB):
    return NL + NK * PD * NB


class BNet:
    def __init__(self, V, H, K, NB, rng=None, params=None):
        self.K, self.NB = K, NB
        if params is not None:
            self.p = params
            return
        out = K * slot_size(NB) + 3 * NB
        self.p = {"W1": rng.normal(0, np.sqrt(2 / V), (V, H)).astype(np.float32), "b1": np.zeros(H, np.float32),
                  "W2": rng.normal(0, 0.3 / np.sqrt(H), (H, out)).astype(np.float32), "b2": np.zeros(out, np.float32)}

    def split(self, O):
        B, K, NB = len(O), self.K, self.NB
        S = O[:, :K * slot_size(NB)].reshape(B, K, slot_size(NB))
        return S[:, :, :NL], S[:, :, NL:].reshape(B, K, NK, PD, NB), O[:, K * slot_size(NB):].reshape(B, 3, NB)

    def forward(self, X):
        z1 = X @ self.p["W1"] + self.p["b1"]
        h = np.maximum(z1, 0)
        return z1, h, h @ self.p["W2"] + self.p["b2"]

    def predict(self, X):
        return self.split(self.forward(X)[2])

    def loss_grad(self, X, kind, par, msk, bg_t, lam=1.0, lam_bg=1.0, sigma=0.8):
        B, K, NB = len(X), self.K, self.NB
        bi, ki = np.arange(B)[:, None], np.arange(K)[None, :]
        z1, h, O = self.forward(X)
        logits, bl, bgl = self.split(O)

        e = np.exp(logits - logits.max(-1, keepdims=True))
        sm = e / e.sum(-1, keepdims=True)
        ce = -np.log(sm[bi, ki, kind] + 1e-9).mean()
        dlog = sm.copy()
        dlog[bi, ki, kind] -= 1
        dlog /= B * K

        ci = np.maximum(kind - 1, 0)
        sel = bl[bi, ki, ci]                                   # (B,K,PD,NB): the true kind's numbers
        sel = sel - sel.max(-1, keepdims=True)
        ps = np.exp(sel)
        ps /= ps.sum(-1, keepdims=True)
        centres = np.arange(NB) + 0.5
        tgt = np.exp(-0.5 * ((centres - par[..., None] * NB) / sigma) ** 2)   # soft one-hot (neighbouring bins share)
        tgt /= tgt.sum(-1, keepdims=True) + 1e-12
        w = (msk * (kind > 0)[..., None])[..., None]
        cnt = max(1.0, float(w[..., 0].sum()))
        cb = float((-(tgt * np.log(ps + 1e-9)).sum(-1) * w[..., 0]).sum() / cnt)
        dsel = (ps - tgt) * w / cnt * lam
        dbl = np.zeros_like(bl)
        dbl[bi, ki, ci] = dsel

        eb = np.exp(bgl - bgl.max(-1, keepdims=True))
        pb = eb / eb.sum(-1, keepdims=True)
        tb = np.exp(-0.5 * ((centres - bg_t[..., None] * NB) / sigma) ** 2)
        tb /= tb.sum(-1, keepdims=True)
        cbg = float((-(tb * np.log(pb + 1e-9)).sum(-1)).mean())
        dbg = (pb - tb) / (B * 3) * lam_bg

        dO = np.zeros_like(O)
        dO[:, :K * slot_size(NB)] = np.concatenate([dlog, dbl.reshape(B, K, NK * PD * NB)], -1).reshape(B, -1)
        dO[:, K * slot_size(NB):] = dbg.reshape(B, -1)
        dh = dO @ self.p["W2"].T
        dz1 = dh * (z1 > 0)
        grads = {"W2": h.T @ dO, "b2": dO.sum(0), "W1": X.T @ dz1, "b1": dz1.sum(0)}
        return ce + lam * cb + lam_bg * cbg, (ce, cb, cbg), grads


def _pick(logp, temp, rng):
    """logits over bins (..., NB) -> bin centre in 0..1 (sampled, or the mode when temp == 0)"""
    NB = logp.shape[-1]
    if temp <= 0:
        b = logp.argmax(-1)
        return (b + 0.5) / NB
    z = logp / temp
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    cum = p.cumsum(-1)
    u = rng.random(p.shape[:-1] + (1,))
    b = np.minimum((u > cum).sum(-1), NB - 1)
    return (b + rng.random(b.shape)) / NB                    # random point inside the chosen bin


def decode_b(logits, bl, bgl, size, temp, rng, order="spatial"):
    K = logits.shape[0]
    bg = _pick(bgl, temp, rng)
    P = np.zeros((K, NK, PD), np.float32)
    for k in range(K):
        P[k] = _pick(bl[k], temp, rng)
    # reuse the v2 decoder: kind logits are sampled there; numbers are already concrete -> noise 0
    logits_t = logits
    return N.decode(logits_t, P, bg, size, temp, 0.0, rng, order)


def train(a):
    items = N.load_data(a.data)
    if len(items) < 4:
        raise SystemExit("Need at least a few captioned samples (fill in data/captions/*.txt).")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(items))
    n_val = max(1, len(items) // 10)
    val = [items[i] for i in order[:n_val]]
    tr = [items[i] for i in order[n_val:]]
    size = N.parse(items[0][1])[0]
    if a.slots == 0:
        a.slots = min(400, max(len(N.parse(p)[2]) for _, p, _ in items))
    vocab = {f: i for i, f in enumerate(sorted({f for c, _, _ in tr for f in N.feats(c)}))}
    X, Xv = N.featurize([c for c, _, _ in tr], vocab), N.featurize([c for c, _, _ in val], vocab)
    st = lambda ts: tuple(np.stack(z) for z in zip(*ts))
    kind, par, msk, bg = st([N.targets(p, a.slots, a.order) for _, p, _ in tr])
    net = BNet(len(vocab), a.hidden, a.slots, a.bins, rng)
    opt = N.Adam(net.p, a.lr)
    print(f"{len(tr)} train / {len(val)} val, {len(vocab)} caption features, {a.bins} bins, "
          f"{sum(v.size for v in net.p.values()) / 1e6:.2f}M weights, order={a.order}")
    steps = a.epochs * int(np.ceil(len(tr) / a.batch))
    t = 0
    for ep in range(1, a.epochs + 1):
        perm, tot = rng.permutation(len(tr)), 0.0
        for s in range(0, len(tr), a.batch):
            idx = perm[s:s + a.batch]
            xb = X[idx]
            if a.word_drop > 0:
                xb = xb * (rng.random(xb.shape) > a.word_drop)
            lr = a.lr * 0.5 * (1 + np.cos(np.pi * t / steps))
            loss, _, g = net.loss_grad(xb, kind[idx], par[idx], msk[idx], bg[idx])
            opt.step(net.p, g, lr)
            tot += loss * len(idx)
            t += 1
        if ep % a.log_every == 0 or ep == a.epochs:
            sims, edges, sharp = evaluate(net, Xv, val, size, a.order, temp=0.0, rng=rng)
            print(f"epoch {ep}/{a.epochs} train loss {tot / len(tr):.3f} | val (mode) similarity {sims:.4f} "
                  f"edge match {edges:.3f} sharpness {sharp:.2f}x")
    meta = json.dumps(dict(vocab=sorted(vocab, key=vocab.get), K=a.slots, size=size, H=a.hidden, NB=a.bins,
                           v=VERSION, order=a.order))
    np.savez(a.out, meta=np.array(meta), **net.p)
    print(f"saved {a.out}")


def evaluate(net, Xv, val, size, order, temp, rng):
    lg, bl, bgl = net.predict(Xv)
    s, e, h = [], [], []
    for i, (_, _, img) in enumerate(val):
        prog = decode_b(lg[i], bl[i], bgl[i], size, temp, rng, order)
        tgt = Image.open(img).convert("RGB")
        r = render(prog)
        s.append(similarity(r, tgt)); e.append(edge_similarity(r, tgt)); h.append(sharpness_ratio(r, tgt))
    return float(np.mean(s)), float(np.mean(e)), float(np.mean(h))


def gen(a):
    d = np.load(a.model, allow_pickle=False)
    meta = json.loads(d["meta"].item())
    if meta.get("v") != VERSION:
        raise SystemExit("this is not an nn_binned model (train one with: python nn_binned.py train ...)")
    vocab = {f: i for i, f in enumerate(meta["vocab"])}
    net = BNet(0, 0, meta["K"], meta["NB"], params={k: d[k] for k in ("W1", "b1", "W2", "b2")})
    X = N.featurize([a.prompt], vocab)
    if not X.any():
        raise SystemExit("None of your prompt's words were in the training captions. Use words your captions use.")
    lg, bl, bgl = net.predict(X)
    rng = np.random.default_rng(a.seed)
    for n in range(1, a.n + 1):
        prog = decode_b(lg[0], bl[0], bgl[0], meta["size"], a.temp, rng, meta.get("order", "spatial"))
        Path(f"genb_{n}.ipg").write_text(prog)
        render(prog, scale=4, aa=True).save(f"genb_{n}.png")
        print(f"wrote genb_{n}.ipg and genb_{n}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("data")
    t.add_argument("--slots", type=int, default=0)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--bins", type=int, default=16)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--word-drop", type=float, default=0.1)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--order", choices=["spatial", "draw"], default="spatial")
    t.add_argument("--out", default="model_bin.npz")
    g = sub.add_parser("gen")
    g.add_argument("model")
    g.add_argument("prompt")
    g.add_argument("--n", type=int, default=1)
    g.add_argument("--temp", type=float, default=0.7)
    g.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else gen(a)
