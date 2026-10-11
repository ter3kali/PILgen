"""Tiny Vision Transformer in pure NumPy: forward, manual backward, loss, AdamW.

Pre-LN transformer, mean-pooled tokens -> linear head producing tag logits.
CPU only. No PyTorch / TensorFlow / JAX.
"""
import json
import numpy as np

import tags as T

# ----------------------------------------------------------------- helpers
_C = np.sqrt(2.0 / np.pi)


def gelu(x):
    return .5 * x * (1 + np.tanh(_C * (x + .044715 * x ** 3)))


def gelu_grad(x):
    u = _C * (x + .044715 * x ** 3)
    t = np.tanh(u)
    return .5 * (1 + t) + .5 * x * (1 - t * t) * _C * (1 + 3 * .044715 * x * x)


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def ln_fwd(x, g, b, eps=1e-5):
    xc = x - x.mean(-1, keepdims=True)
    inv = 1 / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps)
    z = xc * inv
    return z * g + b, (z, inv)


def ln_bwd(dy, cache, g):
    z, inv = cache
    d = z.shape[-1]
    dg = (dy * z).reshape(-1, d).sum(0)
    db = dy.reshape(-1, d).sum(0)
    dz = dy * g
    dx = inv * (dz - dz.mean(-1, keepdims=True) - z * (dz * z).mean(-1, keepdims=True))
    return dx, dg, db


def lin_fwd(x, W, b):
    y = x.reshape(-1, x.shape[-1]) @ W + b
    return y.reshape(*x.shape[:-1], W.shape[1])


def lin_bwd(dy, x, W):
    x2 = x.reshape(-1, x.shape[-1])
    dy2 = dy.reshape(-1, dy.shape[-1])
    dx = (dy2 @ W.T).reshape(x.shape)
    return dx, x2.T @ dy2, dy2.sum(0)


# ------------------------------------------------------------------- model
class TinyViT:
    def __init__(self, img=64, patch=8, dim=64, depth=3, heads=4, mlp=2,
                 n_out=None, seed=0, dtype=np.float32):
        n_out = T.N_OUT if n_out is None else n_out
        assert img % patch == 0 and dim % heads == 0
        self.cfg = dict(img=img, patch=patch, dim=dim, depth=depth, heads=heads,
                        mlp=mlp, n_out=n_out)
        rng = np.random.default_rng(seed)
        n_tok, pd = (img // patch) ** 2, patch * patch * 3
        P = {}

        def lin(name, i, o, scale=1.0):
            P[name + ".W"] = rng.normal(0, scale / np.sqrt(i), (i, o)).astype(dtype)
            P[name + ".b"] = np.zeros(o, dtype)

        def ln(name):
            P[name + ".g"] = np.ones(dim, dtype)
            P[name + ".b"] = np.zeros(dim, dtype)

        lin("patch", pd, dim)
        P["pos"] = rng.normal(0, .02, (n_tok, dim)).astype(dtype)
        for l in range(depth):
            p = f"b{l}."
            ln(p + "ln1"); lin(p + "qkv", dim, 3 * dim); lin(p + "proj", dim, dim, .5)
            ln(p + "ln2"); lin(p + "fc1", dim, dim * mlp); lin(p + "fc2", dim * mlp, dim, .5)
        ln("lnf")
        lin("head", dim, n_out, .1)
        self.P = P

    def n_params(self):
        return sum(v.size for v in self.P.values())

    # -- preprocessing
    def patchify(self, x):
        B, H, W, C = x.shape
        p = self.cfg["patch"]
        x = x.reshape(B, H // p, p, W // p, p, C).transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(B, (H // p) * (W // p), p * p * C)

    def forward(self, imgs):
        """imgs: uint8 or float [B,H,W,3] in 0..255. Returns logits, cache."""
        P, c = self.P, self.cfg
        dt = P["pos"].dtype
        x = (np.asarray(imgs, dt) / 255. - .5) / .5
        pt = self.patchify(x)
        t = lin_fwd(pt, P["patch.W"], P["patch.b"]) + P["pos"]
        B, N, D = t.shape
        Hh, dh = c["heads"], D // c["heads"]
        scale = 1 / np.sqrt(dh)
        cache = {"pt": pt, "blocks": []}
        for l in range(c["depth"]):
            p = f"b{l}."
            h1, c1 = ln_fwd(t, P[p + "ln1.g"], P[p + "ln1.b"])
            qkv = lin_fwd(h1, P[p + "qkv.W"], P[p + "qkv.b"])
            q, k, v = qkv.reshape(B, N, 3, Hh, dh).transpose(2, 0, 3, 1, 4)
            a = softmax((q @ k.transpose(0, 1, 3, 2)) * scale)
            o = (a @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
            t2 = t + lin_fwd(o, P[p + "proj.W"], P[p + "proj.b"])
            h2, c2 = ln_fwd(t2, P[p + "ln2.g"], P[p + "ln2.b"])
            m1 = lin_fwd(h2, P[p + "fc1.W"], P[p + "fc1.b"])
            g = gelu(m1)
            t3 = t2 + lin_fwd(g, P[p + "fc2.W"], P[p + "fc2.b"])
            cache["blocks"].append(dict(h1=h1, c1=c1, q=q, k=k, v=v, a=a, o=o,
                                        h2=h2, c2=c2, m1=m1, g=g))
            t = t3
        hf, cf = ln_fwd(t, P["lnf.g"], P["lnf.b"])
        pooled = hf.mean(1)
        logits = pooled @ P["head.W"] + P["head.b"]
        cache.update(cf=cf, pooled=pooled, N=N)
        return logits, cache

    def backward(self, dlogits, cache):
        P, c = self.P, self.cfg
        G = {}
        N = cache["N"]
        G["head.W"] = cache["pooled"].T @ dlogits
        G["head.b"] = dlogits.sum(0)
        dpooled = dlogits @ P["head.W"].T
        dhf = np.repeat(dpooled[:, None, :] / N, N, axis=1)
        dt, G["lnf.g"], G["lnf.b"] = ln_bwd(dhf, cache["cf"], P["lnf.g"])
        B, _, D = dt.shape
        Hh, dh = c["heads"], D // c["heads"]
        scale = 1 / np.sqrt(dh)
        for l in reversed(range(c["depth"])):
            p, s = f"b{l}.", cache["blocks"][l]
            # MLP branch (dt flows to both the skip and the branch)
            dg, G[p + "fc2.W"], G[p + "fc2.b"] = lin_bwd(dt, s["g"], P[p + "fc2.W"])
            dm1 = dg * gelu_grad(s["m1"])
            dh2, G[p + "fc1.W"], G[p + "fc1.b"] = lin_bwd(dm1, s["h2"], P[p + "fc1.W"])
            dln, G[p + "ln2.g"], G[p + "ln2.b"] = ln_bwd(dh2, s["c2"], P[p + "ln2.g"])
            dt2 = dt + dln
            # attention branch
            do, G[p + "proj.W"], G[p + "proj.b"] = lin_bwd(dt2, s["o"], P[p + "proj.W"])
            do = do.reshape(B, N, Hh, dh).transpose(0, 2, 1, 3)
            a, q, k, v = s["a"], s["q"], s["k"], s["v"]
            da = do @ v.transpose(0, 1, 3, 2)
            dv = a.transpose(0, 1, 3, 2) @ do
            ds = a * (da - (da * a).sum(-1, keepdims=True)) * scale
            dq = ds @ k
            dk = ds.transpose(0, 1, 3, 2) @ q
            dqkv = np.stack([dq, dk, dv]).transpose(1, 3, 0, 2, 4).reshape(B, N, 3 * D)
            dh1, G[p + "qkv.W"], G[p + "qkv.b"] = lin_bwd(dqkv, s["h1"], P[p + "qkv.W"])
            dln, G[p + "ln1.g"], G[p + "ln1.b"] = ln_bwd(dh1, s["c1"], P[p + "ln1.g"])
            dt = dt2 + dln
        G["pos"] = dt.sum(0)
        _, G["patch.W"], G["patch.b"] = lin_bwd(dt, cache["pt"], P["patch.W"])
        return G

    # -- persistence
    def save(self, path, **extra):
        np.savez(path, config=np.array(json.dumps(self.cfg)), **extra, **self.P)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        m = cls(**json.loads(str(d["config"])))
        for k in m.P:
            m.P[k] = d[k].astype(np.float32)
        extra = {k: d[k] for k in d.files if k not in m.P and k != "config"}
        return m, extra


# -------------------------------------------------------------------- loss
def loss_and_grad(logits, y_ex, y_mu, pos_w=None):
    """Sum over heads of: softmax-CE per exclusive group + weighted BCE per tag,
    averaged over the batch. Returns (loss, dlogits)."""
    B = logits.shape[0]
    d = np.zeros_like(logits)
    total, ar = 0., np.arange(B)
    for gi, (s, e) in enumerate(T.GROUPS):
        p = softmax(logits[:, s:e])
        ok = (y_ex[:, gi] >= 0)            # -1 = unlabeled, ignored
        yy = np.where(ok, y_ex[:, gi], 0)
        total += (-np.log(p[ar, yy] + 1e-12) * ok).sum() / B
        g = p.copy()
        g[ar, yy] -= 1
        d[:, s:e] = g * ok[:, None] / B
    s, e = T.MULTI_SLICE
    p = sigmoid(logits[:, s:e])
    w = 1.0 if pos_w is None else 1 + (pos_w - 1) * y_mu
    bce = -(y_mu * np.log(p + 1e-12) + (1 - y_mu) * np.log(1 - p + 1e-12))
    total += (w * bce).sum(1).mean()
    d[:, s:e] = w * (p - y_mu) / B
    return total, d


# --------------------------------------------------------------- optimizer
class AdamW:
    def __init__(self, params, lr=2e-3, betas=(.9, .999), eps=1e-8, wd=.01):
        self.P, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, *betas, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, g in grads.items():
            m, v = self.m[k], self.v[k]
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            upd = (m / c1) / (np.sqrt(v / c2) + self.eps)
            if k.endswith(".W"):  # decay weight matrices only
                upd = upd + self.wd * self.P[k]
            self.P[k] -= lr * upd
