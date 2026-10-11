"""Numerical gradient check (float64). Run this after ANY change to vit.py.

    python gradcheck.py
"""
import numpy as np

import tags as T
from vit import TinyViT, loss_and_grad


def main(samples_per_param=4, eps=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    m = TinyViT(img=16, patch=8, dim=16, depth=2, heads=2, mlp=2, seed=1, dtype=np.float64)
    B = 3
    x = rng.integers(0, 256, (B, 16, 16, 3)).astype(np.float64)
    y_ex = np.stack([rng.integers(0, e - s, B) for s, e in T.GROUPS], 1)
    y_mu = (rng.random((B, len(T.MULTI))) < .3).astype(np.float64)
    pw = rng.uniform(1, 4, len(T.MULTI))

    def loss():
        lg, _ = m.forward(x)
        return loss_and_grad(lg, y_ex, y_mu, pw)[0]

    lg, cache = m.forward(x)
    _, dl = loss_and_grad(lg, y_ex, y_mu, pw)
    G = m.backward(dl, cache)
    assert set(G) == set(m.P), "missing/extra grads: %s" % (set(m.P) ^ set(G))

    worst = 0.0
    for k, p in m.P.items():
        flat = p.reshape(-1)
        idx = rng.choice(flat.size, min(samples_per_param, flat.size), replace=False)
        errs = []
        for i in idx:
            old = flat[i]
            flat[i] = old + eps; lp = loss()
            flat[i] = old - eps; lm = loss()
            flat[i] = old
            num, ana = (lp - lm) / (2 * eps), G[k].reshape(-1)[i]
            errs.append(abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
        worst = max(worst, max(errs))
        print(f"{k:14s} max rel err {max(errs):.2e}")
    print(f"\nWORST relative error: {worst:.2e}  ->", "PASS" if worst < 1e-5 else "FAIL")
    return worst < 1e-5


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
