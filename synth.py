"""Synthetic anime-style data generator (PIL only, no external data).

Draws simple characters with known attributes, so labels are exact and free.
Purpose: prove the training loop, the tag->prompt step and the eval harness
before investing in a real tagged dataset. It will NOT transfer to real art.

Usage:  python synth.py --n 8000 --out train.npz
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw

import tags as T

HAIR = {"black": (30, 30, 36), "brown": (112, 72, 42), "blonde": (236, 202, 92),
        "pink": (240, 140, 190), "blue": (70, 110, 210), "red": (200, 50, 52),
        "white": (240, 240, 244)}
EYES = {"blue": (60, 120, 230), "red": (210, 50, 60), "green": (60, 170, 90),
        "brown": (120, 76, 40), "purple": (140, 70, 190)}
CLOTH = {"black": (35, 35, 42), "white": (242, 242, 246), "red": (205, 50, 55),
         "blue": (60, 100, 205), "green": (55, 160, 90)}
BG = {"white": (250, 250, 250), "black": (22, 22, 28), "pink": (250, 192, 212),
      "blue": (152, 192, 250), "green": (162, 226, 172)}
SKIN = [(255, 224, 205), (245, 205, 180), (235, 190, 160), (255, 236, 222)]
ORN = [(255, 255, 255), (255, 220, 60), (255, 90, 90), (120, 200, 255)]


def _jit(rng, c, a=8):
    return tuple(int(np.clip(v + rng.integers(-a, a + 1), 0, 255)) for v in c)


def _rand_subject(rng):
    return dict(
        hair=rng.choice(list(HAIR)), long=bool(rng.random() < .5),
        eyes=rng.choice(list(EYES)), cloth=rng.choice(list(CLOTH)),
        glasses=bool(rng.random() < .2), orn=bool(rng.random() < .25),
        blush=bool(rng.random() < .4), bangs=bool(rng.random() < .6),
        mouth=rng.choice(["neutral", "smile", "open"], p=[.4, .4, .2]),
        skin=_jit(rng, SKIN[rng.integers(len(SKIN))], 6),
        orn_col=ORN[rng.integers(len(ORN))],
    )


def _draw_subject(d, cx, cy, r, a, S, rng):
    hair, skin, cloth = _jit(rng, HAIR[a["hair"]]), a["skin"], _jit(rng, CLOTH[a["cloth"]])
    # back hair
    if a["long"]:
        d.rounded_rectangle([cx - 1.15 * r, cy - 1.0 * r, cx + 1.15 * r, cy + 2.2 * r],
                            radius=0.6 * r, fill=hair)
    else:
        d.ellipse([cx - 1.12 * r, cy - 1.15 * r, cx + 1.12 * r, cy + 0.55 * r], fill=hair)
    # neck + body
    d.rectangle([cx - .28 * r, cy + .7 * r, cx + .28 * r, cy + 1.3 * r], fill=skin)
    d.polygon([(cx - .45 * r, cy + 1.15 * r), (cx + .45 * r, cy + 1.15 * r),
               (cx + 1.7 * r, cy + 1.6 * r), (cx + 1.9 * r, S + 4),
               (cx - 1.9 * r, S + 4), (cx - 1.7 * r, cy + 1.6 * r)], fill=cloth)
    if a["long"]:  # side locks over the body
        d.rectangle([cx - 1.15 * r, cy - .2 * r, cx - .8 * r, cy + 2.0 * r], fill=hair)
        d.rectangle([cx + .8 * r, cy - .2 * r, cx + 1.15 * r, cy + 2.0 * r], fill=hair)
    # head
    d.ellipse([cx - .95 * r, cy - r, cx + .95 * r, cy + .95 * r], fill=skin)
    d.chord([cx - 1.08 * r, cy - 1.13 * r, cx + 1.08 * r, cy + 1.0 * r], 195, 345, fill=hair)
    if a["bangs"]:
        d.polygon([(cx - .98 * r, cy - .45 * r), (cx + .98 * r, cy - .45 * r),
                   (cx + .9 * r, cy - .05 * r), (cx + .5 * r, cy - .2 * r),
                   (cx + .15 * r, cy - .02 * r), (cx - .2 * r, cy - .2 * r),
                   (cx - .55 * r, cy - .02 * r), (cx - .9 * r, cy - .05 * r)], fill=hair)
    ey = cy + .25 * r
    for sx in (-1, 1):
        ex = cx + sx * .42 * r
        d.ellipse([ex - .26 * r, ey - .22 * r, ex + .26 * r, ey + .22 * r],
                  fill=(250, 250, 250), outline=(40, 30, 40))
        d.ellipse([ex - .2 * r, ey - .2 * r, ex + .2 * r, ey + .2 * r], fill=EYES[a["eyes"]])
        d.ellipse([ex - .08 * r, ey - .08 * r, ex + .08 * r, ey + .08 * r], fill=(20, 20, 30))
        if a["blush"]:
            bx, by = cx + sx * .64 * r, cy + .6 * r
            d.ellipse([bx - .2 * r, by - .1 * r, bx + .2 * r, by + .1 * r], fill=(250, 140, 152))
    my, w = cy + .74 * r, max(1, int(r * .06))
    if a["mouth"] == "neutral":
        d.line([(cx - .13 * r, my), (cx + .13 * r, my)], fill=(150, 60, 70), width=w)
    elif a["mouth"] == "smile":
        d.arc([cx - .22 * r, my - .16 * r, cx + .22 * r, my + .1 * r], 20, 160,
              fill=(150, 60, 70), width=w)
    else:
        d.ellipse([cx - .1 * r, my - .07 * r, cx + .1 * r, my + .1 * r], fill=(120, 30, 50))
    if a["glasses"]:
        gw = max(1, int(r * .07))
        for sx in (-1, 1):
            ex = cx + sx * .42 * r
            d.ellipse([ex - .36 * r, ey - .32 * r, ex + .36 * r, ey + .32 * r],
                      outline=(30, 30, 35), width=gw)
        d.line([(cx - .06 * r, ey), (cx + .06 * r, ey)], fill=(30, 30, 35), width=gw)
    if a["orn"]:
        ox, oy, orad = cx + .75 * r, cy - .72 * r, .2 * r
        d.ellipse([ox - orad, oy - orad, ox + orad, oy + orad], fill=a["orn_col"],
                  outline=(90, 90, 90))


def labels_for(subjects, bg):
    ex = np.array([0 if len(subjects) == 1 else 1, list(T.EXCLUSIVE["background"]).index(bg)])
    mu = np.zeros(len(T.MULTI), np.float32)

    def on(t):
        mu[T.MULTI_INDEX[t]] = 1

    for a in subjects:
        on("hair_" + a["hair"]); on("eyes_" + a["eyes"]); on("clothes_" + a["cloth"])
        on("long_hair" if a["long"] else "short_hair")
        for k in ("glasses", "blush", "bangs"):
            if a[k]:
                on(k)
        if a["orn"]:
            on("hair_ornament")
        if a["mouth"] == "smile":
            on("smile")
        if a["mouth"] == "open":
            on("open_mouth")
    return ex, mu


def render(rng, size=64, ss=2):
    S = size * ss
    bg = rng.choice(list(BG))
    img = Image.new("RGB", (S, S), _jit(rng, BG[bg]))
    d = ImageDraw.Draw(img)
    if rng.random() < .6:
        subs = [_rand_subject(rng)]
        spec = [(S * .5 + rng.normal(0, S * .03), S * .44 + rng.normal(0, S * .03),
                 .24 * S * rng.uniform(.88, 1.1))]
    else:
        subs = [_rand_subject(rng), _rand_subject(rng)]
        spec = [(S * (.29 + rng.normal(0, .02)), S * (.46 + rng.normal(0, .02)),
                 .15 * S * rng.uniform(.9, 1.12)),
                (S * (.71 + rng.normal(0, .02)), S * (.46 + rng.normal(0, .02)),
                 .15 * S * rng.uniform(.9, 1.12))]
    for a, (cx, cy, r) in zip(subs, spec):
        _draw_subject(d, cx, cy, r, a, S, rng)
    img = img.resize((size, size), Image.LANCZOS)
    ex, mu = labels_for(subs, bg)
    return np.asarray(img, np.uint8), ex, mu


def make_dataset(n, size=64, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size, 3), np.uint8)
    EX = np.zeros((n, len(T.EXCLUSIVE)), np.int64)
    MU = np.zeros((n, len(T.MULTI)), np.float32)
    for i in range(n):
        X[i], EX[i], MU[i] = render(rng, size)
    return X, EX, MU


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="train.npz")
    a = ap.parse_args()
    X, EX, MU = make_dataset(a.n, a.size, a.seed)
    np.savez_compressed(a.out, X=X, EX=EX, MU=MU)
    print("saved", a.out, X.shape)
