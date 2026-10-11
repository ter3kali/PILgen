"""
Image-program toolkit v4: greedy shape fitting with
  * semi-transparent shapes (alpha),
  * error-guided placement (new shapes go where the picture is most wrong),
  * an edge-weighted score (so thin lines and outlines matter, not just flat areas),
  * optional fixed slot schedule, and a smooth high-resolution renderer.

Program file format (.ipg), one command per line:
    size N
    bg R G B
    circle X Y R            R G B [A]
    rect X1 Y1 X2 Y2        R G B [A]
    tri X1 Y1 X2 Y2 X3 Y3   R G B [A]
    line X1 Y1 X2 Y2 WIDTH  R G B [A]
    stroke X1 Y1 X2 Y2 X3 Y3 W1 W2  R G B [A]   (curved brush stroke, tapers from width W1 to W2;
                                                 the curve bends toward the middle point X2 Y2)
    skip                    (an empty slot; drawn as nothing)
A = opacity 0..255 (left out = 255, fully solid). Shapes are drawn in file order.

Run:
    python imgprog.py target.png --shapes 4000 --size 256 --alpha --edge 4 --strokes
    python imgprog.py target.png --shapes 12000 --size 512 --alpha --edge 4 --strokes \
                      --time-limit 250 --out big          # stops after 250 s, saves, ...
    python imgprog.py target.png --shapes 12000 --size 512 --alpha --edge 4 --strokes \
                      --time-limit 250 --out big --resume big.ipg      # ... and continues
    python imgprog.py target.png --levels 1,2,4,8 --size 128          # fixed slots
"""
import argparse
import math
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 64
KINDS = ["circle", "rect", "tri", "line"]
NCOORD = {"circle": 3, "rect": 4, "tri": 6, "line": 5, "stroke": 8}
ALPHAS = (255, 255, 200, 150, 100, 60)       # opacities the fitter may pick (biased to solid)


def clamp(n, lo, hi):
    return max(lo, min(hi, n))


# ---------------- drawing primitives ----------------
def stroke_polygon(v, s=1):
    """Outline (list of x, y floats) of a tapered curved stroke, on a canvas s times larger.
    v = x1 y1 x2 y2 x3 y3 w1 w2: a quadratic curve from P1 to P3 bending toward P2."""
    x1, y1, x2, y2, x3, y3, w1, w2 = v
    n = max(6, int((math.hypot(x2 - x1, y2 - y1) + math.hypot(x3 - x2, y3 - y2)) / 2))
    left, right = [], []
    for i in range(n + 1):
        t = i / n
        px = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * x2 + t * t * x3
        py = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * y2 + t * t * y3
        tx = 2 * (1 - t) * (x2 - x1) + 2 * t * (x3 - x2)
        ty = 2 * (1 - t) * (y2 - y1) + 2 * t * (y3 - y2)
        ln = math.hypot(tx, ty) or 1.0
        h = max(0.5, (w1 * (1 - t) + w2 * t) / 2)             # half width, never thinner than 1 px
        nx, ny = -ty / ln * h, tx / ln * h
        left.append(((px + nx + 0.5) * s, (py + ny + 0.5) * s))
        right.append(((px - nx + 0.5) * s, (py - ny + 0.5) * s))
    return left + right[::-1]


def draw_shape(d, kind, v, fill):
    """Pixel-exact drawing; `fill` is a color tuple (RGB image) or an int (mask image)."""
    if kind == "circle":
        x, y, r = v
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    elif kind == "rect":
        x1, y1, x2, y2 = v
        d.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=fill)
    elif kind == "tri":
        d.polygon(list(v), fill=fill)
    elif kind == "line":
        d.line(list(v[:4]), fill=fill, width=max(1, v[4]))
    elif kind == "stroke":
        d.polygon(stroke_polygon(v, 1), fill=fill)


def draw_shape_scaled(d, kind, v, fill, s):
    """Same shapes on a canvas s times larger (programs are resolution independent)."""
    if kind == "circle":
        x, y, r = v
        d.ellipse([(x - r) * s, (y - r) * s, (x + r + 1) * s - 1, (y + r + 1) * s - 1], fill=fill)
    elif kind == "rect":
        x1, y1, x2, y2 = v
        d.rectangle([min(x1, x2) * s, min(y1, y2) * s,
                     (max(x1, x2) + 1) * s - 1, (max(y1, y2) + 1) * s - 1], fill=fill)
    elif kind == "tri":
        d.polygon([(v[i] + 0.5) * s for i in range(6)], fill=fill)
    elif kind == "line":
        d.line([(v[i] + 0.5) * s for i in range(4)], fill=fill, width=max(1, v[4]) * s)
    elif kind == "stroke":
        d.polygon(stroke_polygon(v, s), fill=fill)


def bbox(kind, v, size, pad=2):
    """Pixel box (x0, y0, x1, y1), end exclusive, that contains the whole shape."""
    if kind == "circle":
        x, y, r = v
        xs, ys = (x - r, x + r), (y - r, y + r)
    else:
        n = 6 if kind in ("tri", "stroke") else 4
        xs = (min(v[0:n:2]), max(v[0:n:2]))
        ys = (min(v[1:n:2]), max(v[1:n:2]))
        if kind == "line":
            pad += v[4]
        elif kind == "stroke":
            pad += max(v[6], v[7]) // 2 + 1
    x0, x1 = max(0, xs[0] - pad), min(size, xs[1] + pad + 1)
    y0, y1 = max(0, ys[0] - pad), min(size, ys[1] + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def shift(kind, v, dx, dy):
    """Move a shape's coordinates by (-dx, -dy) (line width is left alone)."""
    if kind == "circle":
        return [v[0] - dx, v[1] - dy, v[2]]
    n = 6 if kind in ("tri", "stroke") else 4
    out = [c - (dx if i % 2 == 0 else dy) for i, c in enumerate(v[:n])]
    if kind == "line":
        return out + [v[4]]
    if kind == "stroke":
        return out + v[6:8]
    return out


# ---------------- program text <-> items ----------------
def parse_program(text, size=SIZE):
    """-> (size, background [r,g,b], items). item = [kind, coords, [r,g,b], alpha]."""
    bg, items = [0, 0, 0], []
    for line in text.strip().splitlines():
        p = line.split()
        if not p:
            continue
        op = p[0]
        if op == "size":
            size = int(p[1])
        elif op == "bg":
            bg = [int(x) for x in p[1:4]]
        elif op == "skip":
            items.append(["skip", [], [0, 0, 0], 255])
        elif op in NCOORD:
            nums = [int(x) for x in p[1:]]
            n = NCOORD[op]
            if len(nums) in (n + 3, n + 4):                    # color, optional alpha
                items.append([op, nums[:n], nums[n:n + 3], nums[n + 3] if len(nums) > n + 3 else 255])
    return size, bg, items


def items_to_text(size, bg, items):
    lines = [f"size {size}", "bg {} {} {}".format(*bg)]
    for kind, v, c, a in items:
        if kind == "skip":
            lines.append("skip")
        else:
            lines.append(" ".join([kind] + [str(x) for x in v + c] + ([str(a)] if a < 255 else [])))
    return "\n".join(lines)


def paint_items(size, bg, items, scale=1, aa=False):
    s = scale * (2 if aa else 1)
    W = size * s
    img = Image.new("RGB", (W, W), tuple(bg))
    d = ImageDraw.Draw(img)
    for kind, v, color, alpha in items:
        if kind == "skip":
            continue
        if alpha >= 255:
            if s == 1:
                draw_shape(d, kind, v, tuple(color))
            else:
                draw_shape_scaled(d, kind, v, tuple(color), s)
        else:                                                  # semi-transparent: paste through a mask
            b = bbox(kind, v, size)
            if b is None:
                continue
            x0, y0, x1, y1 = b
            m = Image.new("L", ((x1 - x0) * s, (y1 - y0) * s), 0)
            vv = shift(kind, v, x0, y0)
            if s == 1:
                draw_shape(ImageDraw.Draw(m), kind, vv, alpha)
            else:
                draw_shape_scaled(ImageDraw.Draw(m), kind, vv, alpha, s)
            img.paste(tuple(color), (x0 * s, y0 * s), m)
    if aa:
        img = img.resize((size * scale, size * scale), Image.LANCZOS)
    return img


def render(program: str, size: int = SIZE, scale: int = 1, aa: bool = False) -> Image.Image:
    """scale=4, aa=True gives a smooth image 4x larger than the program's grid."""
    size, bg, items = parse_program(program, size)
    return paint_items(size, bg, items, scale, aa)


# ---------------- scores ----------------
def similarity(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    return 1.0 - ((x - y) ** 2).mean() / (255.0 ** 2)


def edge_similarity(a: Image.Image, b: Image.Image) -> float:
    """How well the edges/lines of two images agree (1 = same edges, 0 = unrelated).
    Flat blobs score low here even when the plain similarity looks high."""
    def grad(im):
        gy, gx = np.gradient(np.asarray(im.convert("L"), dtype=np.float32))
        return np.hypot(gx, gy).ravel()
    ga, gb = grad(a), grad(b)
    if ga.std() == 0 or gb.std() == 0:
        return 0.0
    return float(np.corrcoef(ga, gb)[0, 1])


def sharpness_ratio(a: Image.Image, b: Image.Image) -> float:
    """Average edge strength of `a` divided by that of `b` (1.0 = as crisp as b)."""
    def strength(im):
        gy, gx = np.gradient(np.asarray(im.convert("L"), dtype=np.float32))
        return float(np.hypot(gx, gy).mean())
    return strength(a) / max(strength(b), 1e-6)


def edge_weights(t, edge):
    """Per-pixel weights: 1 on flat areas, up to 1+edge on strong edges."""
    if edge <= 0:
        return np.ones(t.shape[:2], np.float32)
    lum = t @ np.array([0.299, 0.587, 0.114], np.float32)
    gy, gx = np.gradient(lum)
    g = np.hypot(gx, gy)
    g = np.clip(g / max(float(np.percentile(g, 99)), 1e-6), 0, 1)
    return (1 + edge * g).astype(np.float32)


# ---------------- fitter helpers ----------------
def shape_region(kind, v, size):
    """(x0, y0, x1, y1, mask) of the pixels a shape covers, drawn only inside its box (fast)."""
    b = bbox(kind, v, size)
    if b is None:
        return None
    x0, y0, x1, y1 = b
    m = Image.new("L", (x1 - x0, y1 - y0), 0)
    draw_shape(ImageDraw.Draw(m), kind, shift(kind, v, x0, y0), 255)
    mask = np.asarray(m) > 0
    if not mask.any():
        return None
    return x0, y0, x1, y1, mask


def orientation_field(t, size):
    """For every pixel: a unit vector pointing ALONG the local edge/strand direction
    (perpendicular to the brightness gradient), plus the gradient strength."""
    lum = np.clip(t @ np.array([0.299, 0.587, 0.114], np.float32), 0, 255).astype(np.uint8)
    blur = Image.fromarray(lum).filter(ImageFilter.GaussianBlur(max(1.0, size / 128)))
    gy, gx = np.gradient(np.asarray(blur, dtype=np.float32))
    mag = np.hypot(gx, gy)
    norm = np.maximum(mag, 1e-6)
    return -gy / norm, gx / norm, mag


def random_stroke(size, cx, cy, dim, field):
    """A curved stroke through (cx, cy), lined up with the local edge direction when there is one."""
    L = max(3, int(dim * random.uniform(1.0, 3.0)))
    ix, iy = min(cy, size - 1), min(cx, size - 1)
    if field is not None and field[2][ix, iy] > 1.5:
        ux, uy = float(field[0][ix, iy]), float(field[1][ix, iy])
    else:
        a = random.uniform(0, math.pi)
        ux, uy = math.cos(a), math.sin(a)
    if random.random() < 0.5:
        ux, uy = -ux, -uy
    nx, ny = -uy, ux
    bend = random.uniform(-0.3, 0.3) * L
    pts = [cx - ux * L / 2, cy - uy * L / 2, cx + nx * bend, cy + ny * bend,
           cx + ux * L / 2, cy + uy * L / 2]
    w1 = random.randint(1, 1 + size // 128)
    return [clamp(int(round(p)), 0, size) for p in pts] + [w1, random.randint(0, w1)]


def random_params(kind, size, max_dim, center=None, field=None):
    cx, cy = center if center else (random.randint(0, size), random.randint(0, size))

    def pt():
        return clamp(cx + random.randint(-max_dim, max_dim), 0, size)

    if kind == "stroke":
        return random_stroke(size, cx, cy, max_dim, field)

    if kind == "circle":
        return [cx, cy, random.randint(1, max_dim)]
    if kind == "rect":
        return [cx, cy, pt(), pt()]
    if kind == "tri":
        return [cx, cy, pt(), pt(), pt(), pt()]
    return [cx, cy, pt(), pt(), random.randint(1, 3)]


def random_params_in(kind, size, region, max_dim):
    """Random shape whose center lies inside region = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = region
    cx, cy = int(random.uniform(x0, x1)), int(random.uniform(y0, y1))

    def c(n):
        return clamp(n, 0, size)

    if kind == "circle":
        return [cx, cy, random.randint(1, max_dim)]
    if kind == "rect":
        hw, hh = random.randint(1, max_dim), random.randint(1, max_dim)
        return [c(cx - hw), c(cy - hh), c(cx + hw), c(cy + hh)]
    if kind == "tri":
        return [c(cx + random.randint(-max_dim, max_dim)) for _ in range(6)]
    dx, dy = random.randint(-max_dim, max_dim), random.randint(-max_dim, max_dim)
    return [c(cx - dx), c(cy - dy), c(cx + dx), c(cy + dy), random.randint(1, 3)]


def center(kind, v):
    if kind == "circle":
        return v[0], v[1]
    if kind == "stroke":
        return 0.25 * v[0] + 0.5 * v[2] + 0.25 * v[4], 0.25 * v[1] + 0.5 * v[3] + 0.25 * v[5]
    if kind == "tri":
        return (v[0] + v[2] + v[4]) / 3, (v[1] + v[3] + v[5]) / 3
    return (v[0] + v[2]) / 2, (v[1] + v[3]) / 2


def extent(kind, v):
    if kind == "circle":
        return 2 * v[2]
    n = 6 if kind in ("tri", "stroke") else 4
    xs, ys = v[0:n:2], v[1:n:2]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def mutate_params(kind, v, step, size):
    v = v[:]
    i = random.randrange(len(v))
    if kind == "line" and i == 4:
        v[i] = clamp(v[i] + random.choice([-1, 1]), 1, 6)
    elif kind == "stroke" and i >= 6:
        v[i] = clamp(v[i] + random.choice([-1, 1]), 0, 10)
        if max(v[6], v[7]) < 1:
            v[6] = 1
    elif kind == "circle" and i == 2:
        v[i] = clamp(v[i] + random.randint(-step, step), 1, size)
    else:
        v[i] = clamp(v[i] + random.randint(-step, step), 0, size)
    return v


def make_sampler(err, size):
    """Returns a function that picks a pixel, more likely where the error is large."""
    b = max(1, size // 32)
    n = size // b
    e = err[:n * b, :n * b].reshape(n, b, n, b).sum((1, 3)).astype(np.float64).ravel()
    total = e.sum()
    if total <= 0:
        return lambda: (random.randint(0, size - 1), random.randint(0, size - 1))
    cdf = np.cumsum(e) / total

    def sample():
        k = min(int(np.searchsorted(cdf, random.random())), len(cdf) - 1)
        gy, gx = divmod(k, n)
        return gx * b + random.randint(0, b - 1), gy * b + random.randint(0, b - 1)

    return sample


def evaluate(kind, v, alpha, t, canvas, err, W):
    """Return (weighted error change, region, best color). Negative change = improvement."""
    reg = shape_region(kind, v, t.shape[0])
    if reg is None:
        return None
    x0, y0, x1, y1, mask = reg
    T = t[y0:y1, x0:x1][mask]
    C = canvas[y0:y1, x0:x1][mask]
    E = err[y0:y1, x0:x1][mask]
    if alpha >= 255:
        color = np.round(T.mean(axis=0))                      # optimal flat color for MSE
        newC = color
    else:
        a = alpha / 255.0                                     # optimal color under that opacity
        color = np.clip(np.round(((T - (1 - a) * C) / a).mean(axis=0)), 0, 255)
        newC = np.round((color * alpha + C * (255 - alpha)) / 255)
    new = (W[y0:y1, x0:x1][mask] * ((T - newC) ** 2).sum(-1)).sum()
    return new - E.sum(), reg, color.astype(int)


def commit(canvas, err, W, t, reg, color, alpha):
    x0, y0, x1, y1, mask = reg
    sub = canvas[y0:y1, x0:x1]
    if alpha >= 255:
        sub[mask] = color
    else:
        sub[mask] = np.round((color * alpha + sub[mask] * (255 - alpha)) / 255)
    err[y0:y1, x0:x1][mask] = W[y0:y1, x0:x1][mask] * ((sub[mask] - t[y0:y1, x0:x1][mask]) ** 2).sum(-1)


def slot_plan(size, levels):
    """One slot per grid cell per level. Level g splits the image into g x g cells,
    so slot k always means 'a shape of this size, centered in this cell'."""
    plan = []
    for g in levels:
        cell = size / g
        md = max(3, int(cell * 0.7))
        for gy in range(g):
            for gx in range(g):
                plan.append((gx * cell, gy * cell, (gx + 1) * cell, (gy + 1) * cell, md))
    return plan


def best_shape(t, canvas, err, W, size, region, md, cands, climb, sampler=None, alphas=(255,),
               kinds=KINDS, field=None):
    """Best of `cands` random shapes, then hill-climbed for `climb` steps.
    Returns (delta, kind, coords, alpha, region, color) or None."""
    def valid(kind, v):
        if region is None:
            return True
        cx, cy = center(kind, v)
        x0, y0, x1, y1 = region
        return x0 <= cx <= x1 and y0 <= cy <= y1 and extent(kind, v) <= 3 * md

    best = None
    for _ in range(cands):
        kind = random.choice(kinds)
        if region is None:
            dim = max(2, int(md * 2 ** random.uniform(-3, 0))) if sampler else md
            v = random_params(kind, size, dim, sampler() if sampler else None, field)
        else:
            v = random_params_in(kind, size, region, md)
        if not valid(kind, v):
            continue
        alpha = random.choice(alphas)
        r = evaluate(kind, v, alpha, t, canvas, err, W)
        if r and (best is None or r[0] < best[0]):
            best = [r[0], kind, v, alpha, r[1], r[2]]
    if best is None:
        return None
    delta, kind, v, alpha, reg, color = best
    for _ in range(climb):
        if len(alphas) > 1 and random.random() < 0.15:
            v2, a2 = v, random.choice(alphas)
        else:
            v2, a2 = mutate_params(kind, v, max(1, extent(kind, v) // 5), size), alpha
        if not valid(kind, v2):
            continue
        r = evaluate(kind, v2, a2, t, canvas, err, W)
        if r and r[0] < delta:
            delta, v, alpha, reg, color = r[0], v2, a2, r[1], r[2]
    return delta, kind, v, alpha, reg, color


# ---------------- the fitter ----------------
def fit(target: Image.Image, n_shapes: int = 100, cands: int = 30, climb: int = 60,
        size: int = SIZE, verbose: bool = True, levels=None,
        guided: bool = True, alpha: bool = False, edge: float = 0.0, strokes: bool = False,
        resume: str = None, time_limit: float = None, checkpoint=None, status: dict = None) -> str:
    """Add shapes one at a time, big to small.
      guided     : put new shapes where the error is largest
      alpha      : allow semi-transparent shapes (soft shading)
      edge       : weight edges up to (1+edge)x in the score, so lines and outlines get fitted
      strokes    : also try curved tapered brush strokes, lined up with the image's edges
      resume     : program text to continue from (n_shapes is the TOTAL wanted)
      time_limit : stop (and return what exists) after this many seconds
      checkpoint : function called with the program text every 500 shapes
      status     : optional dict; gets status['done'] = False if the time limit stopped the fit
    With `levels` (e.g. [1,2,4,8]): fixed slots, one per grid cell per level (no alpha/guide);
    a slot that cannot improve the image is written as `skip`."""
    t0 = time.time()
    t = np.asarray(target.convert("RGB").resize((size, size)), dtype=np.float32)
    W = edge_weights(t, edge)
    kinds = KINDS + ["stroke", "stroke"] if strokes else KINDS
    field = orientation_field(t, size) if strokes else None

    if resume:
        _, bg, items = parse_program(resume, size)
        canvas = np.asarray(paint_items(size, bg, items), dtype=np.float32).copy()
        lines = items_to_text(size, bg, items).splitlines()
    else:
        avg = t.reshape(-1, 3).mean(axis=0).round().astype(int)
        canvas = np.empty_like(t)
        canvas[:] = avg
        lines = [f"size {size}", f"bg {avg[0]} {avg[1]} {avg[2]}"]
    err = W * ((canvas - t) ** 2).sum(-1)

    if levels:
        plan, cands, guided, alpha = slot_plan(size, levels), max(cands, 40), False, False
        start = 0
    else:
        start = len(lines) - 2                                  # shapes already in the program
        plan = [None] * n_shapes
    md0 = size * 0.6
    alphas = ALPHAS if alpha else (255,)

    def plain_sim():
        return 1 - ((canvas - t) ** 2).mean() / (255.0 ** 2)

    for i in range(start, len(plan)):
        slot = plan[i]
        if time_limit and time.time() - t0 > time_limit:
            if verbose:
                print(f"time limit reached at shape {i}; continue with --resume", flush=True)
            if status is not None:
                status["done"] = False
            break
        if slot is None:
            region = None
            md = max(3, int(md0 * (3 / md0) ** (i / max(1, n_shapes))))   # big -> small
            sampler = make_sampler(err, size) if guided else None
        else:
            region, md, sampler = slot[:4], slot[4], None
        r = best_shape(t, canvas, err, W, size, region, md, cands, climb, sampler, alphas, kinds, field)
        if r is not None and r[0] < 0:
            _, kind, v, a, reg, color = r
            commit(canvas, err, W, t, reg, color, a)
            lines.append(" ".join([kind] + [str(n) for n in v] + [str(int(c)) for c in color]
                                  + ([str(a)] if a < 255 else [])))
        elif levels:
            lines.append("skip")
        if not levels and (i + 1) % 500 == 0:
            if checkpoint:
                checkpoint("\n".join(lines))
            if verbose:
                print(f"shape {i + 1}/{n_shapes}: similarity {plain_sim():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    if status is not None:
        status.setdefault("done", True)
    if verbose and levels:
        print(f"{len(plan)} slots, {sum(1 for l in lines if l == 'skip')} empty, similarity {plain_sim():.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--shapes", type=int, default=100, help="total number of shapes wanted")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--levels", default="", help="e.g. 1,2,4,8 for fixed slots")
    ap.add_argument("--alpha", action="store_true", help="allow semi-transparent shapes")
    ap.add_argument("--edge", type=float, default=0.0, help="edge weighting, e.g. 4")
    ap.add_argument("--strokes", action="store_true", help="also use curved brush strokes (hair!)")
    ap.add_argument("--no-guide", action="store_true", help="place shapes at random")
    ap.add_argument("--cands", type=int, default=30, help="random tries per shape")
    ap.add_argument("--climb", type=int, default=60, help="improvement steps per shape")
    ap.add_argument("--time-limit", type=float, default=0, help="stop after this many seconds")
    ap.add_argument("--resume", default="", help="continue from this .ipg file")
    ap.add_argument("--out", default="output")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed + (len(open(a.resume).read().splitlines()) if a.resume else 0))
    target = Image.open(a.image).convert("RGB").resize((a.size, a.size))
    levels = [int(x) for x in a.levels.split(",")] if a.levels else None

    def save(text):
        with open(a.out + ".ipg", "w") as f:
            f.write(text)

    program = fit(target, n_shapes=a.shapes, size=a.size, levels=levels, cands=a.cands, climb=a.climb,
                  guided=not a.no_guide, alpha=a.alpha, edge=a.edge, strokes=a.strokes,
                  resume=open(a.resume).read() if a.resume else None,
                  time_limit=a.time_limit or None, checkpoint=save)
    save(program)
    out = render(program)
    n = len(program.splitlines()) - 2
    print(f"{n} shapes   final similarity {similarity(out, target):.4f}   edge match {edge_similarity(out, target):.4f}"
          f"   sharpness {sharpness_ratio(out, target):.2f}x of original")
    render(program, scale=2 if a.size >= 384 else 4, aa=True).save(a.out + ".png")
    print(f"wrote {a.out}.ipg and {a.out}.png (smooth)")
