"""Image -> prompt. One entry point, three engines, always moderated.

    python image_prompt.py pic.jpg                                   # basic: works on ANY image, offline
    python image_prompt.py pic.jpg --wd14 model.onnx tags.csv        # real tagger (needs onnxruntime)
    python image_prompt.py pic.jpg --vit my_weights.npz              # your own trained NumPy ViT

basic : numpy+Pillow only. Reads palette, background, lighting, orientation, flat-vs-shaded style.
        It cannot recognise characters/objects - it describes the *look* of the image.
wd14  : optional. Any WD14/SmilingWolf-style ONNX tagger (model.onnx + selected_tags.csv).
        Gives real content tags (hair, eyes, clothes, pose...) AND a rating for moderation.
vit   : the NumPy ViT from train.py (only useful after training on a real dataset).
"""
import argparse
import csv
import functools
import json
import sys

import numpy as np
from PIL import Image

import moderation as M
from imgio import load_rgb
import prompt_builder as PB

NAMED = {
    "black": (15, 15, 15), "white": (240, 240, 240), "gray": (128, 128, 128), "red": (200, 30, 40),
    "orange": (240, 140, 30), "yellow": (240, 220, 60), "green": (50, 160, 70), "teal": (30, 150, 150),
    "blue": (50, 90, 200), "navy": (20, 30, 90), "purple": (120, 60, 160), "pink": (240, 130, 180),
    "brown": (110, 70, 40), "beige": (225, 205, 170), "skin tone": (225, 170, 140),
}
_NAMES = list(NAMED)
_RGB = np.array([NAMED[n] for n in _NAMES], float)


def _name(rgb):
    """HSV-based colour name; much more stable than nearest-RGB for anime palettes."""
    import colorsys
    r, g, b = (float(x) / 255 for x in rgb)
    h, sat, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360
    if v < 0.18:
        return "black"
    if sat < 0.14:
        return "white" if v > 0.82 else "gray" if v > 0.35 else "dark gray"
    if sat < 0.40 and 8 <= h <= 40 and v > 0.55:
        return "skin tone" if sat > 0.15 else "beige"
    if h < 12 or h >= 345:
        name = "red" if v > 0.45 else "dark red"
    elif h < 40:
        name = "orange" if v > 0.6 else "brown"
    elif h < 70:
        name = "yellow" if v > 0.6 else "olive"
    elif h < 165:
        name = "green"
    elif h < 200:
        name = "teal"
    elif h < 255:
        name = "blue" if v > 0.5 else "navy"
    elif h < 295:
        name = "purple"
    else:
        name = "pink" if v > 0.6 else "magenta"
    return name


def _kmeans(px, k=5, iters=8, seed=0):
    rng = np.random.default_rng(seed)
    c = px[rng.choice(len(px), k, replace=False)].astype(float)
    for _ in range(iters):
        lab = ((px[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = px[lab == j].mean(0)
    return c, np.bincount(lab, minlength=k) / len(px)


def guess_hair(small, bg_rgb):
    """HEURISTIC: dominant non-background colour in the top-centre of the frame.
    Right for most portraits, wrong for crops where the head is not at the top."""
    h, w, _ = small.shape
    region = small[: max(4, int(h * 0.28)), int(w * 0.2): int(w * 0.8)].reshape(-1, 3).astype(float)
    if _name(bg_rgb) != "black":           # on a black backdrop dark pixels are probably the hair
        far = np.sqrt(((region - np.asarray(bg_rgb, float)) ** 2).sum(1)) > 60
        region = region[far]
    if len(region) < 50:
        return []
    cents, frac = _kmeans(region, k=3)
    out = []
    for i in np.argsort(-frac):
        n = _name(cents[i])
        if frac[i] > (0.25 if out else 0.15) and n not in out and n not in ("skin tone", "beige", "white"):
            out.append(n)
    return out[:2]


def framing(small, bg_rgb):
    far = np.sqrt(((small.astype(float) - np.asarray(bg_rgb, float)) ** 2).sum(2)) > 60
    cover = far.mean()
    return "close-up, subject fills the frame" if cover > 0.8 else "centered subject" if cover > 0.35 else "small subject, wide shot"


def analyse_basic(path, guess=False):
    im = load_rgb(path)
    w, h = im.size
    small = np.asarray(im.resize((128, int(128 * h / w)) if w >= h else (int(128 * w / h), 128)), np.uint8)
    px = small.reshape(-1, 3)
    cents, frac = _kmeans(px)
    order = np.argsort(-frac)
    palette, seen = [], set()
    for i in order:
        n = _name(cents[i])
        if n not in seen and frac[i] > 0.06:
            seen.add(n)
            palette.append(n)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]]).astype(float)
    bc, bf = _kmeans(border, k=3)          # dominant border colour, not the mean
    bg_rgb = bc[int(np.argmax(bf))]
    bg = _name(bg_rgb)
    simple_bg = border.std(0).mean() < 28
    gray = small.mean(2)
    bright = gray.mean()
    gy, gx = np.abs(np.diff(gray, axis=0)), np.abs(np.diff(gray, axis=1))
    edge = (gy.mean() + gx.mean()) / 2
    q = (small // 32).reshape(-1, 3)
    codes = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    top8 = np.sort(np.bincount(codes, minlength=512))[::-1][:8].sum() / len(codes)
    mx, mn = small.max(2).astype(float), small.min(2).astype(float)
    sat = ((mx - mn) / np.maximum(mx, 1)).mean()
    tags = []
    ar = w / h
    tags.append("portrait orientation" if ar < 0.85 else "landscape orientation" if ar > 1.2 else "square composition")
    tags.append("flat colors, illustration" if top8 > 0.55 else "soft shading, detailed rendering")
    if edge > 14:
        tags.append("strong linework")
    tags.append("dark, low-key lighting" if bright < 85 else "bright, high-key lighting" if bright > 170 else "balanced lighting")
    tags.append("vibrant colors" if sat > 0.5 else "muted colors" if sat < 0.22 else "")
    if simple_bg:
        tags += ["simple background", f"{bg} background"]
    else:
        tags.append(f"{bg} toned background")
    tags.append(framing(small, bg_rgb))
    tags += [f"{c} color accents" for c in palette[:4] if c != bg]
    hair = guess_hair(small, bg_rgb) if guess else []
    tags = [f"{c} hair" for c in hair] + tags
    return [t for t in tags if t], dict(size=[w, h], palette=palette, background=bg, hair_guess=hair)


@functools.lru_cache(maxsize=2)
def load_wd14(model_path, csv_path):
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("wd14 mode needs onnxruntime: pip install onnxruntime  (basic mode needs nothing extra)")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return sess, [r["name"] for r in rows], [int(r["category"]) for r in rows]


def wd14_probs(sess, path):
    inp = sess.get_inputs()[0]
    size = inp.shape[1]
    im = load_rgb(path)
    s = max(im.size)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    a = np.asarray(canvas.resize((size, size), Image.BICUBIC), np.float32)[:, :, ::-1]  # RGB->BGR
    return sess.run(None, {inp.name: a[None]})[0][0]


def split_wd14(probs, names, cats, thr=0.35, char_thr=0.85):
    """Category 0 = general, 4 = character, 9 = rating (WD14 convention)."""
    rating = [(n, float(p)) for n, p, c in zip(names, probs, cats) if c == 9]
    general = sorted(((n, float(p)) for n, p, c in zip(names, probs, cats) if c == 0 and p >= thr), key=lambda x: -x[1])
    chars = [(n, float(p)) for n, p, c in zip(names, probs, cats) if c == 4 and p >= char_thr]
    return general, chars, rating


def run(path, wd14=None, vit=None, withhold=True, guess=False):
    img_mod = {"action": "review", "checked": False, "note": "no rating model - image NOT assessed"}
    if wd14:
        sess, names, cats = load_wd14(*tuple(wd14))
        general, chars, rating = split_wd14(wd14_probs(sess, path), names, cats)
        tags = [n.replace("_", " ").replace("(", r"\(").replace(")", r"\)") for n, _ in chars + general]
        if rating:
            order = ["general", "sensitive", "questionable", "explicit"]
            rp = dict(rating)
            img_mod = M.check_image([rp.get(k, 0.0) for k in order], order)
        info, engine = {}, "wd14"
    elif vit:
        import predict
        r = predict.run(path, vit, withhold=withhold)
        return dict(engine="vit", **r)
    else:
        tags, info = analyse_basic(path, guess)
        engine = "basic"
    prompt = ", ".join(PB.QUALITY + tags)
    txt_mod = M.check_text(prompt)
    action = M.combine(img_mod, txt_mod)
    out = dict(engine=engine, action=action, image_moderation=img_mod,
               prompt_moderation={k: txt_mod[k] for k in ("action", "categories")}, info=info)
    if action in ("flag", "block") and withhold:
        out.update(prompt="[withheld: flagged by moderation]", negative_prompt="", tags=[])
    else:
        out.update(prompt=prompt, negative_prompt=", ".join(PB.NEGATIVE), tags=tags)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--wd14", nargs=2, metavar=("MODEL_ONNX", "TAGS_CSV"))
    ap.add_argument("--vit", metavar="WEIGHTS_NPZ")
    ap.add_argument("--show-flagged", action="store_true")
    ap.add_argument("--guess", action="store_true", help="basic engine: also guess hair colour (heuristic)")
    a = ap.parse_args()
    r = run(a.image, a.wd14, a.vit, withhold=not a.show_flagged, guess=a.guess)
    if r["action"] == "block":
        r.update(prompt="[withheld: flagged by moderation]", negative_prompt="", tags=[])
    print(json.dumps(r, indent=2))
