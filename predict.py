"""Image -> tags -> prompt, fully offline (NumPy + Pillow only), with moderation.

    python predict.py image.png --weights v11_weights.npz [--fit pad|crop|squash]

If the weights have a `rating` head, the image is rated and flagged. The generated prompt is
also run through the text flagger. Prompts for flagged/blocked images are withheld.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

import tags as T
from imgio import load_rgb
import moderation as M
from vit import TinyViT, softmax, sigmoid
from prompt_builder import select_tags, build_prompt


def load_image(path, size, fit="pad"):
    im = load_rgb(path)
    w, h = im.size
    if fit == "crop":      # center-crop to square
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    elif fit == "pad":     # pad to square (edge-average color), keeps aspect ratio
        s = max(w, h)
        a = np.asarray(im)
        border = np.concatenate([a[0], a[-1]]).reshape(-1, 3).mean(0)
        canvas = Image.new("RGB", (s, s), tuple(int(v) for v in border))
        canvas.paste(im, ((s - w) // 2, (s - h) // 2))
        im = canvas
    return np.asarray(im.resize((size, size), Image.LANCZOS), np.uint8)[None]


_MODELS = {}


def tag_image(path, weights, fit="pad"):
    """Tags + exclusive-group picks for one image (model cached). Returns (on_tags, ex, probs_by_rating)."""
    if weights not in _MODELS:
        _MODELS[weights] = TinyViT.load(weights)
    model, extra = _MODELS[weights]
    if "vocab" in extra:
        T.load_vocab_json(extra["vocab"])
    else:
        T.reset_default()
    x = load_image(path, model.cfg["img"], fit)
    logits = model.forward(x)[0][0]
    probs = np.zeros_like(logits)
    for gs, ge in T.GROUPS:
        probs[gs:ge] = softmax(logits[gs:ge])
    s, e = T.MULTI_SLICE
    probs[s:e] = sigmoid(logits[s:e])
    thr = extra.get("thresholds", np.full(len(T.MULTI), .5))
    on, ex = select_tags(probs, thr)
    return on, ex, extra


def run(image, weights, fit="pad", withhold=True):
    model, extra = TinyViT.load(weights)
    if "vocab" in extra:
        T.load_vocab_json(extra["vocab"])
    else:
        T.reset_default()
    x = load_image(image, model.cfg["img"], fit)
    logits = model.forward(x)[0][0]
    probs = np.zeros_like(logits)
    for gs, ge in T.GROUPS:
        probs[gs:ge] = softmax(logits[gs:ge])
    s, e = T.MULTI_SLICE
    probs[s:e] = sigmoid(logits[s:e])
    thr = extra.get("thresholds", np.full(len(T.MULTI), .5))
    on, ex = select_tags(probs, thr)
    prompt, negative = build_prompt(on, ex)

    if "rating" in T.GROUP_NAMES:
        gs, ge = T.GROUPS[T.GROUP_NAMES.index("rating")]
        img_mod = M.check_image(probs[gs:ge], T.EXCLUSIVE["rating"],
                                flag_threshold=float(extra.get("flag_threshold", .5)))
    else:
        img_mod = {"action": "review", "checked": False,
                   "note": "weights have no rating head - image NOT assessed"}
    txt_mod = M.check_text(prompt)
    action = M.combine(img_mod, txt_mod)
    out = dict(action=action, image_moderation=img_mod,
               prompt_moderation={k: txt_mod[k] for k in ("action", "categories")})
    if action in ("flag", "block") and withhold:
        out.update(prompt="[withheld: flagged by moderation]", negative_prompt="")
    else:
        out.update(prompt=prompt, negative_prompt=negative)
    shown = [t for t in on]
    out.update(tags=shown if not (action in ("flag", "block") and withhold) else [],
               exclusive={k: [v[0], round(v[1], 3)] for k, v in ex.items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default="v11_weights.npz")
    ap.add_argument("--fit", default="pad", choices=["crop", "pad", "squash"])
    ap.add_argument("--show-flagged", action="store_true",
                    help="show prompt for 'flag' (not 'block') results, e.g. for a reviewer UI")
    a = ap.parse_args()
    r = run(a.image, a.weights, a.fit, withhold=not a.show_flagged)
    if r["action"] == "block":  # hard rule: never override
        r.update(prompt="[withheld: flagged by moderation]", negative_prompt="", tags=[])
    print(json.dumps(r, indent=2))
