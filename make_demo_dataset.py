"""Write a tiny Danbooru-style folder (images/ + tags/) from the synthetic renderer.

PLUMBING TEST ONLY: ratings are assigned by a harmless visual rule
(blush -> sensitive, blush+glasses -> questionable, else general) so the rating
head / flag threshold / moderation path can be exercised end to end.
Real moderation quality requires a real rated dataset.

    python make_demo_dataset.py --n 5000 --out demo
"""
import argparse
import os
import numpy as np
from PIL import Image

import tags as T
import synth


def to_dan(name):
    if name == "hair_ornament":
        return name
    if name.startswith("hair_"):
        return name[5:] + "_hair"
    if name.startswith("eyes_"):
        return name[5:] + "_eyes"
    if name.startswith("clothes_"):
        return name[8:] + "_clothes"
    return name


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--out", default="demo")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
T.reset_default()
os.makedirs(f"{a.out}/images", exist_ok=True)
os.makedirs(f"{a.out}/tags", exist_ok=True)
rng = np.random.default_rng(a.seed)
bg_names = T.EXCLUSIVE["background"]
for i in range(a.n):
    img, ex, mu = synth.render(rng, 64)
    on = {T.MULTI[j] for j in np.nonzero(mu)[0]}
    rating = "questionable" if {"blush", "glasses"} <= on else "sensitive" if "blush" in on else "general"
    tags = (["solo", "1girl"] if ex[0] == 0 else ["2girls"]) + [to_dan(t) for t in sorted(on)] \
        + [bg_names[ex[1]] + "_background", "highres", f"rating:{rating}"]
    Image.fromarray(img).save(f"{a.out}/images/{i:05d}.png")
    with open(os.path.join(a.out, "tags", f"{i:05d}.txt"), "w", encoding="utf-8") as _f:
        _f.write(", ".join(tags))
print("wrote", a.n, "samples to", a.out)
