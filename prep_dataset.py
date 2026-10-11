"""Turn a folder of tagged images into a training file for train.py.

Expected layout (Danbooru/safebooru-style exports):
    images/123.jpg            tags/123.txt   -> "1girl, solo, black_hair, rating:general"
                                       or   tags/123.json -> {"tags": [...] | "tag_string": "...",
                                                              "rating": "general"}
Rating can be a `rating:xxx` tag or a json field. Accepted values:
    general|safe|g, sensitive|s, questionable|q, explicit|e

    python prep_dataset.py --images images --tags tags --size 64 --top-k 300 \
        --min-count 20 --out prepared.npz

Output npz: X uint8 [N,S,S,3] | EX int [N,2] (subject_count, rating) | MU float [N,K] | vocab
Safety handling:
  * tags in sexual/nudity/suggestive/minor/age-ambiguous lexicon categories are NOT put in the
    prompt vocabulary (the `rating` head carries that signal for flagging instead)
  * samples that carry a minor-related tag AND a questionable/explicit rating are dropped
    from training entirely (count reported)
"""
import argparse
import json
import os
import sys
import re
from collections import Counter

import numpy as np
from PIL import Image

import moderation as M
from imgio import load_rgb

RATINGS = ["general", "sensitive", "questionable", "explicit"]
RATING_ALIAS = {"general": 0, "safe": 0, "g": 0, "sensitive": 1, "s": 1,
                "questionable": 2, "q": 2, "explicit": 3, "e": 3}
SUBJECTS = ["none", "solo", "duo", "group"]
META = {"highres", "absurdres", "lowres", "commentary", "commentary_request", "translated",
        "translation_request", "artist_name", "signature", "watermark", "web_address",
        "bad_id", "bad_pixiv_id", "english_commentary", "check_commentary", "md5_mismatch",
        "official_art", "scan", "jpeg_artifacts", "dated", "twitter_username", "protected_link"}
META_PREFIX = ("rating:", "artist:", "copyright:", "character:", "meta:", "score:", "year:")
CATEGORY_PATTERNS = [  # only used to order tags inside generated prompts
    ("subject", r"^(\d+(girl|boy|girls|boys)|solo|multiple_)"),
    ("hair", r"hair|bangs|ponytail|twintails|braid|ahoge|bob_cut|sidelocks"),
    ("eyes", r"eyes|eyebrows|eyelashes|pupils"),
    ("expression", r"smile|blush|open_mouth|closed_mouth|frown|expressionless|tears|looking_"),
    ("clothes", r"shirt|dress|skirt|uniform|jacket|kimono|coat|sweater|necktie|tie$|hat|"
                r"gloves|pants|shorts|socks|thighhighs|scarf|hoodie|apron|ribbon|bow|collar"),
    ("accessory", r"glasses|earrings|jewelry|necklace|flower|hairpin|hair_ornament|mask|bag"),
    ("style", r"monochrome|greyscale|sketch|lineart|traditional_media|colored|realistic"),
    ("background", r"background|indoors|outdoors|sky|simple"),
]


def category_of(tag):
    for name, pat in CATEGORY_PATTERNS:
        if re.search(pat, tag):
            return name
    return "other"


def read_tags(tags_dir, stem):
    p_txt, p_json = (os.path.join(tags_dir, stem + e) for e in (".txt", ".json"))
    rating, tags = None, []
    if os.path.exists(p_json):
        d = json.load(open(p_json, encoding="utf-8"))
        t = d.get("tags") or d.get("tag_string") or ""
        tags = t.split() if isinstance(t, str) else list(t)
        rating = d.get("rating")
    elif os.path.exists(p_txt):
        raw = open(p_txt, encoding="utf-8", errors="ignore").read()
        tags = [x for x in re.split(r"[,\n]", raw)]
    else:
        return None, None
    tags = [re.sub(r"\s+", "_", t.strip().lower()) for t in tags if t.strip()]
    for t in tags:
        if t.startswith("rating:"):
            rating = t.split(":", 1)[1]
    rating = RATING_ALIAS.get(str(rating).strip().lower()) if rating is not None else None
    return tags, rating


def subject_of(tags):
    s = set(tags)
    if s & {"solo"} or s & {"1girl", "1boy"}:
        return 1
    if s & {"2girls", "2boys", "1girl_1boy"} or any(t.startswith("2") for t in s):
        return 2
    if any(re.match(r"^([3-9]|\d{2,})(girls|boys)$", t) for t in s) or "multiple_girls" in s \
            or "multiple_boys" in s or "6+girls" in s:
        return 3
    return 0


def load_image(path, size):
    im = load_rgb(path)
    w, h = im.size
    s = max(w, h)
    border = np.concatenate([np.asarray(im)[0], np.asarray(im)[-1]]).reshape(-1, 3)
    canvas = Image.new("RGB", (s, s), tuple(int(v) for v in border.mean(0)))
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))  # pad (keeps aspect ratio)
    return np.asarray(canvas.resize((size, size), Image.LANCZOS), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="prepared.npz")
    a = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith(exts))
    np.random.default_rng(a.seed).shuffle(files)
    if a.limit:
        files = files[:a.limit]

    stats = Counter()
    recs, counts = [], Counter()
    for f in files:
        stem = os.path.splitext(f)[0]
        tags, rating = read_tags(a.tags, stem)
        if tags is None:
            stats["no_tag_file"] += 1
            continue
        if rating is None:
            stats["no_rating"] += 1
            continue
        if "minor" in M.check_text(" ".join(tags))["categories"] and rating >= 2:
            stats["dropped_minor_plus_nsfw"] += 1
            continue
        base = [t for t in dict.fromkeys(tags) if t not in META and not t.startswith(META_PREFIX)]
        keep = [t for t in base if not M.tag_is_flagged(t)]
        stats["flagged_tags_hidden_from_vocab"] += len(base) - len(keep)
        recs.append((f, subject_of(tags), rating, keep))
        counts.update(keep)

    vocab = [t for t, c in counts.most_common() if c >= a.min_count
             and t not in {"solo", "1girl", "1boy", "2girls", "2boys"}][:a.top_k]
    idx = {t: i for i, t in enumerate(vocab)}
    n = len(recs)
    X = np.zeros((n, a.size, a.size, 3), np.uint8)
    EX = np.zeros((n, 2), np.int64)
    MU = np.zeros((n, len(vocab)), np.float32)
    ok = 0
    for i, (f, subj, rating, keep) in enumerate(recs):
        try:
            X[ok] = load_image(os.path.join(a.images, f), a.size)
        except Exception:
            stats["unreadable_image"] += 1
            continue
        EX[ok] = (subj, rating)
        for t in keep:
            if t in idx:
                MU[ok, idx[t]] = 1
        ok += 1
    X, EX, MU = X[:ok], EX[:ok], MU[:ok]
    if ok == 0:
        sys.exit("0 usable samples. Checked: tag files named like the images (cat.jpg <-> cat.txt) with a "
                 f"rating:xxx tag, and readable images. Details: {dict(stats)}")

    vocab_json = json.dumps({"exclusive": {"subject_count": SUBJECTS, "rating": RATINGS},
                             "multi": vocab})
    np.savez_compressed(a.out, X=X, EX=EX, MU=MU, vocab=np.array(vocab_json))
    json.dump({"exclusive": {"subject_count": SUBJECTS, "rating": RATINGS}, "multi": vocab,
               "categories": {t: category_of(t) for t in vocab}},
              open(os.path.splitext(a.out)[0] + "_vocab.json", "w", encoding="utf-8"), indent=1)

    print(f"saved {a.out}: {ok} samples, {len(vocab)} tags, size {a.size}")
    print("rating counts:", {r: int((EX[:, 1] == i).sum()) for i, r in enumerate(RATINGS)})
    print("subject counts:", {s: int((EX[:, 0] == i).sum()) for i, s in enumerate(SUBJECTS)})
    print("skipped/dropped:", dict(stats))
    print("top tags:", [(t, int(MU[:, idx[t]].sum())) for t in vocab[:15]])


if __name__ == "__main__":
    main()
