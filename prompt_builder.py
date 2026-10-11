"""Turn predicted tag probabilities into a text-to-image prompt."""
import numpy as np

import tags as T

QUALITY = ["masterpiece", "best quality", "anime style"]
NEGATIVE = ["lowres", "bad anatomy", "bad hands", "extra fingers", "worst quality",
            "blurry", "jpeg artifacts"]
COLORS = ["black", "brown", "blonde", "pink", "blue", "red", "white"]


def select_tags(probs, thresholds):
    """probs: [n_out] sigmoid/softmax outputs split by layout. Returns dict."""
    s, e = T.MULTI_SLICE
    mp = probs[s:e]
    on = [t for t, p, th in zip(T.MULTI, mp, thresholds) if p >= th]
    ex = {}
    for name, (gs, ge) in zip(T.GROUP_NAMES, T.GROUPS):
        i = int(np.argmax(probs[gs:ge]))
        ex[name] = (T.EXCLUSIVE[name][i], float(probs[gs + i]))
    return on, ex


_ORDER = ["subject", "hair", "eyes", "expression", "clothes", "accessory", "style",
          "other", "background"]


def build_prompt_generic(on, ex):
    """Prompt for dataset-derived vocabularies: tags grouped by category, underscores -> spaces."""
    from prep_dataset import category_of
    parts = list(QUALITY)
    sc = ex.get("subject_count", ("none", 0))[0]
    parts += {"solo": ["1girl", "solo"], "duo": ["2girls"], "group": ["multiple girls"]}.get(sc, [])
    for cat in _ORDER:
        parts += [t.replace("_", " ") for t in sorted(on) if category_of(t) == cat]
    return ", ".join(parts), ", ".join(NEGATIVE)


def build_prompt(on, ex):
    if "hair_black" not in T.MULTI:  # dataset-derived vocabulary
        return build_prompt_generic(on, ex)
    on = set(on)
    parts = list(QUALITY)
    parts.append("1girl, solo" if ex["subject_count"][0] == "solo" else "2girls, duo")

    colors = [c for c in COLORS if "hair_" + c in on]
    length = "long" if "long_hair" in on and "short_hair" not in on else \
             "short" if "short_hair" in on and "long_hair" not in on else ""
    for c in colors:
        parts.append(f"{length + ' ' if length else ''}{c} hair")
    if not colors and length:
        parts.append(f"{length} hair")
    if "bangs" in on:
        parts.append("bangs")
    for t in sorted(on):
        if t.startswith("eyes_"):
            parts.append(t[5:] + " eyes")
    for t in sorted(on):
        if t.startswith("clothes_"):
            parts.append(t[8:] + " clothes")
    for t, w in (("glasses", "glasses"), ("hair_ornament", "hair ornament"),
                 ("blush", "blush"), ("smile", "smile"), ("open_mouth", "open mouth")):
        if t in on:
            parts.append(w)
    parts.append("simple background")
    parts.append(ex["background"][0] + " background")
    return ", ".join(parts), ", ".join(NEGATIVE)
