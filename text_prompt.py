"""Text -> prompt, fully offline (stdlib only), with moderation.

    python text_prompt.py "a girl with long pink hair at a beach, smiling" --style anime
    python text_prompt.py "knight in a misty forest" --style cinematic --detail high

Pipeline: moderate input -> split into phrases -> categorise -> reorder -> add style/quality
presets -> moderate output. Flagged/blocked input never produces a prompt (block can't be overridden).
"""
import argparse
import json
import re

import moderation as M

STYLES = {
    "anime": dict(pre=["masterpiece", "best quality", "anime style"],
                  post=["clean lineart", "vibrant colors"],
                  neg=["lowres", "bad anatomy", "bad hands", "extra fingers", "worst quality", "blurry", "jpeg artifacts"]),
    "realistic": dict(pre=["photorealistic", "high detail", "sharp focus"],
                      post=["natural lighting", "85mm lens", "depth of field"],
                      neg=["cartoon", "painting", "lowres", "blurry", "deformed", "bad anatomy", "watermark"]),
    "cinematic": dict(pre=["cinematic still", "dramatic composition"],
                      post=["volumetric lighting", "film grain", "anamorphic lens", "color graded"],
                      neg=["lowres", "blurry", "flat lighting", "oversaturated", "watermark", "text"]),
    "watercolor": dict(pre=["watercolor painting", "soft washes"],
                       post=["paper texture", "gentle gradients", "pastel palette"],
                       neg=["lowres", "harsh outlines", "photo", "blurry", "watermark"]),
    "pixel": dict(pre=["pixel art", "16-bit", "limited palette"],
                  post=["crisp pixels", "dithering"],
                  neg=["blurry", "antialiasing", "photo", "3d", "watermark"]),
    "3d": dict(pre=["3d render", "octane render", "high detail"],
               post=["soft global illumination", "subsurface scattering"],
               neg=["lowres", "flat", "sketch", "blurry", "watermark"]),
    "sketch": dict(pre=["pencil sketch", "loose linework"],
                   post=["cross hatching", "paper texture", "monochrome"],
                   neg=["color", "photo", "blurry", "watermark"]),
}
DETAIL = {"low": [], "medium": ["highly detailed"],
          "high": ["highly detailed", "intricate details", "ultra high resolution"]}

BUCKETS = [  # (category, keyword regex) first match wins; order here = order in prompt
    ("subject", r"\b(girl|boy|woman|man|lady|knight|wizard|witch|warrior|elf|dragon|robot|cat|dog|fox|"
                r"person|character|samurai|ninja|pirate|princess|prince|child-friendly)s?\b"),
    ("hair", r"\bhair|ponytail|braid|bangs|bob cut|twintail|bun\b"),
    ("eyes", r"\beyes?\b"),
    ("expression", r"\b(smil|laugh|blush|angry|sad|crying|happy|serious|calm|surprised|determined|sleepy)\w*"),
    ("clothes", r"\b(dress|kimono|armor|armour|uniform|hoodie|jacket|coat|cloak|robe|skirt|shirt|suit|hat|"
                r"scarf|glasses|boots|gloves|cape|bikini|swimsuit)\b"),
    ("pose", r"\b(standing|sitting|running|jumping|walking|looking|holding|flying|kneeling|leaning|dancing|"
             r"fighting|reading|sleeping)\b"),
    ("setting", r"\b(beach|forest|city|street|room|castle|mountain|garden|school|temple|ocean|sea|desert|"
                r"night|sky|rain|snow|field|cafe|library|village|space|ruins|background)\b"),
    ("lighting", r"\b(sunset|sunrise|moonlight|sunlight|golden hour|backlit|neon|lantern|glow|fog|mist|misty|shadow)\w*"),
]
_FILLER = re.compile(r"^(a|an|the|of|with|and|in|at|on|who|that|is|are|wearing|has|have)\s+", re.I)
_SPLIT = re.compile(r"\s*(?:,|;|\.|\bwith\b|\band\b|\bwhile\b|\bwho is\b|\bwho\b|\bthat is\b)\s*", re.I)


_VERBS = ("standing|sitting|running|jumping|walking|looking|holding|flying|kneeling|leaning|dancing|"
          "fighting|reading|sleeping|wearing")
_SPLIT2 = re.compile(r"\s+(?=(?:%s|at|on|in|near|by|under|beside|against|under)\b)" % _VERBS, re.I)


def _pieces(text):
    for chunk in _SPLIT.split(text):
        yield from _SPLIT2.split(chunk)


def _clean(p):
    p = p.strip().lower()
    for _ in range(3):
        p = _FILLER.sub("", p)
    return re.sub(r"\s+", " ", p).strip(" -_")


def categorise(phrases):
    out = {name: [] for name, _ in BUCKETS}
    out["other"] = []
    seen = set()
    for p in phrases:
        if not p or p in seen:
            continue
        seen.add(p)
        for name, rx in BUCKETS:
            if re.search(rx, p):
                out[name].append(p)
                break
        else:
            out["other"].append(p)
    return out


def subject_tags(text):
    t = text.lower()
    m = re.search(r"\b(two|2)\s+(girls|boys|women|men)\b", t)
    if m:
        return ["2girls" if m.group(2) in ("girls", "women") else "2boys"]
    if re.search(r"\b(group|crowd|several|many)\b", t):
        return ["multiple characters"]
    if re.search(r"\b(girl|woman|lady)\b", t):
        return ["1girl", "solo"]
    if re.search(r"\b(boy|man)\b", t):
        return ["1boy", "solo"]
    return []


def build(text, style="anime", detail="medium"):
    cfg = STYLES[style]
    phrases = [_clean(p) for p in _pieces(text)]
    cat = categorise([p for p in phrases if p])
    parts = list(cfg["pre"]) + DETAIL[detail] + subject_tags(text)
    for name in ["subject", "hair", "eyes", "expression", "clothes", "pose", "other", "setting", "lighting"]:
        for p in cat[name]:
            if p not in parts and p not in ("girl", "boy", "woman", "man", "lady"):
                parts.append(p)
    parts += cfg["post"]
    seen, final = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            final.append(p)
    return ", ".join(final), ", ".join(cfg["neg"])


def run(text, style="anime", detail="medium", withhold=True):
    inp = M.check_text(text)
    res = dict(action=inp["action"], input_moderation={k: inp[k] for k in ("action", "categories")})
    if inp["action"] in ("flag", "block"):
        res.update(prompt="[withheld: flagged by moderation]", negative_prompt="")
        return res
    prompt, neg = build(text, style, detail)
    out = M.check_text(prompt)
    action = M.combine(inp, out)
    res["action"] = action
    res["prompt_moderation"] = {k: out[k] for k in ("action", "categories")}
    if action in ("flag", "block"):
        res.update(prompt="[withheld: flagged by moderation]", negative_prompt="")
    else:
        res.update(prompt=prompt, negative_prompt=neg)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--style", default="anime", choices=sorted(STYLES))
    ap.add_argument("--detail", default="medium", choices=sorted(DETAIL))
    a = ap.parse_args()
    print(json.dumps(run(a.text, a.style, a.detail), indent=2))
