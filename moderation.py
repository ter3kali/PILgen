"""Detection / flagging layer. It never generates explicit content.

  check_text(prompt)   -> flags prompts that ask for sexual / nude / suggestive content
  check_image(probs)   -> flags images using the model's `rating` head
  combine(a, b)        -> worst action wins

Actions (low -> high): allow < review < flag < block
  block  : any sexual/nude/suggestive request that also involves a minor or an
           age-ambiguous term (hard rule, no override)
  flag   : sexual_explicit or nudity
  review : suggestive only, or age-ambiguous + suggestive

The lexicon (flag_lexicon.json) is a starter list - audit it for your use case.
"""
import json
import os
import re
import unicodedata

import numpy as np

ORDER = {"allow": 0, "review": 1, "flag": 2, "block": 3}
_HERE = os.path.dirname(os.path.abspath(__file__))
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})
_ZW = re.compile("[\u200b-\u200f\u2060\ufeff]")
_AGE = re.compile(r"(?<!\d)(\d{1,2})\s*[- ]?\s*(?:y\.?o\.?|yo|yrs?|years?)(?:[- ]?\s*olds?)?(?![a-z])"
                  r"|\bage[ds]?\s*:?\s*(\d{1,2})\b")
_cache = {}


def load_lexicon(path=None):
    path = path or os.path.join(_HERE, "flag_lexicon.json")
    if path not in _cache:
        with open(path, encoding="utf-8") as f:
            lex = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        _cache[path] = lex
    return _cache[path]


def _norm(text):
    t = unicodedata.normalize("NFKC", str(text))
    t = _ZW.sub("", t).lower()
    return re.sub(r"[_\-.,;:/\\|()\[\]{}\"'*+=~^<>?]+", " ", t)


_SPACED = re.compile(r"\b(?:[a-z]\s){2,}[a-z]\b")


def _variants(text):
    base = _norm(text)
    deleet = _norm(str(text).translate(_LEET))
    out = [re.sub(r"\s+", " ", v).strip() for v in (base, deleet)]
    out += [_SPACED.sub(lambda m: m.group(0).replace(" ", ""), v) for v in out]  # "n u d e"
    return out


def _term_re(term):
    t = _norm(term).strip()
    if t.endswith(" "):
        t = t.strip()
    star = term.endswith("*")
    body = re.escape(t.rstrip()).replace(r"\ ", r"\s+")
    return re.compile(r"\b" + body + (r"\w*" if star else r"\b"))


def _compile(lex):
    key = id(lex)
    if ("re", key) not in _cache:
        _cache[("re", key)] = {c: [(t, _term_re(t)) for t in terms] for c, terms in lex.items()}
    return _cache[("re", key)]


def load_allowlist(path=None):
    """SFW library: benign phrases masked out before sexual/nudity/suggestive matching."""
    path = path or os.path.join(_HERE, "sfw_allowlist.json")
    if ("allow", path) not in _cache:
        try:
            with open(path, encoding="utf-8") as f:
                phrases = json.load(f).get("phrases", [])
        except FileNotFoundError:
            phrases = []
        _cache[("allow", path)] = [_term_re(p) for p in phrases]
    return _cache[("allow", path)]


def _mask(variants, allow):
    out = []
    for v in variants:
        for rx in allow:
            v = rx.sub(" ", v)
        out.append(re.sub(r"\s+", " ", v).strip())
    return out


_NEVER_MASK = {"minor", "age_ambiguous"}  # SFW phrases can never hide age signals


def check_text(text, lexicon=None, allowlist=True):
    lex = lexicon or load_lexicon()
    comp = _compile(lex)
    variants = _variants(text)
    masked = _mask(variants, load_allowlist()) if allowlist else variants
    squashed_raw = [v.replace(" ", "") for v in variants]  # catches "n u d e"
    squashed_msk = [v.replace(" ", "") for v in masked]
    cats = {}
    for cat, items in comp.items():
        vs, sq = (variants, squashed_raw) if cat in _NEVER_MASK else (masked, squashed_msk)
        for term, rx in items:
            hit = any(rx.search(v) for v in vs)
            plain = re.sub(r"[^a-z]", "", term.lower())
            if not hit and len(plain) >= 6 and not term.endswith("*"):
                hit = any(plain in s for s in sq)
            if hit:
                cats.setdefault(cat, []).append(term)
    for v in variants:
        for m in _AGE.finditer(v):
            n = int(m.group(1) or m.group(2))
            if n < 18:
                cats.setdefault("minor", []).append(f"age {n}")
    sexual = set(cats) & {"sexual_explicit", "nudity", "suggestive", "nonconsensual"}
    hard = set(cats) & {"sexual_explicit", "nudity", "nonconsensual"}
    if ("minor" in cats and sexual) or ("age_ambiguous" in cats and hard):
        action = "block"
    elif hard:
        action = "flag"
    elif "suggestive" in cats:
        action = "review"
    else:
        action = "allow"
    return {"action": action, "categories": sorted(cats), "matches": cats}


def tag_is_flagged(tag, lexicon=None):
    """True if a dataset tag is in a sexual/nudity/suggestive/minor/age-ambiguous category."""
    r = check_text(tag, lexicon)
    return bool(r["categories"])


def rating_of(tags_text_or_list):
    return check_text(" ".join(tags_text_or_list) if not isinstance(tags_text_or_list, str)
                      else tags_text_or_list)


def check_image(rating_probs, names=("general", "sensitive", "questionable", "explicit"),
                flag_threshold=0.5, review_threshold=0.5):
    p = dict(zip(names, [float(x) for x in rating_probs]))
    nsfw = p.get("questionable", 0) + p.get("explicit", 0)
    sens = p.get("sensitive", 0)
    if nsfw >= flag_threshold:
        action = "flag"
    elif sens + nsfw >= review_threshold:
        action = "review"
    else:
        action = "allow"
    return {"action": action, "nsfw_probability": round(nsfw, 3),
            "sensitive_probability": round(sens, 3), "checked": True}


def combine(*results):
    worst = max((r["action"] for r in results if r), key=lambda a: ORDER[a], default="allow")
    return worst


if __name__ == "__main__":
    import sys
    print(json.dumps(check_text(" ".join(sys.argv[1:])), indent=2))
