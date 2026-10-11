"""Tag vocabulary. Default = built-in synthetic vocab; `configure()` swaps in a
dataset-derived vocab (written by prep_dataset.py, stored inside weight files).

  EXCLUSIVE  - exactly one value true (softmax)   e.g. background, rating
  MULTI      - any number true (sigmoid per tag)  e.g. hair colors

Logit layout: [exclusive groups in order..., multi tags...]
Other modules must read T.MULTI / T.GROUPS etc. at call time (not import time).
"""
import json

_DEFAULT_EXCLUSIVE = {
    "subject_count": ["solo", "duo"],
    "background": ["white", "black", "pink", "blue", "green"],
}
_DEFAULT_MULTI = [
    "hair_black", "hair_brown", "hair_blonde", "hair_pink",
    "hair_blue", "hair_red", "hair_white",
    "short_hair", "long_hair",
    "eyes_blue", "eyes_red", "eyes_green", "eyes_brown", "eyes_purple",
    "clothes_black", "clothes_white", "clothes_red", "clothes_blue", "clothes_green",
    "glasses", "hair_ornament", "blush", "bangs", "smile", "open_mouth",
]

EXCLUSIVE, MULTI = {}, []
GROUPS, MULTI_SLICE, N_OUT, GROUP_NAMES, MULTI_INDEX = [], (0, 0), 0, [], {}


def configure(exclusive, multi):
    global EXCLUSIVE, MULTI, GROUPS, MULTI_SLICE, N_OUT, GROUP_NAMES, MULTI_INDEX
    EXCLUSIVE = {k: list(v) for k, v in exclusive.items()}
    MULTI = list(multi)
    GROUPS, pos = [], 0
    for vals in EXCLUSIVE.values():
        GROUPS.append((pos, pos + len(vals)))
        pos += len(vals)
    MULTI_SLICE = (pos, pos + len(MULTI))
    N_OUT = pos + len(MULTI)
    GROUP_NAMES = list(EXCLUSIVE)
    MULTI_INDEX = {t: i for i, t in enumerate(MULTI)}


def reset_default():
    configure(_DEFAULT_EXCLUSIVE, _DEFAULT_MULTI)


def vocab_json():
    return json.dumps({"exclusive": EXCLUSIVE, "multi": MULTI})


def load_vocab_json(s):
    d = json.loads(str(s))
    configure(d["exclusive"], d["multi"])


reset_default()
