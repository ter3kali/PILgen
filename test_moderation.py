"""Sanity tests for the text flagger. Extend with cases from your own moderation logs."""
import moderation as M

CASES = [
    ("1girl, solo, black hair, white background, kimono", "allow"),
    ("kids playing in a park", "allow"),
    ("a 25 yo woman, smiling", "allow"),
    ("Essex county map", "allow"),
    ("a girl in lingerie", "review"),
    ("nsfw nude woman", "flag"),
    ("n u d e portrait", "flag"),
    ("n4k3d girl", "flag"),
    ("15 year old girl, topless", "block"),
    ("loli, sexy", "block"),
    ("schoolgirl, nude", "block"),
    ("girl in bikini on the beach", "allow"),
    ("chicken breast recipe", "allow"),
    ("graduated cum laude from Essex", "allow"),
    ("peacock in a cockpit with a cocktail", "allow"),
    ("exposed brick wall, comic strips", "allow"),
    ("breast cancer awareness poster", "allow"),
    ("sex education class", "allow"),
    ("huge breasts, cleavage", "review"),
    ("n s f w uncensored", "flag"),
    ("rape scene", "flag"),
    ("cum laude, 14 year old, nude", "block"),
    ("teen girl, sexy", "block"),
    ("petite student, topless", "block"),
]
bad = 0
for text, want in CASES:
    got = M.check_text(text)["action"]
    ok = got == want
    bad += not ok
    print(("ok  " if ok else "FAIL"), f"{want:6s} got {got:6s} <- {text}")
raise SystemExit(bad)
