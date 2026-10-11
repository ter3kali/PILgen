import text_prompt as TP

CASES = [
    ("a girl with long pink hair standing on a beach at sunset, smiling", "allow"),
    ("knight in a misty forest holding a lantern", "allow"),
    ("two girls reading in a library", "allow"),
    ("girl in lingerie", "review"),
    ("nsfw nude woman", "flag"),
    ("15 year old girl, topless", "block"),
]
bad = 0
for text, want in CASES:
    r = TP.run(text)
    ok = r["action"] == want and (want in ("allow", "review") or r["prompt"].startswith("[withheld"))
    bad += not ok
    print("ok  " if ok else "FAIL", f"{want:6s} got {r['action']:6s} <- {text}")
r = TP.run("a girl with long pink hair standing on a beach at sunset, smiling")
print("\n" + r["prompt"])
raise SystemExit(bad)
