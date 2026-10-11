"""Bridge: caption your shape-program dataset automatically with the image->prompt engine.

Your pipeline (imgprog.py / rebuild_dataset.py / nn_numpy.py) learns  caption -> shape program,
and needs data/captions/<id>.txt. Instead of writing those by hand, this fills them in:

    python bridge_captions.py --raw raw_images --data data_sharp
    python bridge_captions.py --raw raw_images --data data_sharp --vit my_weights.npz --guess

For every id in data/programs/*.ipg it finds the ORIGINAL image (via data/index.csv "source", the
same way rebuild_dataset.py does; falls back to sorted order), runs the prompt generator on it, and
writes the tags as the caption:   data/captions/0007.txt  ->  "black hair, purple hair, red background, ..."

* Existing captions are kept (your manual ones win). --overwrite replaces them.
* Moderation applies to the training data: flagged/blocked images get NO caption, so nn_numpy
  (which skips uncaptioned samples) never trains on them. Listed in data/captions_report.csv.
* Quality words ("masterpiece, best quality, anime style") are left out by default - they would be the
  same in every caption and teach the network nothing. --keep-style puts them back.
* The caption words are exactly what you should type later:
      python nn_numpy.py gen model_np.npz "black hair, red background" --n 4
  (nn_numpy only understands words that occur in the training captions.)
* For REAL content words (hair, eyes, clothes...) train the tagger first and pass --vit (see README).
  Without it the captions describe colours/lighting/framing only.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff")


def all_images(raw):
    out = []
    for d, dirs, files in os.walk(raw):
        dirs.sort()
        out += [os.path.join(d, f) for f in sorted(files) if f.lower().endswith(EXTS)]
    return out


def find_sources(raw, data):
    """id -> image path (same rules as rebuild_dataset.find_sources)."""
    files = all_images(raw)
    by_name = {os.path.basename(p): p for p in files}
    prog_dir = os.path.join(data, "programs")
    if not os.path.isdir(prog_dir):
        sys.exit(f"{prog_dir} not found - is {data} a dataset folder (programs/, captions/, images/)?")
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(prog_dir) if f.endswith(".ipg"))
    mapping, idx = {}, os.path.join(data, "index.csv")
    if os.path.exists(idx):
        with open(idx, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("source") in by_name:
                    mapping[row["id"]] = by_name[row["source"]]
    else:
        print("note: no index.csv; matching ids to images by sorted order")
        mapping = {f"{i:04d}": p for i, p in enumerate(files, 1)}
    return [(i, mapping[i]) for i in ids if i in mapping], [i for i in ids if i not in mapping]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder with your original images")
    ap.add_argument("--data", required=True, help="dataset folder (programs/, captions/, index.csv)")
    ap.add_argument("--vit", metavar="WEIGHTS_NPZ", help="use your trained tagger for real content tags")
    ap.add_argument("--wd14", nargs=2, metavar=("MODEL_ONNX", "TAGS_CSV"))
    ap.add_argument("--guess", action="store_true", help="basic engine: add hair-colour guess")
    ap.add_argument("--keep-style", action="store_true", help="keep 'masterpiece, best quality, anime style'")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.raw):
        sys.exit(f"raw folder not found: {a.raw}")
    jobs, missing = find_sources(a.raw, a.data)
    if missing:
        print(f"warning: no source image for ids {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''} (skipped)")
    if not jobs:
        sys.exit("nothing to do: no program ids matched an image in the raw folder")

    sys.path.insert(0, HERE)
    import image_prompt as IP
    import prompt_builder as PB
    cap_dir = os.path.join(a.data, "captions")
    os.makedirs(cap_dir, exist_ok=True)
    rows, n = [], {"written": 0, "kept": 0, "withheld": 0, "error": 0}
    for k, (i, path) in enumerate(jobs, 1):
        target = os.path.join(cap_dir, f"{i}.txt")
        if os.path.exists(target) and not a.overwrite:
            n["kept"] += 1
            rows.append((i, os.path.basename(path), "kept", ""))
            continue
        try:
            r = IP.run(path, a.wd14, a.vit, withhold=True, guess=a.guess)
        except Exception as e:
            n["error"] += 1
            rows.append((i, os.path.basename(path), "error", str(e)[:100]))
            print(f"[{k}/{len(jobs)}] {i}: ERROR {e}")
            continue
        if r["action"] in ("flag", "block"):
            n["withheld"] += 1
            rows.append((i, os.path.basename(path), "withheld", r["action"]))
            print(f"[{k}/{len(jobs)}] {i}: withheld ({r['action']}) - no caption, excluded from training")
            continue
        tags = list(r.get("tags") or [])
        if a.keep_style:
            tags = list(PB.QUALITY) + tags
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(", ".join(tags) + "\n")
        n["written"] += 1
        rows.append((i, os.path.basename(path), "written", r["action"]))
        if k % 50 == 0 or k == len(jobs):
            print(f"[{k}/{len(jobs)}] {i} -> {', '.join(tags)[:80]}")
    rp = os.path.join(a.data, "captions_report.csv")
    with open(rp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "status", "detail"])
        w.writerows(rows)
    print(f"\ndone: {n}\nreport: {rp}\nnext: python nn_numpy.py train {a.data} --slots 40 --hidden 256 --epochs 300")
    return 0 if n["written"] or n["kept"] else 1


if __name__ == "__main__":
    sys.exit(main())
