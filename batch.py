"""Batch: every image in a folder -> one prompt file per image in another folder.

    python batch.py --images my_images --out prompts
    python ipai.py batch --images my_images --out prompts --recursive --guess

    prompts/cat.txt          <- prompt for my_images/cat.jpg   (one file per image, same name)
    prompts/_report.csv      <- image, status, action, prompt_file (what was written / skipped / why)

Options
  --recursive        also read sub-folders (sub-folder layout is mirrored in --out)
  --guess            basic engine: add the hair-colour guess
  --wd14 MODEL TAGS  use an ONNX tagger instead (optional, needs onnxruntime)
  --vit WEIGHTS      use your own trained NumPy ViT
  --negative         also write the negative prompt as a second line ("Negative prompt: ...")
  --format json      write .json (prompt, negative_prompt, tags, moderation) instead of .txt
  --overwrite        redo images that already have a prompt file (default: skip them, so you can resume)
Flagged/blocked images get NO prompt file (listed in _report.csv). Unreadable files are reported, not fatal.
"""
import argparse
import csv
import json
import os
import sys
import time

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")


def find_images(root, recursive):
    out = []
    if recursive:
        for d, dirs, files in os.walk(root):
            dirs.sort()
            out += [os.path.join(d, f) for f in sorted(files) if f.lower().endswith(EXTS)]
    else:
        out = [os.path.join(root, f) for f in sorted(os.listdir(root))
               if f.lower().endswith(EXTS) and os.path.isfile(os.path.join(root, f))]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder with your images")
    ap.add_argument("--out", required=True, help="folder to write the prompt files into (created if missing)")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--guess", action="store_true")
    ap.add_argument("--wd14", nargs=2, metavar=("MODEL_ONNX", "TAGS_CSV"))
    ap.add_argument("--vit", metavar="WEIGHTS_NPZ")
    ap.add_argument("--negative", action="store_true")
    ap.add_argument("--format", choices=["txt", "json"], default="txt")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.images):
        sys.exit(f"images folder not found: {a.images}")
    in_abs, out_abs = os.path.abspath(a.images), os.path.abspath(a.out)
    if out_abs == in_abs:
        sys.exit("--out must be a different folder from --images")
    os.makedirs(a.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import image_prompt as IP

    files = [f for f in find_images(a.images, a.recursive)
             if not os.path.abspath(f).startswith(out_abs + os.sep)]   # never re-read our own output
    if not files:
        sys.exit(f"no images found in {a.images}" + ("" if a.recursive else " (use --recursive for sub-folders)"))

    ext = "." + a.format
    rows, counts, t0 = [], {"written": 0, "skipped": 0, "withheld": 0, "error": 0}, time.time()
    for n, path in enumerate(files, 1):
        rel = os.path.relpath(path, a.images)
        target = os.path.join(a.out, os.path.splitext(rel)[0] + ext)
        label = f"[{n}/{len(files)}] {rel}"
        if os.path.exists(target) and not a.overwrite:
            counts["skipped"] += 1
            rows.append((rel, "skipped", "", target))
            print(label, "-> exists, skipped")
            continue
        try:
            r = IP.run(path, a.wd14, a.vit, withhold=True, guess=a.guess)
        except Exception as e:                      # corrupt/unsupported file: keep going
            counts["error"] += 1
            rows.append((rel, "error", str(e)[:120], ""))
            print(label, "-> ERROR:", e)
            continue
        if r["action"] in ("flag", "block"):
            counts["withheld"] += 1
            rows.append((rel, "withheld", r["action"], ""))
            print(label, f"-> withheld ({r['action']})")
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            if a.format == "json":
                json.dump({k: r.get(k) for k in ("prompt", "negative_prompt", "tags", "action", "engine",
                                                 "image_moderation", "prompt_moderation")}, f, indent=2)
            else:
                f.write(r["prompt"] + "\n")
                if a.negative:
                    f.write("Negative prompt: " + r["negative_prompt"] + "\n")
        counts["written"] += 1
        rows.append((rel, "written", r["action"], target))
        print(label, f"-> {os.path.relpath(target, a.out)}" + ("  (review)" if r["action"] == "review" else ""))

    # report for THIS run (skipped = prompt file already existed from an earlier run)
    rp = os.path.join(a.out, "_report.csv")
    with open(rp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "status", "action_or_reason", "prompt_file"])
        w.writerows(rows)
    print(f"\ndone in {time.time() - t0:.1f}s: {counts}\nreport: {rp}")
    return 1 if counts["error"] and not counts["written"] else 0


if __name__ == "__main__":
    sys.exit(main())
