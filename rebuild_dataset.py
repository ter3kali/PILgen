"""
Rebuild a dataset at high quality from your ORIGINAL images, keeping ids and captions.

Why: a program fitted at 64px or 128px cannot hold fine detail, and upscaling the small stored
images does not bring detail back. Re-fitting from the originals does.

Usage
    python rebuild_dataset.py raw_images data data_sharp --quality sharp --workers 4
    python rebuild_dataset.py raw_images data data_sharp --quality sharp --time-limit 7200
    python rebuild_dataset.py raw_images data data_max   --quality max --crop top

    raw_images : folder with your original images (the same ones you built `data` from)
    data       : the old dataset (its captions and index.csv are reused)
    data_sharp : the new dataset (can be used directly: retrieval.py data_sharp "prompt")

Quality presets (one image on one CPU core, rough guide; the script prints a live ETA):
    fast  : 128px,  ~800 shapes    tens of seconds
    sharp : 256px,  ~8500 shapes   a few minutes     <- good default
    max   : 384px,  ~20000 shapes  10+ minutes       <- sharpest
  Or set --size and --shapes (or --density = shapes per pixel) yourself.

Stopping and resuming
    Press Ctrl-C or use --time-limit SECONDS at any time. Unfinished images are saved as
    .part files; run the same command again and it carries on where it stopped.

Square images only: --crop says how to make your image square.
    center  (default) cut equally from both sides
    top     keep the top (good for portraits: faces are usually in the upper part)
    pad     keep everything, add a border
    stretch squash to a square (what build_dataset.py did, so content matches the old dataset)
"""
import argparse
import csv
import random
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from imgprog import edge_similarity, fit, render, sharpness_ratio, similarity

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PRESETS = {                       # size, shapes per pixel, edge weight, tries, climb steps
    "fast": (128, 0.05, 8.0, 30, 60),
    "sharp": (256, 0.13, 12.0, 40, 100),
    "max": (384, 0.14, 12.0, 40, 100),
}


def to_square(img, mode):
    w, h = img.size
    if mode == "stretch" or w == h:
        return img
    if mode == "pad":
        s = max(w, h)
        edge = np.concatenate([np.asarray(img)[0], np.asarray(img)[-1],
                               np.asarray(img)[:, 0], np.asarray(img)[:, -1]])
        canvas = Image.new("RGB", (s, s), tuple(int(x) for x in np.median(edge, axis=0)))
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas
    s = min(w, h)
    if w > h:
        x0 = (w - s) // 2
        return img.crop((x0, 0, x0 + s, s))
    y0 = (h - s) // 2 if mode == "center" else int((h - s) * 0.08)
    return img.crop((0, y0, s, y0 + s))


def find_sources(raw, old):
    """id -> source image path. Uses the old index.csv; falls back to sorted order."""
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in EXTS)
    by_name = {p.name: p for p in files}
    ids = sorted(p.stem for p in (old / "programs").glob("*.ipg"))
    mapping, idx_file = {}, old / "index.csv"
    if idx_file.exists():
        with open(idx_file, newline="") as f:
            for row in csv.DictReader(f):
                if row["source"] in by_name:
                    mapping[row["id"]] = by_name[row["source"]]
    else:
        print("note: no index.csv in the old dataset; matching ids to images by sorted order")
        mapping = {f"{i:04d}": p for i, p in enumerate(files, 1)}
    return [(i, mapping[i]) for i in ids if i in mapping], [i for i in ids if i not in mapping]


def work(job):
    name, src, out, size, shapes, edge, cands, climb, crop, deadline = job
    prog_path = out / "programs" / f"{name}.ipg"
    part = out / "programs" / f"{name}.part"
    if prog_path.exists():
        return None
    if deadline and time.time() >= deadline:
        return ("skipped", name)

    target = to_square(Image.open(src).convert("RGB"), crop).resize((size, size), Image.LANCZOS)
    target.save(out / "images" / f"{name}.png")
    resume = part.read_text() if part.exists() else None
    random.seed(int(name) * 7919 + (len(resume.splitlines()) if resume else 0))

    status = {}
    program = fit(target, n_shapes=shapes, cands=cands, climb=climb, size=size, verbose=False,
                  alpha=True, edge=edge, strokes=True, resume=resume,
                  time_limit=(deadline - time.time()) if deadline else None,
                  checkpoint=lambda text: part.write_text(text), status=status)
    if not status.get("done", True):
        part.write_text(program)
        return ("partial", name)
    prog_path.write_text(program)
    if part.exists():
        part.unlink()
    img = render(program)
    return ("done", name, src.name, size, len(program.splitlines()) - 2,
            similarity(img, target), edge_similarity(img, target), sharpness_ratio(img, target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--quality", choices=list(PRESETS), default="sharp")
    ap.add_argument("--size", type=int)
    ap.add_argument("--shapes", type=int)
    ap.add_argument("--density", type=float, help="shapes per pixel, e.g. 0.13")
    ap.add_argument("--edge", type=float)
    ap.add_argument("--crop", choices=["center", "top", "pad", "stretch"], default="center")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=0, help="total seconds, then stop and save")
    a = ap.parse_args()

    size0, dens0, edge0, cands, climb = PRESETS[a.quality]
    size = a.size or size0
    shapes = a.shapes or int((a.density or dens0) * size * size)
    edge = a.edge if a.edge is not None else edge0

    jobs_src, missing = find_sources(a.raw, a.old)
    if missing:
        print(f"warning: no source image found for ids {', '.join(missing[:10])}"
              f"{' ...' if len(missing) > 10 else ''} (they are skipped)")
    if not jobs_src:
        raise SystemExit("Nothing to do: no old programs matched images in the raw folder.")

    for sub in ("images", "programs", "captions"):
        (a.new / sub).mkdir(parents=True, exist_ok=True)
    for i, _ in jobs_src:                                      # keep captions
        c = a.old / "captions" / f"{i}.txt"
        if c.exists() and not (a.new / "captions" / f"{i}.txt").exists():
            shutil.copy(c, a.new / "captions" / f"{i}.txt")

    todo = [(i, s) for i, s in jobs_src if not (a.new / "programs" / f"{i}.ipg").exists()]
    print(f"{len(jobs_src)} images, {len(todo)} to do at {size}px, {shapes} shapes each, "
          f"{a.workers} workers (ETA appears after the first image finishes)")
    if not todo:
        print("all done already")
        return

    deadline = time.time() + a.time_limit if a.time_limit else None
    jobs = [(i, s, a.new, size, shapes, edge, cands, climb, a.crop, deadline) for i, s in todo]

    idx_path = a.new / "index.csv"
    new_file = not idx_path.exists()
    done = partial = 0
    t_start = time.time()
    try:
        with open(idx_path, "a", newline="") as f, Pool(a.workers) as pool:
            w = csv.writer(f)
            if new_file:
                w.writerow(["id", "source", "size", "shapes", "similarity", "edge_match", "sharpness"])
            for r in pool.imap_unordered(work, jobs):
                if r is None:
                    continue
                if r[0] == "done":
                    _, name, src, sz, n, sim, em, sh = r
                    w.writerow([name, src, sz, n, f"{sim:.4f}", f"{em:.4f}", f"{sh:.3f}"])
                    f.flush()
                    done += 1
                    left = (len(todo) - done - partial) * (time.time() - t_start) / done / max(1, a.workers)
                    print(f"{name} {src}: {n} shapes, edge match {em:.3f}, sharpness {sh:.2f}x original"
                          f"   [{done}/{len(todo)} done, about {left / 60:.0f} min left]", flush=True)
                elif r[0] == "partial":
                    partial += 1
                    print(f"{r[1]}: stopped by the time limit (saved, will resume)", flush=True)
    except KeyboardInterrupt:
        print("\nstopped. Unfinished images were saved as .part files; run the same command to resume.")
        return
    print(f"finished {done} images; {partial} partial; {len(todo) - done - partial} not started."
          f"{' Run the same command again to continue.' if partial or len(todo) > done + partial else ''}")


if __name__ == "__main__":
    main()
