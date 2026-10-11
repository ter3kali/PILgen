# shape_prompt_ai

An offline toolkit, in plain Python + NumPy + Pillow (no PyTorch, no cloud), that does two connected things:

1. **Image to prompt.** It describes an image as text tags and moderates the result.
2. **Text to picture.** A small neural network learns *caption to shape program*, so you can type words and get a picture made of circles, rectangles, triangles, lines and brush strokes.

The bridge between the two: the prompt engine writes the captions that the shape network trains on.

```
original images
   |  rebuild_dataset.py (uses imgprog.py)         fit each image as a shape program
   v
data_sharp/programs/*.ipg  + images/ + index.csv
   |  bridge_captions.py (uses the prompt engine)  caption each image, skipping flagged ones
   v
data_sharp/captions/*.txt
   |  nn_binned.py train   (or nn_numpy.py train)  learn caption -> program
   v
model_bin.npz
   |  nn_binned.py gen "black hair, white background"
   v
genb_1.png, genb_1.ipg, ...
```

The folder `examples/` holds a sample result: `compare.png` (original on the left, 8,480-shape fit on the right) and its program `portrait_sharp.ipg`.

---

## 1. Install

Requires **Python 3.8+**.

```
pip install -r requirements.txt        # numpy, Pillow
python ipai.py doctor                  # checks Python, numpy, Pillow, platform
python ipai.py test                    # self-tests, should print 3x PASS
```

All scripts live together in this one folder and import each other by name, so **run every command from inside it** and keep the file names as they are.

| Platform | Notes |
|---|---|
| Windows | use `py` instead of `python` |
| macOS / Linux | `python3` |
| Termux (Android) | `pkg install python python-numpy libjpeg-turbo`, then `pip install Pillow` |

---

## 2. The shape-program format (`.ipg`)

A picture is a text file, one command per line, drawn in file order:

```
size 256
bg 245 245 245
circle X Y R              R G B [A]
rect X1 Y1 X2 Y2          R G B [A]
tri X1 Y1 X2 Y2 X3 Y3     R G B [A]
line X1 Y1 X2 Y2 WIDTH    R G B [A]
stroke X1 Y1 X2 Y2 X3 Y3 W1 W2   R G B [A]    curved brush stroke, tapers W1 to W2
skip                                           an empty slot
```

`A` is opacity 0 to 255 (left out means solid). Programs are rendered smoothly at any scale, so a 256 px fit can be drawn at 1024 px.

---

## 3. Step-by-step workflow

### Step 0: collect images
Put your original images in `raw_images/` (a sample portrait is already there). Use **unique file names**: images are matched to ids by file name. Keep the folder flat (no sub-folders) for the simple bootstrap below.

### Step 1: fit shape programs (`rebuild_dataset.py`)
`rebuild_dataset.py` re-fits programs from the originals and reuses ids and captions from an "old" dataset. On a first run there is no old dataset, so create a stub one that just reserves one id per image. Paste this into a Python prompt (or save it as a file and run it) once:

```python
import os
EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
n = len([f for f in os.listdir('raw_images') if f.lower().endswith(EXTS)])
os.makedirs('data_init/programs', exist_ok=True)
for i in range(1, n + 1):
    open(f'data_init/programs/{i:04d}.ipg', 'w').write('size 64\nbg 0 0 0\n')
```

Then fit:

```
python rebuild_dataset.py raw_images data_init data_sharp --quality sharp --crop top --workers 4
```

| Preset | Size | Shapes | Time per image (one core) |
|---|---|---|---|
| `fast` | 128 px | ~800 | tens of seconds |
| `sharp` (default) | 256 px | ~8,500 | a few minutes |
| `max` | 384 px | ~20,000 | 10+ minutes |

Or set `--size`, `--shapes`, or `--density` (shapes per pixel) yourself.

* **`--crop`** makes non-square images square: `center`, `top` (good for portraits), `pad`, or `stretch`.
* **Stopping and resuming:** press Ctrl-C or use `--time-limit SECONDS`. Unfinished images are saved as `.part` files; run the same command again to continue.
* **Output:** `data_sharp/images/`, `programs/`, `captions/` and `index.csv` (id, source file, size, shapes, similarity, edge match, sharpness).

To fit a single image directly:
```
python imgprog.py picture.png --shapes 8500 --size 256 --alpha --edge 12 --strokes --out result
python imgprog.py picture.png --shapes 12000 --size 512 --alpha --edge 4 --strokes --time-limit 250 --out big
python imgprog.py picture.png --shapes 12000 --size 512 --alpha --edge 4 --strokes --time-limit 250 --out big --resume big.ipg
```
How the fitter works: it adds shapes one at a time, big to small. New shapes go where the picture is most wrong (`--guided`, on by default). A shape is kept only if it lowers the error. `--alpha` allows semi-transparent shapes, `--edge N` weights edges up to (1+N) times so outlines count, and `--strokes` adds curved strokes lined up with the image's edges (good for hair).

How to read the scores: **similarity** is dominated by flat areas and is always high. **Edge match** and **sharpness** (relative to the original) tell you how much fine detail survived.

### Step 2: caption the images (`bridge_captions.py`)
```
python bridge_captions.py --raw raw_images --data data_sharp --guess
python bridge_captions.py --raw raw_images --data data_sharp --vit my_weights.npz
```
For every program id it finds the original image, runs the prompt engine, and writes tags to `data_sharp/captions/<id>.txt`, for example `black hair, dark gray hair, portrait orientation, flat colors, strong linework, ...`.

* Existing captions are kept (your manual ones win); `--overwrite` replaces them.
* **Moderation applies to training data.** Flagged or blocked images get no caption, so the networks never train on them. See `captions_report.csv`.
* Quality words ("masterpiece, best quality, anime style") are left out because they would be identical in every caption. `--keep-style` restores them.
* Without `--vit`, the engine only describes **colour, lighting and framing**. For real content tags (hair, eyes, clothes, accessories) train the tagger first (Step 5).

You can also write or edit captions by hand: they are just text files named after the id.

### Step 3: train the network
Two options, same dataset folder.

**`nn_binned.py` (recommended, sharp output).** Every number (x, y, radius, colour, opacity) is predicted as a probability over 16 bins and then *sampled*, so each output is one crisp program instead of a blurry average.
```
python nn_binned.py train data_sharp --slots 40 --hidden 128 --epochs 200
```

**`nn_numpy.py` (baseline, regression).** Predicts numbers with squared error, so ambiguous captions come out as a soft average. Useful as a comparison.
```
python nn_numpy.py train data_sharp --slots 40 --hidden 256 --epochs 300
```

Both read captions as bag-of-words plus word pairs, use one hidden ReLU layer, and have backprop and Adam written by hand in NumPy. Each program is reduced to its first K shapes (`--slots`), which are the big ones. With `--order spatial` (default) the shapes are sorted top-to-bottom, left-to-right, so slot 7 always means the same place. Useful options: `--batch`, `--lr`, `--word-drop`, `--seed`, `--out`. Models from older versions cannot be loaded; retrain.

### Step 4: generate
```
python nn_binned.py gen model_bin.npz "black hair, white background" --n 4 --temp 0.7
python nn_numpy.py  gen model_np.npz  "black hair, white background" --n 4 --temp 0.5
```
* Writes `genb_N.ipg/png` (binned) or `gen_N.ipg/png` (numpy) in the current folder.
* `--temp 0` gives the most likely choice every time (sharp but always the same). `--temp 1` follows the learned probabilities; higher is wilder.
* **Only words that occur in your training captions are understood.** If none of your words are known, the script tells you.

### Step 5 (optional): train the tagger for real content tags
This is the image-to-prompt engine's own learning loop. Details are in `README_ipai.md`; the short version:
```
python ipai.py label   --images images --tags tags --open              # tap-to-label web tool
python ipai.py retrain --images images --tags tags --out my_weights.npz
python ipai.py label   --images images --tags tags --weights my_weights.npz --open   # faster, model pre-fills
```
Repeat as the label count grows (about 1,000 to 3,000 labelled images is usable; a few hundred shows weak learning). Then pass `--vit my_weights.npz` to Step 2 and re-caption with `--overwrite`.

---

## 4. The prompt engine on its own

```
python ipai.py image photo.jpg --guess                  # image -> prompt (JSON)
python ipai.py text "a girl with long pink hair on a beach" --style anime
python ipai.py batch --images my_images --out prompts   # a folder of images -> prompt files
python ipai.py bridge --raw raw_images --data data_sharp   # same as bridge_captions.py
```
Engines: **basic** (default; palette, lighting, framing, optional hair-colour guess; cannot recognise objects), **vit** (your trained tagger; content tags and a moderation rating), **wd14** (optional ONNX tagger, needs `pip install onnxruntime`, untested here). Full command reference, moderation rules and the labelling guide are in `README_ipai.md`.

## 5. Moderation in one paragraph

Every prompt and every image passes a flagger with four actions, `allow < review < flag < block`, and the worst result wins. Sexual or suggestive terms combined with a minor or age-ambiguous term are **always blocked**, with no override. Flagged and blocked images are never captioned and never reach the shape networks. The word lists (`flag_lexicon.json`, `sfw_allowlist.json`) are a starter set; audit them for your own use. Image moderation needs a trained rating head and should be treated as triage, with a human reviewing results.

---

## 6. File map

| File | Purpose |
|---|---|
| `imgprog.py` | shape fitter and renderer (`.ipg` format, scoring, strokes, alpha) |
| `rebuild_dataset.py` | re-fit a dataset at high quality from the original images (parallel, resumable) |
| `bridge_captions.py` | auto-caption the dataset with the prompt engine |
| `nn_numpy.py` | caption to program network (regression baseline) |
| `nn_binned.py` | caption to program network (binned, sharp); needs `nn_numpy.py` and `imgprog.py` |
| `ipai.py` | launcher for the prompt engine (`image`, `text`, `batch`, `label`, `retrain`, `bridge`, `prep`, `train`, `predict`, `check`, `doctor`, `test`) |
| `image_prompt.py`, `text_prompt.py`, `prompt_builder.py`, `batch.py` | image to prompt, text to prompt, prompt assembly, folder runs |
| `label_tool.py`, `label_vocab.json` | tap-to-label web tool and its tag buttons |
| `prep_dataset.py`, `train.py`, `retrain.py`, `predict.py`, `vit.py`, `tags.py`, `imgio.py` | tagger dataset prep, training, prediction, NumPy vision transformer, tag vocabulary, safe image loading |
| `moderation.py`, `flag_lexicon.json`, `sfw_allowlist.json` | moderation engine and word libraries |
| `synth.py`, `make_demo_dataset.py` | synthetic data for plumbing tests only |
| `gradcheck.py`, `test_moderation.py`, `test_text_prompt.py` | self-tests (`python ipai.py test`) |
| `raw_images/` | your original images (one sample portrait included) |
| `examples/` | sample fit: `compare.png`, `portrait_sharp.ipg` |
| `README_ipai.md` | full reference for the prompt engine (v22) |

Folders created while you work: `data_init/` (id stubs), `data_sharp/` (the dataset), plus `model_*.npz`, `gen*_N.png/ipg`, `images/` and `tags/` for labelling.

---

## 7. Realistic expectations

* **Dataset size matters most.** Dozens of near-identical images teach the network almost nothing; use hundreds or more with real variety.
* The basic engine's captions are about colour and framing only, so generation can only respond to those words until you add a trained tagger.
* The network predicts only the first K (large) shapes, so output is a rough composition, not a copy of a trained image. Retrieval from the dataset is the sharper alternative.
* Fine detail (hair strands, chains, lace) needs the `sharp` or `max` preset and is still the first thing lost.
* Everything trains on CPU: seconds to minutes for the shape networks, minutes to an hour for the tagger.

## 8. Troubleshooting

* **`ModuleNotFoundError: imgprog` / `nn_numpy`**: you are not running from the project folder, or a file kept a download suffix such as `imgprog-7.py`. Rename it to `imgprog.py`.
* **`MISSING numpy/PIL`**: `pip install -r requirements.txt`, then `python ipai.py doctor`.
* **`Need at least a few captioned samples`**: Step 2 has not run, or most images were withheld; check `data_sharp/captions_report.csv`.
* **`None of your prompt's words were in the training captions`**: use words that appear in `data_sharp/captions/*.txt`.
* **`Nothing to do: no old programs matched images`**: the stub dataset (Step 1) is missing, or the image names in `raw_images/` changed.
* **All images show `review`**: no rating model is loaded; expected until you train one.
* **Fit is slow**: use `--quality fast`, more `--workers`, or `--time-limit` and run the same command again to resume.
* **Windows garbled text**: all files are read and written as UTF-8; use a current Python and the `py` launcher.
