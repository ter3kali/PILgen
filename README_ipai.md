# image_prompt_ai v22

Offline **image → prompt** and **text → prompt** generator with built-in content moderation.
No cloud, no external AI model. Pure Python + NumPy + Pillow. Runs on Windows, macOS, Linux,
Android (Termux) and iOS (a-Shell / Pythonista).

---

## 1. Install

Requires **Python 3.8+**.

```
pip install -r requirements.txt        # numpy, Pillow
python ipai.py doctor                  # checks Python, numpy, Pillow, platform
python ipai.py test                    # runs all self-tests (should print 3x PASS)
```

| Platform | Notes |
|---|---|
| Windows | use `py ipai.py ...` and `py -m pip install -r requirements.txt` |
| macOS / Linux | `python3 ipai.py ...` |
| Termux (Android) | `pkg install python python-numpy libjpeg-turbo`, then `pip install Pillow` |
| iOS | a-Shell or Pythonista; numpy + Pillow must be available |

Every command below is run through the single launcher `ipai.py` (or call the script directly, e.g. `python batch.py ...`).

---

## 2. Quick start

```
# image -> prompt
python ipai.py image photo.jpg --guess

# text -> prompt
python ipai.py text "a girl with long pink hair on a beach at sunset, smiling" --style anime

# a whole folder of images -> one prompt file per image
python ipai.py batch --images my_images --out prompts
```

---

## 3. What it can and cannot do (read this)

| Engine | What you get | Needs |
|---|---|---|
| **basic** (default) | Describes the *look* of any image: palette, background colour, lighting, orientation, flat-vs-shaded style, framing, optional hair-colour guess. **Cannot recognise** eyes, clothes, accessories, poses, objects. | nothing extra |
| **vit** (`--vit weights.npz`) | Real content tags (hair, eyes, clothes, expression, background...) **and** a rating for moderation. | weights you train yourself (section 7) |
| **wd14** (`--wd14 model.onnx tags.csv`) | Optional: use a WD14-style ONNX tagger. | `pip install onnxruntime` and a model file. Untested here. |

* `v11_weights.npz` (full version only) was trained on **synthetic drawings** and does not work on real art.
* The **hair-colour guess** (`--guess`) looks at the top-centre of the frame. It was right on 4 of 5 test portraits and is unreliable when the head is not at the top.
* Without a rating model, images cannot be assessed and are reported as `review (not assessed)`.
* The moderation word lists are a **starter set**. Audit them against your own use.

---

## 4. Commands

### 4.1 `image` - image to prompt
```
python ipai.py image PIC [--guess] [--vit WEIGHTS.npz] [--wd14 MODEL.onnx TAGS.csv] [--show-flagged]
```
* `--guess` add hair-colour guess (basic engine)
* `--vit` use your trained tagger
* `--wd14` use an ONNX tagger
* `--show-flagged` show the prompt for *flag* results (reviewer use). *block* results are **never** shown.

Output is JSON: `engine`, `action`, `image_moderation`, `prompt_moderation`, `prompt`, `negative_prompt`, `tags`.

### 4.2 `text` - text to prompt
```
python ipai.py text "your idea" [--style STYLE] [--detail LEVEL]
```
* `--style`: `anime` (default), `realistic`, `cinematic`, `watercolor`, `pixel`, `3d`, `sketch`
* `--detail`: `low`, `medium` (default), `high`

It splits your text into phrases, sorts them (subject, hair, eyes, expression, clothes, pose, setting, lighting), then adds style/quality presets and a negative prompt. Input and output are both moderated; flagged text produces no prompt.

### 4.3 `batch` - folder of images to prompt files
```
python ipai.py batch --images IN_FOLDER --out OUT_FOLDER [options]
```
| Option | Meaning |
|---|---|
| `--recursive` | also read sub-folders (structure mirrored in the output folder) |
| `--guess` | hair-colour guess (basic engine) |
| `--vit W.npz` / `--wd14 M T` | use a trained / ONNX tagger |
| `--negative` | add a second line `Negative prompt: ...` |
| `--format json` | write `.json` (prompt, negative, tags, moderation) instead of `.txt` |
| `--overwrite` | redo images that already have a prompt file |

Behaviour:
* `IN/cat.jpg` becomes `OUT/cat.txt` (one file per image).
* Re-running **skips finished images**, so you can stop and resume.
* **Flagged/blocked images get no prompt file.** *Review* images are written.
* Corrupt or unsupported files are reported and skipped; the run continues.
* `OUT/_report.csv` lists every image: `image, status, action_or_reason, prompt_file`.
* `--out` must be a different folder from `--images`.

### 4.4 `label` - tap-to-label tool (build your own training set)
```
python ipai.py label --images images --tags tags [--open] [--lan] [--port 8765] [--weights W.npz] [--stats]
```
Starts a small web page (default `http://127.0.0.1:8765`).
* Tap tags, pick a **rating (required)**, press **Save & Next**. Back / Skip buttons included; it resumes where you left off.
* Type extra tags in the box. Edit `label_vocab.json` to change the tag buttons.
* Suggestions appear with **dashed borders**: hair/background guesses, or the model's guesses when `--weights` is given. The **rating is only hinted, never auto-selected.**
* Images appear in a **fixed random order** so your labelled subset is a fair sample (use `--sorted` for alphabetical). `--stats` also warns if you have too few questionable/explicit examples.
* `--open` opens your browser. `--lan` lets another device on your Wi-Fi open it (only on a trusted network). `--stats` prints progress and tag/rating counts instead.
* Output: `tags/<image-name>.txt` such as `1girl, solo, black_hair, rating:general`.

Rating guide (this trains the moderation head, so label honestly):
`general` nothing suggestive | `sensitive` swimwear, cleavage, mildly suggestive | `questionable` partial nudity / strongly suggestive | `explicit` nudity or sexual content.

### 4.5 `retrain` - labelled folder to trained weights (one command)
```
python ipai.py retrain --images images --tags tags --out my_weights.npz [--epochs 30] [--size 64]
```
Runs `prep` then `train` with settings chosen from how many images you labelled. Refuses to run below 20 labelled images.

### 4.6 `prep`, `train`, `predict` - the manual steps
```
python ipai.py prep --images images --tags tags --size 64 --top-k 300 --min-count 20 --out prepared.npz
python ipai.py train --data prepared.npz --n-val 100 --epochs 30 --out my_weights.npz
python ipai.py predict pic.png --weights my_weights.npz [--fit pad|crop|squash] [--show-flagged]
```
* `prep` options: `--size`, `--top-k`, `--min-count`, `--limit`, `--seed`, `--out`
* `train` options: `--n-train --n-val --epochs --batch --lr --dim --depth --heads --size --patch --seed --data --out --max-steps`
* Tag files may be `.txt` (comma list) or `.json`; the rating can be a `rating:xxx` tag or JSON field.

### 4.7 `check`, `test`, `doctor`
```
python ipai.py check     # gradient check of the model (run after any change to vit.py)
python ipai.py test      # moderation tests + text-prompt tests + gradcheck
python ipai.py doctor    # environment check
python moderation.py "some text"    # see how the text flagger rates a string
```

---

## 5. Training workflow (bootstrap loop)

1. **Label** the first ~100-200 images by hand: `python ipai.py label --images images --tags tags --open`
2. **Train**: `python ipai.py retrain --images images --tags tags --out my_weights.npz`
3. **Label faster**: `python ipai.py label --images images --tags tags --weights my_weights.npz --open` (model pre-fills; you only fix mistakes)
4. Repeat 2-3 as the label count grows, then generate prompts:
   `python ipai.py batch --images my_images --out prompts --vit my_weights.npz`

Realistic expectations:
* A few hundred labelled images shows weak learning; **1,000-3,000** is usable.
* Include examples of **every rating**, or the flagger cannot learn those classes.
* The model is small and sees images at 64x64, so broad tags (hair colour, background, subject count, clothing colour) work best; fine details (earrings, ornaments) will be weak.
* Training runs on CPU: minutes to about an hour, depending on device and data size.
* Training does not resume; re-run it after adding labels. (An interrupted run does resume from its checkpoint; finished runs clean it up.)

---

## 6. Moderation

Every prompt and every image passes through a flagger. It **detects and flags only; it never generates explicit content.**

**Actions** (lowest to highest): `allow` < `review` < `flag` < `block`. The worst result wins.

| Action | Triggered by |
|---|---|
| `block` | any sexual / nudity / suggestive / non-consensual term **together with** a minor or age-ambiguous term, or a stated age under 18. **Hard rule, no override.** |
| `flag` | `sexual_explicit`, `nudity` or `nonconsensual` terms; image rating questionable/explicit above the threshold |
| `review` | `suggestive` terms; image rating sensitive; image not assessed |
| `allow` | everything else |

**Word libraries**
* `flag_lexicon.json` - the **NSFW library**: `sexual_explicit`, `nudity`, `nonconsensual`, `suggestive`, `sensitive` (reported only, e.g. bikini), `minor`, `age_ambiguous`. A trailing `*` means prefix match.
* `sfw_allowlist.json` - the **SFW library**: harmless phrases containing flagged words ("chicken breast", "cum laude", "Essex", "breast cancer"). They are masked before matching. They **never** mask minor or age terms.
* Detection also handles upper/lower case, leetspeak (`n4k3d`), spaced letters (`n u d e`), zero-width characters and Unicode tricks.

**Image moderation** needs a trained rating head. `train.py` tunes a flag threshold for at least 95% recall on questionable + explicit. Treat model output as a **triage signal**, not a final decision, and have a human review the validation set.

**Dataset safety** (`prep_dataset.py`): flagged-category tags are hidden from the prompt vocabulary, and samples with a minor-related tag **and** a questionable/explicit rating are dropped from training.

Extend the libraries from your own logs and re-run `python ipai.py test`. Add cases to `test_moderation.py`.

---

## 7. Output examples

Image (basic engine):
```
masterpiece, best quality, anime style, black hair, purple hair, portrait orientation,
flat colors, illustration, strong linework, balanced lighting, purple toned background,
centered subject, black color accents, gray color accents
```
Text:
```
masterpiece, best quality, anime style, highly detailed, 1girl, solo, long pink hair,
smiling, standing, beach, sunset, clean lineart, vibrant colors
```

---

## 8. File map

| File | Purpose |
|---|---|
| `ipai.py` | the launcher for every command |
| `image_prompt.py` | image to prompt (basic / vit / wd14 engines) |
| `text_prompt.py` | text to prompt with style presets |
| `batch.py` | folder of images to prompt files |
| `label_tool.py`, `label_vocab.json` | tap-to-label web tool and its tag buttons |
| `retrain.py` | labelled folder to trained weights |
| `bridge_captions.py` | auto-captions for a shape-program dataset (imgprog / nn_numpy) |
| `prep_dataset.py`, `train.py`, `predict.py` | dataset prep, training, single-image prediction |
| `vit.py`, `tags.py`, `prompt_builder.py` | NumPy ViT model, tag vocabulary, prompt assembly |
| `moderation.py`, `flag_lexicon.json`, `sfw_allowlist.json` | moderation engine and word libraries |
| `imgio.py` | safe image loading (EXIF rotation, transparent PNGs) |
| `synth.py`, `make_demo_dataset.py` | synthetic data for plumbing tests only |
| `gradcheck.py`, `test_moderation.py`, `test_text_prompt.py` | self-tests |
| `v11_weights.npz` | synthetic demo weights (full version only; not for real art) |
| `requirements.txt` | numpy, Pillow (onnxruntime optional) |

---

## 9. Troubleshooting

* **`MISSING numpy/PIL`** - run `pip install -r requirements.txt`, then `python ipai.py doctor`.
* **`wd14 mode needs onnxruntime`** - `pip install onnxruntime`, or use the default engine.
* **All images show `review`** - no rating model is loaded; expected until you train one.
* **`ZeroDivisionError` in train.py** - fixed in v20 (it happened when the data had fewer images than the batch size or `--n-val` was bigger than the dataset). Update to v20.
* **`only N labelled images`** - label at least ~50 (a few hundred for real results) before `retrain`.
* **Prompt shows `[withheld: flagged by moderation]`** - the input or output was flagged; this is by design. *block* can never be overridden.
* **Windows garbled text** - all files are read/written as UTF-8; use a current Python and `py` launcher.
* **Label page won't load from a phone** - use `--lan` and open the printed address on the same Wi-Fi.
