"""Tap-to-label tool for building your own training set (stdlib only, runs in Termux).

    python label_tool.py --images images --tags tags
    then open  http://127.0.0.1:8765  in your phone browser.

For each image: tap tags, pick a rating (required), Save & Next. Output matches prep_dataset.py:
    tags/<image-stem>.txt   ->  "1girl, solo, black_hair, rating:general"
Suggestions from the built-in analyser (hair colour guess, background colour) are pre-highlighted;
tap to confirm or remove. Edit label_vocab.json to change the tag buttons. Type extra tags in the box.

Rating guide (this trains the moderation head):
  general = nothing suggestive | sensitive = swimwear, cleavage, mildly suggestive
  questionable = partial nudity / strongly suggestive | explicit = nudity or sexual content
Label honestly - the flagger can only be as good as these labels.

    python label_tool.py --images images --tags tags --stats     # progress + tag/rating counts
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
EXTS = (".jpg", ".jpeg", ".png", ".webp")
RATINGS = ["general", "sensitive", "questionable", "explicit"]
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def clean_tag(t):
    return re.sub(r"[^a-z0-9_:\-()]", "", re.sub(r"\s+", "_", str(t).strip().lower()))


def read_label(tags_dir, stem):
    p = os.path.join(tags_dir, stem + ".txt")
    if not os.path.exists(p):
        return None
    parts = [x.strip() for x in re.split(r"[,\n]", open(p, encoding="utf-8").read()) if x.strip()]
    rating = next((x.split(":", 1)[1] for x in parts if x.startswith("rating:")), None)
    return [x for x in parts if not x.startswith("rating:")], rating


def save_label(tags_dir, stem, tags, rating):
    if rating not in RATINGS:
        raise ValueError("rating required")
    tags = sorted({clean_tag(t) for t in tags if clean_tag(t) and not clean_tag(t).startswith("rating:")})
    os.makedirs(tags_dir, exist_ok=True)
    with open(os.path.join(tags_dir, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write(", ".join(tags + [f"rating:{rating}"]))


WEIGHTS = None   # set by --weights: the model pre-fills tags (bootstrap loop)


def model_suggestions(path):
    """Tags (and a rating hint) from your trained ViT. Returns (tags, rating_or_None)."""
    sys.path.insert(0, HERE)
    import predict
    on, ex, _ = predict.tag_image(path, WEIGHTS)
    tags = [t for t in on]
    for k, (v, p) in ex.items():
        if k == "subject_count" and v == "solo":
            tags += ["solo"]
        if k == "background":
            tags.append(f"{v}_background")
    rating = ex.get("rating", (None, 0))[0]
    return list(dict.fromkeys(tags)), rating


def suggestions(path):
    try:
        sys.path.insert(0, HERE)
        import image_prompt as IP
        _, info = IP.analyse_basic(path, guess=True)
        fix = {"dark red": "red", "dark gray": "grey", "gray": "grey", "navy": "blue", "magenta": "pink",
               "orange": "red", "olive": "green", "teal": "green"}
        out = [fix.get(c, c).replace(" ", "_") + "_hair" for c in info.get("hair_guess", [])]
        bg = info.get("background", "")
        if bg in ("white", "black", "red", "purple", "blue", "pink", "gray"):
            out.append(("grey" if bg == "gray" else bg) + "_background")
        return list(dict.fromkeys(out))
    except Exception:
        return []


PAGE = r"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>label</title><style>
body{font-family:system-ui;margin:0;background:#111;color:#eee}
#top{position:sticky;top:0;background:#000;padding:8px;z-index:5;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
img{width:100%;max-height:60vh;object-fit:contain;background:#000}
h4{margin:10px 8px 4px;color:#9ab}.row{padding:0 6px;display:flex;flex-wrap:wrap;gap:6px}
.t{padding:9px 12px;border-radius:18px;background:#2a2a2a;border:1px solid #444;color:#eee;font-size:15px}
.t.on{background:#2b6cb0;border-color:#63b3ed}.t.sug{border-color:#f6ad55;border-style:dashed}
.r.on{background:#c05621}.btn{padding:12px 16px;border-radius:8px;border:0;font-size:16px;color:#fff;background:#38a169}
.btn.g{background:#555}input{padding:10px;font-size:16px;width:60%;border-radius:8px;border:1px solid #555;background:#222;color:#eee}
</style><div id=top><span id=prog></span><button class="btn g" onclick=go(-1)>Back</button>
<button class="btn g" onclick=go(1)>Skip</button><button class=btn onclick=save()>Save &amp; Next</button></div>
<img id=im><div id=root></div>
<div class=row style="margin:10px 0"><input id=extra placeholder="extra tags, comma separated"></div><div style="height:80px"></div>
<script>
let L=[],i=0,sel=new Set(),sug=new Set(),rating=null,srat=null,V={};
const $=s=>document.querySelector(s);
async function init(){V=await (await fetch('/api/vocab')).json();L=await (await fetch('/api/list')).json();
 i=Math.max(0,L.findIndex(x=>!x.done));load()}
async function load(){const n=L[i];if(!n){$('#root').innerHTML='<h4>All done 🎉</h4>';return}
 const d=await (await fetch('/api/item?name='+encodeURIComponent(n.name))).json();
 sel=new Set(d.tags||[]);sug=new Set(d.tags?[]:d.suggest);if(!d.tags)d.suggest.forEach(t=>sel.add(t));
 rating=d.rating;srat=d.suggest_rating;$('#im').src='/img/'+encodeURIComponent(n.name);$('#extra').value='';draw();
 $('#prog').textContent=(i+1)+'/'+L.length+' · done '+L.filter(x=>x.done).length}
function draw(){let h='';const known=new Set(Object.values(V).flat());const ex=[...sel].filter(t=>!known.has(t));
 if(ex.length){h+='<h4>other</h4><div class=row>';for(const t of ex)h+=`<button class="t on ${sug.has(t)?'sug':''}" onclick="tg('${t}')">${t.replace(/_/g,' ')}</button>`;h+='</div>'}
 for(const [k,tags] of Object.entries(V)){h+='<h4>'+k+'</h4><div class=row>';
 for(const t of tags)h+=`<button class="t ${sel.has(t)?'on':''} ${sug.has(t)?'sug':''}" onclick="tg('${t}')">${t.replace(/_/g,' ')}</button>`;h+='</div>'}
 h+='<h4>rating (required)</h4><div class=row>';
 for(const r of ['general','sensitive','questionable','explicit'])h+=`<button class="t r ${rating==r?'on':''} ${(!rating&&srat==r)?'sug':''}" onclick="rt('${r}')">${r}${(!rating&&srat==r)?' (model guess)':''}</button>`;
 $('#root').innerHTML=h+'</div>'}
function tg(t){sel.has(t)?sel.delete(t):sel.add(t);draw()}function rt(r){rating=r;draw()}
function go(d){i=Math.min(L.length-1,Math.max(0,i+d));load();scrollTo(0,0)}
async function save(){if(!rating){alert('pick a rating');return}
 const extra=$('#extra').value.split(',').map(s=>s.trim()).filter(Boolean);
 const r=await fetch('/api/save',{method:'POST',body:JSON.stringify({name:L[i].name,tags:[...sel,...extra],rating})});
 if(!r.ok){alert('save failed');return}L[i].done=true;go(1)}
init()</script>"""


ORDER_SEED = 1   # None = alphabetical


def list_images(images):
    names = sorted(f for f in os.listdir(images) if f.lower().endswith(EXTS))
    if ORDER_SEED is not None:      # fixed random order: your labelled subset is a fair sample, not one folder/artist
        import random
        random.Random(ORDER_SEED).shuffle(names)
    return names


def make_handler(images, tags_dir):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _safe(self, name):
            name = os.path.basename(name)
            return name if name.lower().endswith(EXTS) and os.path.isfile(os.path.join(images, name)) else None

        def do_GET(self):
            from urllib.parse import urlparse, parse_qs, unquote
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if u.path == "/api/vocab":
                return self._send(200, open(os.path.join(HERE, "label_vocab.json"), encoding="utf-8").read())
            if u.path == "/api/list":
                names = list_images(images)
                return self._send(200, json.dumps([{"name": n, "done": read_label(tags_dir, os.path.splitext(n)[0]) is not None} for n in names]))
            if u.path == "/api/item":
                n = self._safe(parse_qs(u.query).get("name", [""])[0])
                if not n:
                    return self._send(404, "{}")
                lab = read_label(tags_dir, os.path.splitext(n)[0])
                sug, srat = [], None
                if not lab:
                    p = os.path.join(images, n)
                    if WEIGHTS:
                        try:
                            sug, srat = model_suggestions(p)
                        except Exception as e:
                            print("model suggestion failed:", e)
                    if not sug:
                        sug = suggestions(p)
                return self._send(200, json.dumps({"tags": lab[0] if lab else None, "rating": lab[1] if lab else None,
                                                   "suggest": sug, "suggest_rating": srat}))
            if u.path.startswith("/img/"):
                n = self._safe(unquote(u.path[5:]))
                if not n:
                    return self._send(404, b"", "text/plain")
                return self._send(200, open(os.path.join(images, n), "rb").read(), MIME[os.path.splitext(n)[1].lower()])
            self._send(404, "{}")

        def do_POST(self):
            if self.path != "/api/save":
                return self._send(404, "{}")
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                n = self._safe(d["name"])
                if not n:
                    raise ValueError("bad name")
                save_label(tags_dir, os.path.splitext(n)[0], d.get("tags", []), d.get("rating"))
                self._send(200, '{"ok":true}')
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)}))
    return H


def stats(images, tags_dir):
    names = sorted(f for f in os.listdir(images) if f.lower().endswith(EXTS))
    tc, rc, done = Counter(), Counter(), 0
    for n in names:
        lab = read_label(tags_dir, os.path.splitext(n)[0])
        if lab:
            done += 1
            tc.update(lab[0])
            rc[lab[1]] += 1
    print(f"labelled {done}/{len(names)}\nratings: {dict(rc)}\ntop tags: {tc.most_common(25)}")
    for r in ("questionable", "explicit"):
        if done and rc.get(r, 0) < 50:
            print(f"warning: only {rc.get(r, 0)} '{r}' examples - the flagger needs ~50+ of each to learn it "
                  f"(use --sorted or search your folder for such images and label them on purpose)")
    if done < 300:
        print("note: a few hundred+ labelled images are needed before the tagger is usable; thousands for good results.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--tags", default="tags")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--sorted", action="store_true", help="show images alphabetically instead of in a fixed random order")
    ap.add_argument("--weights", help="trained weights (.npz): the model pre-fills tags so you only correct mistakes")
    ap.add_argument("--lan", action="store_true", help="listen on your Wi-Fi so a phone can open it (only on a network you trust)")
    ap.add_argument("--open", action="store_true", help="open the browser automatically")
    a = ap.parse_args()
    if not os.path.isdir(a.images):
        sys.exit(f"no such folder: {a.images}")
    if a.sorted:
        ORDER_SEED = None
    if a.weights:
        if not os.path.exists(a.weights):
            sys.exit(f"weights not found: {a.weights}")
        WEIGHTS = a.weights
        print("model suggestions ON (dashed = guess; the rating is never auto-selected)")
    if a.stats:
        stats(a.images, a.tags)
    else:
        host = "0.0.0.0" if a.lan else "127.0.0.1"
        url = f"http://127.0.0.1:{a.port}"
        if a.lan:
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("10.255.255.255", 1))
                print(f"on your phone/other device open: http://{s.getsockname()[0]}:{a.port}")
                s.close()
            except OSError:
                print("LAN address not found; use your computer's IP address")
        print(f"labelling {a.images} -> {a.tags}\nopen {url}  (Ctrl+C to stop)")
        if a.open:
            import webbrowser
            webbrowser.open(url)
        try:
            HTTPServer((host, a.port), make_handler(a.images, a.tags)).serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
