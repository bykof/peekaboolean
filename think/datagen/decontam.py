"""Drop training items whose image matches an eval image by sha256 or 8x8 DCT perceptual hash.

Eval images: every image file under /root/big/evals/{realjev,heldout} and /root/big/imajev-bench/assets.
python decontam.py  -> decontam/drop_ids.txt, decontam/report.json
"""
import glob, hashlib, io, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = "/root/big/data"
EVAL_DIRS = ["/root/big/evals/realjev", "/root/big/evals/heldout", "/root/big/imajev-bench/assets"]
EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")
N = 32
C = np.array([[np.cos(np.pi * (2 * j + 1) * i / (2 * N)) for j in range(N)] for i in range(N)])


def phash(im):
    a = np.asarray(im.convert("L").resize((N, N), Image.LANCZOS), dtype=np.float64)
    d = (C @ a @ C.T)[:8, :8]
    bits = (d > np.median(d)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hashes(path):
    try:
        b = open(path, "rb").read()
        return path, hashlib.sha256(b).hexdigest(), phash(Image.open(io.BytesIO(b)))
    except Exception as e:  # unreadable file: report, do not crash
        return path, None, None


def eval_images():
    out = []
    for d in EVAL_DIRS:
        out += [p for p in glob.glob(d + "/**/*", recursive=True) if p.lower().endswith(EXT)]
    return out


def main():
    items = [json.loads(l) for f in sorted(glob.glob(f"{D}/items/*.jsonl")) for l in open(f)]
    train_imgs = sorted({p for it in items for p in it["images"]})
    evals = eval_images()
    with ProcessPoolExecutor(16) as ex:
        ev = list(ex.map(hashes, evals, chunksize=64))
        tr = list(ex.map(hashes, train_imgs, chunksize=64))
    ev_sha = {s for _, s, _ in ev if s}
    ev_ph = {h for _, _, h in ev if h is not None}
    bad = {p: ("sha256" if s in ev_sha else "phash") for p, s, h in tr if s in ev_sha or h in ev_ph}
    drop = sorted({it["id"] for it in items if any(p in bad for p in it["images"])})
    os.makedirs(f"{D}/decontam", exist_ok=True)
    open(f"{D}/decontam/drop_ids.txt", "w").write("\n".join(drop) + "\n")
    rep = {"eval_images": len(evals), "eval_unreadable": sum(s is None for _, s, _ in ev), "train_images": len(train_imgs),
           "matched_images": len(bad), "by": {k: list(bad.values()).count(k) for k in ("sha256", "phash")},
           "dropped_items": len(drop), "examples": list(bad.items())[:20], "eval_dirs": {d: os.path.isdir(d) for d in EVAL_DIRS}}
    json.dump(rep, open(f"{D}/decontam/report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:  # self-check: re-encoding keeps the phash, a different image does not
        im = Image.effect_noise((256, 256), 60).convert("RGB")
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=70)
        h2 = phash(Image.open(io.BytesIO(buf.getvalue())).resize((300, 300)))
        assert bin(phash(im) ^ h2).count("1") <= 6 and phash(im) != phash(im.rotate(90)), "phash broken"
        print("ok")
    else:
        main()
