"""The UI server's one check: a real round-trip through the HTTP layer, with the real
request validation and image decode behind a stub scorer.

  uv run python tests/test_ui.py
"""

import base64
import io
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer

from PIL import Image

from peekaboolean.data import load_image
from peekaboolean.serve import request_examples
from peekaboolean.ui import WEB, make_handler

QUESTIONS = {"kind": {"type": "choice", "instructions": "What kind of image is this?",
                      "criteria": {"photo": "", "chart": ""}}}


def fake_answer(state, questions, image):
    if state == "boom":
        raise RuntimeError("boom")
    request_examples(state, questions, "upload")   # the validation evaluate runs first
    img = load_image(image, 512)
    return {"answers": {"kind": {"type": "choice", "choice": "photo"}}, "size": list(img.size)}


def post(base, body, headers=None, path="/api/answer"):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def png_b64(size=(40, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, "teal").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_ui_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(fake_answer, {"model": "stub"}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/") as r:
            assert r.status == 200 and b"<html" in r.read().lower()
        with urllib.request.urlopen(base + "/api/info") as r:
            assert json.loads(r.read()) == {"model": "stub"}
        for path in ("/../pyproject.toml", "/web/index.html", "/index.html", "/nope"):
            try:
                urllib.request.urlopen(base + path)
                raise AssertionError(f"{path} should 404")
            except urllib.error.HTTPError as e:
                assert e.code == 404, path

        status, out = post(base, {"state": "", "questions": QUESTIONS, "image": png_b64()})
        assert status == 200 and out["answers"]["kind"]["choice"] == "photo" and out["size"] == [40, 30], out

        one_option = {"kind": {"type": "choice", "instructions": "?", "criteria": {"only": ""}}}
        status, out = post(base, {"questions": one_option, "image": png_b64()})
        assert (status, out["scope"]) == (400, "request") and "2..255" in out["error"], out

        status, out = post(base, {"questions": QUESTIONS, "image": base64.b64encode(b"not an image").decode()})
        assert (status, out["scope"]) == (400, "image") and "BytesIO" not in out["error"], out

        status, out = post(base, {"state": "boom", "questions": QUESTIONS, "image": png_b64()})
        assert (status, out["scope"]) == (500, "server"), out

        status, out = post(base, {"questions": QUESTIONS, "image": png_b64()}, {"Content-Type": "text/plain"})
        assert (status, out["scope"]) == (415, "request"), out

        status, out = post(base, {"questions": QUESTIONS, "image": png_b64()}, {"Host": "evil.example"})
        assert (status, out["scope"]) == (403, "request"), out

        status, out = post(base, {"questions": QUESTIONS})
        assert (status, out["scope"]) == (400, "request") and "image" in out["error"], out

        status, out = post(base, {"questions": {"kind": {"type": "noul"}}, "image": png_b64()})
        assert (status, out["scope"]) == (400, "request") and "instructions" in out["error"], out

        status, out = post(base, {"questions": {"kind": None}, "image": png_b64()})
        assert (status, out["scope"]) == (400, "request"), out

        # Jev's envelope, with imajev's `images` extension as data URLs; none means a blank image.
        url = "data:image/png;base64," + png_b64()
        status, out = post(base, {"state": {"record": 1}, "questions": QUESTIONS, "images": [url]}, path="/v1/systemone")
        assert status == 200 and set(out) == {"model", "answers", "usage"}, out
        assert out["answers"]["kind"] == {"type": "choice", "choice": "photo", "unknown_probability": 0.0,
                                          "abstained": False}, out
        status, out = post(base, {"questions": QUESTIONS}, path="/v1/systemone")
        assert status == 200 and out["answers"]["kind"]["choice"] == "photo", out
        status, out = post(base, {"questions": QUESTIONS, "images": [url, url]}, path="/v1/systemone")
        assert (status, out["scope"]) == (400, "request") and "one image" in out["error"], out
    finally:
        server.shutdown()


def test_zip():
    """The page's zip writer under node, read back by zipfile, which checks every CRC."""
    if not shutil.which("node"):
        return print("skip test_zip: no node")
    html = (WEB / "index.html").read_text()
    start = html.index("const CRC =")
    js = html[start:html.index("\n}\n", html.index("async function zipBlob")) + 3] + """
const f = (body, t) => new File([body], 'x', { lastModified: t });
const blob = await zipBlob([['photo/a.jpg', f('abc', new Date(2020, 0, 2, 3, 4, 6).getTime())],
                            ['L0 very blurry/sub/ü.png', f(new Uint8Array(70000).fill(7), 0)]]);
process.stdout.write(Buffer.from(await blob.arrayBuffer()));
"""
    out = subprocess.run(["node", "--input-type=module", "-e", js], capture_output=True, check=True).stdout
    z = zipfile.ZipFile(io.BytesIO(out))
    assert z.testzip() is None
    assert z.namelist() == ["photo/a.jpg", "L0 very blurry/sub/ü.png"], z.namelist()
    assert z.read("photo/a.jpg") == b"abc" and z.read("L0 very blurry/sub/ü.png") == b"\x07" * 70000
    assert z.getinfo("photo/a.jpg").date_time == (2020, 1, 2, 3, 4, 6)
    assert z.getinfo("L0 very blurry/sub/ü.png").date_time[0] == 1980


if __name__ == "__main__":
    test_ui_server()
    test_zip()
    print("ok")
