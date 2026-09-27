"""The UI server's one check: a real round-trip through the HTTP layer, with the real
request validation and image decode behind a stub scorer.

  uv run python tests/test_ui.py
"""

import base64
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

from PIL import Image

from peekaboolean.data import load_image
from peekaboolean.serve import request_examples
from peekaboolean.ui import make_handler

QUESTIONS = {"kind": {"type": "choice", "instructions": "What kind of image is this?",
                      "criteria": {"photo": "", "chart": ""}}}


def fake_answer(state, questions, image):
    request_examples(state, questions, "upload")   # the validation evaluate runs first
    img = load_image(image, 512)
    return {"answers": {"kind": {"type": "choice", "choice": "photo"}}, "size": list(img.size)}


def post(base, body):
    req = urllib.request.Request(base + "/api/answer", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
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
    server = HTTPServer(("127.0.0.1", 0), make_handler(fake_answer, {"model": "stub"}))
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
        assert (status, out["scope"]) == (400, "image"), out

        status, out = post(base, {"questions": QUESTIONS})
        assert (status, out["scope"]) == (400, "request") and "image" in out["error"], out

        status, out = post(base, {"questions": {"kind": {"type": "noul"}}, "image": png_b64()})
        assert (status, out["scope"]) == (400, "request") and "instructions" in out["error"], out

        status, out = post(base, {"questions": {"kind": None}, "image": png_b64()})
        assert (status, out["scope"]) == (400, "request"), out
    finally:
        server.shutdown()


if __name__ == "__main__":
    test_ui_server()
    print("ok")
