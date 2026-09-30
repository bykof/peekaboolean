#!/usr/bin/env python3
"""A local web page over `serve.evaluate`: drop images, write typed questions, read answers.

  uv run python -m peekaboolean.ui --adapter peekaboolean-500m

The page (web/index.html) sends one image per POST /api/answer, base64 in a JSON body
beside the request, and gets back exactly what `serve` prints. The model is loaded once;
each request gets a thread, and a lock lets the model score one image at a time.
"""

from __future__ import annotations

import argparse
import base64
import errno
import io
import json
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

WEB = Path(__file__).with_name("web")
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/archivo.woff2": ("archivo.woff2", "font/woff2")}
MAX_BODY = 64 * 1024 * 1024


def make_handler(answer, info: dict):
    """`answer(state, questions, image_file) -> dict` does the scoring; the test passes a stub."""

    class Handler(BaseHTTPRequestHandler):
        timeout = 30

        def parse_request(self):
            if not super().parse_request():
                return False
            port = self.server.server_port
            if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self._send(403, {"error": "forbidden host", "scope": "request"})
                return False
            return True

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/info":
                return self._send(200, info)
            if path not in STATIC:
                return self._send(404, {"error": "not found", "scope": "request"})
            name, ctype = STATIC[path]
            self._send(200, (WEB / name).read_bytes(), ctype)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path != "/api/answer":
                return self._send(404, {"error": "not found", "scope": "request"})
            if self.headers.get_content_type() != "application/json":
                return self._send(415, {"error": "request body must be application/json", "scope": "request"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:     # base64 is 4/3 of the image
                return self._send(413, {"error": f"image is over {MAX_BODY * 3 // 4 >> 20} MiB", "scope": "image"})
            try:
                req = json.loads(self.rfile.read(length))
                image = io.BytesIO(base64.b64decode(req["image"], validate=True))
                state, questions = req.get("state", ""), req["questions"]
            except KeyError as e:
                return self._send(400, {"error": f"request body is missing {e}", "scope": "request"})
            except (ValueError, TypeError) as e:
                return self._send(400, {"error": f"malformed request body: {e}", "scope": "request"})
            try:
                out = answer(state, questions, image)
            except Image.DecompressionBombError:
                return self._send(400, {"error": "Image is too large to decode (decompression bomb guard).", "scope": "image"})
            except OSError:     # Pillow could not read the bytes as an image
                return self._send(400, {"error": "Not a readable image. The file may be truncated or mislabelled.",
                                        "scope": "image"})
            except KeyError as e:
                return self._send(400, {"error": f"a question is missing {e}", "scope": "request"})
            except (AttributeError, ValueError, TypeError) as e:
                return self._send(400, {"error": str(e), "scope": "request"})
            except Exception as e:
                return self._send(500, {"error": repr(e), "scope": "server"})
            self._send(200, out)

        def _send(self, status, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Local web UI for typed visual questions")
    ap.add_argument("--adapter", default="peekaboolean-500m", help="checkpoint directory or Hub repo id")
    ap.add_argument("--model", default=None, help="inferred from adapter when omitted")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--max-edge", type=int, default=512, help="calibrated at 256, 384 and 512")
    ap.add_argument("--mode", choices=["auto", "tree", "shared", "single", "naive"], default="auto")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    hub_id = re.fullmatch(r"[\w.-]+/[\w.-]+", args.adapter) and not args.adapter.startswith(".")
    if not Path(args.adapter).is_dir() and not hub_id:
        raise SystemExit(f"no checkpoint at {args.adapter!r}: fetch it with the README quickstart "
                         "(curl ... | tar xz) or pass --adapter <dir or hub id>")

    lock, info = threading.Lock(), {}

    def answer(state, questions, image):
        with lock:
            t0 = time.perf_counter()
            out = evaluate(model, processor, state, questions, image, calib, args.mode, args.max_edge)
        out["usage"] = {"seconds": round(time.perf_counter() - t0, 3)}
        return out

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(answer, info))
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        raise SystemExit(f"port {args.port} is in use: is peekaboolean.ui already running? pass --port")

    from .model import select_device
    from .serve import evaluate, load

    device = select_device(args.device)
    print(f"[ui] loading {args.adapter} on {device} ...", flush=True)
    model, processor, calib = load(args.adapter, args.model, device, merge=True)
    info.update(model=model.model_id, adapter=args.adapter, device=str(device), max_edge=args.max_edge)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"[ui] {url}  (ctrl+c stops)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
