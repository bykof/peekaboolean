#!/usr/bin/env python3
"""A local web page over `serve.evaluate`: drop images, write typed questions, read answers.

  uv run python -m peekaboolean.ui --adapter peekaboolean-500m

The page (web/index.html) sends one image per POST /api/answer, base64 in a JSON body
beside the request, and gets back exactly what `serve` prints. The model is loaded once;
HTTPServer answers one request at a time, which is also the order the page sends them.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image

WEB = Path(__file__).with_name("web")
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/archivo.woff2": ("archivo.woff2", "font/woff2")}
MAX_BODY = 64 * 1024 * 1024


def make_handler(answer, info: dict):
    """`answer(state, questions, image_file) -> dict` does the scoring; the test passes a stub."""

    class Handler(BaseHTTPRequestHandler):
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
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, {"error": f"image is over {MAX_BODY >> 20} MiB", "scope": "image"})
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
            except (OSError, Image.DecompressionBombError) as e:     # Pillow could not read the bytes as an image
                return self._send(400, {"error": str(e), "scope": "image"})
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
    ap.add_argument("--mode", choices=["auto", "shared", "single", "naive"], default="auto")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not Path(args.adapter).is_dir() and "/" not in args.adapter:
        raise SystemExit(f"no checkpoint at {args.adapter!r}: fetch it with the README quickstart "
                         "(curl ... | tar xz) or pass --adapter <dir or hub id>")

    from .model import select_device
    from .serve import evaluate, load

    device = select_device(args.device)
    print(f"[ui] loading {args.adapter} on {device} ...", flush=True)
    model, processor, calib = load(args.adapter, args.model, device, merge=True)

    def answer(state, questions, image):
        t0 = time.perf_counter()
        out = evaluate(model, processor, state, questions, image, calib, args.mode, args.max_edge)
        out["usage"] = {"seconds": round(time.perf_counter() - t0, 3)}
        return out

    info = {"model": model.model_id, "adapter": args.adapter, "device": str(device), "max_edge": args.max_edge}
    server = HTTPServer(("127.0.0.1", args.port), make_handler(answer, info))
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
