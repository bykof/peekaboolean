"""jevsrv on a Mac: mlx-vlm generates instead of vLLM; prompt, readout and response are jevsrv's.

python jevmlx.py --model-path DIR --port 8765 --samples 4
"""
import argparse, base64, os, tempfile, threading
from http.server import ThreadingHTTPServer

import mlx.core as mx
from mlx_vlm import load, stream_generate

import jevsrv


class MLXModel(jevsrv.Model):
    def __init__(self, args):
        super().__init__(args)
        self.model, self.processor = load(args.model_path)
        self.tok = self.processor.tokenizer
        self.lock = threading.Lock()  # one generation at a time on the GPU

    def chat(self, content):
        tmp = tempfile.mkdtemp()
        images = []
        for i, part in enumerate(content[:-1]):
            head, b64 = part["image_url"]["url"].split(",", 1)
            path = os.path.join(tmp, f"{i}.{head.split('/')[1].split(';')[0]}")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            images.append(path)
        messages = [{"role": "user", "content": [{"type": "image"}] * len(images)
                     + [{"type": "text", "text": content[-1]["text"]}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        choices = []
        for _ in range(self.a.samples):
            toks, text = [], ""
            with self.lock:
                for r in stream_generate(self.model, self.processor, prompt, image=images or None,
                                         max_tokens=self.a.max_tokens, temperature=self.a.temperature, top_p=0.95):
                    piece = self.tok.decode([r.token])
                    top = []
                    if jevsrv.ANSWER.search(text):  # only the answer position needs alternatives
                        lp = r.logprobs.reshape(-1)
                        ids = mx.argpartition(-lp, 20)[:20].tolist()
                        top = [{"token": self.tok.decode([i]), "logprob": lp[i].item()} for i in ids]
                    toks.append({"token": piece, "top_logprobs": top})
                    text += piece
            choices.append({"logprobs": {"content": toks}})
        return {"choices": choices, "usage": {}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--name", default="q36-think-mlx")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--calibration", help="calib.py fit output")
    args = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), jevsrv.make_handler(MLXModel(args))).serve_forever()
