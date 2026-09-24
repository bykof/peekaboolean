"""Detached jobs with logs, exit status, and PID-reuse protection (Linux host)."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if stat[0] == "Z" else stat[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2)); temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "status", "stop", "worker"])
    ap.add_argument("--run-dir", required=True)
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = ap.parse_args(argv[:split])
    args.command = argv[split + 1:]
    root = Path(args.run_dir).resolve()
    info = root / "process.json"
    if args.action == "start":
        root.mkdir(parents=True, exist_ok=True)
        if info.exists():
            old = json.loads(info.read_text())
            if identity(old["pid"]) == old["identity"]:
                raise SystemExit(f"job already running: {old['pid']}")
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command: raise SystemExit("supply a command after --")
        env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                   HF_HOME=str(Path.cwd() / ".cache" / "huggingface"))
        with (root / "job.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "peekaboolean.background", "worker",
                                      "--run-dir", str(root), "--", *command],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env)
        data = {"pid": child.pid, "identity": identity(child.pid), "command": command,
                "started": time.time(), "cwd": str(Path.cwd())}
        write_json(info, data)
        print(json.dumps(data, indent=2))
    elif args.action == "worker":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        print(f"[job] started {time.ctime()}: {command}", flush=True)
        code = 1
        try:
            child = subprocess.Popen(command)
            # The process group receives TERM together. Keep supervising while the
            # trainer flushes its resumable checkpoint.
            signal.signal(signal.SIGTERM, lambda *_: None)
            code = child.wait()
        finally:
            write_json(root / "exit.json", {"exit_code": code, "finished": time.time()})
            print(f"[job] exited {code}: {time.ctime()}", flush=True)
        raise SystemExit(code)
    else:
        data = json.loads(info.read_text())
        alive = identity(data["pid"]) == data["identity"]
        if args.action == "stop" and alive:
            os.killpg(data["pid"], signal.SIGTERM)
            print("Graceful stop requested; inspect job.log for checkpoint completion.")
        else:
            print(json.dumps({**data, "running": alive,
                "exit": json.loads((root / "exit.json").read_text()) if not alive and (root / "exit.json").exists() else None}, indent=2))


if __name__ == "__main__":
    main()
