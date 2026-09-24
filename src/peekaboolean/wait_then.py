"""Wait for another background job to exit successfully, then run a command.

Chains detached stages (teacher labelling -> v6 mixture) without a shell loop that
would die with the SSH session: start this itself under peekaboolean.background.
"""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    run, command = Path(sys.argv[1]), sys.argv[sys.argv.index("--") + 1:]
    started = json.loads((run / "process.json").read_text())["started"]
    while True:
        exit_file = run / "exit.json"
        if exit_file.exists():
            status = json.loads(exit_file.read_text())
            if status["finished"] >= started:
                if status["exit_code"]: raise SystemExit(f"{run} failed with exit code {status['exit_code']}")
                break
        time.sleep(60)
    print(f"[wait] {run} finished; running {command}", flush=True)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__": main()
