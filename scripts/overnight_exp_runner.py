# run_many.py
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import tempfile
import copy
import signal
import threading

import yaml  # pip install pyyaml


# =========================
# Configuration (edit these)
# =========================
SCRIPT = Path("/home/williaar/projects/delphi/run_genetic_evolution.py")  # the script you want to run
CONFIG = Path("/home/williaar/projects/delphi/configs/genetic_evolution_mediator_smooth_o3_full.yml")  # base config
SEEDS = [1, 2, 3, 4, 5]  # seeds to run
timestamp_all = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path(
    f"/home/williaar/projects/delphi/results/genetic_evolution_mediator_smooth_40_o3/logs/{timestamp_all}/"
)

# If your script uses a different CLI, change this factory to build the command.
# For example, Hydra or argparse might expect "--config", "--cfg", or "-c", etc.
def build_cmd(tmp_cfg_path: Path):
    return ["uv", "run", str(SCRIPT), "--config", str(tmp_cfg_path)]


# =========================
# Utilities
# =========================
def read_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(p: Path, data: dict) -> None:
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def set_nested(d: dict, path, value):
    cur = d
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _pump(src, sinks):
    """Copy bytes from src (a pipe) to each sink (binary file-like), line-buffered."""
    for chunk in iter(lambda: src.readline(8192), b""):
        for s in sinks:
            s.write(chunk)
            s.flush()


# =========================
# Main
# =========================
def main():
    if not SCRIPT.exists():
        print(f"Script not found: {SCRIPT}", file=sys.stderr)
        sys.exit(1)
    if not CONFIG.exists():
        print(f"Config not found: {CONFIG}", file=sys.stderr)
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = read_yaml(CONFIG)

    for seed in SEEDS:
        # 1) make a temp config with the seed override
        cfg_copy = copy.deepcopy(base_cfg)
        set_nested(cfg_copy, ["experiment", "seed"], int(seed))

        with tempfile.TemporaryDirectory() as td:
            tmp_cfg_path = Path(td) / f"config_seed_{seed}.yaml"
            write_yaml(tmp_cfg_path, cfg_copy)

            # 2) build log filename (per-seed, per-run timestamp)
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"{SCRIPT.stem}_seed{seed}_{run_ts}.log"
            log_path = LOG_DIR / log_name

            # 3) construct command
            cmd = build_cmd(tmp_cfg_path)

            print(f"[seed {seed}] Running: {' '.join(cmd)}")
            print(f"[seed {seed}] Logging -> {log_path}")

            # 4) run the job, tee stdout+stderr to both file and console; handle Ctrl-C
            with log_path.open("wb") as logf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(SCRIPT.parent) if str(SCRIPT.parent) != "" else None,
                    bufsize=1,
                    text=False,  # read bytes; we pump as bytes
                )
                t = threading.Thread(
                    target=_pump, args=(proc.stdout, [logf, sys.stdout.buffer]), daemon=True
                )
                t.start()
                try:
                    ret = proc.wait()
                except KeyboardInterrupt:
                    # Propagate SIGINT to child; wait a bit; then kill if needed
                    proc.send_signal(signal.SIGINT)
                    try:
                        ret = proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        ret = proc.wait()
                t.join()

            if ret != 0:
                print(f"[seed {seed}] EXIT CODE {ret} (see {log_path})", file=sys.stderr)
            else:
                print(f"[seed {seed}] Done.")


if __name__ == "__main__":
    main()
