"""
Overnight experiment runner (updated):
- Runs a target script with uv across multiple seeds
- Writes per-seed temp configs
- Logs to the experiment's output_dir from the config
- Supports passing extra args to the target after '--'
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import tempfile
import copy
import signal
import threading
import argparse
import yaml


def read_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(p: Path, data: dict) -> None:
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def set_nested(d: dict, path, value):
    cur = d
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _pump(src, sinks):
    for chunk in iter(lambda: src.readline(8192), b""):
        for s in sinks:
            s.write(chunk)
            s.flush()


def _filter_config_args(args_list: list[str]) -> list[str]:
    """Remove any --config occurrences from passthrough args; we provide our own."""
    out = []
    skip_next = False
    for i, a in enumerate(args_list):
        if skip_next:
            skip_next = False
            continue
        if a == "--config":
            skip_next = True
            continue
        if a.startswith("--config="):
            continue
        out.append(a)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run a script across seeds via 'uv run' and tee output to logs.")
    parser.add_argument("--script", type=Path, required=True, help="Path to Python script to run with 'uv run'")
    parser.add_argument("--config", type=Path, required=True, help="Base YAML config path")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="List of integer seeds")
    parser.add_argument("--cwd", type=Path, default=None, help="Working directory for the process. Defaults to current.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Extra args to pass to the target script after '--'.")

    args = parser.parse_args()

    script_path: Path = args.script
    config_path: Path = args.config
    seeds: list[int] = args.seeds

    if not script_path.exists():
        print(f"Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    base_cfg = read_yaml(config_path)
    out_dir_str = (
        base_cfg.get("experiment", {}).get("output_dir")
        or base_cfg.get("experiment", {}).get("output_path")
    )
    if not out_dir_str:
        print("Config missing experiment.output_dir; cannot determine log directory.", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(out_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group logs under output_dir/logs/<timestamp>
    timestamp_all = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = output_dir / "logs" / timestamp_all
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Passthrough args after '--', without any '--config' payload
    extra = []
    if args.args:
        extra = [a for a in args.args if a != "--"]
        extra = _filter_config_args(extra)

    script_abs = script_path.resolve()
    run_cwd = args.cwd.resolve() if args.cwd is not None else None
    if run_cwd is not None and not run_cwd.exists():
        print(f"Working directory not found: {run_cwd}", file=sys.stderr)
        sys.exit(1)

    for seed in seeds:
        cfg_copy = copy.deepcopy(base_cfg)
        set_nested(cfg_copy, ["experiment", "seed"], int(seed))

        with tempfile.TemporaryDirectory() as td:
            tmp_cfg_path = Path(td) / f"config_seed_{seed}.yaml"
            write_yaml(tmp_cfg_path, cfg_copy)

            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = logs_dir / f"{script_path.stem}_seed{seed}_{run_ts}.log"

            cmd = ["uv", "run", str(script_abs), "--config", str(tmp_cfg_path)] + extra

            print(f"[seed {seed}] Running: {' '.join(cmd)}")
            print(f"[seed {seed}] Logging -> {log_path}")

            with log_path.open("wb") as logf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(run_cwd) if run_cwd is not None else None,
                    bufsize=1,
                    text=False,
                )
                t = threading.Thread(target=_pump, args=(proc.stdout, [logf, sys.stdout.buffer]), daemon=True)
                t.start()
                try:
                    ret = proc.wait()
                except KeyboardInterrupt:
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
