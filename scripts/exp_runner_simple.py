import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse
import signal
import threading
import yaml


def _pump(src, sinks):
    """Copy bytes from src (a pipe) to each sink (binary file-like), line-buffered."""
    for chunk in iter(lambda: src.readline(8192), b""):
        for s in sinks:
            s.write(chunk)
            s.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Run a script via 'uv run' and tee output to a log file."
    )
    parser.add_argument(
        "--script",
        type=Path,
        required=True,
        help="Path to the Python script to run with 'uv run'",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to log file. If not provided, uses results/logs/<timestamp>/<script>_<ts>.log",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Working directory for the target process. Defaults to current directory.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional args passed through to the target script (everything after '--').",
    )

    args = parser.parse_args()

    script_path: Path = args.script
    if not script_path.exists():
        print(f"Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    # Determine extra args (everything after '--')
    extra = []
    if args.args:
        extra = [a for a in args.args if a != "--"]

    # Try to detect a config path to place logs inside the experiment's output_dir
    config_path = None
    for i, a in enumerate(extra):
        if a.startswith("--config="):
            config_path = a.split("=", 1)[1]
            break
        if a == "--config" and i + 1 < len(extra):
            config_path = extra[i + 1]
            break

    output_dir_from_config: Path | None = None
    if config_path is not None:
        cfg_p = Path(config_path)
        if cfg_p.exists():
            try:
                with cfg_p.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                out_dir_str = (
                    cfg.get("experiment", {}).get("output_dir")
                    or cfg.get("experiment", {}).get("output_path")
                )
                if out_dir_str:
                    output_dir_from_config = Path(out_dir_str)
                    output_dir_from_config.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    # Build log path
    if args.log is not None:
        log_path = args.log
        log_path.parent.mkdir(parents=True, exist_ok=True)
    elif output_dir_from_config is not None:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = output_dir_from_config / f"{script_path.stem}_{run_ts}.log"
    else:
        root = Path("results/logs")
        root_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = root / root_ts
        log_dir.mkdir(parents=True, exist_ok=True)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{script_path.stem}_{run_ts}.log"

    # Build command
    script_abs = script_path.resolve()
    cmd = ["uv", "run", str(script_abs)] + extra

    print(f"Running: {' '.join(cmd)}")
    print(f"Logging -> {log_path}")

    # Run and tee
    with log_path.open("wb") as logf:
        run_cwd = args.cwd.resolve() if args.cwd is not None else None
        if run_cwd is not None and not run_cwd.exists():
            print(f"Working directory not found: {run_cwd}", file=sys.stderr)
            sys.exit(1)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Use requested working directory or inherit current
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
        print(f"EXIT CODE {ret} (see {log_path})", file=sys.stderr)
        sys.exit(ret)
    else:
        print("Done.")


if __name__ == "__main__":
    main()
