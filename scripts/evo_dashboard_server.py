#!/usr/bin/env python3
"""
Flask-based interactive dashboard for evolution runs.

Launch:
  uv run scripts/evo_dashboard_server.py --run-dir <path/to/run_dir> --host 127.0.0.1 --port 5000

Then open http://127.0.0.1:5000 in your browser.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List

from flask import Flask, jsonify, request, Response
import logging
import traceback
import yaml
import numpy as np


app = Flask(__name__)
RUN_DIR: Path | None = None  # Currently selected run dir (timestamp folder or similar)
RUNS_ROOT: Path | None = (
    None  # Root to discover selectable runs (defaults near RUN_DIR)
)


def _is_run_dir(p: Path) -> bool:
    # A run dir has monitor.csv or evolved_prompts (optionally under dry_run)
    return (
        (p / "monitor.csv").exists()
        or (p / "evolved_prompts").exists()
        or (p / "dry_run" / "monitor.csv").exists()
        or (p / "dry_run" / "evolved_prompts").exists()
    )


def _pick_latest_child_run(p: Path) -> Path | None:
    # Choose latest child dir that looks like a run (by mtime)
    if not p.exists() or not p.is_dir():
        return None
    candidates = [d for d in p.iterdir() if d.is_dir() and _is_run_dir(d)]
    if candidates:
        return max(candidates, key=lambda d: d.stat().st_mtime)
    # Fallback: pick subdirs containing a marker file
    markers = [
        d
        for d in p.iterdir()
        if d.is_dir()
        and ((d / "config_used").exists() or (d / "config_used.yml").exists())
    ]
    if markers:
        return max(markers, key=lambda d: d.stat().st_mtime)
    return None


def _discover_runs(root: Path) -> list[Path]:
    """Recursively find run-like directories beneath root.

    Heuristic: directories containing monitor.csv or evolved_prompts (or dry_run variants),
    or having config_used/config_used.yml marker files.
    """
    runs: list[Path] = []
    if not root.exists() or not root.is_dir():
        return runs
    for d in root.rglob("*"):
        try:
            if d.is_dir() and _is_run_dir(d):
                runs.append(d)
        except Exception:
            continue
    # De-duplicate nested picks: keep deepest paths
    runs_sorted = sorted(set(runs), key=lambda p: (len(p.parts), p.as_posix()))
    return runs_sorted


def require_run_dir() -> Path:
    if RUN_DIR is None:
        raise RuntimeError("Server not configured with --run-dir")
    # If RUN_DIR isn't a run dir, try to select a latest child
    if not _is_run_dir(RUN_DIR):
        child = _pick_latest_child_run(RUN_DIR)
        if child is not None:
            logging.info("Auto-selected child run dir: %s", child)
            return child
    return RUN_DIR


@app.before_request
def _log_request_info():
    logging.info("%s %s", request.method, request.path)


@app.errorhandler(Exception)
def _handle_exception(e: Exception):
    logging.exception("Unhandled exception at %s: %s", request.path, e)
    return jsonify(
        {
            "error": "server_error",
            "message": str(e),
            "path": request.path,
            "trace": traceback.format_exc().splitlines()[-6:],
        }
    ), 500


def read_monitor_csv(path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    if not path.exists():
        return data
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except Exception:
                    data[k].append(float("nan"))
    return data


@app.get("/api/summary")
def api_summary():
    run = require_run_dir()
    # Prefer direct monitor.csv, else dry_run/monitor.csv
    mon_path = run / "monitor.csv"
    if not mon_path.exists():
        alt = run / "dry_run" / "monitor.csv"
        if alt.exists():
            mon_path = alt
    mon = read_monitor_csv(mon_path)
    # Derive available dynamic metric bases
    bases = set()
    for k in mon.keys():
        if k.startswith("train_"):
            bases.add(k[len("train_") :])
        if k.startswith("val_"):
            bases.add(k[len("val_") :])
    resp = {
        "columns": list(mon.keys()),
        "data": mon,
        "metric_bases": sorted(bases),
    }
    return jsonify(resp)


@app.get("/api/runs")
def api_runs():
    """List selectable run directories under the configured root.

    Returns list of objects: { path, label, mtime }
    """
    global RUNS_ROOT
    base = RUNS_ROOT or (RUN_DIR.parent if RUN_DIR else Path("results"))
    # If base itself is a run dir, prefer its parent
    if _is_run_dir(base):
        base = base.parent
    found = _discover_runs(base)
    import re

    ts_re = re.compile(r"^(\d{8})_(\d{6})(?:_.+)?$")

    def _ts_val(seg: str) -> int | None:
        m = ts_re.match(seg)
        if not m:
            return None
        try:
            return int(m.group(1) + m.group(2))  # YYYYMMDDHHMMSS
        except Exception:
            return None

    def _label_sort_key(label: str):
        # Sort path segments alphabetically, except timestamp-like segments
        # which sort by recency (newest first).
        try:
            parts = Path(label).parts
        except Exception:
            parts = tuple(str(label).split("/"))
        key = []
        for seg in parts:
            ts = _ts_val(seg)
            if ts is not None:
                key.append((1, -ts, ""))  # Timestamp: sort by descending ts
            else:
                key.append((0, 0, seg.lower()))  # Text: alphabetical
        return tuple(key)

    items = []
    for p in found:
        try:
            rel = p.relative_to(base)
        except Exception:
            rel = p.name
        items.append(
            {
                "path": str(p.resolve()),
                "label": str(rel),
                "mtime": p.stat().st_mtime,
            }
        )
    # Sort by path parts alpha, but timestamp segments by recency
    items.sort(key=lambda x: _label_sort_key(x["label"]))
    return jsonify(
        {
            "root": str(base.resolve()),
            "runs": items,
            "current": str(require_run_dir().resolve()),
        }
    )


@app.post("/api/select_run")
def api_select_run():
    """Select current run directory by absolute path from /api/runs."""
    data = request.get_json(silent=True) or {}
    path = data.get("path") or request.args.get("path")
    if not path:
        return jsonify({"error": "missing_path"}), 400
    p = Path(path).resolve()
    if not p.exists():
        return jsonify({"error": "not_found", "path": str(p)}), 404
    # Accept both run dir and parents that contain runs (auto-pick latest child)
    chosen = p
    if not _is_run_dir(chosen):
        child = _pick_latest_child_run(chosen)
        if child is None:
            return jsonify({"error": "no_run_under_path", "path": str(p)}), 400
        chosen = child
    global RUN_DIR
    RUN_DIR = chosen
    return jsonify({"ok": True, "run_dir": str(RUN_DIR)})


@app.get("/api/health")
def api_health():
    run = require_run_dir()
    data = {
        "run_dir": str(run),
        "has_monitor": (run / "monitor.csv").exists(),
        "has_monitor_dry": (run / "dry_run" / "monitor.csv").exists(),
        "has_evolved_prompts": (run / "evolved_prompts").exists(),
        "has_evolved_prompts_dry": (run / "dry_run" / "evolved_prompts").exists(),
    }
    logging.info("Health: %s", data)
    return jsonify(data)


@app.get("/api/generations")
def api_generations():
    run = require_run_dir()
    base = run / "evolved_prompts"
    if not base.exists():
        dr = run / "dry_run" / "evolved_prompts"
        if dr.exists():
            base = dr
    gens = []
    if base.exists():
        for p in base.iterdir():
            if p.is_dir() and p.name.startswith("gen_"):
                try:
                    gens.append(int(p.name.split("_")[1]))
                except Exception:
                    pass
    gens.sort()
    return jsonify({"generations": gens})


@app.get("/api/candidates")
def api_candidates():
    run = require_run_dir()
    gen = int(request.args.get("gen", "0"))
    base = run / "evolved_prompts"
    if not base.exists():
        dr = run / "dry_run" / "evolved_prompts"
        if dr.exists():
            base = dr
    gen_dir = base / f"gen_{gen:03d}"
    cands = []
    if gen_dir.exists():
        cands = [
            p.name
            for p in gen_dir.iterdir()
            if p.is_dir() and p.name.startswith("cand_")
        ]
        cands.sort()
    return jsonify({"candidates": cands})


@app.get("/api/questions")
def api_questions():
    run = require_run_dir()
    gen = int(request.args.get("gen", "0"))
    cand = request.args.get("cand") or ""
    split = request.args.get("split", "val")
    base = run / "evolved_prompts"
    if not base.exists():
        dr = run / "dry_run" / "evolved_prompts"
        if dr.exists():
            base = dr
    split_dir = base / f"gen_{gen:03d}" / cand / "delphi_logs" / split
    items = []
    if split_dir.exists():
        for p in split_dir.glob("*.json"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                items.append(
                    {
                        "question_id": p.stem,
                        "question_text": (obj.get("question_text") or "")[:160],
                    }
                )
            except Exception:
                items.append({"question_id": p.stem, "question_text": ""})
    items.sort(key=lambda x: x["question_id"])
    return jsonify({"questions": items})


@app.get("/api/delphi")
def api_delphi():
    run = require_run_dir()
    gen = int(request.args.get("gen", "0"))
    cand = request.args.get("cand") or ""
    split = request.args.get("split", "val")
    qid = request.args.get("question_id") or ""
    base = run / "evolved_prompts"
    if not base.exists():
        dr = run / "dry_run" / "evolved_prompts"
        if dr.exists():
            base = dr
    path = base / f"gen_{gen:03d}" / cand / "delphi_logs" / split / f"{qid}.json"
    if not path.exists():
        return jsonify({"error": f"Not found: {path}"}), 404
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    # Attach resolution info for convenience in the UI
    try:
        qid_eff = obj.get("question_id") or qid
        # Prefer resolution_date in the log, else from config_used
        res_date = (obj.get("config", {}) or {}).get("resolution_date")
        if not res_date:
            cfg = _read_config_used(run)
            res_date = (cfg.get("data", {}) or {}).get("resolution_date")
        if qid_eff and res_date:
            from dataset.dataloader import ForecastDataLoader

            loader = ForecastDataLoader()
            res = loader.get_resolution(qid_eff, res_date)
            if res is not None:
                obj["_resolution"] = {
                    "date": res_date,
                    "resolved": bool(getattr(res, "resolved", False)),
                    "actual_outcome": float(getattr(res, "resolved_to", 0.0)),
                    "direction": getattr(res, "direction", None),
                }
    except Exception:
        pass
    return jsonify(obj)


@app.get("/api/prompt")
def api_prompt():
    run = require_run_dir()
    gen = int(request.args.get("gen", "0"))
    cand = request.args.get("cand") or ""
    base = run / "evolved_prompts"
    if not base.exists():
        dr = run / "dry_run" / "evolved_prompts"
        if dr.exists():
            base = dr
    path = base / f"gen_{gen:03d}" / cand / "prompt.md"
    if not path.exists():
        return Response("", mimetype="text/plain")
    return Response(path.read_text(encoding="utf-8"), mimetype="text/plain")


def _read_config_used(run: Path) -> dict:
    for name in ("config_used.yml", "config_used"):
        f = run / name
        if f.exists():
            try:
                return yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception:
                continue
    # Try dry_run subdir
    for name in ("config_used.yml", "config_used"):
        f = run / "dry_run" / name
        if f.exists():
            try:
                return yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def _iter_delphi_logs(
    run: Path, split: str, gen: int | None = None, cand: str | None = None
):
    base = run / "evolved_prompts"
    if not base.exists():
        alt = run / "dry_run" / "evolved_prompts"
        if alt.exists():
            base = alt
    if not base.exists():
        return
    gen_dirs = (
        [base / f"gen_{gen:03d}"]
        if gen is not None
        else [d for d in base.iterdir() if d.is_dir() and d.name.startswith("gen_")]
    )
    for gdir in gen_dirs:
        if not gdir.exists():
            continue
        cand_dirs = (
            [gdir / cand]
            if cand
            else [
                d for d in gdir.iterdir() if d.is_dir() and d.name.startswith("cand_")
            ]
        )
        for cdir in cand_dirs:
            sdir = cdir / "delphi_logs" / split
            if not sdir.exists():
                continue
            for fp in sdir.glob("*.json"):
                yield fp


def _compute_avg_brier_by_round(
    run: Path, split: str, gen: int | None = None, cand: str | None = None
):
    """Compute average Brier per round using median-of-experts each round.

    Also computes baseline Briers: median superforecaster and median public forecast
    per question (static across rounds), averaged across questions.
    """
    cfg = _read_config_used(run)
    resolution_date = ((cfg or {}).get("data") or {}).get("resolution_date")
    if not resolution_date:
        # Try nested shape
        resolution_date = (cfg.get("data", {}) or {}).get("resolution_date")
    # Lazy import to avoid heavy startup
    from dataset.dataloader import ForecastDataLoader

    loader = ForecastDataLoader()

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    seen_qids: set[str] = set()  # unique questions processed for baselines
    sf_sum = 0.0
    sf_n = 0
    public_sum = 0.0
    public_n = 0

    for fp in _iter_delphi_logs(run, split, gen, cand):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        qid = obj.get("question_id") or fp.stem
        try:
            res = loader.get_resolution(qid, resolution_date)
            if not res or not getattr(res, "resolved", False):
                continue
            actual = float(res.resolved_to)
        except Exception:
            continue
        rounds = obj.get("rounds") or []
        if not rounds:
            continue
        # Baselines (one per unique question)
        if qid not in seen_qids:
            seen_qids.add(qid)
            try:
                sfs = loader.get_super_forecasts(
                    question_id=qid, resolution_date=resolution_date
                )
                if sfs:
                    sf_median = float(np.median([float(f.forecast) for f in sfs]))
                    sf_sum += (sf_median - actual) ** 2
                    sf_n += 1
            except Exception:
                pass
            try:
                pubs = loader.get_public_forecasts(
                    question_id=qid, resolution_date=resolution_date
                )
                if pubs:
                    public_median = float(np.median([float(f.forecast) for f in pubs]))
                    public_sum += (public_median - actual) ** 2
                    public_n += 1
            except Exception:
                pass
        for idx, rd in enumerate(rounds):
            experts = rd.get("experts") or {}
            if not experts:
                continue
            probs = [
                float(v.get("prob"))
                for v in experts.values()
                if v is not None and isinstance(v.get("prob"), (int, float))
            ]
            if not probs:
                continue
            pred = float(np.median(probs))
            brier = (pred - actual) ** 2
            sums[idx] = sums.get(idx, 0.0) + brier
            counts[idx] = counts.get(idx, 0) + 1

    rounds_sorted = sorted(sums.keys())
    avg = [sums[i] / max(counts.get(i, 1), 1) for i in rounds_sorted]
    sf_avg = (sf_sum / sf_n) if sf_n > 0 else None
    public_avg = (public_sum / public_n) if public_n > 0 else None
    return {
        "rounds": rounds_sorted,
        "avg_brier": avg,
        "n_questions": len(seen_qids),
        "sf_avg_brier": sf_avg,
        "public_avg_brier": public_avg,
    }


@app.get("/api/avg_brier_rounds")
def api_avg_brier_rounds():
    run = require_run_dir()
    split = request.args.get("split", "val")
    gen = request.args.get("gen")
    cand = request.args.get("cand")
    gen_i = int(gen) if gen is not None else None
    data = _compute_avg_brier_by_round(run, split, gen_i, cand)
    return jsonify({"split": split, **data})


@app.get("/api/avg_brier_rounds_all")
def api_avg_brier_rounds_all():
    run = require_run_dir()
    gen = request.args.get("gen")
    cand = request.args.get("cand")
    gen_i = int(gen) if gen is not None else None
    out = {}
    for split in ("train", "val", "test"):
        out[split] = _compute_avg_brier_by_round(run, split, gen_i, cand)
    return jsonify(out)


INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Evolution Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
      .row { display: flex; gap: 24px; margin-bottom: 24px; flex-wrap: wrap; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; flex: 1; min-width: 380px; }
      label { font-size: 12px; color: #555; }
      select, input { margin: 0 8px 8px 0; }
      pre { white-space: pre-wrap; background: #fafafa; padding: 8px; border-radius: 6px; max-height: 200px; overflow: auto; }
      canvas { max-height: 360px; }
      /* Conversation role styling */
      .msg-header { font-weight: 600; margin-bottom: 6px; }
      .msg-expert { color: blue; }
      .msg-mediator { color: red; }
      .msg-other { color: #444; }
      .error { background: #fee; border: 1px solid #f99; color: #900; padding: 8px; border-radius: 6px; display: none; }
    </style>
  </head>
  <body>
    <h2>Evolution Dashboard</h2>
    <div class="row">
      <div class="card" style="max-width: 100%">
        <h3>Run</h3>
        <label>Run Directory</label>
        <select id="runSelect" style="min-width: 420px;"></select>
        <button id="switchRunBtn">Load Run</button>
        <span id="runRootLabel" style="margin-left:12px;color:#666;font-size:12px;"></span>
      </div>
    </div>
    <div id="errorBox" class="error"></div>
    <div class="row">
      <div class="card">
        <h3>Summary</h3>
        <canvas id="summaryChart"></canvas>
      </div>
      <div class="card">
        <h3>Metric Means</h3>
        <label>Metrics:</label>
        <select id="metricsSelect" multiple size="6" style="min-width: 220px;"></select>
        <canvas id="metricsChart"></canvas>
      </div>
    </div>
    <div class="row">
      <div class="card">
        <h3>Prompt / Question Explorer</h3>
        <div>
          <label>Generation</label>
          <select id="genSelect"></select>
          <label>Candidate</label>
          <select id="candSelect"></select>
          <label>Split</label>
          <select id="splitSelect">
            <option>train</option>
            <option selected>val</option>
            <option>test</option>
          </select>
          <label>Question</label>
          <select id="qidSelect" style="min-width: 320px;"></select>
          <button id="loadBtn">Load</button>
        </div>
        <div class="row">
          <div class="card">
            <h4>Expert Probabilities</h4>
            <div id="resolutionInfo" style="margin-bottom:6px;font-size:12px;color:#555;"></div>
            <canvas id="delphiChart"></canvas>
          </div>
          <div class="card">
            <h4>Prompt Text</h4>
            <pre id="promptText"></pre>
          </div>
          <div class="card">
            <h4>All Experts (Conversations)</h4>
            <div id="allExpertsConvs"></div>
          </div>
        </div>
      </div>
    </div>
    <div class="row">
      <div class="card">
        <h3>Avg Brier by Round</h3>
        <div>
          <label>Split:</label>
          <label><input type="checkbox" id="chkTrain" checked> train</label>
          <label><input type="checkbox" id="chkVal" checked> val</label>
          <label><input type="checkbox" id="chkTest" checked> test</label>
          <button id="refreshBrierBtn">Refresh</button>
        </div>
        <canvas id="avgBrierChart"></canvas>
      </div>
    </div>

    <script>
      const summaryCtx = document.getElementById('summaryChart').getContext('2d');
      const metricsCtx = document.getElementById('metricsChart').getContext('2d');
      const delphiCtx = document.getElementById('delphiChart').getContext('2d');
      const avgBrierCtx = document.getElementById('avgBrierChart').getContext('2d');
      let summaryChart, metricsChart, delphiChart, avgBrierChart;

      async function fetchJSON(url) {
        const r = await fetch(url);
        if (!r.ok) { throw new Error(`HTTP ${r.status} for ${url}`); }
        return r.json();
      }

      function showError(msg) {
        const box = document.getElementById('errorBox');
        if (msg) {
          box.textContent = msg;
          box.style.display = 'block';
          console.error(msg);
        } else {
          box.textContent = '';
          box.style.display = 'none';
        }
      }

      async function loadSummary() {
        try {
          const s = await fetchJSON('/api/summary');
          const gen = s.data?.generation ?? [];
          const bestTrain = s.data?.best_train ?? [];
          const meanTrain = s.data?.mean_train ?? [];
          const bestVal = s.data?.best_val ?? [];
          const meanVal = s.data?.mean_val ?? [];
          const gap = s.data?.gap ?? [];
          const mut = s.data?.mutation_rate ?? [];
          if (summaryChart) summaryChart.destroy();
          summaryChart = new Chart(summaryCtx, {
            type: 'line',
            data: {
              labels: gen,
              datasets: [
                { label: 'best_train', data: bestTrain, borderColor: 'blue', tension: 0.2 },
                { label: 'mean_train', data: meanTrain, borderColor: 'blue', borderDash: [5,4], tension: 0.2 },
                { label: 'best_val', data: bestVal, borderColor: 'orange', tension: 0.2 },
                { label: 'mean_val', data: meanVal, borderColor: 'orange', borderDash: [5,4], tension: 0.2 },
                { label: 'gap', data: gap, borderColor: 'red', tension: 0.2 },
                { label: 'mutation_rate', data: mut, borderColor: 'green', tension: 0.2 },
              ]
            },
            options: { responsive: true, animation: false }
          });

          // Fill metrics multiselect
          const select = document.getElementById('metricsSelect');
          select.innerHTML = '';
          (s.metric_bases || []).forEach(m => {
            const opt = document.createElement('option');
            opt.value = m; opt.textContent = m; select.appendChild(opt);
          });
        } catch (e) {
          showError('Failed to load summary: ' + (e?.message || e));
        }
      }

      async function loadRuns() {
        try {
          const r = await fetchJSON('/api/runs');
          const select = document.getElementById('runSelect');
          const rootLbl = document.getElementById('runRootLabel');
          select.innerHTML = '';
          rootLbl.textContent = `root: ${r.root}`;
          (r.runs || []).forEach(item => {
            const opt = document.createElement('option');
            opt.value = item.path; opt.textContent = item.label;
            if (item.path === r.current) opt.selected = true;
            select.appendChild(opt);
          });
        } catch (e) {
          console.warn('Failed to load runs', e);
        }
      }

      async function switchRun() {
        try {
          const sel = document.getElementById('runSelect');
          const path = sel.value;
          if (!path) return;
          await fetch('/api/select_run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) });
          // Reload all visuals for new run
          await loadSummary();
          await populateExplorer();
          await loadAvgBrierRounds();
          showError('');
        } catch (e) {
          showError('Failed to switch run: ' + (e?.message || e));
        }
      }

      async function loadMetrics() {
        try {
          const s = await fetchJSON('/api/summary');
          const gen = s.data?.generation ?? [];
          const select = document.getElementById('metricsSelect');
          const chosen = Array.from(select.selectedOptions).map(o => o.value);
          const datasets = [];
          chosen.forEach(base => {
            const tkey = 'train_' + base;
            const vkey = 'val_' + base;
            if (s.data?.[tkey]) datasets.push({ label: tkey, data: s.data[tkey], borderColor: '#3366cc', tension: 0.2 });
            if (s.data?.[vkey]) datasets.push({ label: vkey, data: s.data[vkey], borderColor: '#dc3912', tension: 0.2 });
          });
          if (metricsChart) metricsChart.destroy();
          metricsChart = new Chart(metricsCtx, {
            type: 'line', data: { labels: gen, datasets }, options: { responsive: true, animation: false }
          });
        } catch (e) {
          showError('Failed to load metrics: ' + (e?.message || e));
        }
      }

      async function populateExplorer() {
        const gens = await fetchJSON('/api/generations');
        const genSel = document.getElementById('genSelect');
        genSel.innerHTML = '';
        gens.generations.forEach(g => { const opt = document.createElement('option'); opt.value = g; opt.textContent = g; genSel.appendChild(opt); });
        await refreshCandidates();
        document.getElementById('genSelect').addEventListener('change', refreshCandidates);
        document.getElementById('candSelect').addEventListener('change', refreshQuestions);
        document.getElementById('splitSelect').addEventListener('change', refreshQuestions);
        document.getElementById('loadBtn').addEventListener('click', loadDelphi);
      }

      async function refreshCandidates() {
        const gen = document.getElementById('genSelect').value;
        const cands = await fetchJSON('/api/candidates?gen=' + gen);
        const candSel = document.getElementById('candSelect');
        candSel.innerHTML = '';
        cands.candidates.forEach(c => { const opt = document.createElement('option'); opt.value = c; opt.textContent = c; candSel.appendChild(opt); });
        await refreshQuestions();
      }

      async function refreshQuestions() {
        const gen = document.getElementById('genSelect').value;
        const cand = document.getElementById('candSelect').value;
        const split = document.getElementById('splitSelect').value;
        const qs = await fetchJSON(`/api/questions?gen=${gen}&cand=${encodeURIComponent(cand)}&split=${split}`);
        const qSel = document.getElementById('qidSelect');
        qSel.innerHTML = '';
        qs.questions.forEach(q => { const opt = document.createElement('option'); opt.value = q.question_id; opt.textContent = `${q.question_id.slice(0,8)} | ${q.question_text}`; qSel.appendChild(opt); });
      }

      async function loadDelphi() {
        const gen = document.getElementById('genSelect').value;
        const cand = document.getElementById('candSelect').value;
        const split = document.getElementById('splitSelect').value;
        const qid = document.getElementById('qidSelect').value;
        if (!gen || !cand || !qid) return;
        try {
          const log = await fetchJSON(`/api/delphi?gen=${gen}&cand=${encodeURIComponent(cand)}&split=${split}&question_id=${qid}`);
          // Build expert series across rounds
          const rounds = log.rounds || [];
          const expertIds = new Set();
          rounds.forEach(r => { Object.keys(r.experts || {}).forEach(k => expertIds.add(k)); });
          const xs = rounds.map(r => r.round || 0);
          const datasets = [];
          Array.from(expertIds).sort().forEach((sfid, idx) => {
            const vals = rounds.map(r => ((r.experts || {})[sfid] || {}).prob ?? NaN);
            const color = `hsl(${(idx*47)%360},60%,45%)`;
            datasets.push({ label: sfid, data: vals, borderColor: color });
          });
          if (delphiChart) delphiChart.destroy();
          delphiChart = new Chart(delphiCtx, { type: 'line', data: { labels: xs, datasets }, options: { responsive: true, animation: false } });
          // Show resolution info if available
          const rinfo = document.getElementById('resolutionInfo');
          if (rinfo) {
            const meta = log?._resolution || {};
            if (meta && meta.actual_outcome != null) {
              const val = Number(meta.actual_outcome);
              const date = meta.date || '';
              rinfo.textContent = `Resolution${date ? ' ('+date+')' : ''}: ${val.toFixed(2)}`;
            } else {
              rinfo.textContent = 'Resolution: (unknown)';
            }
          }
          // Load prompt text
          const promptResp = await fetch(`/api/prompt?gen=${gen}&cand=${encodeURIComponent(cand)}`);
          document.getElementById('promptText').textContent = await promptResp.text();
          // Render one box per expert
          renderAllExpertConversations(log);
          showError('');
        } catch (e) {
          showError('Failed to load Delphi log: ' + (e?.message || e));
        }
      }

      

      function renderAllExpertConversations(log) {
        const container = document.getElementById('allExpertsConvs');
        if (!container) return;
        try {
          container.innerHTML = '';
          const histExperts = log?.histories?.experts || {};
          let sfids = Object.keys(histExperts);
          if (!sfids.length) {
            const rounds = Array.isArray(log?.rounds) ? log.rounds : [];
            if (rounds.length && rounds[0]?.experts) sfids = Object.keys(rounds[0].experts);
          }
          if (!sfids.length) { container.textContent = '(No expert histories found in log)'; return; }
          sfids.sort();
          let totalLen = 0;
          for (const sfid of sfids) {
            const hdr = document.createElement('div');
            hdr.className = 'msg-header';
            hdr.textContent = `Expert ${sfid}`;
            container.appendChild(hdr);
            const pre = document.createElement('pre');
            pre.style.maxHeight = '200px';
            pre.style.overflow = 'auto';
            const msgs = Array.isArray(histExperts[sfid]) ? histExperts[sfid] : [];
            // HTML-escape helper
            const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
            let html = '';
            for (let i = 0; i < msgs.length; i++) {
              const m = msgs[i];
              const role = (m && typeof m.role === 'string') ? m.role : '';
              let who = '';
              // First message: [PROMPT] red; Expert (assistant): black; Mediator (user): blue
              let color = 'black';
              if (i === 0) { who = '[PROMPT]'; color = 'red'; }
              else if (role === 'assistant') { who = '[EXPERT]'; color = 'black'; }
              else if (role === 'user') { who = '[MEDIATOR]'; color = 'blue'; }
              else { who = `[${(role||'').toString().toUpperCase()}]`; color = '#444'; }
              let content = '';
              if (m && typeof m.content === 'string') content = m.content; else content = String(m?.content ?? '');
              content = content.replace(/\u0000/g, '\\0');
              html += `<span style="color:${color}"><strong>${esc(who)}</strong> ${esc(content)}</span>\n\n`;
              totalLen += content.length;
              if (totalLen > 60000) break;
            }
            pre.innerHTML = html || '(No conversation history)';
            container.appendChild(pre);
            if (totalLen > 60000) { const cut = document.createElement('div'); cut.textContent = '... (truncated)'; container.appendChild(cut); break; }
          }
        } catch (e) {
          container.textContent = '(Failed to render expert conversations)';
          console.error(e);
        }
      }

      async function loadAvgBrierRounds() {
        try {
          const r = await fetchJSON('/api/avg_brier_rounds_all');
          const showTrain = document.getElementById('chkTrain').checked;
          const showVal = document.getElementById('chkVal').checked;
          const showTest = document.getElementById('chkTest').checked;
          const datasets = [];
          const xs = (r.val?.rounds || r.train?.rounds || r.test?.rounds || []).map(i => `R${(i??0)+1}`);
          function add(split, color) {
            const obj = r[split]; if (!obj) return;
            datasets.push({ label: `${split} (n=${obj.n_questions||0})`, data: obj.avg_brier || [], borderColor: color, tension: 0.2 });
            // Baselines: SF and Public (flat across rounds)
            const n = xs.length;
            if (obj.sf_avg_brier != null) {
              const arr = Array(n).fill(obj.sf_avg_brier);
              datasets.push({ label: `${split} SF median`, data: arr, borderColor: color, borderDash: [6,4], tension: 0.0 });
            }
            if (obj.public_avg_brier != null) {
              const arr = Array(n).fill(obj.public_avg_brier);
              datasets.push({ label: `${split} Public median`, data: arr, borderColor: color, borderDash: [2,3], tension: 0.0 });
            }
          }
          if (showTrain) add('train', 'blue');
          if (showVal) add('val', 'orange');
          if (showTest) add('test', 'green');
          if (avgBrierChart) avgBrierChart.destroy();
          avgBrierChart = new Chart(avgBrierCtx, { type: 'line', data: { labels: xs, datasets }, options: { responsive: true, animation: false, scales: { y: { reverse: false, title: { display: true, text: 'Avg Brier (lower is better)' } } } } });
        } catch (e) {
          showError('Failed to load avg brier by round: ' + (e?.message || e));
        }
      }

      // Init
      (async () => {
        await loadRuns();
        await loadSummary();
        await populateExplorer();
        await loadAvgBrierRounds();
        document.getElementById('metricsSelect').addEventListener('change', loadMetrics);
        document.getElementById('switchRunBtn').addEventListener('click', switchRun);
        document.getElementById('refreshBrierBtn').addEventListener('click', loadAvgBrierRounds);
        document.getElementById('chkTrain').addEventListener('change', loadAvgBrierRounds);
        document.getElementById('chkVal').addEventListener('change', loadAvgBrierRounds);
        document.getElementById('chkTest').addEventListener('change', loadAvgBrierRounds);
      })();
    </script>
  </body>
  </html>
"""


@app.get("/")
def index():
    # Serve the single-page dashboard
    return Response(INDEX_HTML, mimetype="text/html")


def main():
    parser = argparse.ArgumentParser(description="Evolution Flask dashboard")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to a run directory or its parent (seed dir)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Root folder to discover selectable runs (defaults near --run-dir)",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    global RUN_DIR
    RUN_DIR = args.run_dir.resolve()
    if not RUN_DIR.exists():
        raise SystemExit(f"Run dir not found: {RUN_DIR}")
    # If a seed or base dir was given, auto-pick the latest child run dir
    if not _is_run_dir(RUN_DIR):
        picked = _pick_latest_child_run(RUN_DIR)
        if picked is not None:
            print(f"Auto-selected latest run dir: {picked}")
            RUN_DIR = picked

    # Configure root for run discovery
    global RUNS_ROOT
    if args.runs_root is not None:
        RUNS_ROOT = args.runs_root.resolve()
    else:
        # If given a run dir, use its parent; if a seed/base, use itself
        RUNS_ROOT = RUN_DIR.parent if _is_run_dir(RUN_DIR) else RUN_DIR

    print(f"Serving dashboard for run: {RUN_DIR}")
    print(f"Runs root: {RUNS_ROOT}")
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
