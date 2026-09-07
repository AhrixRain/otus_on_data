#!/usr/bin/env python
"""Live cms_Joint dashboard and periodic plot generator.

This is a dependency-free W&B-style dashboard for the cms_Joint training run.
It serves a local webpage at http://127.0.0.1:7878 and watches history.json.
Every `--eval-every` epochs it automatically generates the latest/best model
plots using the existing joint plot scripts.

Usage:
    python scripts_joint/dashboard.py --run-dir outputs/cms_Joint/<Run> --device cuda
"""


from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_JOINT = REPO_ROOT / "scripts_joint"
DEFAULT_PORT = 7878

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>cms_Joint Dashboard</title>
<style>
:root { color-scheme: dark; }
body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 16px; }
h1 { font-size: 1.3rem; }
h2 { font-size: 1rem; margin-top: 1.2rem; color: #94a3b8; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 12px; }
.mono { font-family: ui-monospace, monospace; font-size: 0.85rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
th, td { border: 1px solid #334155; padding: 4px 6px; text-align: left; }
canvas { width: 100%; height: 220px; }
img { max-width: 100%; border: 1px solid #334155; border-radius: 6px; margin: 4px 0; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; }
.ok { background: #065f46; color: #a7f3d0; }
.fail { background: #7f1d1d; color: #fecaca; }
.run { background: #1e3a8a; color: #bfdbfe; }
#status { font-size: 0.9rem; }
</style>
</head>
<body>
<h1 id="title">cms_Joint Dashboard</h1>
<div id="status" class="card mono">Waiting for training data...</div>
<div class="grid">
  <div class="card"><h2>Total loss</h2><canvas id="loss_total"></canvas></div>
  <div class="card"><h2>Region loss</h2><canvas id="loss_region"></canvas></div>
  <div class="card"><h2>Raw worst-region score</h2><canvas id="selection"></canvas></div>
  <div class="card"><h2>Gate-pass history</h2><canvas id="gates"></canvas></div>
  <div class="card"><h2>Noise / LR</h2><canvas id="noise"></canvas></div>
</div>
<div class="grid">
  <div class="card"><h2>Latest validation</h2><div id="validation"></div></div>
  <div class="card"><h2>Recent epochs</h2><div id="recent"></div></div>
</div>
<div class="card"><h2>Generated plots</h2><div id="plots">No plots yet.</div></div>
<script>
const $ = id => document.getElementById(id);
const state = { history: [], plots: [] };

async function refresh() {
  try {
    const r = await fetch('/api/state');
    const data = await r.json();
    state.history = data.history || [];
    state.plots = data.plots || [];
    renderStatus(data);
    renderLoss();
    renderSelection();
    renderNoise();
    renderValidation(data);
    renderRecent(data);
    renderPlots();
  } catch (e) {
    $('status').textContent = 'Waiting for server/training to start...';
  }
}

function renderStatus(d) {
  const runLabel = d.run_label || 'cms_Joint';
  document.title = runLabel + ' Dashboard';
  const titleEl = document.getElementById('title');
  if (titleEl) titleEl.textContent = runLabel + ' Dashboard';
  const h = d.history;
  const last = h.length ? h[h.length-1] : null;
  const best = d.best;
  let html = `Run: ${runLabel} | Port: ${d.port} | Polling: ${d.interval}s`;
  if (last) {
    html += `<br>Epoch: <b>${last.global_epoch}</b> | Stage: ${last.stage} | LR: ${Number(last.lr).toExponential(2)}`;
    if (last.joint_selection) {
      const ok = last.joint_selection.all_gates_passed;
      html += ` | Gates: <span class="badge ${ok ? 'ok' : 'fail'}">${ok ? 'PASS' : 'FAIL'}</span>`;
      html += ` | Score: ${Number(last.joint_selection.selection_score).toFixed(4)}`;
    }
  }
  if (best) {
    html += `<br>Best checkpoint: epoch ${best.global_epoch} | ${best.stage} | score ${Number(best.selection_score).toFixed(4)}`;
  }
  $('status').innerHTML = html;
}

function drawChart(canvasId, labels, seriesList) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(300, rect.width) * dpr;
  canvas.height = 220 * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = canvas.width / dpr, H = 220;
  ctx.clearRect(0,0,W,H);
  if (!labels.length) return;
  const allVals = seriesList.flatMap(s => s.values.filter(v => Number.isFinite(v)));
  if (!allVals.length) return;
  let min = Math.min(...allVals), max = Math.max(...allVals);
  if (min === max) { min -= 1; max += 1; }
  const pad = 10;
  const x = i => labels.length === 1 ? W/2 : pad + (W - 2*pad) * i / (labels.length - 1);
  const y = v => H - pad - (H - 2*pad) * (v - min) / (max - min);
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
  for (let i=0;i<=4;i++) {
    const yy = pad + (H - 2*pad) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(W-pad, yy); ctx.stroke();
  }
  const colors = ['#38bdf8','#34d399','#fbbf24','#f472b6','#a78bfa'];
  seriesList.forEach((s, si) => {
    ctx.strokeStyle = colors[si % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.values.forEach((v, i) => {
      const xx = x(i), yy = y(v);
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
    ctx.fillStyle = colors[si % colors.length];
    ctx.font = '11px system-ui';
    ctx.fillText(s.name, pad + 4, 16 + si * 14);
  });
}

function renderLoss() {
  const h = state.history;
  const labels = h.map(r => r.global_epoch);
  drawChart('loss_total', labels, [
    { name: 'total', values: h.map(r => r.train && r.train.loss) },
  ]);
  drawChart('loss_region', labels, [
    { name: 'jpsi', values: h.map(r => r.train && r.train.jpsi_loss) },
    { name: 'z', values: h.map(r => r.train && r.train.z_loss) },
  ]);
}

function renderSelection() {
  const rows = state.history.filter(r => r.joint_selection);
  const labels = rows.map(r => r.global_epoch);
  // The penalized selection score jumps by gate_fail_penalty (1e6) whenever a
  // gate fails, which flattens the raw curve to a straight line if the two are
  // drawn on one axis. Plot the raw score alone and show gate state separately.
  drawChart('selection', labels, [
    { name: 'raw', values: rows.map(r => r.joint_selection.raw_worst_region_score) },
  ]);
  drawChart('gates', labels, [
    { name: 'all gates', values: rows.map(r => r.joint_selection.all_gates_passed ? 1 : 0) },
  ]);
}

function renderNoise() {
  const h = state.history;
  const labels = h.map(r => r.global_epoch);
  drawChart('noise', labels, [
    { name: 'core', values: h.map(r => r.core_noise_multiplier) },
    { name: 'tail', values: h.map(r => r.tail_noise_multiplier) },
    { name: 'lr', values: h.map(r => r.lr) },
  ]);
}

function renderValidation(d) {
  const rows = d.history.filter(r => r.joint_selection);
  const last = rows.length ? rows[rows.length-1] : null;
  if (!last) { $('validation').textContent = 'No validation yet.'; return; }
  const sel = last.joint_selection;
  let html = `<div class="mono">Epoch ${last.global_epoch} | ${last.stage}</div>`;
  html += `<table><tr><th>Region</th><th>Worst metric</th><th>Score</th><th>Gates</th></tr>`;
  for (const [name, rep] of Object.entries(sel.regions)) {
    html += `<tr><td>${name}</td><td>${rep.worst_metric}</td><td>${Number(rep.score).toFixed(4)}</td><td>${rep.gate_passed ? 'PASS' : 'FAIL'}</td></tr>`;
  }
  html += `</table>`;
  $('validation').innerHTML = html;
}

function renderRecent(d) {
  const h = d.history;
  const recent = h.slice(-20).reverse();
  let html = '<table><tr><th>Epoch</th><th>Stage</th><th>Loss</th><th>Gates</th></tr>';
  for (const r of recent) {
    const ok = r.joint_selection ? (r.joint_selection.all_gates_passed ? 'PASS' : 'FAIL') : '—';
    html += `<tr><td>${r.global_epoch}</td><td>${r.stage}</td><td>${r.train ? Number(r.train.loss).toFixed(4) : '—'}</td><td>${ok}</td></tr>`;
  }
  html += '</table>';
  $('recent').innerHTML = html;
}

function renderPlots() {
  const plots = state.plots;
  if (!plots.length) { $('plots').textContent = 'No plots yet. They will appear after every eval checkpoint.'; return; }
  let html = '';
  for (const p of plots) {
    html += `<div style="margin-bottom:12px"><div class="mono">${p.label}</div>`;
    for (const img of p.images) {
      html += `<a href="${img.url}" target="_blank"><img src="${img.url}" alt="${img.name}"></a>`;
    }
    html += '</div>';
  }
  $('plots').innerHTML = html;
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, dashboard=None, **kwargs):
        self.dashboard = dashboard
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            payload = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/api/state":
            payload = json.dumps(self.dashboard.state()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.startswith("/plots/"):
            relative = self.path[len("/plots/"):]
            file_path = (self.dashboard.dashboard_dir / relative).resolve()
            if not str(file_path).startswith(str(self.dashboard.dashboard_dir.resolve())):
                self.send_error(403)
                return
            try:
                data = file_path.read_bytes()
            except OSError:
                self.send_error(404)
                return
            ctype = "image/png" if file_path.suffix == ".png" else "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def log_message(self, fmt, *args):
        pass


class Dashboard:
    def __init__(self, run_dir: Path, config: Path, device: str, port: int,
                 interval: float, eval_every: int, auto_eval: bool,
                 run_label: str | None = None):
        self.run_dir = run_dir
        self.config = config
        self.device = device
        self.port = port
        self.interval = interval
        self.eval_every = eval_every
        self.auto_eval = auto_eval
        self.run_label = run_label or run_dir.name
        self.dashboard_dir = run_dir / "dashboard"
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = run_dir / "history.json"
        self.last_epoch = 0
        self._lock = threading.Lock()
        self._eval_lock = threading.Lock()
        self._plot_index: list[dict] = []

    def load_history(self) -> list[dict]:
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def state(self) -> dict:
        history = self.load_history()
        best = None
        for row in reversed(history):
            sel = row.get("joint_selection") or {}
            if sel.get("all_gates_passed"):
                best = {
                    "global_epoch": row.get("global_epoch"),
                    "stage": row.get("stage"),
                    "selection_score": sel.get("selection_score"),
                }
                break
        return {
            "run_label": self.run_label,
            "port": self.port,
            "interval": self.interval,
            "history": history,
            "best": best,
            "plots": self._plot_index,
        }

    def monitor(self) -> None:
        while True:
            history = self.load_history()
            if history:
                latest_epoch = int(history[-1]["global_epoch"])
                with self._lock:
                    new_epochs = [
                        row["global_epoch"] for row in history
                        if int(row["global_epoch"]) > self.last_epoch
                    ]
                    if new_epochs:
                        self.last_epoch = latest_epoch
                if self.auto_eval:
                    for epoch in new_epochs:
                        if int(epoch) % self.eval_every == 0:
                            self.trigger_eval(int(epoch))
            time.sleep(self.interval)

    def trigger_eval(self, epoch: int) -> None:
        if not self._eval_lock.acquire(blocking=False):
            return
        try:
            marker = self.dashboard_dir / f".done_{epoch}.json"
            if marker.exists():
                return
            latest_ckpt = self.run_dir / "last_model.pt"
            best_ckpt = self.run_dir / "best_model.pt"
            tasks = []
            if latest_ckpt.exists():
                tasks.append(("latest", latest_ckpt, epoch))
            if best_ckpt.exists():
                tasks.append(("best", best_ckpt, epoch))
            all_ok = True
            for label, ckpt, ep in tasks:
                out_dir = self.dashboard_dir / f"{label}_epoch_{ep}"
                if (out_dir / "plot_manifest.json").exists():
                    continue
                if not self._run_plot_scripts(ckpt, out_dir, ep, label):
                    all_ok = False
            if all_ok:
                marker.write_text(json.dumps({"epoch": epoch, "time": time.time()}), encoding="utf-8")
            else:
                print(f"[dashboard] Plot generation incomplete for epoch {epoch}; will retry.")
        finally:
            self._eval_lock.release()

    def _run_plot_scripts(self, checkpoint: Path, output_dir: Path, epoch: int, label: str) -> bool:
        print(f"[dashboard] Generating {label} plots for epoch {epoch} -> {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        ok = True
        cmds = [
            [
                sys.executable,
                SCRIPTS_JOINT / "plot_joint_run.py",
                "--config", self.config,
                "--checkpoint", checkpoint,
                "--output-dir", output_dir / "elements",
                "--device", self.device,
            ],
            [
                sys.executable,
                SCRIPTS_JOINT / "plot_joint_paperstyle.py",
                "--config", self.config,
                "--checkpoint", checkpoint,
                "--output-dir", output_dir / "paperstyle",
                "--device", self.device,
            ],
        ]
        env = os.environ.copy()
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("OMP_NUM_THREADS", "1")
        for cmd in cmds:
            try:
                subprocess.check_call([str(item) for item in cmd], env=env)
            except Exception as exc:
                ok = False
                print(f"[dashboard] Plot command failed for epoch {epoch}: {exc}")
        self._refresh_plot_index()
        return ok

    def _refresh_plot_index(self) -> None:
        entries = []
        for folder in sorted(self.dashboard_dir.iterdir(), reverse=True):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            images = []
            for sub in ("elements", "paperstyle"):
                subdir = folder / sub
                if subdir.is_dir():
                    for png in sorted(subdir.glob("*.png"))[:24]:
                        rel = folder.name + "/" + sub + "/" + png.name
                        images.append({"name": png.name, "url": "/plots/" + rel})
            if images:
                entries.append({"label": folder.name, "images": images})
        self._plot_index = entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--label", default=None, help="Run label shown in the dashboard; defaults to run-dir name.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--no-auto-eval", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config = args.config.expanduser().resolve()
    dashboard = Dashboard(
        run_dir=run_dir,
        config=config,
        device=args.device,
        port=args.port,
        interval=args.interval,
        eval_every=max(1, args.eval_every),
        auto_eval=not args.no_auto_eval,
        run_label=args.label,
    )

    handler = lambda *hargs, **hkwargs: DashboardHandler(
        *hargs, dashboard=dashboard, **hkwargs
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler)

    monitor_thread = threading.Thread(target=dashboard.monitor, daemon=True)
    monitor_thread.start()

    url = f"http://127.0.0.1:{args.port}"
    print(f"{dashboard.run_label} dashboard: {url}")
    print(f"Watching {dashboard.history_path}")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
