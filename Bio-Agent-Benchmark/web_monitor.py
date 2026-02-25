#!/usr/bin/env python3
"""
Web-based real-time monitor for Biomni benchmark logs.

Usage:
    python web_monitor.py /tmp/biomni_streaming_run.log --port 8080
"""

import argparse
import os
import threading
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from monitor import Monitor, parse_log_line  # pyright: ignore[reportImplicitRelativeImport]


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Biomni Benchmark Monitor</title>
  <style>
    :root {
      --bg: #0d1117;
      --bg-soft: #111722;
      --panel: #141b27;
      --panel-2: #1a2331;
      --line: #263246;
      --text: #d6deeb;
      --muted: #90a0b8;
      --green: #3ddc84;
      --red: #ff5c7c;
      --amber: #f6c549;
      --cyan: #4fd4ff;
      --blue: #66b3ff;
      --shadow: 0 10px 35px rgba(0, 0, 0, 0.35);
      --mono: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      padding: 20px;
      color: var(--text);
      font-family: var(--mono);
      background:
        radial-gradient(1100px 600px at 90% -10%, rgba(79, 212, 255, 0.08), transparent 60%),
        radial-gradient(900px 500px at -10% 120%, rgba(61, 220, 132, 0.06), transparent 60%),
        linear-gradient(180deg, #0b1018 0%, #0d1117 100%);
      min-height: 100vh;
      letter-spacing: 0.1px;
    }

    .container {
      max-width: 1320px;
      margin: 0 auto;
      display: grid;
      gap: 14px;
    }

    .panel {
      background: linear-gradient(180deg, rgba(30, 40, 58, 0.25), rgba(12, 18, 29, 0.8));
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: var(--shadow);
      padding: 14px;
    }

    .header {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px;
      border-left: 4px solid var(--cyan);
    }

    h1 {
      margin: 0;
      font-size: clamp(18px, 2.6vw, 28px);
      color: #eff6ff;
      text-transform: uppercase;
      letter-spacing: 0.7px;
    }

    .header-meta {
      display: flex;
      gap: 18px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }

    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 6px;
      background: var(--red);
      box-shadow: 0 0 0 0 rgba(255, 92, 124, 0.6);
      animation: pulseDot 2s infinite;
    }

    .status-dot.connected {
      background: var(--green);
      box-shadow: 0 0 0 0 rgba(61, 220, 132, 0.6);
    }

    @keyframes pulseDot {
      0% { box-shadow: 0 0 0 0 rgba(255, 92, 124, 0.6); }
      70% { box-shadow: 0 0 0 9px rgba(255, 92, 124, 0); }
      100% { box-shadow: 0 0 0 0 rgba(255, 92, 124, 0); }
    }

    .status-dot.connected {
      animation-name: pulseDotGreen;
    }

    @keyframes pulseDotGreen {
      0% { box-shadow: 0 0 0 0 rgba(61, 220, 132, 0.55); }
      70% { box-shadow: 0 0 0 9px rgba(61, 220, 132, 0); }
      100% { box-shadow: 0 0 0 0 rgba(61, 220, 132, 0); }
    }

    .progress-wrap {
      padding: 10px 4px 4px;
    }

    .progress-track {
      position: relative;
      width: 100%;
      height: 28px;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #314158;
      background: linear-gradient(180deg, #0f1724, #0c121d);
    }

    .progress-fill {
      height: 100%;
      width: 0%;
      transition: width 0.45s ease;
      background:
        repeating-linear-gradient(
          -45deg,
          rgba(61, 220, 132, 0.8),
          rgba(61, 220, 132, 0.8) 10px,
          rgba(53, 194, 116, 0.9) 10px,
          rgba(53, 194, 116, 0.9) 20px
        );
      box-shadow: inset 0 0 12px rgba(0, 0, 0, 0.35);
    }

    .progress-label {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      color: #f4faff;
      text-shadow: 0 1px 0 rgba(0, 0, 0, 0.4);
      letter-spacing: 0.5px;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      gap: 10px;
    }

    .card {
      background: linear-gradient(180deg, rgba(22, 32, 48, 0.9), rgba(17, 24, 36, 0.95));
      border: 1px solid #2a374c;
      border-radius: 8px;
      padding: 12px;
      min-height: 92px;
      display: grid;
      gap: 6px;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .card:hover {
      transform: translateY(-1px);
      border-color: #3a4e6c;
    }

    .card .k {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }

    .card .v {
      font-size: clamp(18px, 2.4vw, 30px);
      font-weight: 600;
      transition: color 0.3s ease;
    }

    .ok .v { color: var(--green); }
    .bad .v { color: var(--red); }
    .lat .v { color: var(--cyan); }
    .ttft .v { color: var(--amber); }
    .acc .v { color: #c084fc; }

    .sub-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 10px;
    }

    .kv {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      font-size: 13px;
    }

    .kv .label { color: var(--muted); }
    .kv .value { color: #f4faff; font-weight: 600; }

    .section-title {
      margin: 0 0 10px;
      font-size: 13px;
      color: #f4faff;
      text-transform: uppercase;
      letter-spacing: 0.7px;
    }

    .workers-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 10px;
    }

    .worker-card {
      border: 1px solid #2e3f58;
      border-radius: 8px;
      background: linear-gradient(180deg, rgba(22, 30, 43, 0.9), rgba(13, 19, 30, 0.95));
      padding: 10px;
      display: grid;
      gap: 7px;
      position: relative;
      overflow: hidden;
    }

    .worker-card.generating {
      border-color: #3f7ec6;
      animation: pulseWorker 1.8s infinite;
    }

    .worker-card.stalled {
      border-color: #9f3045;
    }

    @keyframes pulseWorker {
      0% { box-shadow: 0 0 0 0 rgba(102, 179, 255, 0.30); }
      70% { box-shadow: 0 0 0 12px rgba(102, 179, 255, 0); }
      100% { box-shadow: 0 0 0 0 rgba(102, 179, 255, 0); }
    }

    .worker-top {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }

    .badge {
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }

    .badge.generating { color: #9bd2ff; border-color: #4979aa; }
    .badge.waiting { color: #ffe08b; border-color: #8f7440; }
    .badge.stalled { color: #ff9fb2; border-color: #8d3e50; animation: pulseBadge 1.4s infinite; }

    @keyframes pulseBadge {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.56; }
    }

    .worker-line {
      font-size: 12px;
      color: #d4deed;
      word-break: break-word;
    }

    .dist-wrap {
      display: grid;
      gap: 8px;
    }

    .dist-row {
      display: grid;
      grid-template-columns: 70px 1fr 42px;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }

    .dist-label { color: var(--muted); }

    .dist-track {
      position: relative;
      height: 14px;
      border-radius: 5px;
      border: 1px solid #2a374a;
      background: #0f1521;
      overflow: hidden;
    }

    .dist-fill {
      position: absolute;
      inset: 0 auto 0 0;
      width: 0;
      transition: width 0.4s ease;
      background: linear-gradient(90deg, #4fd4ff, #3ddc84);
    }

    .dist-value { text-align: right; color: #e7f2ff; }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }

    th, td {
      border-bottom: 1px solid #2a3648;
      padding: 7px 8px;
      text-align: left;
    }

    th {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-size: 11px;
    }

    td.num {
      text-align: right;
      color: #f4faff;
    }

    details {
      border: 1px solid #4a2833;
      border-radius: 8px;
      background: linear-gradient(180deg, rgba(42, 18, 24, 0.55), rgba(26, 11, 15, 0.7));
      padding: 8px 10px;
    }

    summary {
      cursor: pointer;
      color: #ffafc0;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      user-select: none;
    }

    .error-item {
      margin-top: 8px;
      border: 1px solid #663542;
      background: rgba(29, 12, 17, 0.65);
      border-radius: 6px;
      padding: 8px;
      font-size: 12px;
      color: #ffd3dd;
      word-break: break-word;
    }

    .hidden { display: none !important; }

    .muted-empty {
      color: var(--muted);
      font-size: 12px;
    }

    @media (max-width: 960px) {
      body { padding: 12px; }
      .stats-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      .sub-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 560px) {
      .stats-grid { grid-template-columns: 1fr; }
      .header-meta { width: 100%; justify-content: space-between; }
    }
  </style>
</head>
<body>
  <div class="container">
    <section class="panel header">
      <h1>Biomni Benchmark Monitor</h1>
      <div class="header-meta">
        <div>elapsed: <span id="elapsed">0.0s</span></div>
        <div><span id="statusDot" class="status-dot"></span><span id="statusText">Disconnected</span></div>
      </div>
    </section>

    <section class="panel progress-wrap">
      <div class="progress-track">
        <div id="progressFill" class="progress-fill"></div>
        <div id="progressLabel" class="progress-label">0/0 (0.0%)</div>
      </div>
    </section>

    <section class="stats-grid">
      <div class="card ok"><div class="k">Completed</div><div id="completed" class="v">0</div></div>
      <div class="card bad"><div class="k">Errors</div><div id="errors" class="v">0</div></div>
      <div class="card acc"><div class="k">Accuracy</div><div id="accuracy" class="v">-</div><div style="color:var(--muted);font-size:11px;"><span id="correctCount">0</span>/<span id="totalScored">0</span> correct</div></div>
      <div class="card lat"><div class="k">Avg Latency</div><div id="avgLatency" class="v">0.00s</div></div>
      <div class="card ttft"><div class="k">Avg TTFT</div><div id="avgTTFT" class="v">0.00s</div></div>
    </section>

    <section class="sub-grid">
      <div class="panel">
        <div class="section-title">Pipeline Pace</div>
        <div class="kv"><span class="label">ETA</span><span id="eta" class="value">0s</span></div>
        <div class="kv"><span class="label">Throughput</span><span id="throughput" class="value">0.00 tasks/min</span></div>
      </div>
      <div class="panel">
        <div class="section-title">Token Budget</div>
        <div class="kv"><span class="label">Prompt Tokens</span><span id="promptTokens" class="value">0</span></div>
        <div class="kv"><span class="label">Completion Tokens</span><span id="completionTokens" class="value">0</span></div>
      </div>
      <div class="panel">
        <div class="section-title">Slowest Task</div>
        <div class="kv"><span class="label">Task ID</span><span id="slowestTask" class="value">-</span></div>
        <div class="kv"><span class="label">Latency</span><span id="slowestLatency" class="value">0.00s</span></div>
      </div>
    </section>

    <section class="panel">
      <h2 class="section-title">Active Workers</h2>
      <div id="workers" class="workers-grid"></div>
    </section>

    <section class="panel">
      <h2 class="section-title">Latency Distribution</h2>
      <div id="latencyDist" class="dist-wrap"></div>
    </section>

    <section class="panel">
      <h2 class="section-title">Task Type Breakdown</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Task Type</th>
              <th style="text-align:right;">Count</th>
              <th style="text-align:right;">Accuracy</th>
              <th style="text-align:right;">Correct</th>
              <th style="text-align:right;">Avg Latency (s)</th>
            </tr>
          </thead>
          <tbody id="taskTableBody"></tbody>
        </table>
      </div>
    </section>

    <section id="errorsSection" class="hidden">
      <details open>
        <summary>Recent Errors</summary>
        <div id="errorsList"></div>
      </details>
    </section>
  </div>

  <script>
    const state = {
      pollMs: 2000,
      connected: false,
      last: {}
    };

    function setConnection(ok) {
      state.connected = ok;
      const dot = document.getElementById('statusDot');
      const text = document.getElementById('statusText');
      if (ok) {
        dot.classList.add('connected');
        text.textContent = 'Connected';
      } else {
        dot.classList.remove('connected');
        text.textContent = 'Disconnected';
      }
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = String(text);
      return div.innerHTML;
    }

    function formatNumber(n, decimals) {
      const value = Number(n || 0);
      if (decimals > 0) {
        return value.toFixed(decimals);
      }
      return Math.round(value).toString();
    }

    function formatInt(n) {
      return Math.round(Number(n || 0)).toLocaleString();
    }

    function formatSeconds(sec) {
      const value = Number(sec || 0);
      if (value >= 3600) {
        const h = Math.floor(value / 3600);
        const m = Math.floor((value % 3600) / 60);
        const s = Math.floor(value % 60);
        return h + 'h ' + m + 'm ' + s + 's';
      }
      if (value >= 60) {
        const m = Math.floor(value / 60);
        const s = Math.floor(value % 60);
        return m + 'm ' + s + 's';
      }
      return value.toFixed(1) + 's';
    }

    function animateValue(id, nextValue, decimals, suffix, integerWithCommas) {
      const el = document.getElementById(id);
      if (!el) {
        return;
      }
      const startValue = Number(el.dataset.value || 0);
      const endValue = Number(nextValue || 0);
      const duration = 450;
      const startedAt = performance.now();

      function draw(now) {
        const ratio = Math.min((now - startedAt) / duration, 1);
        const eased = 1 - Math.pow(1 - ratio, 3);
        const current = startValue + (endValue - startValue) * eased;
        if (integerWithCommas) {
          el.textContent = formatInt(current);
        } else {
          el.textContent = formatNumber(current, decimals) + (suffix || '');
        }
        if (ratio < 1) {
          requestAnimationFrame(draw);
        }
      }

      requestAnimationFrame(draw);
      el.dataset.value = String(endValue);
    }

    function renderProgress(stats) {
      const done = Number(stats.completed || 0) + Number(stats.errors || 0);
      const total = Number(stats.total_tasks || 0);
      const pct = total > 0 ? (done / total) * 100 : 0;
      const fill = document.getElementById('progressFill');
      const label = document.getElementById('progressLabel');
      fill.style.width = Math.max(0, Math.min(100, pct)).toFixed(1) + '%';
      label.textContent = done + '/' + total + ' (' + pct.toFixed(1) + '%)';
    }

    function renderWorkers(stats) {
      const wrap = document.getElementById('workers');
      const workers = Array.isArray(stats.active_workers) ? stats.active_workers : [];
      if (!workers.length) {
        wrap.innerHTML = '<div class="muted-empty">No active workers.</div>';
        return;
      }

      const cards = workers.map(function (worker) {
        const status = worker.status || 'waiting';
        const cardClass = 'worker-card ' + escapeHtml(status);
        return (
          '<div class="' + cardClass + '">' +
            '<div class="worker-top">' +
              '<strong>Worker ' + escapeHtml(worker.worker || '-') + '</strong>' +
              '<span class="badge ' + escapeHtml(status) + '">' + escapeHtml(status) + '</span>' +
            '</div>' +
            '<div class="worker-line">task: ' + escapeHtml(worker.task_id || '-') + '</div>' +
            '<div class="worker-line">elapsed: ' + formatSeconds(worker.elapsed || 0) + '</div>' +
            '<div class="worker-line">tokens: ' + formatInt(worker.tokens || 0) + ' <span style="color:#c9a0dc;font-size:0.85em">(reasoning: ' + formatInt(worker.reasoning_tokens || 0) + ')</span></div>' +
            '<div class="worker-line">ttft: ' + (worker.ttft == null ? '-' : Number(worker.ttft).toFixed(2) + 's') + '</div>' +
            '<div class="worker-line">prompt chars: ' + formatInt(worker.prompt_chars || 0) + '</div>' +
          '</div>'
        );
      });

      wrap.innerHTML = cards.join('');
    }

    function renderLatencyDist(stats) {
      const dist = stats.latency_distribution || {};
      const order = ['0-5s', '5-10s', '10-30s', '30-60s', '1-5m', '5-10m', '10m+'];
      const values = order.map(function (key) { return Number(dist[key] || 0); });
      const maxValue = Math.max(1, ...values);
      const rows = order.map(function (bucket, idx) {
        const val = values[idx];
        const width = (val / maxValue) * 100;
        return (
          '<div class="dist-row">' +
            '<div class="dist-label">' + bucket + '</div>' +
            '<div class="dist-track"><div class="dist-fill" style="width:' + width.toFixed(1) + '%"></div></div>' +
            '<div class="dist-value">' + val + '</div>' +
          '</div>'
        );
      });
      document.getElementById('latencyDist').innerHTML = rows.join('');
    }

    function renderTaskTable(stats) {
      const tbody = document.getElementById('taskTableBody');
      const entries = Object.entries(stats.task_type_stats || {});
      if (!entries.length) {
        tbody.innerHTML = '<tr><td colspan="3" class="muted-empty">No completed tasks yet.</td></tr>';
        return;
      }

      entries.sort(function (a, b) {
        const countA = Number((a[1] || {}).count || 0);
        const countB = Number((b[1] || {}).count || 0);
        return countB - countA;
      });

      tbody.innerHTML = entries.map(function (entry) {
        const name = entry[0];
        const data = entry[1] || {};
        const accDisplay = data.accuracy != null ? (data.accuracy * 100).toFixed(1) + '%' : '-';
        const correctDisplay = (data.correct || 0) + '/' + (data.scored || 0);
        const accColor = data.accuracy != null ? (data.accuracy >= 0.5 ? 'var(--green)' : 'var(--red)') : 'var(--muted)';
        return (
          '<tr>' +
            '<td>' + escapeHtml(name) + '</td>' +
            '<td class="num">' + formatInt(data.count || 0) + '</td>' +
            '<td class="num" style="color:' + accColor + '">' + accDisplay + '</td>' +
            '<td class="num">' + correctDisplay + '</td>' +
            '<td class="num">' + Number(data.avg_latency || 0).toFixed(2) + '</td>' +
          '</tr>'
        );
      }).join('');
    }

    function renderErrors(stats) {
      const section = document.getElementById('errorsSection');
      const list = document.getElementById('errorsList');
      const errors = Array.isArray(stats.recent_errors) ? stats.recent_errors : [];
      if (!errors.length || Number(stats.errors || 0) === 0) {
        section.classList.add('hidden');
        list.innerHTML = '';
        return;
      }

      section.classList.remove('hidden');
      list.innerHTML = errors.map(function (item) {
        return (
          '<div class="error-item">' +
            '<strong>' + escapeHtml(item.task_id || '-') + '</strong> — ' +
            escapeHtml(item.error || '') +
          '</div>'
        );
      }).join('');
    }

    function renderStats(stats) {
      renderProgress(stats);

      animateValue('completed', stats.completed || 0, 0, '', true);
      animateValue('errors', stats.errors || 0, 0, '', true);
      animateValue('avgLatency', stats.avg_latency || 0, 2, 's', false);
      animateValue('avgTTFT', stats.avg_ttft || 0, 2, 's', false);

      // Accuracy card
      var accEl = document.getElementById('accuracy');
      if (stats.accuracy != null) {
        accEl.textContent = (stats.accuracy * 100).toFixed(1) + '%';
        accEl.style.color = stats.accuracy >= 0.5 ? '#c084fc' : '#ff5c7c';
      } else {
        accEl.textContent = '-';
      }
      document.getElementById('correctCount').textContent = formatInt(stats.correct_count || 0);
      document.getElementById('totalScored').textContent = formatInt(stats.total_scored || 0);

      document.getElementById('elapsed').textContent = formatSeconds(stats.elapsed || 0);
      document.getElementById('eta').textContent = formatSeconds(stats.eta || 0);
      document.getElementById('throughput').textContent = Number(stats.throughput || 0).toFixed(2) + ' tasks/min';
      document.getElementById('promptTokens').textContent = formatInt(stats.total_prompt_tokens || 0);
      document.getElementById('completionTokens').textContent = formatInt(stats.total_completion_tokens || 0);

      const slowestTask = stats.slowest_task || {};
      document.getElementById('slowestTask').textContent = slowestTask.id || '-';
      document.getElementById('slowestLatency').textContent = Number(slowestTask.latency || 0).toFixed(2) + 's';

      renderWorkers(stats);
      renderLatencyDist(stats);
      renderTaskTable(stats);
      renderErrors(stats);
    }

    async function pollStats() {
      try {
        const response = await fetch('/api/stats', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error('HTTP ' + response.status);
        }
        const stats = await response.json();
        setConnection(true);
        renderStats(stats);
        state.last = stats;
      } catch (err) {
        setConnection(false);
      }
    }

    pollStats();
    setInterval(pollStats, state.pollMs);
  </script>
</body>
</html>
"""


class DashboardRuntime:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.monitor = Monitor()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.tail_thread: Optional[threading.Thread] = None

    def bootstrap_from_file(self) -> None:
        try:
            with open(self.log_path, "r") as log_file:
                for line in log_file:
                    event = parse_log_line(line)
                    if event:
                        with self.lock:
                            self.monitor.process_event(event)
        except FileNotFoundError as exc:
            raise RuntimeError("Log file not found: %s" % self.log_path) from exc

    def _process_line(self, line: str) -> None:
        event = parse_log_line(line)
        if event:
            with self.lock:
                self.monitor.process_event(event)

    def _tail_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                with open(self.log_path, "r") as log_file:
                    log_file.seek(0, os.SEEK_END)
                    while not self.stop_event.is_set():
                        line = log_file.readline()
                        if line:
                            self._process_line(line)
                            continue

                        try:
                            if os.path.getsize(self.log_path) < log_file.tell():
                                break
                        except OSError:
                            break

                        time.sleep(0.5)
            except FileNotFoundError:
                time.sleep(0.5)

    def start_tail_thread(self) -> None:
        if self.tail_thread and self.tail_thread.is_alive():
            return
        self.tail_thread = threading.Thread(target=self._tail_loop, daemon=True)
        self.tail_thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _compute_latency_distribution(self, latencies: List[float]) -> Dict[str, int]:
        buckets: Dict[str, int] = {
            "0-5s": 0,
            "5-10s": 0,
            "10-30s": 0,
            "30-60s": 0,
            "1-5m": 0,
            "5-10m": 0,
            "10m+": 0,
        }
        for latency in latencies:
            if latency < 5:
                buckets["0-5s"] += 1
            elif latency < 10:
                buckets["5-10s"] += 1
            elif latency < 30:
                buckets["10-30s"] += 1
            elif latency < 60:
                buckets["30-60s"] += 1
            elif latency < 300:
                buckets["1-5m"] += 1
            elif latency < 600:
                buckets["5-10m"] += 1
            else:
                buckets["10m+"] += 1

        return buckets

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self.lock:
            completed = int(self.monitor.completed)
            errors = int(self.monitor.errors)
            stalls = int(self.monitor.stalls)
            total_tasks = int(self.monitor.total_tasks)
            latencies = list(self.monitor.latencies)
            ttfts = list(self.monitor.ttfts)
            total_prompt_tokens = int(self.monitor.total_prompt_tokens)
            total_completion_tokens = int(self.monitor.total_completion_tokens)
            slowest_task = self.monitor.slowest_task
            error_list = list(self.monitor.error_list)
            active_map = dict(self.monitor.active_tasks)
            task_type_raw = dict(self.monitor.task_type_stats)
            start_time = float(self.monitor.start_time)
            correct_count = int(self.monitor.correct_count)
            total_scored = int(self.monitor.total_scored)
            task_type_accuracy_raw = {
                k: dict(v) for k, v in self.monitor.task_type_accuracy.items()
            }

        elapsed = max(0.0, now - start_time)
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
        avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else 0.0
        min_ttft = min(ttfts) if ttfts else 0.0
        max_ttft = max(ttfts) if ttfts else 0.0

        total_done = completed + errors
        throughput = (float(total_done) / elapsed * 60.0) if elapsed > 0 else 0.0
        eta = 0.0
        if total_tasks > 0 and throughput > 0:
            remaining = max(0, total_tasks - total_done)
            eta = float(remaining) / throughput * 60.0

        active_workers: List[Dict[str, Any]] = []
        for worker_name in sorted(active_map.keys()):
            info = active_map[worker_name]
            worker_elapsed = max(0.0, now - float(info.get("start_time", now)))

            worker_label = str(worker_name)
            if worker_label.lower().startswith("thread-"):
                worker_label = worker_label.split("-", 1)[1]
            elif "_" in worker_label:
                worker_label = worker_label.rsplit("_", 1)[-1]

            active_workers.append(
                {
                    "worker": worker_label,
                    "task_id": str(info.get("task_id", "")),
                    "elapsed": worker_elapsed,
                    "tokens": int(info.get("tokens", 0)),
                    "reasoning_tokens": int(info.get("reasoning_tokens", 0)),
                    "status": str(info.get("status", "waiting")),
                    "ttft": info.get("ttft"),
                    "prompt_chars": int(info.get("prompt_chars", 0)),
                }
            )

        task_type_pairs: List[Any] = sorted(
            task_type_raw.items(),
            key=lambda pair: pair[1].get("count", 0),
            reverse=True,
        )
        task_type_stats: Dict[str, Dict[str, Any]] = {}
        for task_name, stats in task_type_pairs:
            count = int(stats.get("count", 0))
            total_latency = float(stats.get("total_latency", 0.0))
            avg = (total_latency / count) if count > 0 else 0.0
            acc_info = task_type_accuracy_raw.get(str(task_name), {})
            tt_correct = int(acc_info.get("correct", 0))
            tt_total = int(acc_info.get("total", 0))
            tt_accuracy = (float(tt_correct) / tt_total) if tt_total > 0 else None
            task_type_stats[str(task_name)] = {
                "count": count,
                "avg_latency": avg,
                "correct": tt_correct,
                "scored": tt_total,
                "accuracy": tt_accuracy,
            }

        recent_errors: List[Dict[str, str]] = []
        for task_id, err in error_list[-10:]:
            recent_errors.append({"task_id": task_id, "error": err})

        return {
            "completed": completed,
            "errors": errors,
            "stalls": stalls,
            "total_tasks": total_tasks,
            "avg_latency": avg_latency,
            "avg_ttft": avg_ttft,
            "min_ttft": min_ttft,
            "max_ttft": max_ttft,
            "throughput": throughput,
            "elapsed": elapsed,
            "eta": eta,
            "slowest_task": {"id": slowest_task[0], "latency": slowest_task[1]},
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "active_workers": active_workers,
            "task_type_stats": task_type_stats,
            "recent_errors": recent_errors,
            "latency_distribution": self._compute_latency_distribution(latencies),
            "accuracy": (float(correct_count) / total_scored) if total_scored > 0 else None,
            "correct_count": correct_count,
            "total_scored": total_scored,
        }


def create_app(log_path: str) -> FastAPI:
    app = FastAPI(title="Biomni Benchmark Monitor", version="1.0.0")
    runtime = DashboardRuntime(log_path)
    app.state.runtime = runtime

    @app.on_event("startup")
    def on_startup() -> None:
        runtime.bootstrap_from_file()
        runtime.start_tail_thread()

    @app.on_event("shutdown")
    def on_shutdown() -> None:
        runtime.stop()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(content=DASHBOARD_HTML)

    @app.get("/api/stats")
    def api_stats() -> JSONResponse:
        return JSONResponse(content=runtime.snapshot())

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web monitor for benchmark logs")
    parser.add_argument("logfile", help="Path to benchmark log file")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to serve dashboard"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.logfile)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
