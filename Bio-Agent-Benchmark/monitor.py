#!/usr/bin/env python3
"""
벤치마크 실시간 모니터링 스크립트.
로그 파일을 tail -f 하면서 핵심 지표를 실시간 요약합니다.

사용법:
    python monitor.py /tmp/biomni_parallel_run.log
    python monitor.py /tmp/biomni_parallel_run.log --refresh 5
"""

import sys
import re
import time
import argparse
from collections import defaultdict
from datetime import datetime


def parse_log_line(line: str) -> dict | None:
    """로그 한 줄을 파싱하여 이벤트 딕셔너리 반환."""

    # [REQ] task=gwas_134 | worker=Thread-1 | prompt_chars=1523
    req_match = re.search(
        r"\[REQ\] task=(\S+) \| worker=(\S+) \| prompt_chars=(\d+)", line
    )
    if req_match:
        return {
            "event": "req",
            "task_id": req_match.group(1),
            "worker": req_match.group(2),
            "prompt_chars": int(req_match.group(3)),
        }

    # [RES] task=gwas_134 | worker=Thread-1 | latency=3.2s | prompt_tokens=412 | completion_tokens=8 | answer=rs123
    res_match = re.search(
        r"\[RES\] task=(\S+) \| worker=(\S+) \| "
        r"latency=([\d.]+)s \| prompt_tokens=(\d+) \| "
        r"completion_tokens=(\d+) \| answer=(.*)",
        line,
    )
    if res_match:
        return {
            "event": "res",
            "task_id": res_match.group(1),
            "worker": res_match.group(2),
            "latency": float(res_match.group(3)),
            "prompt_tokens": int(res_match.group(4)),
            "completion_tokens": int(res_match.group(5)),
            "answer": res_match.group(6).strip(),
        }

    # [ERR] task=hle_55 | worker=Thread-3 | latency=60.0s | error=TimeoutError: ...
    err_match = re.search(
        r"\[ERR\] task=(\S+) \| worker=(\S+) \| "
        r"latency=([\d.]+)s \| error=(.*)",
        line,
    )
    if err_match:
        return {
            "event": "err",
            "task_id": err_match.group(1),
            "worker": err_match.group(2),
            "latency": float(err_match.group(3)),
            "error": err_match.group(4).strip(),
        }

    # tqdm progress: Running tasks (x4):  25%|██▌       | 108/433 [05:23<15:02, 2.78s/it]
    tqdm_match = re.search(r"(\d+)/(\d+) \[(\S+)<(\S+)", line)
    if tqdm_match:
        return {
            "event": "progress",
            "done": int(tqdm_match.group(1)),
            "total": int(tqdm_match.group(2)),
            "elapsed": tqdm_match.group(3),
            "remaining": tqdm_match.group(4).rstrip(","),
        }

    return None


class Monitor:
    def __init__(self):
        self.active_tasks = {}  # worker -> {task_id, start_time, prompt_chars}
        self.completed = 0
        self.errors = 0
        self.total_tasks = 0
        self.latencies = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.task_type_stats = defaultdict(
            lambda: {"count": 0, "correct": 0, "total_latency": 0.0}
        )
        self.start_time = time.time()
        self.last_progress = ""
        self.slowest_task = ("", 0.0)
        self.error_list = []

    def process_event(self, event: dict):
        if event["event"] == "req":
            self.active_tasks[event["worker"]] = {
                "task_id": event["task_id"],
                "start_time": time.time(),
                "prompt_chars": event["prompt_chars"],
            }

        elif event["event"] == "res":
            self.completed += 1
            self.latencies.append(event["latency"])
            self.total_prompt_tokens += event["prompt_tokens"]
            self.total_completion_tokens += event["completion_tokens"]

            if event["latency"] > self.slowest_task[1]:
                self.slowest_task = (event["task_id"], event["latency"])

            # task type tracking
            task_type = (
                event["task_id"].rsplit("_", 1)[0]
                if "_" in event["task_id"]
                else event["task_id"]
            )
            self.task_type_stats[task_type]["count"] += 1
            self.task_type_stats[task_type]["total_latency"] += event["latency"]

            self.active_tasks.pop(event["worker"], None)

        elif event["event"] == "err":
            self.errors += 1
            self.error_list.append((event["task_id"], event["error"][:60]))
            self.active_tasks.pop(event["worker"], None)

        elif event["event"] == "progress":
            self.total_tasks = event["total"]
            self.last_progress = f"{event['done']}/{event['total']} [{event['elapsed']}<{event['remaining']}]"

    def render(self) -> str:
        elapsed = time.time() - self.start_time
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        total_done = self.completed + self.errors
        throughput = total_done / elapsed * 60 if elapsed > 0 else 0

        lines = []
        lines.append("=" * 70)
        lines.append(
            f"  BENCHMARK MONITOR  |  {datetime.now().strftime('%H:%M:%S')}  |  elapsed: {elapsed:.0f}s"
        )
        lines.append("=" * 70)

        # Progress
        if self.total_tasks > 0:
            pct = total_done / self.total_tasks * 100
            bar_len = 40
            filled = int(bar_len * total_done / self.total_tasks)
            bar = "█" * filled + "░" * (bar_len - filled)
            eta = (
                (self.total_tasks - total_done) / throughput * 60
                if throughput > 0
                else 0
            )
            lines.append(
                f"  Progress: [{bar}] {pct:.1f}% ({total_done}/{self.total_tasks})"
            )
            lines.append(f"  ETA: {eta:.0f}s | Throughput: {throughput:.1f} tasks/min")
        else:
            lines.append(f"  Progress: {self.last_progress}")

        lines.append("-" * 70)

        # Stats
        lines.append(f"  ✅ Completed: {self.completed}  |  ❌ Errors: {self.errors}")
        lines.append(
            f"  ⏱  Avg latency: {avg_latency:.1f}s  |  Slowest: {self.slowest_task[0]} ({self.slowest_task[1]:.1f}s)"
        )
        lines.append(
            f"  📊 Tokens — prompt: {self.total_prompt_tokens:,}  |  completion: {self.total_completion_tokens:,}"
        )

        # Active workers
        lines.append("-" * 70)
        lines.append("  ACTIVE WORKERS:")
        if self.active_tasks:
            for worker, info in self.active_tasks.items():
                running_for = time.time() - info["start_time"]
                status = "⚠️ SLOW" if running_for > 30 else "🔄"
                worker_short = worker.split("_")[-1] if "_" in worker else worker
                lines.append(
                    f"    {status} {worker_short}: {info['task_id']} "
                    f"({info['prompt_chars']} chars) — {running_for:.0f}s"
                )
        else:
            lines.append("    (none)")

        # Task type breakdown
        if self.task_type_stats:
            lines.append("-" * 70)
            lines.append("  TASK TYPES:")
            for ttype, stats in sorted(self.task_type_stats.items()):
                avg = (
                    stats["total_latency"] / stats["count"] if stats["count"] > 0 else 0
                )
                lines.append(f"    {ttype}: {stats['count']} done, avg {avg:.1f}s")

        # Recent errors
        if self.error_list:
            lines.append("-" * 70)
            lines.append("  RECENT ERRORS:")
            for task_id, err in self.error_list[-3:]:
                lines.append(f"    ❌ {task_id}: {err}")

        lines.append("=" * 70)
        return "\n".join(lines)


def tail_and_monitor(filepath: str, refresh: float = 3.0):
    """로그 파일을 tail -f 하면서 모니터링."""
    monitor = Monitor()

    try:
        with open(filepath, "r") as f:
            # 기존 내용 먼저 처리
            for line in f:
                event = parse_log_line(line)
                if event:
                    monitor.process_event(event)

            # Clear and print initial state
            print("\033[2J\033[H" + monitor.render(), flush=True)

            # tail -f 모드
            while True:
                line = f.readline()
                if line:
                    event = parse_log_line(line)
                    if event:
                        monitor.process_event(event)
                        # Refresh display
                        print("\033[2J\033[H" + monitor.render(), flush=True)
                else:
                    time.sleep(0.5)
                    # 주기적 refresh (active worker 시간 업데이트)
                    print("\033[2J\033[H" + monitor.render(), flush=True)

    except FileNotFoundError:
        print(f"Error: Log file not found: {filepath}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(monitor.render())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Monitor")
    parser.add_argument("logfile", help="Path to benchmark log file")
    parser.add_argument(
        "--refresh", type=float, default=3.0, help="Refresh interval (seconds)"
    )
    args = parser.parse_args()

    tail_and_monitor(args.logfile, args.refresh)
