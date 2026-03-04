#!/usr/bin/env python3
"""
벤치마크 실시간 모니터링 스크립트.
로그 파일을 tail -f 하면서 핵심 지표를 실시간 요약합니다.

Streaming/non-streaming 모드 모두 지원.

사용법:
    python monitor.py /tmp/biomni_parallel_run.log
    python monitor.py /tmp/biomni_parallel_run.log --refresh 5
"""

import sys
import re
import time
import argparse
from typing import Optional
from collections import defaultdict
from datetime import datetime


def parse_log_line(line: str) -> Optional[dict]:
    """로그 한 줄을 파싱하여 이벤트 딕셔너리 반환."""

    # [REQ] task=gwas_134 | worker=Thread-1 | prompt_chars=1523 | mode=stream
    # [MULTI] task=gwas_134 | worker=Thread-1 | task_name=gwas_variant_prioritization | prompt_chars=362
    req_match = re.search(
        r"\[(?:REQ|MULTI)\] task=(\S+) \| worker=(\S+) .*?prompt_chars=(\d+)", line
    )
    if req_match:
        mode_match = re.search(r"mode=(\S+)", line)
        return {
            "event": "req",
            "task_id": req_match.group(1),
            "worker": req_match.group(2),
            "prompt_chars": int(req_match.group(3)),
            "mode": mode_match.group(1) if mode_match else "stream",
        }

    # [STREAM_START] task=gwas_134 | worker=Thread-1 | ttft=0.85s
    stream_start_match = re.search(
        r"\[STREAM_START\] task=(\S+) \| worker=(\S+) \| ttft=([\d.]+)s", line
    )
    if stream_start_match:
        return {
            "event": "stream_start",
            "task_id": stream_start_match.group(1),
            "worker": stream_start_match.group(2),
            "ttft": float(stream_start_match.group(3)),
        }

    # [STREAM_PROGRESS] task=gwas_134 | worker=Thread-1 | tokens_so_far=20 | reasoning=15 | elapsed=3.5s
    stream_progress_match = re.search(
        r"\[STREAM_PROGRESS\] task=(\S+) \| worker=(\S+) \| "
        r"tokens_so_far=(\d+) \| (?:reasoning=(\d+) \| )?elapsed=([\d.]+)s",
        line,
    )
    if stream_progress_match:
        return {
            "event": "stream_progress",
            "task_id": stream_progress_match.group(1),
            "worker": stream_progress_match.group(2),
            "tokens_so_far": int(stream_progress_match.group(3)),
            "reasoning_tokens": int(stream_progress_match.group(4)) if stream_progress_match.group(4) else 0,
            "elapsed": float(stream_progress_match.group(5)),
        }

    # [STREAM_STALL] task=hle_55 | worker=Thread-3 | last_token_age=60.0s | tokens_so_far=12
    stream_stall_match = re.search(r"\[STREAM_STALL\] task=(\S+) \| worker=(\S+)", line)
    if stream_stall_match:
        return {
            "event": "stream_stall",
            "task_id": stream_stall_match.group(1),
            "worker": stream_stall_match.group(2),
        }

    # [STREAM_END] task=X | worker=Y | total_tokens=N | ... | latency=Ns | answer=...
    # Also handles multi-agent format with task_name=..., tools=... fields
    stream_end_match = re.search(
        r"\[STREAM_END\] task=(\S+) \| worker=(\S+) \| "
        r"(?:task_name=\S+ \| )?"
        r"total_tokens=(\d+) \| "
        r"(?:reasoning_tokens=(\d+) \| )?"
        r"(?:prompt_tokens=([^|]+) \| )?"
        r"(?:ttft=([\d.-]+)s \| )?"
        r"latency=([\d.]+)s \| "
        r"(?:tools=\S+ \| )?"
        r"answer=(.*)",
        line,
    )
    if stream_end_match:
        prompt_tokens_raw = stream_end_match.group(5)
        ttft_str = stream_end_match.group(6)
        prompt_tokens_val = None
        if prompt_tokens_raw:
            stripped = prompt_tokens_raw.strip()
            if stripped.isdigit():
                prompt_tokens_val = int(stripped)
        ttft_val = None
        if ttft_str and ttft_str != "-1":
            ttft_val = float(ttft_str)
        reasoning_str = stream_end_match.group(4)
        return {
            "event": "stream_end",
            "task_id": stream_end_match.group(1),
            "worker": stream_end_match.group(2),
            "total_tokens": int(stream_end_match.group(3)),
            "reasoning_tokens": int(reasoning_str) if reasoning_str else 0,
            "prompt_tokens": prompt_tokens_val,
            "ttft": ttft_val,
            "latency": float(stream_end_match.group(7)),
            "answer": stream_end_match.group(8).strip(),
        }

    # [RES] (non-streaming mode) task=gwas_134 | worker=Thread-1 | latency=3.2s | ...
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

    # [SPECIALIST_START] task=X | worker=Y | ttft=Zs
    # [SPECIALIST_V2_START] task=X | worker=Y | ttft=Zs
    specialist_start_match = re.search(
        r"\[SPECIALIST(?:_V2)?_START\] task=(\S+) \| worker=(\S+) \| ttft=([\d.]+)s", line
    )
    if specialist_start_match:
        return {
            "event": "stream_start",
            "task_id": specialist_start_match.group(1),
            "worker": specialist_start_match.group(2),
            "ttft": float(specialist_start_match.group(3)),
        }

    # [SPECIALIST_PROGRESS] task=X | tokens=N | reasoning=N | elapsed=Ns
    # [SPECIALIST_V2_PROGRESS] task=X | tokens=N | reasoning=N | elapsed=Ns
    specialist_prog_match = re.search(
        r"\[SPECIALIST(?:_V2)?_PROGRESS\] task=(\S+) \| tokens=(\d+) \| reasoning=(\d+) \| elapsed=([\d.]+)s",
        line,
    )
    if specialist_prog_match:
        return {
            "event": "stream_progress",
            "task_id": specialist_prog_match.group(1),
            "worker": "specialist",
            "tokens_so_far": int(specialist_prog_match.group(2)),
            "reasoning_tokens": int(specialist_prog_match.group(3)),
            "elapsed": float(specialist_prog_match.group(4)),
        }

    # [SPECIALIST_END] task=X | specialist=Y | tokens=N | reasoning=N | ttft=Ns | latency=Ns | steps=N | tools=X | answer=...
    # [SPECIALIST_V2_END] task=X | specialist=Y | tokens=N | reasoning=N | ttft=Ns | latency=Ns | steps=N | tools=X | answer=...
    specialist_end_match = re.search(
        r"\[SPECIALIST(?:_V2)?_END\] task=(\S+) \| specialist=(\S+) \| "
        r"tokens=(\d+) \| reasoning=(\d+) \| "
        r"ttft=([\d.-]+)s \| latency=([\d.]+)s \| "
        r"steps=(\d+) \| tools=(\S+) \| answer=(.*)",
        line,
    )
    if specialist_end_match:
        ttft_str = specialist_end_match.group(5)
        ttft_val = float(ttft_str) if ttft_str != "-1" else None
        return {
            "event": "stream_end",
            "task_id": specialist_end_match.group(1),
            "worker": "specialist",
            "total_tokens": int(specialist_end_match.group(3)),
            "reasoning_tokens": int(specialist_end_match.group(4)),
            "prompt_tokens": 0,  # Not tracked in specialist pipeline; streaming chunks only
            "ttft": ttft_val,
            "latency": float(specialist_end_match.group(6)),
            "answer": specialist_end_match.group(9).strip(),
        }

    # [SPECIALIST_ERR] task=X | latency=Ns | error=...
    specialist_err_match = re.search(
        r"\[SPECIALIST(?:_V2)?_ERR\] task=(\S+) \| latency=([\d.]+)s \| error=(.*)",
        line,
    )
    if specialist_err_match:
        return {
            "event": "err",
            "task_id": specialist_err_match.group(1),
            "worker": "specialist",
            "latency": float(specialist_err_match.group(2)),
            "error": specialist_err_match.group(3).strip(),
        }

    # [SCORE] task=gwas_variant_prioritization_134 | score=1.0 | prediction=rs4253311 | ground_truth=rs4253311
    score_match = re.search(
        r"\[SCORE\] task=(\S+) \| score=([\d.]+) \| prediction=(.*?) \| ground_truth=(.*)",
        line,
    )
    if score_match:
        return {
            "event": "score",
            "task_id": score_match.group(1),
            "score": float(score_match.group(2)),
            "prediction": score_match.group(3).strip(),
            "ground_truth": score_match.group(4).strip(),
        }

    # tqdm progress: Running tasks (x4):  25%|...| 108/433 [05:23<15:02, 2.78s/it]
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
        # worker -> {task_id, start_time, prompt_chars, tokens, status, ttft}
        self.active_tasks = {}
        self.completed = 0
        self.errors = 0
        self.stalls = 0
        self.total_tasks = 0
        self.latencies = []
        self.ttfts = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.task_type_stats = defaultdict(lambda: {"count": 0, "total_latency": 0.0})
        self.start_time = time.time()
        self.last_progress = ""
        self.slowest_task = ("", 0.0)
        self.error_list = []
        # Accuracy tracking
        self.correct_count = 0
        self.total_scored = 0
        self.task_type_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})

    def process_event(self, event):
        evt = event["event"]

        if evt == "req":
            self.active_tasks[event["worker"]] = {
                "task_id": event["task_id"],
                "start_time": time.time(),
                "prompt_chars": event["prompt_chars"],
                "tokens": 0,
                "reasoning_tokens": 0,
                "status": "waiting",  # waiting -> generating -> done
                "ttft": None,
                "mode": event.get("mode", "unknown"),
            }

        elif evt == "stream_start":
            worker_info = self.active_tasks.get(event["worker"])
            if worker_info:
                worker_info["status"] = "generating"
                worker_info["ttft"] = event["ttft"]

        elif evt == "stream_progress":
            worker_info = self.active_tasks.get(event["worker"])
            if worker_info:
                worker_info["tokens"] = event["tokens_so_far"]
                worker_info["reasoning_tokens"] = event.get("reasoning_tokens", 0)
                worker_info["status"] = "generating"

        elif evt == "stream_stall":
            self.stalls += 1
            worker_info = self.active_tasks.get(event["worker"])
            if worker_info:
                worker_info["status"] = "stalled"

        elif evt == "stream_end":
            self.completed += 1
            self.latencies.append(event["latency"])
            if event.get("prompt_tokens") is not None:
                self.total_prompt_tokens += event["prompt_tokens"]
            self.total_completion_tokens += event["total_tokens"]

            if event["ttft"] is not None:
                self.ttfts.append(event["ttft"])

            if event["latency"] > self.slowest_task[1]:
                self.slowest_task = (event["task_id"], event["latency"])

            task_type = (
                event["task_id"].rsplit("_", 1)[0]
                if "_" in event["task_id"]
                else event["task_id"]
            )
            self.task_type_stats[task_type]["count"] += 1
            self.task_type_stats[task_type]["total_latency"] += event["latency"]

            self.active_tasks.pop(event["worker"], None)

        elif evt == "res":
            # Non-streaming completion
            self.completed += 1
            self.latencies.append(event["latency"])
            self.total_prompt_tokens += event["prompt_tokens"]
            self.total_completion_tokens += event["completion_tokens"]

            if event["latency"] > self.slowest_task[1]:
                self.slowest_task = (event["task_id"], event["latency"])

            task_type = (
                event["task_id"].rsplit("_", 1)[0]
                if "_" in event["task_id"]
                else event["task_id"]
            )
            self.task_type_stats[task_type]["count"] += 1
            self.task_type_stats[task_type]["total_latency"] += event["latency"]

            self.active_tasks.pop(event["worker"], None)

        elif evt == "err":
            self.errors += 1
            self.error_list.append((event["task_id"], event["error"][:60]))
            self.active_tasks.pop(event["worker"], None)

        elif evt == "progress":
            self.total_tasks = event["total"]
            self.last_progress = "%d/%d [%s<%s]" % (
                event["done"],
                event["total"],
                event["elapsed"],
                event["remaining"],
            )


        elif evt == "score":
            self.total_scored += 1
            is_correct = event["score"] >= 1.0
            if is_correct:
                self.correct_count += 1
            task_type = (
                event["task_id"].rsplit("_", 1)[0]
                if "_" in event["task_id"]
                else event["task_id"]
            )
            self.task_type_accuracy[task_type]["total"] += 1
            if is_correct:
                self.task_type_accuracy[task_type]["correct"] += 1
    def render(self):
        elapsed = time.time() - self.start_time
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        avg_ttft = sum(self.ttfts) / len(self.ttfts) if self.ttfts else 0
        total_done = self.completed + self.errors
        throughput = total_done / elapsed * 60 if elapsed > 0 else 0

        lines = []
        lines.append("=" * 70)
        lines.append(
            "  BENCHMARK MONITOR  |  %s  |  elapsed: %.0fs"
            % (datetime.now().strftime("%H:%M:%S"), elapsed)
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
                "  Progress: [%s] %.1f%% (%d/%d)"
                % (bar, pct, total_done, self.total_tasks)
            )
            lines.append(
                "  ETA: %.0fs | Throughput: %.1f tasks/min" % (eta, throughput)
            )
        else:
            lines.append("  Progress: %s" % self.last_progress)

        lines.append("-" * 70)

        # Stats
        lines.append(
            "  ✅ Completed: %d  |  ❌ Errors: %d  |  ⛔ Stalls: %d"
            % (self.completed, self.errors, self.stalls)
        )
        lines.append(
            "  ⏱  Avg latency: %.1fs  |  Slowest: %s (%.1fs)"
            % (avg_latency, self.slowest_task[0], self.slowest_task[1])
        )
        lines.append("  ⚡ Avg TTFT: %.2fs" % avg_ttft)
        lines.append(
            "  📊 Tokens — completion: %s  |  prompt: %s"
            % (
                "{:,}".format(self.total_completion_tokens),
                "{:,}".format(self.total_prompt_tokens)
                if self.total_prompt_tokens
                else "N/A (streaming)",
            )
        )

        # Active workers
        lines.append("-" * 70)
        lines.append("  ACTIVE WORKERS:")
        if self.active_tasks:
            for worker, info in self.active_tasks.items():
                running_for = time.time() - info["start_time"]
                tokens = info.get("tokens", 0)
                status_str = info.get("status", "waiting")

                if status_str == "stalled":
                    icon = "⛔ STALL"
                elif status_str == "generating":
                    icon = "🔄"
                elif running_for > 30 and status_str == "waiting":
                    icon = "⚠️ SLOW"
                else:
                    icon = "🔄"

                worker_short = worker.split("_")[-1] if "_" in worker else worker

                if status_str == "generating":
                    lines.append(
                        "    %s %s: %s — %.0fs, %d tokens, generating..."
                        % (
                            icon,
                            worker_short,
                            info["task_id"],
                            running_for,
                            tokens,
                        )
                    )
                elif status_str == "stalled":
                    lines.append(
                        "    %s %s: %s — %.0fs, %d tokens, STALLED!"
                        % (icon, worker_short, info["task_id"], running_for, tokens)
                    )
                else:
                    lines.append(
                        "    %s %s: %s (%d chars) — %.0fs, waiting for first token..."
                        % (
                            icon,
                            worker_short,
                            info["task_id"],
                            info["prompt_chars"],
                            running_for,
                        )
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
                lines.append(
                    "    %s: %d done, avg %.1fs" % (ttype, stats["count"], avg)
                )

        # Recent errors
        if self.error_list:
            lines.append("-" * 70)
            lines.append("  RECENT ERRORS:")
            for task_id, err in self.error_list[-3:]:
                lines.append("    ❌ %s: %s" % (task_id, err))

        lines.append("=" * 70)
        return "\n".join(lines)


def tail_and_monitor(filepath, refresh=3.0):
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
        print("Error: Log file not found: %s" % filepath)
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
