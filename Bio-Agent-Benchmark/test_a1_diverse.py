#!/usr/bin/env python3
"""Quick diverse test: 2 tasks per category with Biomni A1 agent.

This picks 2 tasks from each benchmark category to test A1's answer quality
and the _extract_answer() parsing across all task types.
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

from benchmarks.biomni import BiomniBenchmark
from agent.biomni_a1 import BiomniA1Agent

BIOMNI_PATH = os.getenv("BIOMNI_PATH", "/Users/admin/Projects/for_stu/Biomni")
DATA_PATH = os.getenv("BIOMNI_DATA_PATH", os.path.join(BIOMNI_PATH, "data"))

TASKS_PER_CATEGORY = 2


def main():
    print("=" * 70)
    print("Biomni A1 Agent — Diverse Category Test")
    print("=" * 70)

    # Load benchmark
    benchmark = BiomniBenchmark()
    all_tasks = benchmark.load_tasks()
    print(f"\nLoaded {len(all_tasks)} total tasks")

    # Group by task_name
    by_category = {}
    for task in all_tasks:
        cat = task["task_name"]
        by_category.setdefault(cat, []).append(task)

    # Select N tasks per category
    selected = []
    for cat, tasks in sorted(by_category.items()):
        selected.extend(tasks[:TASKS_PER_CATEGORY])
        print(f"  {cat}: selected {min(TASKS_PER_CATEGORY, len(tasks))}/{len(tasks)}")

    print(f"\nTotal selected: {len(selected)} tasks")

    # Initialize A1 agent
    print("\nInitializing Biomni A1 agent...")
    agent = BiomniA1Agent(
        biomni_path=BIOMNI_PATH,
        data_path=DATA_PATH,
        skip_datalake_download=True,
    )

    # Run tasks
    results = []
    total_start = time.time()

    for i, task in enumerate(selected):
        task_id = task["id"]
        task_name = task["task_name"]
        prompt = task["prompt"]
        gt = task["ground_truth"]

        print(f"\n--- [{i + 1}/{len(selected)}] {task_id} ({task_name}) ---")
        print(f"  GT: {str(gt)[:80]}")

        t0 = time.time()
        prediction = agent.predict(prompt, task_id=task_id)
        elapsed = time.time() - t0

        # Compute score
        score = benchmark._compute_reward(task_name, prediction, gt)

        results.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "prediction": prediction,
                "ground_truth": gt,
                "score": score,
                "elapsed": elapsed,
            }
        )

        status = "✅" if score > 0 else "❌"
        print(f"  Pred: {prediction[:80]}")
        print(f"  {status} Score={score:.1f} | {elapsed:.1f}s")

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")

    # Per-category accuracy
    cat_stats = {}
    for r in results:
        cat = r["task_name"]
        cat_stats.setdefault(cat, {"correct": 0, "total": 0})
        cat_stats[cat]["total"] += 1
        cat_stats[cat]["correct"] += r["score"]

    correct_total = sum(r["score"] for r in results)
    print(
        f"Overall: {correct_total:.0f}/{len(results)} ({100 * correct_total / len(results):.1f}%)"
    )

    for cat, stats in sorted(cat_stats.items()):
        acc = stats["correct"] / stats["total"] * 100
        print(f"  {cat}: {stats['correct']:.0f}/{stats['total']} ({acc:.0f}%)")


if __name__ == "__main__":
    main()
