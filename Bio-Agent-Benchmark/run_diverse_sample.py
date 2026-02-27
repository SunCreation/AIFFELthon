#!/usr/bin/env python3
"""Run a diverse sample of tasks across ALL task types for tool usage analysis.

Selects N tasks per task type to ensure full coverage of the benchmark.
"""

import sys
import os
import json
import time
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.biomni import BiomniBenchmark
from agent.biomni_a1 import BiomniA1Agent

N_PER_TYPE = 5  # tasks per type
PARALLEL = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Load all tasks
    bench = BiomniBenchmark()
    all_tasks = bench.load_tasks()

    # Group by task type
    by_type = {}
    for t in all_tasks:
        tname = t["task_name"]
        by_type.setdefault(tname, []).append(t)

    logger.info("Task types found: %d", len(by_type))
    for tname, tasks in sorted(by_type.items()):
        logger.info("  %s: %d tasks", tname, len(tasks))

    # Sample N_PER_TYPE from each
    selected = []
    random.seed(42)
    for tname, tasks in sorted(by_type.items()):
        sample = random.sample(tasks, min(N_PER_TYPE, len(tasks)))
        selected.extend(sample)
        logger.info("Selected %d from %s", len(sample), tname)

    logger.info("Total selected tasks: %d", len(selected))

    # Initialize agent with pool
    agent = BiomniA1Agent(
        pool_size=PARALLEL,
        skip_datalake_download=True,
    )

    # Run tasks in parallel
    results = []
    start_time = time.time()

    def run_task(task):
        result = bench.run_task(agent, task)
        if result["status"] == "success":
            task_name = result["metadata"]["task_name"]
            score = bench._compute_reward(task_name, result["prediction"], result["ground_truth"])
            result["score"] = score
        else:
            result["score"] = 0.0
        return result

    with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        futures = {executor.submit(run_task, t): t for t in selected}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(
                    "[DONE %d/%d] task=%s | score=%s",
                    done_count,
                    len(selected),
                    task["id"],
                    result.get("score", "?"),
                )
            except Exception as e:
                logger.error(
                    "[FAIL %d/%d] task=%s | error=%s",
                    done_count,
                    len(selected),
                    task["id"],
                    e,
                )

    elapsed = time.time() - start_time
    logger.info("=== ALL DONE in %.1fs ===", elapsed)

    # Save results
    outdir = os.path.join(os.path.dirname(__file__), "logs", "tool_analysis")
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(
        outdir, f"diverse_sample_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    with open(outfile, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Results saved to %s", outfile)


if __name__ == "__main__":
    main()
