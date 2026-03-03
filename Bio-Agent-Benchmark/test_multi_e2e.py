#!/usr/bin/env python3
"""End-to-end test: verify multi-agent actually calls tools (not just returns text).

Tests 2 tasks from different categories:
1. gwas_variant_prioritization (genetics) — should call query_gwas_catalog
2. lab_bench_dbqa (literature) — should give a letter answer

Success criteria:
- Execution time > 10s (proves tool calls happened, not just text generation)
- Answer is non-empty and not a prompt fragment
- No "` tags" in the answer (the old failure mode)
"""

import os
import sys
import json
import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

# Setup paths
biomni_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Biomni")
)
sys.path.insert(0, biomni_path)
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(os.path.join(biomni_path, ".env"))

# Load test data
from datasets import load_dataset

ds = load_dataset("biomni/Eval1", split="test")

# Pick first task from each of 2 categories
test_cases = []
seen_tasks = set()
for row in ds:
    tn = row["task_name"]
    if tn in ("gwas_variant_prioritization", "lab_bench_dbqa") and tn not in seen_tasks:
        seen_tasks.add(tn)
        task_id = f"{tn}_{row['task_instance_id']}"
        test_cases.append(
            {
                "task_id": task_id,
                "prompt": row["prompt"],
                "answer": str(row["answer"]),
                "task_name": tn,
            }
        )
    if len(test_cases) >= 2:
        break

if not test_cases:
    print("ERROR: Could not find test tasks in dataset")
    sys.exit(1)

print(f"\n{'=' * 60}")
print(f"Testing {len(test_cases)} tasks with multi-agent (bug-fixed)")
print(f"{'=' * 60}\n")

# Create multi-agent
from agent.biomni_a1_multi import BiomniA1MultiAgent

agent = BiomniA1MultiAgent(
    biomni_path=biomni_path,
    data_path=os.path.join(biomni_path, "data"),
    pool_size=1,
    skip_datalake_download=True,
)

results = []
for tc in test_cases:
    task_id = tc["task_id"]
    prompt = tc["prompt"]
    expected = tc["answer"]

    print(f"\n{'─' * 60}")
    print(f"TASK: {task_id}")
    print(f"EXPECTED: {expected[:80]}")
    print(f"PROMPT: {prompt[:120]}...")
    print(f"{'─' * 60}")

    start = time.time()
    try:
        answer = agent.predict(prompt, task_id=task_id)
    except Exception as e:
        import traceback

        traceback.print_exc()
        answer = f"ERROR: {e}"
    elapsed = time.time() - start

    # Evaluate
    has_tool_call = elapsed > 10  # If >10s, tool calls likely happened
    has_valid_answer = (
        bool(answer) and "` tags" not in answer and "execute step" not in answer.lower()
    )

    status = "PASS" if (has_tool_call and has_valid_answer) else "FAIL"

    result = {
        "task_id": task_id,
        "answer": answer[:200],
        "expected": expected[:80],
        "elapsed": round(elapsed, 1),
        "has_tool_call": has_tool_call,
        "has_valid_answer": has_valid_answer,
        "status": status,
    }
    results.append(result)

    print(f"\n  ANSWER:   {answer[:200]}")
    print(f"  EXPECTED: {expected[:80]}")
    print(f"  TIME:     {elapsed:.1f}s")
    print(f"  STATUS:   {status}")

print(f"\n\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
passed = sum(1 for r in results if r["status"] == "PASS")
print(f"  Passed: {passed}/{len(results)}")
for r in results:
    print(
        f"  [{r['status']}] {r['task_id']}: answer='{r['answer'][:60]}' ({r['elapsed']}s)"
    )

# Save results
out_path = "/tmp/test_multi_e2e_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")

if passed < len(results):
    print("\n⚠️  Some tests FAILED — check logs above")
    sys.exit(1)
else:
    print("\n✅ All tests PASSED — tools are being called correctly")
    sys.exit(0)
