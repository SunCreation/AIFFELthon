#!/usr/bin/env python3
"""Debug script: Compare system prompts between baseline and multi-agent.

Prints the system prompt that each approach generates for the same task,
so we can diff them and find what's missing.
"""

import os
import sys
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Biomni"))
from dotenv import load_dotenv

load_dotenv()
load_dotenv(
    os.path.join(os.path.dirname(__file__), "..", "..", "Biomni", ".env"),
    override=False,
)

from biomni.agent.a1 import A1

DATA_PATH = os.getenv(
    "BIOMNI_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "Biomni", "data")
)

# Sample prompt from gwas_variant_prioritization_134
SAMPLE_PROMPT = """Given the following GWAS phenotype and a list of variants, identify the most promising variant that is most likely to be causal for the phenotype based on its association strength, functional annotations, and biological relevance.

GWAS phenotype: LDL cholesterol measurement

Variant list: rs4253311, rs6687813, rs7700133, rs12916, rs2228671, rs4253772, rs10455872, rs646776, rs2073547, rs3757354, rs855791"""

print("=" * 80)
print("1. Creating A1 with use_tool_retriever=TRUE (baseline mode)")
print("=" * 80)
a1_baseline = A1(
    path=DATA_PATH,
    use_tool_retriever=True,
    timeout_seconds=60,
    expected_data_lake_files=[],
)

print(f"\n>>> After __init__ + configure():")
print(f"    system_prompt length: {len(a1_baseline.system_prompt)}")
print(f"    has '<execute>': {'<execute>' in a1_baseline.system_prompt}")
print(f"    has '<solution>': {'<solution>' in a1_baseline.system_prompt}")
print(f"    has 'MUST': {'MUST' in a1_baseline.system_prompt}")
print(f"    has app: {hasattr(a1_baseline, 'app') and a1_baseline.app is not None}")
print(f"    has checkpointer: {hasattr(a1_baseline, 'checkpointer')}")

# Save initial prompt
with open("/tmp/debug_baseline_initial_prompt.txt", "w") as f:
    f.write(a1_baseline.system_prompt)
print(f"    Saved to /tmp/debug_baseline_initial_prompt.txt")

# Now simulate what baseline predict() does
print(f"\n>>> Simulating baseline predict() retrieval step...")
selected = a1_baseline._prepare_resources_for_retrieval(SAMPLE_PROMPT)
print(f"    Selected resources: {json.dumps(selected, indent=2, default=str)[:500]}")
a1_baseline.update_system_prompt_with_selected_resources(selected)

print(f"\n>>> After retrieval + update:")
print(f"    system_prompt length: {len(a1_baseline.system_prompt)}")
print(f"    has '<execute>': {'<execute>' in a1_baseline.system_prompt}")
print(f"    has '<solution>': {'<solution>' in a1_baseline.system_prompt}")

with open("/tmp/debug_baseline_retrieval_prompt.txt", "w") as f:
    f.write(a1_baseline.system_prompt)
print(f"    Saved to /tmp/debug_baseline_retrieval_prompt.txt")

print("\n" + "=" * 80)
print("2. Creating A1 with use_tool_retriever=FALSE (multi-agent mode)")
print("=" * 80)
a1_multi = A1(
    path=DATA_PATH,
    use_tool_retriever=False,
    timeout_seconds=60,
    expected_data_lake_files=[],
)

print(f"\n>>> After __init__ + configure():")
print(f"    system_prompt length: {len(a1_multi.system_prompt)}")
print(f"    has '<execute>': {'<execute>' in a1_multi.system_prompt}")
print(f"    has '<solution>': {'<solution>' in a1_multi.system_prompt}")
print(f"    has app: {hasattr(a1_multi, 'app') and a1_multi.app is not None}")

with open("/tmp/debug_multi_initial_prompt.txt", "w") as f:
    f.write(a1_multi.system_prompt)
print(f"    Saved to /tmp/debug_multi_initial_prompt.txt")

# Now simulate what multi-agent _inject_curated_system_prompt does
print(f"\n>>> Simulating multi-agent _inject_curated_system_prompt()...")

# Import multi-agent functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))
from biomni_a1_multi import get_curated_tool_desc, TASK_PROMPTS
import glob as glob_mod

task_name = "gwas_variant_prioritization"
curated_tool_desc = get_curated_tool_desc(task_name, a1_multi.module2api)
print(
    f"    curated tools: {sum(len(v) for v in curated_tool_desc.values())} tools from {len(curated_tool_desc)} modules"
)
for mod, tools in curated_tool_desc.items():
    print(f"      {mod}: {[t['name'] for t in tools]}")

# Prepare data lake / library / know-how (same as _inject_curated_system_prompt)
data_lake_path = a1_multi.path + "/data_lake"
data_lake_content = glob_mod.glob(data_lake_path + "/*")
data_lake_items = [x.split("/")[-1] for x in data_lake_content]
data_lake_with_desc = []
for item in data_lake_items:
    description = a1_multi.data_lake_dict.get(item, f"Data lake item: {item}")
    data_lake_with_desc.append({"name": item, "description": description})

library_content_list = list(a1_multi.library_content_dict.keys())

know_how_docs = []
if hasattr(a1_multi, "know_how_loader") and a1_multi.know_how_loader.documents:
    for _doc_id, doc in a1_multi.know_how_loader.documents.items():
        know_how_docs.append(
            {
                "id": doc["id"],
                "name": doc["name"],
                "description": doc["description"],
                "content": doc["content_without_metadata"],
                "metadata": doc["metadata"],
            }
        )

system_prompt = a1_multi._generate_system_prompt(
    tool_desc=curated_tool_desc,
    data_lake_content=data_lake_with_desc,
    library_content_list=library_content_list,
    self_critic=False,
    is_retrieval=True,
    know_how_docs=know_how_docs if know_how_docs else None,
)

# Prepend task instruction
task_instruction = TASK_PROMPTS.get(task_name, "")
if task_instruction:
    system_prompt = (
        "=== TASK-SPECIFIC INSTRUCTIONS ===\n"
        + task_instruction
        + "\n=== END TASK-SPECIFIC INSTRUCTIONS ===\n\n"
        + system_prompt
    )

# Add GPT formatting reminder
if hasattr(a1_multi.llm, "model_name") and (
    "gpt" in str(a1_multi.llm.model_name).lower()
    or "openai" in str(type(a1_multi.llm)).lower()
):
    system_prompt += (
        "\n\nIMPORTANT FOR GPT MODELS: You MUST use XML tags "
        "<execute> or <solution> in EVERY response. "
        "Do not use markdown code blocks (```) - use <execute> tags instead."
    )

print(f"\n>>> After curated injection:")
print(f"    system_prompt length: {len(system_prompt)}")
print(f"    has '<execute>': {'<execute>' in system_prompt}")
print(f"    has '<solution>': {'<solution>' in system_prompt}")
print(f"    has task instruction: {bool(task_instruction)}")
print(f"    instruction length: {len(task_instruction)}")

with open("/tmp/debug_multi_injected_prompt.txt", "w") as f:
    f.write(system_prompt)
print(f"    Saved to /tmp/debug_multi_injected_prompt.txt")

# CRITICAL: Check if the prompt actually has tool descriptions
print(f"\n>>> Tool description check:")
for tool_name in [
    "query_gwas_catalog",
    "query_dbsnp",
    "query_gnomad",
    "query_regulomedb",
    "query_opentarget",
]:
    in_baseline = tool_name in a1_baseline.system_prompt
    in_multi = tool_name in system_prompt
    print(f"    {tool_name}: baseline={in_baseline} | multi={in_multi}")

print("\n" + "=" * 80)
print("3. Summary comparison")
print("=" * 80)
print(f"    Baseline retrieval prompt: {len(a1_baseline.system_prompt)} chars")
print(f"    Multi injected prompt:     {len(system_prompt)} chars")
print(
    f"    Difference:                {len(a1_baseline.system_prompt) - len(system_prompt)} chars"
)

# Show first 500 chars of each
print(f"\n>>> Baseline prompt first 500 chars:")
print(a1_baseline.system_prompt[:500])
print(f"\n>>> Multi prompt first 500 chars:")
print(system_prompt[:500])

print("\n\nDone! Compare files in /tmp/debug_*.txt for full diff")
