"""
Biomni A1 Exp#5 — Lightweight Specialist Agents (no _generate_system_prompt).

Key difference from Exp#4:
  - Exp#4: uses a1._generate_system_prompt() → 27K system prompt (no reduction)
  - Exp#5: uses SPECIALIST_SYSTEM_PROMPTS (hand-crafted) → ~3K system prompt

This is a TRUE context reduction experiment. Each specialist gets:
  - Minimal A1 loop instructions (<execute>/<solution> tags)
  - Tool function signatures (only what's needed)
  - Domain-specific API knowledge (from TOOL_KNOWLEDGE)
  - Task strategy (from sub_prompt_template)

Comparison:
  - biomni_a1_baseline.py: Original, 37% accuracy
  - biomni_a1_multi.py: Exp#4, ~28% (27K prompt, no real reduction)
  - biomni_a1_exp5.py: THIS FILE, ~3K specialist prompts

Usage:
    python run.py run --benchmark biomni --agent biomni_a1_exp5 --parallel 8
"""

import os
import sys
import re
import json
import time
import logging
import threading
import queue
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Prompt-based task classifier (100% accuracy on 433/433 eval1 tasks)
# ─────────────────────────────────────────────────────────────────────────────


def classify_prompt(prompt: str) -> str:
    """Classify benchmark prompt into task_name using keyword matching.

    Validated on all 433 Eval1 tasks with 100% accuracy.
    Returns 'unknown' if no pattern matches (falls back to default behavior).
    """
    p = prompt.strip().lower()
    if "crispr delivery" in p:
        return "crispr_delivery"
    if "identify the causal gene" in p and "HP:" in prompt:
        return "patient_gene_detection"
    if "diagnose the rare disease" in p:
        return "rare_disease_diagnosis"
    if "most promising variant" in p and "gwas phenotype" in p:
        return "gwas_variant_prioritization"
    if "causal genes within a locus" in p and "gwas phenotype" in p:
        return "gwas_causal_gene"
    if "strongest perturbation effect" in p:
        return "screen_gene_retrieval"
    if "multiple choice question about biology" in p:
        return "lab_bench"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Task → Agent type mapping
# ─────────────────────────────────────────────────────────────────────────────

TASK_TO_AGENT = {
    "patient_gene_detection": "genetics",
    "rare_disease_diagnosis": "genetics",
    "gwas_variant_prioritization": "genetics",
    "gwas_causal_gene": "genetics",
    "gwas_causal_gene_opentargets": "genetics",
    "gwas_causal_gene_pharmaprojects": "genetics",
    "gwas_causal_gene_gwas_catalog": "genetics",
    "crispr_delivery": "literature",
    "lab_bench": "literature",
    "lab_bench_seqqa": "literature",
    "lab_bench_dbqa": "literature",
    "screen_gene_retrieval": "bioinformatics",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Curated tool sets per task (replaces ToolRetriever)
# ─────────────────────────────────────────────────────────────────────────────

TASK_TOOLS = {
    "patient_gene_detection": [
        "query_monarch",
        "query_ensembl",
        "query_clinvar",
        "query_opentarget",
    ],
    "rare_disease_diagnosis": [
        "query_monarch",
        "query_clinvar",
        "query_opentarget",
        "search_google",
    ],
    "gwas_variant_prioritization": [
        "query_gwas_catalog",
        "query_dbsnp",
        "query_gnomad",
        "query_regulomedb",
        "query_opentarget",
    ],
    "gwas_causal_gene": [
        "query_opentarget",
        "query_ensembl",
        "query_gwas_catalog",
    ],
    "gwas_causal_gene_opentargets": [
        "query_opentarget",
        "query_ensembl",
    ],
    "gwas_causal_gene_pharmaprojects": [
        "query_opentarget",
        "query_ensembl",
        "query_gwas_catalog",
    ],
    "gwas_causal_gene_gwas_catalog": [
        "query_gwas_catalog",
        "query_opentarget",
        "query_ensembl",
    ],
    "crispr_delivery": [
        "query_pubmed",
        "search_google",
        "extract_url_content",
    ],
    "lab_bench": [
        "query_pubmed",
        "query_uniprot",
        "query_ensembl",
        "query_pdb",
        "blast_sequence",
        "search_google",
    ],
    "lab_bench_seqqa": [
        "query_pubmed",
        "query_uniprot",
        "query_ensembl",
        "blast_sequence",
    ],
    "lab_bench_dbqa": [
        "query_pubmed",
        "query_uniprot",
        "query_pdb",
        "search_google",
    ],
    "screen_gene_retrieval": [
        "gene_set_enrichment_analysis",
        "query_opentarget",
        "query_ensembl",
        "query_uniprot",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Curated tool description builder
# ─────────────────────────────────────────────────────────────────────────────


def get_curated_tool_desc(task_name: str, module2api: dict) -> dict:
    """Build a filtered tool_desc dict containing only the curated tools.

    Args:
        task_name: Classified task name (key into TASK_TOOLS).
        module2api: Full {module_path: [tool_dict, ...]} from Biomni.

    Returns:
        Filtered dict in same format: {module_path: [tool_dict, ...]}.
        Only includes tools listed in TASK_TOOLS[task_name].
    """
    tool_names = set(TASK_TOOLS.get(task_name, []))
    if not tool_names:
        # Fallback: return all tools (same as baseline)
        return {
            mod: [t for t in tools if t["name"] != "run_python_repl"]
            for mod, tools in module2api.items()
        }

    curated = {}
    for module_path, tools in module2api.items():
        matched = [t for t in tools if t["name"] in tool_names]
        if matched:
            curated[module_path] = matched
    return curated

def get_curated_tool_desc_by_names(tool_names: list, module2api: dict) -> dict:
    """Build a filtered tool_desc dict containing only the specified tools by name.

    Unlike get_curated_tool_desc() which looks up TASK_TOOLS, this takes an explicit
    list of tool names. Used by _run_specialist() for pipeline-specific tool sets.
    """
    names_set = set(tool_names)
    curated = {}
    for module_path, tools in module2api.items():
        matched = [t for t in tools if t["name"] in names_set]
        if matched:
            curated[module_path] = matched
    return curated


# ─────────────────────────────────────────────────────────────────────────────
# 5. Task-specific instruction prompts
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: TASK_PROMPTS disabled in Exp#3 based on data analysis:
#   - 40/65 degraded tasks were no_tool→no_tool (prompt text caused wrong answers)
#   - Tag leak (71), V2G leak (6), placeholder leak (67) all from prompt injection
#   - Oracle recommendation: 20B model is prompt-sensitive, injection hurts
#   - Tool filtering (TASK_TOOLS) remains active — low-risk, reduces tool confusion
TASK_PROMPTS = {}  # Exp#3: All prompts disabled. Tool filtering via TASK_TOOLS only.


# ─────────────────────────────────────────────────────────────────────────────
# 5a. Tool-specific API knowledge for specialist agents (Exp#4)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_KNOWLEDGE = {
    "gwas_catalog": """You are a GWAS Catalog API specialist. Your ONLY job is to query the GWAS Catalog
and return structured findings about variant-trait associations.

## API Knowledge
- Use query_gwas_catalog(prompt="...") to query. The internal LLM translates your prompt to API calls.
- For looking up associations of a specific rsID, use: query_gwas_catalog(prompt="Find associations for SNP rsXXXXX")
- For trait-based searches, use: query_gwas_catalog(prompt="Find studies for trait XXX")

## Response Structure
The API returns raw JSON (no formatting applied). Key paths:
- result._embedded.associations[] — list of association objects
- Each association has: pvalueMantissa, pvalueExponent (p-value = mantissa * 10^exponent)
- Loci info: loci[0].strongestRiskAlleles[0].riskAlleleName (format: "rsXXXX-?" or "rsXXXX-A")
- Extract rsID: riskAlleleName.split("-")[0]

## Critical Instructions
- When given a list of rsIDs, query EACH rsID individually using query_gwas_catalog(prompt="Find associations for SNP rsXXXX")
- Calculate p-value as: float(pvalueMantissa) * (10 ** float(pvalueExponent))
- The SMALLEST p-value = STRONGEST association = the answer
- If an rsID returns no associations, it is NOT the answer
- Output the rsID with the smallest p-value as your <solution>
""",

    "monarch": """You are a Monarch Initiative API specialist. Your ONLY job is to query Monarch
for phenotype-disease-gene associations.

## API Knowledge
- Use query_monarch(prompt="...") for natural language queries
- Use query_monarch(endpoint="URL") for direct API calls
- Key endpoints:
  - Search: https://api.monarchinitiative.org/v3/api/search?q=TERM&category=biolink:Disease&limit=10
  - Entity details: https://api.monarchinitiative.org/v3/api/entity/MONDO:XXXXXXX
  - Associations: https://api.monarchinitiative.org/v3/api/association?subject=HP:XXXXXXX&category=biolink:DiseaseToPhenotypicFeatureAssociation
  - Semantic similarity: https://api.monarchinitiative.org/v3/api/semsim/search

## For patient_gene_detection tasks:
- You are given HPO terms (HP:XXXXXXX) and candidate genes (ENSG IDs)
- Use Monarch's semantic similarity search to find which diseases match the HPO profile
- Then identify which candidate gene is associated with those diseases
- The answer MUST be one of the given ENSG candidate genes
- Use semsim endpoint: POST to https://api.monarchinitiative.org/v3/api/semsim/search
  with body: {"termset": ["HP:0001250", "HP:0002300", ...], "group": "Human Diseases", "limit": 10}

## For rare_disease_diagnosis tasks:
- You are given HPO terms and a candidate gene (ENSG ID)
- Find what disease is associated with that gene + those phenotypes
- The answer must include disease_name and OMIM_ID
- First look up the gene in Monarch to find associated diseases
- Then match the disease that best fits the HPO phenotype profile
- OMIM IDs look like: 6-digit numbers (e.g., 114300)
- Use query_monarch(endpoint="https://api.monarchinitiative.org/v3/api/entity/ENSGXXX") to find gene info
""",

    "opentarget": """You are an OpenTargets Platform API specialist. Your ONLY job is to query
OpenTargets for gene-disease-drug associations using GraphQL.

## API Knowledge
- Use query_opentarget(prompt="...") for natural language queries
- The API uses GraphQL at https://api.platform.opentargets.org/api/v4/graphql
- Results are automatically formatted (truncated for context).

## For gwas_causal_gene tasks:
- Given a GWAS phenotype and genes in a locus, identify the causal gene
- Use V2G (Variant-to-Gene) scores or L2G (Locus-to-Gene) scores
- Query: query_opentarget(prompt="Find V2G or L2G scores for genes GENE1, GENE2 in TRAIT locus")
- The gene with the highest V2G/L2G score is likely causal
""",

    "ensembl": """You are an Ensembl REST API specialist. Your ONLY job is to look up gene information.

## API Knowledge
- Use query_ensembl(prompt="...") or query_ensembl(endpoint="...")
- Gene symbol lookup: query_ensembl(endpoint="lookup/symbol/homo_sapiens/BRCA2")
- ENSG ID lookup: query_ensembl(endpoint="lookup/id/ENSG00000139618")
- The response includes: id (ENSG ID), display_name (gene symbol), description, biotype

## Critical: ENSG ↔ Gene Symbol mapping
- To convert gene symbol → ENSG: query_ensembl(endpoint="lookup/symbol/homo_sapiens/SYMBOL")
- To convert ENSG → gene symbol: query_ensembl(endpoint="lookup/id/ENSGXXX")
""",
}


# ─────────────────────────────────────────────────────────────────────────────
# 5a-2. Hand-crafted specialist system prompts (Exp#5)
#        ~3K each, replacing _generate_system_prompt() (~27K)
# ─────────────────────────────────────────────────────────────────────────────

# Core A1 loop instructions — every specialist needs this to work with A1's
# generate→execute→generate→... state machine
_A1_CORE_INSTRUCTIONS = """You are an expert biomedical agent that solves problems step-by-step.

For every response, you MUST include exactly ONE of the following XML tags:

1. <execute>your python code here</execute> - For executing Python code
2. <solution>your final answer here</solution> - For providing the final answer

RULES:
- Every response must contain EXACTLY ONE of these tags.
- <execute> blocks run in a Python environment with access to imported tool functions.
- After <execute>, you will receive the result in <observation>result</observation>.
- Use <solution> ONLY when you have the final answer.
- When calling functions, you MUST save the output AND print it: result = func(...); print(result)
- IMPORTANT: You must import functions before using them: from biomni.tool.database import function_name
- Keep code simple. Print results clearly.
- In each response you must include EITHER <execute> or <solution>. Not both. No empty messages.
"""

SPECIALIST_SYSTEM_PROMPTS = {
    # ── GWAS Catalog Specialist (~2.5K) ──
    "gwas_catalog": _A1_CORE_INSTRUCTIONS + """

## Your Role
You are a GWAS Catalog API specialist. Query the GWAS Catalog to find variant-trait associations.

## Available Tool
Function: query_gwas_catalog(prompt: str, endpoint: str = None, max_results: int = 3)
  Import: from biomni.tool.database import query_gwas_catalog
  Description: Query the GWAS Catalog API using natural language or a direct endpoint.
  - prompt (str, required): Natural language query about GWAS data
  - endpoint (str, optional): Endpoint name (e.g., 'studies')
  - max_results (int, optional): Max results per page, default 3

## API Response Structure
The API returns raw JSON (no formatting). Key paths:
- result._embedded.associations[] — list of association objects
- Each association has: pvalueMantissa, pvalueExponent (p-value = mantissa * 10^exponent)
- Loci info: loci[0].strongestRiskAlleles[0].riskAlleleName (format: "rsXXXX-?" or "rsXXXX-A")
- Extract rsID: riskAlleleName.split("-")[0]

## Strategy for Variant Prioritization
When given a GWAS phenotype and candidate rsIDs:
1. Query EACH rsID individually: query_gwas_catalog(prompt="Find associations for SNP rsXXXX")
2. Parse the response to extract p-value: float(pvalueMantissa) * (10 ** float(pvalueExponent))
3. The SMALLEST p-value = STRONGEST association = the answer
4. If an rsID returns no associations, skip it
5. Return the rsID with the smallest p-value in your <solution> tag
""",

    # ── Monarch Specialist (~3K) ──
    "monarch": _A1_CORE_INSTRUCTIONS + """

## Your Role
You are a Monarch Initiative API specialist for phenotype-disease-gene associations.

## Available Tools
Function: query_monarch(prompt: str, endpoint: str = None, max_results: int = 2, verbose: bool = False)
  Import: from biomni.tool.database import query_monarch
  Description: Query Monarch Initiative API using natural language or direct endpoint.
  - prompt (str, required): Natural language query about genes/diseases/phenotypes
  - endpoint (str, optional): Direct endpoint or full URL
  - max_results (int, optional): Max results, default 2

Function: query_ensembl(prompt: str, endpoint: str = None, verbose: bool = True)
  Import: from biomni.tool.database import query_ensembl
  Description: Query Ensembl REST API for gene information.
  - prompt (str, required): Natural language query about genomic data
  - endpoint (str, optional): Direct Ensembl endpoint or full URL

Function: query_opentarget(prompt: str, query: str = None, variables: dict = None, verbose: bool = False)
  Import: from biomni.tool.database import query_opentarget
  Description: Query OpenTargets Platform API using natural language or GraphQL.
  - prompt (str, required): Natural language query about targets/diseases

## Monarch API Endpoints
- Search: https://api.monarchinitiative.org/v3/api/search?q=TERM&category=biolink:Disease&limit=10
- Entity: https://api.monarchinitiative.org/v3/api/entity/MONDO:XXXXXXX
- Associations: https://api.monarchinitiative.org/v3/api/association?subject=HP:XXXXXXX&category=biolink:DiseaseToPhenotypicFeatureAssociation
- Semantic similarity: POST https://api.monarchinitiative.org/v3/api/semsim/search
  body: {"termset": ["HP:0001250", ...], "group": "Human Diseases", "limit": 10}

## Ensembl Endpoints
- Gene symbol lookup: query_ensembl(endpoint="lookup/symbol/homo_sapiens/BRCA2")
- ENSG ID lookup: query_ensembl(endpoint="lookup/id/ENSG00000139618")
- Response includes: id (ENSG ID), display_name (gene symbol)

## Strategy for patient_gene_detection
Given HPO terms and candidate ENSG IDs:
1. Use Monarch semsim to find diseases matching the HPO profile
2. Check which candidate gene associates with top-matching disease
3. Answer MUST be one of the given ENSG IDs
4. Return as: {"causal_gene": ["ENSGXXX"]} in <solution> tag

## Strategy for rare_disease_diagnosis
Given HPO terms and a candidate ENSG gene:
1. Look up gene in Ensembl to get gene symbol: query_ensembl(endpoint="lookup/id/ENSGXXX")
2. Search Monarch for diseases associated with that gene
3. Match disease whose phenotype profile best fits the HPO terms
4. Get the OMIM ID (6-digit number like 114300)
5. Return as: {"disease_name": "XXX", "OMIM_ID": "XXXXXX"} in <solution> tag
""",

    # ── OpenTargets Specialist (~2K) ──
    "opentarget": _A1_CORE_INSTRUCTIONS + """

## Your Role
You are an OpenTargets Platform API specialist for gene-disease associations.

## Available Tool
Function: query_opentarget(prompt: str, query: str = None, variables: dict = None, verbose: bool = False)
  Import: from biomni.tool.database import query_opentarget
  Description: Query OpenTargets Platform API using natural language or GraphQL.
  - prompt (str, required): Natural language query
  - query (str, optional): Direct GraphQL query string
  - variables (dict, optional): Variables for GraphQL

## Strategy for gwas_causal_gene tasks
Given a GWAS phenotype and genes in a locus:
1. Use V2G (Variant-to-Gene) or L2G (Locus-to-Gene) scores
2. Query: query_opentarget(prompt="Find V2G scores for GENE1, GENE2 in TRAIT locus")
3. The gene with highest V2G/L2G score is likely causal
4. Return the gene symbol in <solution> tag
""",

    # ── Ensembl Specialist (~1.5K) ──
    "ensembl": _A1_CORE_INSTRUCTIONS + """

## Your Role
You are an Ensembl REST API specialist for gene information lookup.

## Available Tool
Function: query_ensembl(prompt: str, endpoint: str = None, verbose: bool = True)
  Import: from biomni.tool.database import query_ensembl
  Description: Query Ensembl REST API for gene information.
  - prompt (str, required): Natural language query
  - endpoint (str, optional): Direct endpoint or full URL

## Key Endpoints
- Gene symbol to ENSG: query_ensembl(endpoint="lookup/symbol/homo_sapiens/BRCA2")
- ENSG to gene info: query_ensembl(endpoint="lookup/id/ENSG00000139618")
- Response: id (ENSG ID), display_name (gene symbol), description, biotype
""",
}


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Task pipelines — which specialist handles which task (Exp#4)
#     ONLY failing tasks get specialist treatment. Passing tasks use fallback.
# ─────────────────────────────────────────────────────────────────────────────

TASK_PIPELINES = {
    # Baseline: 3/43 (7%) — wrong rsID from API parsing failure
    "gwas_variant_prioritization": {
        "specialist": "gwas_catalog",
        "tools": ["query_gwas_catalog"],
        "sub_prompt_template": (
            "You are given a GWAS phenotype and a list of candidate rsIDs.\n"
            "Your task: query the GWAS Catalog for EACH rsID to find its association p-value "
            "for the given phenotype, then return the rsID with the STRONGEST association (smallest p-value).\n\n"
            "GWAS phenotype: {trait}\n"
            "Candidate rsIDs: {variants}\n\n"
            "IMPORTANT: Query each rsID individually. Compare p-values. "
            "Return ONLY the single rsID with the smallest p-value in your <solution> tag."
        ),
    },
    # Baseline: 1/50 (2%) — gene symbol output instead of ENSG
    "patient_gene_detection": {
        "specialist": "monarch",
        "tools": ["query_monarch", "query_ensembl", "query_opentarget"],
        "sub_prompt_template": (
            "You are given a patient's phenotypes (HPO terms) and a list of candidate genes (ENSG IDs).\n"
            "Your task: determine which candidate gene is most likely causal for the patient's phenotype profile.\n\n"
            "Patient phenotypes: {phenotypes}\n"
            "Candidate genes: {candidates}\n\n"
            "Strategy:\n"
            "1. Use Monarch's semantic similarity search to find diseases matching the HPO profile\n"
            "2. Check which candidate gene is associated with the top-matching disease\n"
            "3. Your answer MUST be one of the candidate ENSG IDs listed above\n\n"
            "Return ONLY the causal ENSG ID in format: {{\"causal_gene\": [\"ENSGXXX\"]}} in your <solution> tag."
        ),
    },
    # Baseline: 0/30 (0%) — HPO→OMIM mapping failure
    "rare_disease_diagnosis": {
        "specialist": "monarch",
        "tools": ["query_monarch", "query_ensembl", "query_opentarget"],
        "sub_prompt_template": (
            "You are given a patient's phenotypes (HPO terms) and a candidate gene (ENSG ID).\n"
            "Your task: diagnose the rare disease by finding the OMIM disease associated with this gene and phenotype profile.\n\n"
            "Patient phenotypes: {phenotypes}\n"
            "Candidate gene: {candidate_gene}\n\n"
            "Strategy:\n"
            "1. Look up the candidate gene in Monarch/Ensembl to find its gene symbol and associated diseases\n"
            "2. Among associated diseases, find the one whose phenotype profile best matches the patient's HPO terms\n"
            "3. Get the OMIM ID for that disease\n\n"
            "Return in format: {{\"disease_name\": \"XXX\", \"OMIM_ID\": \"XXXXXX\"}} in your <solution> tag."
        ),
    },
}


def _extract_task_params(prompt: str, task_name: str) -> dict:
    """Extract structured parameters from benchmark prompt for specialist sub-prompt."""
    params = {}

    if task_name == "gwas_variant_prioritization":
        # Extract: GWAS phenotype: XXX\nVariants: rs..., rs...
        trait_match = re.search(r"GWAS phenotype:\s*(.+)", prompt)
        variants_match = re.search(r"Variants:\s*(.+)", prompt)
        params["trait"] = trait_match.group(1).strip() if trait_match else "unknown"
        params["variants"] = variants_match.group(1).strip() if variants_match else ""

    elif task_name == "patient_gene_detection":
        # Extract: Phenotypes: HP:XXX, HP:XXX\nCandidate genes: ENSG..., ENSG...
        pheno_match = re.search(r"Phenotypes:\s*(.+)", prompt)
        genes_match = re.search(r"Candidate genes:\s*(.+)", prompt)
        params["phenotypes"] = pheno_match.group(1).strip() if pheno_match else ""
        params["candidates"] = genes_match.group(1).strip() if genes_match else ""

    elif task_name == "rare_disease_diagnosis":
        # Extract: Phenotypes: HP:XXX\nCandidate genes: ['ENSGXXX']
        pheno_match = re.search(r"Phenotypes:\s*(.+)", prompt)
        gene_match = re.search(r"Candidate genes:\s*\[?'?(ENSG\d+)", prompt)
        params["phenotypes"] = pheno_match.group(1).strip() if pheno_match else ""
        params["candidate_gene"] = gene_match.group(1) if gene_match else ""

    return params


# ─────────────────────────────────────────────────────────────────────────────
# 6. Post-processor
# ─────────────────────────────────────────────────────────────────────────────


def postprocess_answer(raw_output: str, task_name: str) -> str:
    """Task-specific output normalization.

    Applied after A1 loop completes but before returning to benchmark scorer.
    """
    if not raw_output:
        return ""

    answer = str(raw_output).strip()

    # Step 1: Extract <solution> tag content if present
    sol_match = re.search(r"<solution>(.*?)</solution>", answer, re.DOTALL)
    if sol_match:
        answer = sol_match.group(1).strip()

    # Step 2: Extract from "Final Answer:" markers
    for marker in ["Final Answer:", "FINAL ANSWER:", "Answer:", "ANSWER:"]:
        if marker in answer:
            answer = answer.split(marker)[-1].strip()
            break

    # Step 3: Strip markdown
    answer = re.sub(r"```[\s\S]*?```", "", answer)
    answer = re.sub(r"`([^`]+)`", r"\1", answer)
    answer = re.sub(r"^>\s?", "", answer, flags=re.MULTILINE)
    answer = answer.strip()

    # Step 4: Task-specific normalization
    if task_name in ("lab_bench_seqqa", "lab_bench_dbqa", "lab_bench"):
        return _extract_mcq_letter(answer, valid="ABCDE")

    if task_name == "crispr_delivery":
        return _extract_mcq_letter(answer, valid="ABCDEF", lowercase=True)

    if task_name == "gwas_variant_prioritization":
        rsids = re.findall(r"rs\d+", answer)
        return rsids[-1] if rsids else answer.strip()

    if task_name.startswith("gwas_causal_gene"):
        return _extract_gene_symbol(answer)

    if task_name == "screen_gene_retrieval":
        return _extract_gene_symbol(answer)

    if task_name == "patient_gene_detection":
        return _extract_patient_gene_json(answer)

    if task_name == "rare_disease_diagnosis":
        return _extract_rare_disease_json(answer)

    return answer


def _extract_mcq_letter(
    text: str, valid: str = "ABCDE", lowercase: bool = False
) -> str:
    """Extract multiple choice letter from text."""
    text = text.replace("[ANSWER]", "").replace("[/ANSWER]", "")
    text = text.replace("**", "").replace("*", "")

    # Check for explicit patterns
    patterns = [
        r"(?:answer|choice)\s*(?:is)?\s*[:=]?\s*([{v}])\b".format(
            v=valid.lower() + valid
        ),
        r"\b([{v}])\)".format(v=valid),
        r"\(([{v}])\)".format(v=valid),
        r"^\s*([{v}{vl}])\s*$".format(v=valid, vl=valid.lower()),
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            letter = match.group(1)
            return letter.lower() if lowercase else letter.upper()

    # Last resort: short text containing a valid letter
    text_clean = text.strip()
    if len(text_clean) <= 5:
        for c in text_clean:
            if c.upper() in valid:
                return c.lower() if lowercase else c.upper()

    return text_clean


def _extract_gene_symbol(text: str) -> str:
    """Extract gene symbol from text."""
    text = text.replace("**", "").replace("*", "")

    # Priority 1: Explicit "gene is X" patterns
    explicit = re.search(
        r"(?:gene|causal gene|target gene)\s*(?:is|:|=)\s*([A-Z][A-Z0-9]{1,9})\b",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).upper()

    # Priority 2: Uppercase gene-like tokens
    genes = re.findall(r"\b([A-Z][A-Z0-9-]{1,15})\b", text)
    noise = {
        "THE",
        "AND",
        "FOR",
        "NOT",
        "WITH",
        "FROM",
        "THIS",
        "THAT",
        "ARE",
        "WAS",
        "HAS",
        "HAD",
        "BUT",
        "ALL",
        "CAN",
        "HER",
        "ONE",
        "OUR",
        "OUT",
        "YOU",
        "ITS",
        "MAY",
        "WHO",
        "NOW",
        "GET",
        "USE",
        "NEW",
        "OLD",
        "YES",
        "NO",
        "GWAS",
        "SNP",
        "DNA",
        "RNA",
        "OMIM",
        "ID",
        "HTTP",
        "API",
        "URL",
        "NONE",
        "NULL",
        "TRUE",
        "FALSE",
        "ERROR",
        "NA",
        "GENE",
        "STUDY",
        "RISK",
        "LOCUS",
        "REGION",
        "VARIANT",
        "BASED",
        "ANALYSIS",
        "RESULT",
        "DATA",
        "TABLE",
        "FIGURE",
        "CONCLUSION",
        "SUMMARY",
        "ANSWER",
        "FINAL",
        "NOTE",
        "MOST",
        "LIKELY",
        "ASSOCIATED",
        "SIGNIFICANT",
        "FOUND",
        "IDENTIFIED",
        "REPORT",
        "EVIDENCE",
        "STRONG",
        "HIGH",
    }
    genes = [g for g in genes if g not in noise]
    if genes:
        return genes[0]

    first_line = text.split("\n")[0].strip()
    return first_line[:50] if len(first_line) > 50 else first_line


def _extract_patient_gene_json(answer: str) -> str:
    """Extract causal_gene ENSG IDs as JSON."""
    try:
        d = json.loads(answer) if isinstance(answer, str) else answer
        genes = d.get("causal_gene", [])
        if isinstance(genes, list) and genes:
            return json.dumps({"causal_gene": genes})
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    # Try Python dict format
    try:
        answer_clean = answer.replace("'", '"')
        d = json.loads(answer_clean)
        genes = d.get("causal_gene", [])
        if isinstance(genes, list) and genes:
            return json.dumps({"causal_gene": genes})
    except (json.JSONDecodeError, ValueError):
        pass

    # Direct ENSG extraction
    ensg = re.findall(r"ENSG\d+", answer)
    if ensg:
        return json.dumps({"causal_gene": ensg[:1]})

    return answer


def _extract_rare_disease_json(answer: str) -> str:
    """Extract disease JSON with OMIM_ID."""
    # Try JSON parse
    for text in [answer, answer.replace("'", '"')]:
        try:
            d = json.loads(text)
            if "OMIM_ID" in d:
                return json.dumps(d)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try extracting OMIM_ID directly
    omim_match = re.search(r"OMIM[_\s]*(?:ID)?[:\s]*(\d{6})", answer, re.IGNORECASE)
    if omim_match:
        omim_id = omim_match.group(1)
        # Try to extract disease name too
        name_match = re.search(
            r"disease[_\s]*name[:\s]*['\"]?([^'\"}\n]+)", answer, re.IGNORECASE
        )
        disease_name = name_match.group(1).strip() if name_match else "Unknown"
        return json.dumps({"disease_name": disease_name, "OMIM_ID": omim_id})

    return answer


# ─────────────────────────────────────────────────────────────────────────────
# 7. BiomniA1MultiAgent class
# ─────────────────────────────────────────────────────────────────────────────


class BiomniA1MultiAgent:
    """Multi-agent adapter that wraps Biomni A1 with task-specific configurations.

    Key differences from baseline (BiomniA1Agent):
    1. Disables ToolRetriever — injects curated tools per task type
    2. Prepends task-specific structured instructions to system prompt
    3. Post-processes raw output for format normalization
    """

    def __init__(
        self,
        biomni_path: Optional[str] = None,
        data_path: Optional[str] = None,
        llm: Optional[str] = None,
        source: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: int = 600,
        skip_datalake_download: bool = False,
        pool_size: int = 1,
    ):
        """Initialize BiomniA1MultiAgent.

        Same interface as BiomniA1Agent but forces use_tool_retriever=False.
        """
        # Resolve Biomni repo path
        if biomni_path is None:
            biomni_path = os.getenv(
                "BIOMNI_REPO_PATH",
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "Biomni"),
            )
        biomni_path = os.path.abspath(biomni_path)

        if not os.path.isdir(biomni_path):
            raise FileNotFoundError(
                "Biomni repository not found at %s. "
                "Set BIOMNI_REPO_PATH or pass biomni_path." % biomni_path
            )

        if biomni_path not in sys.path:
            sys.path.insert(0, biomni_path)
            logger.info("Added Biomni to sys.path: %s", biomni_path)

        # Load Biomni .env
        biomni_env = os.path.join(biomni_path, ".env")
        if os.path.isfile(biomni_env):
            load_dotenv(biomni_env, override=False)
            logger.info("Loaded Biomni .env from %s", biomni_env)

        # Resolve data path
        if data_path is None:
            data_path = os.getenv("BIOMNI_PATH", os.path.join(biomni_path, "data"))

        logger.info("Initializing Biomni A1 Multi-Agent...")
        logger.info("  Biomni repo: %s", biomni_path)
        logger.info("  Data path: %s", data_path)
        logger.info("  Tool retriever: DISABLED (curated tools per task)")
        logger.info("  Timeout: %ds", timeout_seconds)

        # Import Biomni A1
        try:
            from biomni.agent.a1 import A1
        except ImportError as e:
            raise ImportError(
                "Failed to import Biomni A1 agent. "
                "Make sure Biomni is installed: cd %s && pip install -e .\n"
                "Error: %s" % (biomni_path, e)
            ) from e

        # Build kwargs — force use_tool_retriever=False
        a1_kwargs = {
            "path": data_path,
            "use_tool_retriever": False,  # CRITICAL: bypass ToolRetriever
            "timeout_seconds": timeout_seconds,
        }
        if llm is not None:
            a1_kwargs["llm"] = llm
        if source is not None:
            a1_kwargs["source"] = source
        if base_url is not None:
            a1_kwargs["base_url"] = base_url
        if api_key is not None:
            a1_kwargs["api_key"] = api_key
        if skip_datalake_download:
            a1_kwargs["expected_data_lake_files"] = []

        # Create pool of A1 instances
        self._pool = queue.Queue()
        logger.info("Creating %d A1 instance(s) for parallel execution...", pool_size)
        for i in range(pool_size):
            init_start = time.time()
            a1 = A1(**a1_kwargs)
            init_time = time.time() - init_start
            self._pool.put(a1)
            logger.info(
                "  A1 instance %d/%d initialized in %.1fs", i + 1, pool_size, init_time
            )
        logger.info("All %d A1 instance(s) ready (multi-agent mode)", pool_size)

    def predict(self, prompt, task_id="unknown"):
        """Run the multi-agent pipeline: classify → inject tools → A1 loop → postprocess.

        Args:
            prompt: The benchmark question/prompt.
            task_id: Identifier (e.g. 'gwas_variant_prioritization_134').
        Returns:
            str: The agent's normalized answer.
        """
        from langchain_core.messages import HumanMessage

        worker = threading.current_thread().name
        start = time.time()

        # Step 1: Determine task_name
        task_name = self._resolve_task_name(task_id, prompt)
        logger.info(
            "[MULTI] task=%s | worker=%s | task_name=%s | prompt_chars=%d",
            task_id,
            worker,
            task_name,
            len(prompt),
        )

        # Step 1.5: Check if this task has a specialist pipeline (Exp#5 — lightweight)
        pipeline = TASK_PIPELINES.get(task_name)
        if pipeline:
            logger.info(
                "[MULTI] Routing to specialist pipeline: task=%s | specialist=%s",
                task_id, pipeline["specialist"],
            )
            try:
                a1 = self._pool.get()
                try:
                    return self._run_specialist_v2(a1, task_name, pipeline, prompt, task_id)
                finally:
                    self._pool.put(a1)
            except Exception as e:
                latency = time.time() - start
                logger.error(
                    "[SPECIALIST_ERR] task=%s | latency=%.1fs | error=%s: %s",
                    task_id, latency, type(e).__name__, e,
                )
                return "Error: %s" % str(e)

        # Fallback: existing single-agent path (for passing tasks like gwas_causal_gene)
        try:
            # Get an A1 instance from the pool
            a1 = self._pool.get()
            try:
                # Step 2: Inject curated tools into system prompt
                self._inject_curated_system_prompt(a1, task_name, prompt)

                # Step 2.5: Reset MemorySaver to prevent state contamination
                # Each task MUST start with a fresh conversation history.
                # Without this, previous task messages leak into the new task,
                # causing the LLM to skip tool calls (thinks it already ran them).
                from langgraph.checkpoint.memory import MemorySaver
                a1.checkpointer = MemorySaver()
                a1.app.checkpointer = a1.checkpointer

                # Initialize a1.log (normally done in a1.run()/a1.run_stream(),
                # but we call a1.app.stream() directly so must init manually)
                a1.log = []

                # Step 3: Run A1 loop (same as baseline)
                a1.critic_count = 0
                a1.user_task = prompt

                inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
                config = {
                    "recursion_limit": 500,
                    "configurable": {"thread_id": task_id},
                }

                token_count = 0
                reasoning_count = 0
                ttft = None
                step_count = 0
                answer = ""
                progress_interval = 50
                next_progress = progress_interval
                tools_called = []
                seen_execute_blocks = set()

                for mode, data in a1.app.stream(
                    inputs, stream_mode=["messages", "values"], config=config
                ):
                    if mode == "messages":
                        chunk, metadata = data
                        content = getattr(chunk, "content", "") or ""
                        reasoning = getattr(chunk, "reasoning_content", "") or ""

                        if content:
                            token_count += 1
                        if reasoning:
                            reasoning_count += 1

                        if content or reasoning:
                            now = time.time()
                            if ttft is None:
                                ttft = now - start
                                logger.info(
                                    "[STREAM_START] task=%s | worker=%s | ttft=%.2fs",
                                    task_id,
                                    worker,
                                    ttft,
                                )
                            total = token_count + reasoning_count
                            if total >= next_progress:
                                logger.info(
                                    "[STREAM_PROGRESS] task=%s | worker=%s | tokens_so_far=%d | reasoning=%d | elapsed=%.1fs",
                                    task_id,
                                    worker,
                                    total,
                                    reasoning_count,
                                    now - start,
                                )
                                next_progress += progress_interval

                    elif mode == "values":
                        step_count += 1
                        message = data["messages"][-1]
                        answer = message.content

                        # Tool call detection
                        msg_content = str(message.content) if message.content else ""
                        execute_blocks = re.findall(
                            r"<execute>(.*?)</execute>", msg_content, re.DOTALL
                        )
                        for code_block in execute_blocks:
                            block_hash = hash(code_block.strip())
                            if block_hash in seen_execute_blocks:
                                continue
                            seen_execute_blocks.add(block_hash)
                            try:
                                new_tools = a1._parse_tool_calls_from_code(code_block)
                                for tool_name in new_tools:
                                    if tool_name not in tools_called:
                                        tools_called.append(tool_name)
                                        logger.info(
                                            "[TOOL_CALL] task=%s | worker=%s | tool=%s | step=%d | elapsed=%.1fs",
                                            task_id,
                                            worker,
                                            tool_name,
                                            step_count,
                                            time.time() - start,
                                        )
                            except Exception as e:
                                logger.debug(
                                    "[TOOL_PARSE_ERR] task=%s | error=%s", task_id, e
                                )

                        # Preserve A1 log
                        try:
                            from biomni.utils.utils import pretty_print

                            out = pretty_print(message)
                            a1.log.append(out)
                        except ImportError:
                            a1.log.append(str(message.content)[:200])

            finally:
                self._pool.put(a1)

            # Step 4: Post-process answer
            latency = time.time() - start
            clean_answer = postprocess_answer(answer, task_name)

            logger.info(
                "[STREAM_END] task=%s | worker=%s | task_name=%s | total_tokens=%d | reasoning_tokens=%d | "
                "ttft=%.2fs | latency=%.1fs | tools=%s | answer=%s",
                task_id,
                worker,
                task_name,
                token_count,
                reasoning_count,
                ttft if ttft is not None else -1,
                latency,
                ",".join(tools_called) if tools_called else "none",
                clean_answer[:80],
            )
            return clean_answer

        except Exception as e:
            latency = time.time() - start
            logger.error(
                "[ERR] task=%s | worker=%s | latency=%.1fs | error=%s: %s",
                task_id,
                worker,
                latency,
                type(e).__name__,
                e,
            )
            return "Error: %s" % str(e)

    def _resolve_task_name(self, task_id: str, prompt: str) -> str:
        """Resolve task_name from task_id or prompt classification.

        Priority: task_id parse > prompt classification > 'unknown'
        """
        # Try extracting from task_id (e.g. 'gwas_variant_prioritization_134')
        parts = task_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            candidate = parts[0]
            if candidate in TASK_TOOLS or candidate in TASK_PROMPTS:
                return candidate

        # Fall back to prompt classification
        classified = classify_prompt(prompt)
        if classified != "unknown":
            return classified

        return "unknown"

    def _inject_curated_system_prompt(self, a1, task_name: str, prompt: str):
        """Rebuild A1's system prompt with curated tools and task instructions.

        This replaces what ToolRetriever normally does:
        1. Filters module2api to curated tools only
        2. Regenerates system prompt via a1._generate_system_prompt()
        3. Prepends task-specific instructions
        """
        import glob

        # Build curated tool_desc
        curated_tool_desc = get_curated_tool_desc(task_name, a1.module2api)

        # Prepare data lake items (same as configure())
        data_lake_path = a1.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]

        data_lake_with_desc = []
        for item in data_lake_items:
            description = a1.data_lake_dict.get(item, "Data lake item: %s" % item)
            data_lake_with_desc.append({"name": item, "description": description})

        # Prepare library content
        library_content_list = list(a1.library_content_dict.keys())

        # NOTE: know_how_docs deliberately EXCLUDED to reduce prompt size
        # (50K→5K chars). Know-how docs caused LLM to hit context limits
        # and fail to produce <execute> tags. The curated tool descriptions
        # + task-specific instructions provide sufficient guidance.

        # Regenerate system prompt with curated tools (NO know_how_docs)
        system_prompt = a1._generate_system_prompt(
            tool_desc=curated_tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=False,
            is_retrieval=True,
            know_how_docs=None,  # Excluded: too large, causes tool-calling failure
        )

        # Exp#3: TASK_PROMPTS injection DISABLED.
        # Data analysis showed prompt injection harms 20B model performance:
        #   - 65 degraded tasks (BL correct → ML wrong), 40 without any tool change
        #   - Tag/V2G/placeholder leaks account for 144/312 wrong answers
        # Only curated tool filtering (TASK_TOOLS) is active.
        # task_instruction = TASK_PROMPTS.get(task_name, "")
        # if task_instruction:
        #     system_prompt = (
        #         "=== TASK-SPECIFIC INSTRUCTIONS ===\n"
        #         + task_instruction
        #         + "\n=== END TASK-SPECIFIC INSTRUCTIONS ===\n\n"
        #         + system_prompt
        #     )

        # NOTE: GPT formatting reminder is NOT added here because
        # A1's generate() node already appends the identical reminder
        # at inference time (a1.py L1384-1387). Adding it here would be a duplicate.

        a1.system_prompt = system_prompt

        logger.info(
            "[INJECT] task_name=%s | tools=%d | instruction_len=%d | prompt_len=%d",
            task_name,
            sum(len(v) for v in curated_tool_desc.values()),
            0,  # Exp#3: TASK_PROMPTS disabled
            len(system_prompt),
        )

    def _run_specialist(self, a1, task_name, pipeline, prompt, task_id):
        """Run a specialist agent: small context + domain-specific TOOL_KNOWLEDGE.

        This is the core of Exp#4's multi-agent approach:
        1. Extracts structured params from the benchmark prompt
        2. Builds a specialist sub-prompt with clear instructions
        3. Filters tools to ONLY those needed for this pipeline
        4. Injects TOOL_KNOWLEDGE (API parsing guidance) into system prompt
        5. Runs the same A1 loop (generate→execute→generate→...) with reduced context

        The key insight: 28K system prompt → ~5K, with domain expertise added.
        Same 20B model, but with less noise and more relevant knowledge.
        """
        import glob
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import MemorySaver

        worker = threading.current_thread().name
        start = time.time()

        # 1. Extract structured params and build specialist sub-prompt
        params = _extract_task_params(prompt, task_name)
        try:
            sub_prompt = pipeline["sub_prompt_template"].format(**params)
        except KeyError as e:
            logger.warning(
                "[SPECIALIST] param extraction failed for task=%s: %s. Falling back to raw prompt.",
                task_id, e
            )
            sub_prompt = prompt  # Fallback: use original prompt as-is

        # 2. Filter tools to pipeline-specific set
        curated_tool_desc = get_curated_tool_desc_by_names(pipeline["tools"], a1.module2api)

        # 3. Build reduced system prompt (NO know_how_docs → saves ~50K chars)
        data_lake_path = a1.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]
        data_lake_with_desc = []
        for item in data_lake_items:
            description = a1.data_lake_dict.get(item, "Data lake item: %s" % item)
            data_lake_with_desc.append({"name": item, "description": description})
        library_content_list = list(a1.library_content_dict.keys())

        base_prompt = a1._generate_system_prompt(
            tool_desc=curated_tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=False,
            is_retrieval=True,
            know_how_docs=None,  # Excluded: too large
        )

        # 4. Inject TOOL_KNOWLEDGE at the front of system prompt
        specialist_knowledge = TOOL_KNOWLEDGE.get(pipeline["specialist"], "")
        a1.system_prompt = specialist_knowledge + "\n\n" + base_prompt

        prompt_len = len(a1.system_prompt)
        logger.info(
            "[SPECIALIST] task=%s | specialist=%s | tools=%s | prompt_len=%d | sub_prompt_len=%d",
            task_id, pipeline["specialist"],
            ",".join(pipeline["tools"]), prompt_len, len(sub_prompt),
        )

        # 5. Reset conversation state for fresh specialist run
        a1.checkpointer = MemorySaver()
        a1.app.checkpointer = a1.checkpointer
        a1.log = []
        a1.critic_count = 0
        a1.user_task = sub_prompt

        # 6. Run A1 loop with specialist sub-prompt
        inputs = {"messages": [HumanMessage(content=sub_prompt)], "next_step": None}
        config = {
            "recursion_limit": 500,
            "configurable": {"thread_id": task_id},
        }

        token_count = 0
        reasoning_count = 0
        ttft = None
        step_count = 0
        answer = ""
        progress_interval = 50
        next_progress = progress_interval
        tools_called = []
        seen_execute_blocks = set()

        for mode, data in a1.app.stream(
            inputs, stream_mode=["messages", "values"], config=config
        ):
            if mode == "messages":
                chunk, metadata = data
                content = getattr(chunk, "content", "") or ""
                reasoning = getattr(chunk, "reasoning_content", "") or ""

                if content:
                    token_count += 1
                if reasoning:
                    reasoning_count += 1

                if content or reasoning:
                    now = time.time()
                    if ttft is None:
                        ttft = now - start
                        logger.info(
                            "[SPECIALIST_START] task=%s | worker=%s | ttft=%.2fs",
                            task_id, worker, ttft,
                        )
                    total = token_count + reasoning_count
                    if total >= next_progress:
                        logger.info(
                            "[SPECIALIST_PROGRESS] task=%s | tokens=%d | reasoning=%d | elapsed=%.1fs",
                            task_id, total, reasoning_count, now - start,
                        )
                        next_progress += progress_interval

            elif mode == "values":
                step_count += 1
                message = data["messages"][-1]
                answer = message.content

                # Tool call detection
                msg_content = str(message.content) if message.content else ""
                execute_blocks = re.findall(
                    r"<execute>(.*?)</execute>", msg_content, re.DOTALL
                )
                for code_block in execute_blocks:
                    block_hash = hash(code_block.strip())
                    if block_hash in seen_execute_blocks:
                        continue
                    seen_execute_blocks.add(block_hash)
                    try:
                        new_tools = a1._parse_tool_calls_from_code(code_block)
                        for tool_name in new_tools:
                            if tool_name not in tools_called:
                                tools_called.append(tool_name)
                                logger.info(
                                    "[SPECIALIST_TOOL] task=%s | tool=%s | step=%d | elapsed=%.1fs",
                                    task_id, tool_name, step_count, time.time() - start,
                                )
                    except Exception as e:
                        logger.debug("[SPECIALIST_TOOL_ERR] task=%s | error=%s", task_id, e)

                # Preserve A1 log
                try:
                    from biomni.utils.utils import pretty_print
                    out = pretty_print(message)
                    a1.log.append(out)
                except ImportError:
                    a1.log.append(str(message.content)[:200])

        # 7. Post-process and return
        latency = time.time() - start
        clean_answer = postprocess_answer(answer, task_name)

        logger.info(
            "[SPECIALIST_END] task=%s | specialist=%s | tokens=%d | reasoning=%d | "
            "ttft=%.2fs | latency=%.1fs | steps=%d | tools=%s | answer=%s",
            task_id, pipeline["specialist"],
            token_count, reasoning_count,
            ttft if ttft is not None else -1,
            latency, step_count,
            ",".join(tools_called) if tools_called else "none",
            clean_answer[:80],
        )
        return clean_answer

    def _run_specialist_v2(self, a1, task_name, pipeline, prompt, task_id):
        """Run a lightweight specialist agent with hand-crafted system prompt (Exp#5).

        Key difference from _run_specialist (Exp#4):
          - Exp#4: a1._generate_system_prompt() → 27K chars system prompt
          - Exp#5: SPECIALIST_SYSTEM_PROMPTS[specialist] → ~3K chars

        Same A1 loop (generate→execute→generate), but with 9x smaller context.
        """
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import MemorySaver

        worker = threading.current_thread().name
        start = time.time()

        # 1. Extract structured params and build specialist sub-prompt
        params = _extract_task_params(prompt, task_name)
        try:
            sub_prompt = pipeline["sub_prompt_template"].format(**params)
        except KeyError as e:
            logger.warning(
                "[SPECIALIST_V2] param extraction failed for task=%s: %s. Using raw prompt.",
                task_id, e
            )
            sub_prompt = prompt

        # 2. Set hand-crafted system prompt (~3K instead of ~27K)
        specialist_key = pipeline["specialist"]
        system_prompt = SPECIALIST_SYSTEM_PROMPTS.get(specialist_key)
        if not system_prompt:
            # Fallback: use TOOL_KNOWLEDGE + minimal A1 instructions
            logger.warning(
                "[SPECIALIST_V2] No hand-crafted prompt for specialist=%s. Using fallback.",
                specialist_key,
            )
            system_prompt = _A1_CORE_INSTRUCTIONS + "\n" + TOOL_KNOWLEDGE.get(specialist_key, "")

        # 3. Inject into A1 instance
        a1.system_prompt = system_prompt

        prompt_len = len(a1.system_prompt)
        logger.info(
            "[SPECIALIST_V2] task=%s | specialist=%s | tools=%s | prompt_len=%d | sub_prompt_len=%d",
            task_id, specialist_key,
            ",".join(pipeline["tools"]), prompt_len, len(sub_prompt),
        )

        # 4. Reset conversation state for fresh specialist run
        a1.checkpointer = MemorySaver()
        a1.app.checkpointer = a1.checkpointer
        a1.log = []
        a1.critic_count = 0
        a1.user_task = sub_prompt

        # 5. Run A1 loop with specialist sub-prompt
        inputs = {"messages": [HumanMessage(content=sub_prompt)], "next_step": None}
        config = {
            "recursion_limit": 500,
            "configurable": {"thread_id": task_id},
        }

        token_count = 0
        reasoning_count = 0
        ttft = None
        step_count = 0
        answer = ""
        progress_interval = 50
        next_progress = progress_interval
        tools_called = []
        seen_execute_blocks = set()

        for mode, data in a1.app.stream(
            inputs, stream_mode=["messages", "values"], config=config
        ):
            if mode == "messages":
                chunk, metadata = data
                content = getattr(chunk, "content", "") or ""
                reasoning = getattr(chunk, "reasoning_content", "") or ""

                if content:
                    token_count += 1
                if reasoning:
                    reasoning_count += 1

                if content or reasoning:
                    now = time.time()
                    if ttft is None:
                        ttft = now - start
                        logger.info(
                            "[SPECIALIST_V2_START] task=%s | worker=%s | ttft=%.2fs",
                            task_id, worker, ttft,
                        )
                    total = token_count + reasoning_count
                    if total >= next_progress:
                        logger.info(
                            "[SPECIALIST_V2_PROGRESS] task=%s | tokens=%d | reasoning=%d | elapsed=%.1fs",
                            task_id, total, reasoning_count, now - start,
                        )
                        next_progress += progress_interval

            elif mode == "values":
                step_count += 1
                message = data["messages"][-1]
                answer = message.content

                # Tool call detection
                msg_content = str(message.content) if message.content else ""
                execute_blocks = re.findall(
                    r"<execute>(.*?)</execute>", msg_content, re.DOTALL
                )
                for code_block in execute_blocks:
                    block_hash = hash(code_block.strip())
                    if block_hash in seen_execute_blocks:
                        continue
                    seen_execute_blocks.add(block_hash)
                    try:
                        new_tools = a1._parse_tool_calls_from_code(code_block)
                        for tool_name in new_tools:
                            if tool_name not in tools_called:
                                tools_called.append(tool_name)
                                logger.info(
                                    "[SPECIALIST_V2_TOOL] task=%s | tool=%s | step=%d | elapsed=%.1fs",
                                    task_id, tool_name, step_count, time.time() - start,
                                )
                    except Exception as e:
                        logger.debug("[SPECIALIST_V2_TOOL_ERR] task=%s | error=%s", task_id, e)

                # Preserve A1 log
                try:
                    from biomni.utils.utils import pretty_print
                    out = pretty_print(message)
                    a1.log.append(out)
                except ImportError:
                    a1.log.append(str(message.content)[:200])

        # 6. Post-process and return
        latency = time.time() - start
        clean_answer = postprocess_answer(answer, task_name)

        logger.info(
            "[SPECIALIST_V2_END] task=%s | specialist=%s | tokens=%d | reasoning=%d | "
            "ttft=%.2fs | latency=%.1fs | steps=%d | tools=%s | answer=%s",
            task_id, specialist_key,
            token_count, reasoning_count,
            ttft if ttft is not None else -1,
            latency, step_count,
            ",".join(tools_called) if tools_called else "none",
            clean_answer[:80],
        )
        return clean_answer
