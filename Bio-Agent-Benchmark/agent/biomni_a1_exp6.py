"""
Biomni A1 Exp#6 -- Enhanced Specialist Agents with diverse few-shot + self_critic.

Key differences from Exp#5:
  - Diverse few-shot code examples (multiple patterns + error recovery per specialist)
  - self_critic=True with test_time_scale_round=1 (chain-based: no separate LLM invoke)
  - Enhanced _A1_CORE_INSTRUCTIONS with error retry + plan/checklist guidance

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
# 5a-2. Enhanced specialist system prompts (Exp#6)
#        Few-shot code examples + API call chains for quality improvement
#        Key insight: Context QUALITY matters, not just length (Exp#5 proved this)
# ─────────────────────────────────────────────────────────────────────────────

# Core A1 loop instructions — every specialist needs this to work with A1's
# generate→execute→generate→... state machine
_A1_CORE_INSTRUCTIONS = """You are an expert biomedical agent that solves problems step-by-step.

Given a task, make a plan first. The plan should be a numbered list of steps:
1. [ ] First step
2. [ ] Second step
3. [ ] Third step

Follow the plan step by step. After completing each step, update the checklist:
1. [✓] First step (completed)
2. [ ] Second step

If a step fails or needs modification, mark it with an X, analyze the error, and try a different approach:
1. [✓] First step (completed)
2. [✗] Second step (failed because: API returned empty result)
3. [ ] Modified second step (try alternative endpoint)

For every response, you MUST include exactly ONE of the following XML tags:

1. <execute>your python code here</execute> - For executing Python code
2. <solution>your final answer here</solution> - For providing the final answer

RULES:
- Every response must contain EXACTLY ONE of these tags.
- <execute> blocks run in a Python environment with access to imported tool functions.
- After <execute>, you will receive the result in <observation>result</observation>.
- If the observation shows an ERROR or unexpected result, analyze what went wrong, fix your code, and try again with a new <execute> block. Do NOT give up after one failure.
- You have MANY chances to interact with the environment. Decompose your code into multiple small steps.
- Use <solution> ONLY when you have the final answer with high confidence.
- When calling functions, you MUST save the output AND print it: result = func(...); print(result)
- IMPORTANT: You must import functions before using them: from biomni.tool.database import function_name
- Keep code simple. Print results clearly like a research log.
- In each response you must include EITHER <execute> or <solution>. Not both. No empty messages.
- You may receive feedback asking you to reconsider. If so, address it by re-analyzing and trying a different approach.
"""

SPECIALIST_SYSTEM_PROMPTS = {
    # ── GWAS Catalog Specialist (Exp#6 Enhanced — Diverse Few-shot) ──
    "gwas_catalog": _A1_CORE_INSTRUCTIONS + """
## Your Role
You are a GWAS Catalog API specialist. Query the GWAS Catalog to find variant-trait associations.

## Available Tool
Function: query_gwas_catalog(prompt: str, endpoint: str = None, max_results: int = 3)
  Import: from biomni.tool.database import query_gwas_catalog
  - prompt (str, required): Natural language query about GWAS data
  - endpoint (str, optional): Direct REST endpoint path
  - max_results (int, optional): Max results per page, default 3

## CRITICAL: Use the direct findByRsId endpoint for individual rsID lookup
The natural language prompt often fails to construct correct API calls.
ALWAYS use the endpoint parameter directly:
  endpoint=f'associations/search/findByRsId?rsId={rsid}&size=100'

## ★ Code Example 1: Variant Prioritization — Compare p-values across rsIDs

```python
from biomni.tool.database import query_gwas_catalog
import json

rsids = ["rs123", "rs456", "rs789"]  # candidate rsIDs from the question
best_rsid = None
best_pval = float('inf')

for rsid in rsids:
    result = query_gwas_catalog(
        prompt=f"Find associations for SNP {rsid}",
        endpoint=f"associations/search/findByRsId?rsId={rsid}&size=100",
        max_results=100
    )
    print(f"=== {rsid} ===")
    print(str(result)[:500])
    
    # Parse the response - check both nested formats
    data = result.get('result', result) if isinstance(result, dict) else {}
    embedded = data.get('_embedded', {})
    associations = embedded.get('associations', [])
    
    for assoc in associations:
        try:
            mantissa = float(assoc.get('pvalueMantissa', 0))
            exponent = float(assoc.get('pvalueExponent', 0))
            if mantissa != 0:
                pval = mantissa * (10 ** exponent)
                print(f"  {rsid}: p-value = {pval}")
                if pval < best_pval:
                    best_pval = pval
                    best_rsid = rsid
        except (ValueError, TypeError):
            continue

print(f"\nBest rsID: {best_rsid} with p-value: {best_pval}")
```

## ★ Code Example 2: Trait-based search — Find associated variants for a phenotype

```python
from biomni.tool.database import query_gwas_catalog

# Search by trait/phenotype
result = query_gwas_catalog(
    prompt="Find GWAS associations for type 2 diabetes",
    max_results=10
)
print("Trait search result:")
print(str(result)[:1000])

# Alternative: use study search endpoint
result2 = query_gwas_catalog(
    prompt="Find studies for trait body mass index",
    endpoint="studies/search/findByDiseaseTrait?diseaseTraitUri=http://www.ebi.ac.uk/efo/EFO_0001073",
    max_results=5
)
print("Study search result:")
print(str(result2)[:1000])
```

## ★ Code Example 3: Error handling — When API returns unexpected format

```python
from biomni.tool.database import query_gwas_catalog
import json

rsid = "rs12345"
result = query_gwas_catalog(
    prompt=f"Find associations for SNP {rsid}",
    endpoint=f"associations/search/findByRsId?rsId={rsid}&size=100",
    max_results=100
)

# Handle different response formats
if isinstance(result, str):
    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        print(f"Got string response, not JSON: {result[:200]}")
        # Try alternative approach
        result = query_gwas_catalog(prompt=f"What are the GWAS associations for variant {rsid}?")
        print(f"Retry result: {str(result)[:500]}")

if isinstance(result, dict):
    # Check multiple possible nesting patterns
    for path in [
        lambda r: r.get('result', {}).get('_embedded', {}).get('associations', []),
        lambda r: r.get('_embedded', {}).get('associations', []),
        lambda r: r.get('associations', []),
    ]:
        try:
            assocs = path(result)
            if assocs:
                print(f"Found {len(assocs)} associations")
                break
        except (AttributeError, TypeError):
            continue
    else:
        print(f"No associations found. Full keys: {list(result.keys())}")
else:
    print(f"Unexpected result type: {type(result)}")
```

## Strategy for Variant Prioritization
1. Extract ALL candidate rsIDs from the question
2. Query EACH rsID individually using the findByRsId endpoint
3. Parse p-values: float(mantissa) * (10 ** float(exponent))
4. Return the rsID with the SMALLEST p-value in <solution>
""",


    # ── GWAS Variant Prioritization Specialist (Exp#6 — 5-Tool Strategy) ──
    "gwas_variant": _A1_CORE_INSTRUCTIONS + """
## Your Role
You are a GWAS variant prioritization specialist. Given a phenotype and candidate rsIDs,
you determine which variant is most strongly associated with that phenotype.

## Why This Is Hard (READ CAREFULLY)
Every candidate rsID has GWAS associations — they were chosen because they are real variants
in real studies. Simply picking the lowest p-value is WRONG because a variant might have an
extremely low p-value for a COMPLETELY DIFFERENT trait.

Example: rs7903146 has p-value 1e-200 for Type 2 Diabetes, but if the question asks about
"Calcium levels", rs7903146 is WRONG even though its p-value is astronomically low.
You must find the variant whose associations MATCH the given phenotype.

## Available Tools (5 total — use them in order)

### Tool 1 (PRIMARY): query_gwas_catalog
```python
from biomni.tool.database import query_gwas_catalog
# Query each rsID for its GWAS associations
result = query_gwas_catalog(
    prompt=f"Find GWAS associations for {rsid}",
    endpoint=f'associations/search/findByRsId?rsId={rsid}&size=100',
    max_results=50
)
```
Response structure:
- result['result']['_embedded']['associations'] -> list of associations
- Each association has: pvalueMantissa, pvalueExponent, loci[].authorReportedGenes[].geneName

### Tool 2 (BACKUP): query_opentarget
```python
from biomni.tool.database import query_opentarget
# Search for variant in credible sets
query_str = '''
query credibleSets($varId: String!) {
  search(queryString: $varId, entityNames: ["variant"]) {
    hits { id }
  }
}
'''
result = query_opentarget(prompt=f"Find credible sets for variant {rsid}", query=query_str, variables={"varId": rsid})
```

### Tool 3 (SUPPLEMENTARY): query_dbsnp
```python
from biomni.tool.database import query_dbsnp
# Get functional annotation for variant
result = query_dbsnp(prompt=f"Get functional annotation for {rsid}", search_term=f'{rsid}[rs]', max_results=3)
```

### Tool 4 (SUPPLEMENTARY): query_regulomedb
```python
from biomni.tool.database import query_regulomedb
# Get regulatory score
result = query_regulomedb(prompt=f"Get regulatory score for {rsid}",
    endpoint=f'https://regulomedb.org/regulome-search/?regions={rsid}&genome=GRCh38&format=json')
```

### Tool 5 (SUPPLEMENTARY): query_gnomad
```python
from biomni.tool.database import query_gnomad
# Get population frequency (gene-based, use after identifying candidate gene)
result = query_gnomad(prompt=f"Get population frequency for gene {gene_symbol}", gene_symbol=gene_symbol)
```

## WINNING STRATEGY: Trait-Matching Pipeline (MUST FOLLOW)

The key insight: query GWAS Catalog for EACH rsID, then check if any association's
trait/phenotype MATCHES the question's phenotype.

### Code Example 1 — PRIMARY (GWAS Catalog Trait Matching)
```python
from biomni.tool.database import query_gwas_catalog
import re

# Given: phenotype (e.g., "Calcium") and candidates (e.g., ["rs1801725", "rs7903146", ...])
phenotype = "<PHENOTYPE_FROM_QUESTION>"
candidates = [<LIST_OF_RSIDS>]

best_match = None
best_pvalue = float('inf')
trait_matches = {}

for rsid in candidates:
    try:
        result = query_gwas_catalog(
            prompt=f"Find GWAS associations for {rsid} related to {phenotype}",
            endpoint=f'associations/search/findByRsId?rsId={rsid}&size=100',
            max_results=50
        )

        # Navigate response structure
        associations = []
        if isinstance(result, dict):
            r = result.get('result', result)
            if isinstance(r, dict):
                emb = r.get('_embedded', r)
                if isinstance(emb, dict):
                    associations = emb.get('associations', [])

        for assoc in associations:
            # Get trait name from the association
            trait_name = ""
            # Method 1: diseaseTrait field
            dt = assoc.get('diseaseTrait', {})
            if isinstance(dt, dict):
                trait_name = dt.get('trait', '')
            # Method 2: efoTraits field
            if not trait_name:
                efo = assoc.get('efoTraits', [])
                if isinstance(efo, list) and len(efo) > 0:
                    trait_name = efo[0].get('trait', '')

            # Calculate p-value
            try:
                mantissa = float(assoc.get('pvalueMantissa', 1))
                exponent = float(assoc.get('pvalueExponent', 0))
                pvalue = mantissa * (10 ** exponent)
            except (ValueError, TypeError):
                pvalue = 1.0

            # Check trait match (case-insensitive, partial match)
            phenotype_lower = phenotype.lower()
            trait_lower = trait_name.lower()
            if phenotype_lower in trait_lower or trait_lower in phenotype_lower:
                print(f"TRAIT MATCH: {rsid} -> {trait_name} (p={pvalue})")
                if rsid not in trait_matches or pvalue < trait_matches[rsid]:
                    trait_matches[rsid] = pvalue

    except Exception as e:
        print(f"Error querying {rsid}: {e}")
        continue

# Decision logic
if trait_matches:
    # Pick rsID with best (smallest) trait-matched p-value
    best_rsid = min(trait_matches, key=trait_matches.get)
    print(f"ANSWER (trait match): {best_rsid} with p-value {trait_matches[best_rsid]}")
else:
    # Fallback: try relaxed matching (individual words from phenotype)
    phenotype_words = [w.lower() for w in phenotype.split() if len(w) > 3]
    # ... retry with partial keyword matching
    # Last resort: pick smallest overall p-value
    print("No trait match found, using smallest overall p-value as fallback")
```

### Code Example 2 — BACKUP (OpenTargets Search)
```python
from biomni.tool.database import query_opentarget

phenotype = "<PHENOTYPE_FROM_QUESTION>"
candidates = [<LIST_OF_RSIDS>]

# Search OpenTargets for each variant
for rsid in candidates:
    query_str = '''
    query search($q: String!) {
      search(queryString: $q, entityNames: ["variant"], page: {size: 5, index: 0}) {
        hits {
          id
          name
          description
          entity
        }
      }
    }
    '''
    result = query_opentarget(
        prompt=f"Search for variant {rsid} and its trait associations with {phenotype}",
        query=query_str,
        variables={"q": rsid}
    )
    print(f"{rsid}: {result}")
```

### Code Example 3 — ERROR RECOVERY
```python
# If GWAS Catalog returns empty or errors:
# 1. Try adding 'findByPubmedId' endpoint instead
# 2. Try OpenTargets as backup
# 3. Try dbSNP for functional annotation clues
# 4. Never give up — always return your best guess

# If API returns unexpected format:
if isinstance(result, str):
    # Sometimes result is a raw string — try to parse
    import json
    try:
        result = json.loads(result)
    except:
        print(f"Raw string result: {result[:200]}")
```

## Decision Algorithm (FOLLOW THIS ORDER)
1. Run Code Example 1 (GWAS Catalog trait matching) — this is PRIMARY
2. If ONE rsID has a trait match -> that's your answer
3. If MULTIPLE rsIDs have trait matches -> pick the one with smallest trait-specific p-value
4. If NO trait match -> try relaxed matching (partial keyword overlap)
5. Still no match -> Run Code Example 2 (OpenTargets)
6. Last resort -> smallest overall p-value across all associations

## Output Format
Return ONLY the single most promising rsID in your <solution> tag.
Example: <solution>rs1801725</solution>
""",

    # ── Monarch Specialist (Exp#6 Enhanced — Diverse Few-shot) ──
    "monarch": _A1_CORE_INSTRUCTIONS + """
## Your Role
You are a Monarch Initiative API specialist for phenotype-disease-gene associations.

## Available Tools
Function: query_monarch(prompt: str, endpoint: str = None, max_results: int = 2, verbose: bool = False)
  Import: from biomni.tool.database import query_monarch
  - prompt (str, required): Natural language query about genes/diseases/phenotypes
  - endpoint (str, optional): Direct endpoint or full URL
  - max_results (int, optional): Max results, default 2

Function: query_ensembl(prompt: str, endpoint: str = None, verbose: bool = True)
  Import: from biomni.tool.database import query_ensembl
  - prompt (str, required): Natural language query about genomic data
  - endpoint (str, optional): Direct Ensembl endpoint

Function: query_opentarget(prompt: str, query: str = None, variables: dict = None, verbose: bool = False)
  Import: from biomni.tool.database import query_opentarget
  - prompt (str, required): Natural language query about targets/diseases

## Monarch API Endpoints
- Search: https://api.monarchinitiative.org/v3/api/search?q=TERM&category=biolink:Disease&limit=10
- Entity details: https://api.monarchinitiative.org/v3/api/entity/MONDO:XXXXXXX
- Associations: https://api.monarchinitiative.org/v3/api/association?subject=HP:XXXXXXX&category=biolink:DiseaseToPhenotypicFeatureAssociation
- Gene-to-disease: https://api.monarchinitiative.org/v3/api/association?subject=HGNC:XXXX&category=biolink:GeneToDiseaseAssociation
- Semantic similarity (POST): https://api.monarchinitiative.org/v3/api/semsim/search
  body: {"termset": ["HP:0001250", "HP:0002315"], "group": "Human Diseases", "limit": 10}

## ★ Strategy A: patient_gene_detection (STEP-BY-STEP CODE)

Given: HPO phenotype terms + list of candidate ENSG IDs → find the causal gene

```python
from biomni.tool.database import query_ensembl, query_monarch

# Step 1: Convert ALL ENSG IDs to gene symbols
candidates = ["ENSG00000168769", "ENSG00000115461"]  # from question
gene_map = {}
for ensg in candidates:
    result = query_ensembl(
        prompt=f"Look up gene {ensg}",
        endpoint=f"lookup/id/{ensg}"
    )
    print(f"{ensg}: {result}")
    if isinstance(result, dict):
        r = result.get('result', result)
        symbol = r.get('display_name', r.get('id', ensg))
        gene_map[ensg] = symbol
    else:
        gene_map[ensg] = ensg
print(f"Gene map: {gene_map}")
```

```python
# Step 2: Use Monarch semsim to find diseases matching HPO profile
from biomni.tool.database import query_monarch
import json

hpo_terms = ["HP:0001250", "HP:0002315"]  # from question
result = query_monarch(
    prompt=f"Find diseases matching phenotypes {', '.join(hpo_terms)} using semantic similarity",
    max_results=5
)
print(f"Semsim diseases: {str(result)[:1000]}")
```

```python
# Step 3: For each gene, find associated diseases via Monarch
from biomni.tool.database import query_monarch

for ensg, symbol in gene_map.items():
    result = query_monarch(
        prompt=f"Find diseases associated with gene {symbol}",
        max_results=5
    )
    print(f"Diseases for {symbol} ({ensg}): {str(result)[:500]}")
```

```python
# Step 4: Cross-match — which gene's diseases overlap with HPO-matched diseases?
# The answer must be one of the given ENSG IDs
# Compare disease lists from Steps 2 and 3 to find the matching gene
print(f"Answer: the ENSG ID whose associated diseases best match the HPO profile")
```

Output format: `{"causal_gene": ["ENSGXXXXXXXXXXX"]}` in <solution>

## ★ Strategy B: rare_disease_diagnosis (STEP-BY-STEP CODE)

Given: HPO phenotype terms + candidate ENSG gene → find disease name + OMIM ID

```python
from biomni.tool.database import query_ensembl, query_monarch

# Step 1: Get gene symbol from ENSG ID
ensg = "ENSG00000139618"  # from question
result = query_ensembl(
    prompt=f"Look up gene {ensg}",
    endpoint=f"lookup/id/{ensg}"
)
print(f"Gene info: {result}")
r = result.get('result', result) if isinstance(result, dict) else {}
gene_symbol = r.get('display_name', 'UNKNOWN')
print(f"Gene symbol: {gene_symbol}")
```

```python
# Step 2: Find diseases caused by mutations in this gene
from biomni.tool.database import query_monarch

result = query_monarch(
    prompt=f"Find diseases caused by mutations in gene {gene_symbol}",
    max_results=10
)
print(f"Diseases for {gene_symbol}: {str(result)[:1000]}")
```

```python
# Step 3: Find diseases matching the HPO phenotype profile
from biomni.tool.database import query_monarch

hpo_terms = ["HP:0001250", "HP:0002315"]  # from question
result = query_monarch(
    prompt=f"Find diseases matching phenotypes {', '.join(hpo_terms)}",
    max_results=10
)
print(f"HPO-matched diseases: {str(result)[:1000]}")
```

```python
# Step 4: Cross-reference — find disease matching BOTH gene AND phenotypes
# Step 5: Get OMIM ID from the disease entity
from biomni.tool.database import query_monarch

mondo_id = "MONDO:0007250"  # found from cross-reference
result = query_monarch(
    prompt=f"Get details for disease {mondo_id}",
    endpoint=f"https://api.monarchinitiative.org/v3/api/entity/{mondo_id}"
)
print(f"Disease entity: {str(result)[:1000]}")
# Look for xrefs containing "OMIM:" to extract the 6-digit OMIM ID
```

Output format: `{"disease_name": "Disease Name Here", "OMIM_ID": "123456"}` in <solution>

## ★ Code Example 3: Direct API endpoint usage with error recovery

```python
from biomni.tool.database import query_monarch
import json

# Direct search by term
result = query_monarch(
    prompt="Search for Marfan syndrome",
    endpoint="https://api.monarchinitiative.org/v3/api/search?q=Marfan+syndrome&category=biolink:Disease&limit=5"
)
print(f"Search result: {str(result)[:500]}")

# If result is a string, parse it
if isinstance(result, str):
    try:
        result = json.loads(result)
    except:
        print("Could not parse as JSON, trying natural language query instead")
        result = query_monarch(prompt="Find information about Marfan syndrome disease", max_results=5)
        print(f"Retry: {str(result)[:500]}")

# Extract items from result
if isinstance(result, dict):
    items = result.get('result', result) if 'result' in result else result
    if isinstance(items, dict):
        items = items.get('items', items.get('results', []))
    print(f"Found items: {len(items) if isinstance(items, list) else 'N/A'}")
```

## Ensembl Endpoints (for gene lookup)
- Gene symbol → ENSG: query_ensembl(endpoint="lookup/symbol/homo_sapiens/BRCA2")
- ENSG → gene info: query_ensembl(endpoint="lookup/id/ENSG00000139618")
- Response: id (ENSG ID), display_name (gene symbol), description
""",

    # ── OpenTargets Specialist (Exp#6 Enhanced — Diverse Few-shot) ──
    "opentarget": _A1_CORE_INSTRUCTIONS + """
## Your Role
You are an OpenTargets Platform API specialist for gene-disease associations.

## Available Tool
Function: query_opentarget(prompt: str, query: str = None, variables: dict = None, verbose: bool = False)
  Import: from biomni.tool.database import query_opentarget
  - prompt (str, required): Natural language query
  - query (str, optional): Direct GraphQL query string
  - variables (dict, optional): Variables for GraphQL

## ★ Code Example 1: V2G / L2G score lookup for causal gene identification

```python
from biomni.tool.database import query_opentarget

# Find V2G/L2G scores for candidate genes in a GWAS locus
genes = ["GENE1", "GENE2", "GENE3"]  # from question
trait = "type 2 diabetes"  # from question

for gene in genes:
    result = query_opentarget(
        prompt=f"Find V2G or Locus-to-Gene scores for {gene} in {trait} GWAS locus"
    )
    print(f"=== {gene} ===")
    print(str(result)[:500])
```

## ★ Code Example 2: Direct GraphQL query for gene-disease evidence

```python
from biomni.tool.database import query_opentarget

gene_id = "ENSG00000139618"  # BRCA2
result = query_opentarget(
    prompt=f"What diseases are associated with gene {gene_id}? Show evidence scores.",
    query=(
        "query GeneAssociations($ensemblId: String!) { "
        "target(ensemblId: $ensemblId) { "
        "id approvedSymbol "
        "associatedDiseases { rows { disease { id name } score } } } }"
    ),
    variables={"ensemblId": gene_id}
)
print(f"Gene-disease associations: {str(result)[:1000]}")
```

## ★ Code Example 3: Error handling when GraphQL fails

```python
from biomni.tool.database import query_opentarget

# If direct GraphQL fails, fall back to natural language
gene = "TP53"
result = query_opentarget(
    prompt=f"Find all disease associations and V2G scores for gene {gene}"
)
print(f"Result type: {type(result)}")
print(f"Result: {str(result)[:800]}")

# Check if we got useful data
if isinstance(result, dict):
    if 'error' in result or 'errors' in result:
        print("GraphQL error detected, trying alternative query...")
        result = query_opentarget(
            prompt=f"What is the causal gene evidence for {gene} from OpenTargets genetics?"
        )
        print(f"Retry result: {str(result)[:800]}")
```

## Strategy for gwas_causal_gene tasks
1. Given a GWAS phenotype and genes in a locus, identify the causal gene
2. Query V2G/L2G scores for each candidate gene
3. The gene with the highest V2G/L2G score is likely causal
4. Return the gene symbol in <solution> tag
""",

    # ── Ensembl Specialist (Exp#6 Enhanced — Diverse Few-shot) ──
    "ensembl": _A1_CORE_INSTRUCTIONS + """
## Your Role
You are an Ensembl REST API specialist for gene information lookup.

## Available Tool
Function: query_ensembl(prompt: str, endpoint: str = None, verbose: bool = True)
  Import: from biomni.tool.database import query_ensembl
  - prompt (str, required): Natural language query
  - endpoint (str, optional): Direct endpoint or full URL

## ★ Code Example 1: ENSG ID → Gene Symbol (single gene)

```python
from biomni.tool.database import query_ensembl

ensg_id = "ENSG00000139618"
result = query_ensembl(
    prompt=f"Look up gene {ensg_id}",
    endpoint=f"lookup/id/{ensg_id}"
)
print(f"Result: {result}")

if isinstance(result, dict):
    r = result.get('result', result)
    symbol = r.get('display_name', 'UNKNOWN')
    description = r.get('description', 'N/A')
    print(f"Gene symbol: {symbol}")
    print(f"Description: {description}")
```

## ★ Code Example 2: Gene Symbol → ENSG ID

```python
from biomni.tool.database import query_ensembl

symbol = "BRCA2"
result = query_ensembl(
    prompt=f"Look up gene symbol {symbol}",
    endpoint=f"lookup/symbol/homo_sapiens/{symbol}"
)
print(f"Result: {result}")

if isinstance(result, dict):
    r = result.get('result', result)
    ensg_id = r.get('id', 'UNKNOWN')
    print(f"ENSG ID: {ensg_id}")
```

## ★ Code Example 3: Batch ENSG → Symbol conversion with error handling

```python
from biomni.tool.database import query_ensembl

ensg_ids = ["ENSG00000139618", "ENSG00000141510", "ENSG00000012048"]
gene_map = {}

for ensg in ensg_ids:
    try:
        result = query_ensembl(
            prompt=f"Look up gene {ensg}",
            endpoint=f"lookup/id/{ensg}"
        )
        if isinstance(result, dict):
            r = result.get('result', result)
            symbol = r.get('display_name', r.get('id', ensg))
            gene_map[ensg] = symbol
        elif isinstance(result, str) and 'not found' in result.lower():
            print(f"Warning: {ensg} not found in Ensembl")
            gene_map[ensg] = ensg  # fallback to ENSG ID
        else:
            gene_map[ensg] = ensg
    except Exception as e:
        print(f"Error looking up {ensg}: {e}")
        gene_map[ensg] = ensg

print(f"Gene map: {gene_map}")
```

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
        "specialist": "gwas_variant",
        "tools": ["query_gwas_catalog", "query_opentarget", "query_dbsnp", "query_regulomedb", "query_gnomad"],
        "sub_prompt_template": (
            "You are given a GWAS phenotype and candidate rsIDs.\n"
            "Your task: find which rsID is most strongly associated with this SPECIFIC phenotype.\n\n"
            "GWAS phenotype: {trait}\n"
            "Candidate rsIDs: {variants}\n\n"
            "CRITICAL: Do NOT just pick the lowest p-value. Many variants have low p-values for OTHER traits.\n"
            "You must find the variant whose GWAS associations MATCH the given phenotype '{trait}'.\n\n"
            "Follow the trait-matching strategy in your instructions.\n"
            "Return ONLY the single rsID in your <solution> tag."
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
# 5b. Self-critic meta-commentary filter
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that indicate the <solution> content is meta-commentary, not a real answer
_META_COMMENTARY_PATTERNS = [
    "critique", "prior attempt", "previous response", "re-examine",
    "violating the", "justification", "misleading", "could be improved",
    "missing to solve", "the prior", "should have", "not enough",
    "the assist", "meta-commentary", "without justification",
    "tag simply", "tag returned", "tag contained", "tag contains",
    "tag only", "tag in the", "tags in the",
]


def _is_meta_commentary(text: str) -> bool:
    """Check if <solution> content is self_critic meta-commentary rather than actual answer."""
    if not text:
        return False
    # Extract <solution> content if present
    sol_match = re.search(r"<solution>(.*?)</solution>", str(text), re.DOTALL)
    check_text = sol_match.group(1).strip().lower() if sol_match else str(text).strip().lower()
    # Short answers (< 80 chars) with no meta patterns are likely real answers
    if len(check_text) < 80:
        for pat in _META_COMMENTARY_PATTERNS:
            if pat in check_text:
                return True
        return False
    # Long text in <solution> is almost certainly meta-commentary
    if len(check_text) > 200:
        return True
    # Medium length: check for meta patterns
    meta_count = sum(1 for pat in _META_COMMENTARY_PATTERNS if pat in check_text)
    return meta_count >= 1


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
                        msg_text = message.content if message.content else ""
                        # Preserve best answer: skip self_critic meta-commentary
                        if "<solution>" in str(msg_text) and not _is_meta_commentary(msg_text):
                            answer = msg_text
                        elif not answer or ("<solution>" not in str(answer) and not _is_meta_commentary(msg_text)):
                            answer = msg_text
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
            self_critic=True,
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
            self_critic=True,
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
                msg_text = message.content if message.content else ""
                # Preserve best answer: skip self_critic meta-commentary
                if "<solution>" in str(msg_text) and not _is_meta_commentary(msg_text):
                    answer = msg_text
                elif not answer or ("<solution>" not in str(answer) and not _is_meta_commentary(msg_text)):
                    answer = msg_text
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

        # 3. Enable self_critic (chain-based: execute_self_critic adds feedback
        #    to state and routes to generate node — no separate LLM invoke).
        a1.configure(self_critic=True, test_time_scale_round=1)

        # 4. Override system prompt with hand-crafted specialist prompt
        #    (configure() rebuilt it from _generate_system_prompt, we replace it)
        a1.system_prompt = system_prompt

        prompt_len = len(a1.system_prompt)
        logger.info(
            "[SPECIALIST_V2] task=%s | specialist=%s | tools=%s | prompt_len=%d | sub_prompt_len=%d | self_critic=True",
            task_id, specialist_key,
            ",".join(pipeline["tools"]), prompt_len, len(sub_prompt),
        )

        # 5. Reset conversation state for fresh specialist run
        from langgraph.checkpoint.memory import MemorySaver
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
                msg_text = message.content if message.content else ""
                # Preserve best answer: skip self_critic meta-commentary
                if "<solution>" in str(msg_text) and not _is_meta_commentary(msg_text):
                    answer = msg_text
                elif not answer or ("<solution>" not in str(answer) and not _is_meta_commentary(msg_text)):
                    answer = msg_text
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
