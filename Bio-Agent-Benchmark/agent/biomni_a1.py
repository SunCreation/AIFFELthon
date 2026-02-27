"""
Biomni A1 Agent adapter for Bio-Agent-Benchmark.

Wraps the Biomni A1 agent (with tool retrieval and data_lake) into the
simple `predict(prompt, task_id)` interface expected by the benchmark runner.

Usage:
    agent = BiomniA1Agent(
        biomni_path="/path/to/Biomni",
        data_path="/path/to/Biomni/data",
    )
    answer = agent.predict("What gene is causal for ...", task_id="gwas_01")
"""

import os
import sys
import re
import time
import logging
import threading
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # Load CWD .env first

logger = logging.getLogger(__name__)


class BiomniA1Agent:
    """Adapter that wraps Biomni A1 agent for benchmark evaluation.

    The Biomni A1 agent uses LangGraph with a tool registry (223 tools),
    data_lake, and ToolRetriever to answer biomedical questions using
    actual database lookups and computations.
    """

    def __init__(
        self,
        biomni_path: Optional[str] = None,
        data_path: Optional[str] = None,
        llm: Optional[str] = None,
        source: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_tool_retriever: bool = True,
        timeout_seconds: int = 600,
        skip_datalake_download: bool = False,
    ):
        """Initialize BiomniA1Agent.

        Args:
            biomni_path: Path to the Biomni repository root.
                         Defaults to env BIOMNI_REPO_PATH or ../../../Biomni
            data_path: Path where Biomni stores data (data_lake, benchmark).
                       Defaults to env BIOMNI_PATH or biomni_path/data
            llm: LLM model name. Defaults to env BIOMNI_LLM.
            source: LLM source type. Defaults to env BIOMNI_SOURCE.
            base_url: Custom LLM base URL. Defaults to env BIOMNI_CUSTOM_BASE_URL.
            api_key: Custom LLM API key. Defaults to env BIOMNI_CUSTOM_API_KEY.
            use_tool_retriever: Whether to use tool retrieval (recommended True).
            timeout_seconds: Timeout for code execution in seconds.
            skip_datalake_download: If True, skip downloading data_lake files
                                    (assumes they are already present).
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

        # Add Biomni to Python path so we can import it
        if biomni_path not in sys.path:
            sys.path.insert(0, biomni_path)
            logger.info("Added Biomni to sys.path: %s", biomni_path)

        # Load Biomni's .env for server config (BIOMNI_CUSTOM_BASE_URL, etc.)
        biomni_env = os.path.join(biomni_path, ".env")
        if os.path.isfile(biomni_env):
            load_dotenv(biomni_env, override=False)
            logger.info("Loaded Biomni .env from %s", biomni_env)

        # Resolve data path
        if data_path is None:
            data_path = os.getenv("BIOMNI_PATH", os.path.join(biomni_path, "data"))

        logger.info("Initializing Biomni A1 agent...")
        logger.info("  Biomni repo: %s", biomni_path)
        logger.info("  Data path: %s", data_path)
        logger.info("  LLM: %s", llm or os.getenv("BIOMNI_LLM", "(default)"))
        logger.info("  Source: %s", source or os.getenv("BIOMNI_SOURCE", "(auto)"))
        logger.info(
            "  Base URL: %s", base_url or os.getenv("BIOMNI_CUSTOM_BASE_URL", "(none)")
        )
        logger.info("  Tool retriever: %s", use_tool_retriever)
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

        # Build kwargs for A1 constructor
        a1_kwargs = {
            "path": data_path,
            "use_tool_retriever": use_tool_retriever,
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

        # If skipping datalake download, pass empty expected_data_lake_files
        if skip_datalake_download:
            a1_kwargs["expected_data_lake_files"] = []

        # Initialize A1 (this may trigger data_lake download on first run)
        init_start = time.time()
        self._a1 = A1(**a1_kwargs)
        init_time = time.time() - init_start
        logger.info("Biomni A1 initialized in %.1fs", init_time)

        # Thread lock for A1.go() which is not thread-safe
        # (LangGraph state is shared within the A1 instance)
        self._lock = threading.Lock()

    def predict(self, prompt, task_id="unknown"):
        """Run the Biomni A1 agent on the given prompt with token-level streaming.

        Uses LangGraph stream_mode=["messages", "values"] to get token-level
        chunks, emitting [STREAM_START], [STREAM_PROGRESS], [STREAM_END] events
        that the web dashboard monitor can parse for real-time display.

        Args:
            prompt: The benchmark question/prompt.
            task_id: Identifier for logging.
        Returns:
            str: The agent's answer.
        """
        from langchain_core.messages import HumanMessage

        worker = threading.current_thread().name
        logger.info(
            "[REQ] task=%s | worker=%s | prompt_chars=%d | mode=stream",
            task_id,
            worker,
            len(prompt),
        )
        start = time.time()
        task_type = self._get_task_type(task_id)

        try:
            # A1 is NOT thread-safe (LangGraph state), so serialize access
            with self._lock:
                # Replicate A1.go() setup without modifying Biomni source
                self._a1.critic_count = 0
                self._a1.user_task = prompt
                if self._a1.use_tool_retriever:
                    selected = self._a1._prepare_resources_for_retrieval(prompt)
                    self._a1.update_system_prompt_with_selected_resources(selected)

                inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
                config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
                self._a1.log = []

                token_count = 0
                reasoning_count = 0
                ttft = None
                step_count = 0
                answer = ""
                progress_interval = 50
                next_progress = progress_interval

                for mode, data in self._a1.app.stream(
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
                                    task_id, worker, ttft,
                                )

                            total = token_count + reasoning_count
                            if total >= next_progress:
                                logger.info(
                                    "[STREAM_PROGRESS] task=%s | worker=%s | tokens_so_far=%d | reasoning=%d | elapsed=%.1fs",
                                    task_id, worker, total, reasoning_count, now - start,
                                )
                                next_progress += progress_interval

                    elif mode == "values":
                        step_count += 1
                        message = data["messages"][-1]
                        answer = message.content
                        # Preserve A1 log for compatibility
                        try:
                            from biomni.utils.utils import pretty_print
                            out = pretty_print(message)
                            self._a1.log.append(out)
                        except ImportError:
                            self._a1.log.append(str(message.content)[:200])

            latency = time.time() - start
            clean_answer = self._extract_answer(answer, task_type)

            logger.info(
                "[STREAM_END] task=%s | worker=%s | total_tokens=%d | reasoning_tokens=%d | ttft=%.2fs | latency=%.1fs | answer=%s",
                task_id,
                worker,
                token_count,
                reasoning_count,
                ttft if ttft is not None else -1,
                latency,
                clean_answer,
            )
            logger.debug(
                "[A1_DETAIL] task=%s | type=%s | steps=%d | raw=%s | clean=%s",
                task_id,
                task_type,
                step_count,
                str(answer)[:60],
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
    @staticmethod
    def _get_task_type(task_id: str) -> str:
        """Extract the task type from the task_id.

        task_id format: '{task_type}_{number}' e.g. 'gwas_variant_prioritization_134'
        """
        # Remove trailing _<digits>
        parts = task_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return task_id

    def _extract_answer(self, raw_answer, task_type="unknown"):
        """Extract a clean answer from the A1 agent's response.
        The A1 agent returns verbose responses with markdown, explanations, etc.
        This method extracts just the final answer suitable for benchmark evaluation,
        applying task-type-specific parsing.

        Args:
            raw_answer: Raw text from A1.go()
            task_type: The benchmark task type for format-specific extraction

        Returns:
            str: Cleaned answer string
        """
        if not raw_answer:
            return ""
        answer = str(raw_answer).strip()

        # Step 1: Extract <solution> tag content if present
        if "<solution>" in answer and "</solution>" in answer:
            start = answer.index("<solution>") + len("<solution>")
            end = answer.index("</solution>")
            answer = answer[start:end].strip()
        # Step 2: Extract from "Final Answer:" markers
        for marker in ["Final Answer:", "FINAL ANSWER:", "Answer:", "ANSWER:"]:
            if marker in answer:
                answer = answer.split(marker)[-1].strip()
                break

        # Step 2.5: Strip markdown formatting (backticks, quotes)
        answer = re.sub(r'```[\s\S]*?```', '', answer)  # fenced code blocks
        answer = re.sub(r'`([^`]+)`', r'\1', answer)     # inline code
        answer = re.sub(r'^>\s?', '', answer, flags=re.MULTILINE)  # blockquotes
        answer = answer.strip()

        # Step 3: Task-type-specific extraction
        if task_type == "gwas_variant_prioritization":
            answer = self._extract_rsid(answer)
        elif task_type.startswith("gwas_causal_gene"):
            answer = self._extract_gene_symbol(answer)
        elif task_type == "screen_gene_retrieval":
            answer = self._extract_gene_symbol(answer)
        elif task_type == "crispr_delivery":
            answer = self._extract_multiple_choice(answer)
        elif task_type.startswith("lab_bench"):
            answer = self._extract_multiple_choice(answer)
        elif task_type == "hle":
            answer = self._extract_short_answer(answer)
        elif task_type == "rare_disease_diagnosis":
            answer = self._extract_json_answer(answer)
        elif task_type == "patient_gene_detection":
            answer = self._extract_json_answer(answer)

        return answer

    @staticmethod
    def _extract_rsid(text: str) -> str:
        """Extract rsID (e.g., rs1234567) from text."""
        rsids = re.findall(r'rs\d+', text)
        if rsids:
            return rsids[-1]  # prefer last rsID (A1 states conclusions at end)
        return text.strip()

    @staticmethod
    def _extract_gene_symbol(text: str) -> str:
        """Extract gene symbol from text.

        Gene symbols are typically uppercase alphanumeric (e.g., BRCA1, TP53, ACE).
        """
        text = text.replace("**", "").replace("*", "")

        # Priority 1: Explicit "gene is X" or "gene: X" patterns
        explicit = re.search(
            r'(?:gene|causal gene|target gene)\s*(?:is|:|=)\s*([A-Z][A-Z0-9]{1,9})\b',
            text, re.IGNORECASE
        )
        if explicit:
            return explicit.group(1).upper()

        # Priority 2: Regex scan for uppercase gene-like tokens
        genes = re.findall(r'\b([A-Z][A-Z0-9]{1,9})\b', text)
        if genes:
            noise = {
                'THE', 'AND', 'FOR', 'NOT', 'WITH', 'FROM', 'THIS', 'THAT',
                'ARE', 'WAS', 'HAS', 'HAD', 'BUT', 'ALL', 'CAN', 'HER',
                'ONE', 'OUR', 'OUT', 'YOU', 'ITS', 'MAY', 'WHO', 'NOW',
                'GET', 'USE', 'NEW', 'OLD', 'YES', 'NO', 'GWAS', 'SNP',
                'DNA', 'RNA', 'OMIM', 'ID', 'HTTP', 'API', 'URL',
                'NONE', 'NULL', 'TRUE', 'FALSE', 'ERROR', 'NA',
                'GENE', 'STUDY', 'RISK', 'LOCUS', 'REGION', 'VARIANT',
                'BASED', 'ANALYSIS', 'RESULT', 'DATA', 'TABLE', 'FIGURE',
                'CONCLUSION', 'SUMMARY', 'ANSWER', 'FINAL', 'NOTE',
                'MOST', 'LIKELY', 'ASSOCIATED', 'SIGNIFICANT', 'FOUND',
                'IDENTIFIED', 'REPORT', 'EVIDENCE', 'STRONG', 'HIGH',
            }
            genes = [g for g in genes if g not in noise]
            if genes:
                return genes[0]
        first_line = text.split('\n')[0].strip()
        return first_line[:50] if len(first_line) > 50 else first_line

    @staticmethod
    def _extract_multiple_choice(text: str) -> str:
        """Extract multiple choice answer (A, B, C, D, E, or a-f) from text."""
        # Check for [ANSWER]X[/ANSWER] pattern first (A1 common format)
        tag_match = re.search(r'\[ANSWER\]\s*([A-Fa-f])\s*\[/ANSWER\]', text)
        if tag_match:
            return tag_match.group(1).upper()
        patterns = [
            r'(?:answer|choice)\s*(?:is)?\s*[:=]?\s*([A-Ea-e])\b',
            r'\b([A-E])\)\s',
            r'\(([A-E])\)',  # (A) parenthesized pattern
            r'^\s*([A-Ea-e])\s*$',
            r'\*\*([A-E])\*\*',
            r'(?:select\s+one\s+letter\s+[a-f]\s*\)\s*:\s*)([a-f])\b',
            r':\s*([a-f])\s*(?:\(|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        text_clean = text.strip()
        if len(text_clean) <= 5:
            for c in text_clean:
                if c.upper() in 'ABCDEF':
                    return c.upper()
        return text_clean

    @staticmethod
    def _extract_short_answer(text: str) -> str:
        """Extract a short answer from verbose text."""
        text = text.replace("**", "").replace("*", "")
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith(('#', '-', '*')):
                if len(line) <= 100:
                    return line
                sentences = re.split(r'[.!]\s', line)
                if sentences:
                    return sentences[0].strip()
        return text[:100]

    @staticmethod
    def _extract_json_answer(text: str) -> str:
        """Try to extract JSON from text, or return as-is."""
        import json
        match = re.search(r'\{[^{}]+\}', text)
        if match:
            try:
                parsed = json.loads(match.group())
                return json.dumps(parsed)
            except json.JSONDecodeError:
                pass
        try:
            parsed = json.loads(text)
            return json.dumps(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
        return text
