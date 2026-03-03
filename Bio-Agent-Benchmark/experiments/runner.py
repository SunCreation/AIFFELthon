"""
실험 실행을 담당하는 Runner 구현체입니다.
"""

import time
import logging
from typing import List, Dict, Any, Optional
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

from benchmarks.base import BaseBenchmark
from benchmarks.biomni import BiomniBenchmark
from benchmarks.labbench import LabBenchBenchmark
from agent.mock import MockAgent
from agent.llm import LLMAgent
from agent.biomni_a1 import BiomniA1Agent
from agent.biomni_a1_multi import BiomniA1MultiAgent
from agent.biomni_a1_exp5 import BiomniA1MultiAgent as BiomniA1Exp5Agent
from storage.schemas import BenchmarkResult
from storage.saver import ResultSaver

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    벤치마크 실행을 관리하는 클래스
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.saver = ResultSaver()

    def get_benchmark(self, name: str, **kwargs) -> BaseBenchmark:
        name_lower = name.lower()
        if name_lower == "biomni":
            return BiomniBenchmark()
        elif name_lower == "labbench":
            subset = kwargs.get("subset", "all")  # Default to 'all'
            return LabBenchBenchmark(subset=subset)
        elif name_lower == "mock":
            logger.warning(
                "Mock benchmark not fully implemented, using Biomni instead."
            )
            return BiomniBenchmark()
        else:
            raise ValueError(f"Unknown benchmark: {name}")

    def get_agent(self, name: str, **agent_kwargs) -> Any:
        if name.lower() == "mock":
            return MockAgent()
        elif name.lower() == "llm":
            return LLMAgent()
        elif name.lower() == "biomni_a1":
            return BiomniA1Agent(**agent_kwargs)
        elif name.lower() == "biomni_a1_multi":
            return BiomniA1MultiAgent(**agent_kwargs)
        elif name.lower() == "biomni_a1_exp5":
            return BiomniA1Exp5Agent(**agent_kwargs)
        else:
            raise ValueError(f"Unknown agent: {name}")

    def run_benchmark(
        self,
        benchmark_name: str,
        agent_name: str = "MockAgent",
        limit: Optional[int] = None,
        use_wandb: bool = True,
        parallel: int = 0,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        벤치마크 실행 메인 루프
        Args:
            parallel (int): 병렬 워커 수. 0이면 순차 실행, 1 이상이면 ThreadPoolExecutor 사용.
        """
        logger.info(f"Starting benchmark: {benchmark_name} with agent: {agent_name}")
        if parallel > 0:
            logger.info(f"Parallel mode enabled: {parallel} workers")

        if use_wandb:
            wandb_key = os.getenv("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key)
            wandb.init(
                project="bio-agent-benchmark",
                name=f"{benchmark_name}_{agent_name}_{int(time.time())}",
                config={
                    "benchmark": benchmark_name,
                    "agent": agent_name,
                    "limit": limit,
                    "parallel": parallel,
                    **kwargs,
                },
            )
        benchmark = self.get_benchmark(benchmark_name, **kwargs)
        agent = self.get_agent(agent_name, **(agent_kwargs or {}))
        tasks = benchmark.load_tasks()
        if limit:
            tasks = tasks[:limit]
            logger.info(f"Limiting tasks to {limit}")

        logger.info(f"Loaded {len(tasks)} tasks.")

        start_time = time.time()

        if parallel > 0:
            task_results = self._run_parallel(
                benchmark, agent, tasks, parallel, use_wandb
            )
        else:
            task_results = self._run_sequential(
                benchmark, agent, tasks, use_wandb
            )
        total_duration = time.time() - start_time
        logger.info(f"Execution finished in {total_duration:.2f}s")
        logger.info("Evaluating results...")
        # 점수 계산 및 채우기
        for res in task_results:
            task_name = res.get("metadata", {}).get("task_name", "unknown")
            pred = res.get("prediction", "")
            gt = res.get("ground_truth", "")
            score = 0.0
            if isinstance(benchmark, BiomniBenchmark):
                score = benchmark._compute_reward(task_name, pred, gt)
            elif isinstance(benchmark, LabBenchBenchmark):
                p_text = str(pred).strip().upper()
                if "ANSWER:" in p_text:
                    p_text = p_text.split("ANSWER:")[-1].strip()
                p_char = (
                    p_text[0] if len(p_text) > 0 and "A" <= p_text[0] <= "Z" else ""
                )
                score = 1.0 if p_char == str(gt).strip().upper() else 0.0
            res["score"] = score

        metrics = benchmark.evaluate(predictions=task_results, ground_truth=None)
        summary = {
            "benchmark": benchmark_name,
            "agent": agent_name,
            "total_tasks": len(tasks),
            "duration": total_duration,
            "parallel_workers": parallel,
            "metrics": metrics,
        }

        if use_wandb:
            wandb.log(metrics)
            wandb.finish()
        saved_path = self.saver.save_experiment(summary, task_results)
        summary["saved_path"] = saved_path
        logger.info(f"Results saved at: {saved_path}")
        return summary

    def _run_sequential(
        self,
        benchmark: BaseBenchmark,
        agent: Any,
        tasks: List[Dict[str, Any]],
        use_wandb: bool,
    ) -> List[Dict[str, Any]]:
        """순차 실행 (기존 로직)"""
        task_results = []
        for task in tqdm(tasks, desc="Running tasks"):
            task_start = time.time()
            result_dict = benchmark.run_task(agent, task)
            duration = time.time() - task_start
            result_dict["execution_time"] = duration

            if use_wandb:
                wandb.log(
                    {
                        "task_id": result_dict.get("task_id"),
                        "status": result_dict.get("status"),
                        "duration": duration,
                    }
                )

            task_results.append(result_dict)

        return task_results

    def _run_parallel(
        self,
        benchmark: BaseBenchmark,
        agent: Any,
        tasks: List[Dict[str, Any]],
        max_workers: int,
        use_wandb: bool,
    ) -> List[Dict[str, Any]]:
        """병렬 실행 (ThreadPoolExecutor)"""

        def _compute_score_inline(result_dict: Dict[str, Any]) -> float:
            task_name = result_dict.get("metadata", {}).get("task_name", "unknown")
            pred = result_dict.get("prediction", "")
            gt = result_dict.get("ground_truth", "")
            sc = 0.0
            if isinstance(benchmark, BiomniBenchmark):
                sc = benchmark._compute_reward(task_name, pred, gt)
            elif isinstance(benchmark, LabBenchBenchmark):
                p_text = str(pred).strip().upper()
                if "ANSWER:" in p_text:
                    p_text = p_text.split("ANSWER:")[-1].strip()
                p_char = (
                    p_text[0] if len(p_text) > 0 and "A" <= p_text[0] <= "Z" else ""
                )
                sc = 1.0 if p_char == str(gt).strip().upper() else 0.0
            return sc
        def _execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
            task_start = time.time()
            result_dict = benchmark.run_task(agent, task)
            result_dict["execution_time"] = time.time() - task_start
            sc = _compute_score_inline(result_dict)
            result_dict["score"] = sc
            task_id = result_dict.get("task_id", "unknown")
            logger.info(
                "[SCORE] task=%s | score=%.1f | prediction=%s | ground_truth=%s",
                task_id,
                sc,
                str(result_dict.get("prediction", ""))[:60],
                str(result_dict.get("ground_truth", ""))[:60],
            )
            return result_dict
        task_results = []
        errors = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_execute_task, task): task
                for task in tasks
            }

            with tqdm(total=len(tasks), desc=f"Running tasks (x{max_workers})") as pbar:
                for future in as_completed(futures):
                    try:
                        result_dict = future.result()
                        task_results.append(result_dict)

                        if result_dict.get("status") == "error":
                            errors += 1

                        if use_wandb:
                            wandb.log(
                                {
                                    "task_id": result_dict.get("task_id"),
                                    "status": result_dict.get("status"),
                                    "duration": result_dict.get("execution_time", 0),
                                }
                            )
                    except Exception as e:
                        task = futures[future]
                        logger.error(f"Task {task.get('id', '?')} raised: {e}")
                        task_results.append({
                            "task_id": task.get("id", "unknown"),
                            "status": "error",
                            "error_message": str(e),
                            "execution_time": 0.0,
                            "metadata": {"task_name": task.get("task_name", "unknown")},
                        })
                        errors += 1

                    pbar.update(1)
                    if errors > 0:
                        pbar.set_postfix(errors=errors)

        logger.info(f"Parallel execution done: {len(task_results)} completed, {errors} errors")
        return task_results
