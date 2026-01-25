"""
실험 실행을 담당하는 Runner 구현체입니다.
"""

import time
import logging
from typing import List, Dict, Any, Optional
import os
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

from benchmarks.base import BaseBenchmark
from benchmarks.biomni import BiomniBenchmark
from benchmarks.labbench import LabBenchBenchmark
from agent.mock import MockAgent
from agent.llm import LLMAgent
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

    def get_agent(self, name: str) -> Any:
        if name.lower() == "mock":
            return MockAgent()
        elif name.lower() == "llm":
            return LLMAgent()
        else:
            raise ValueError(f"Unknown agent: {name}")

    def run_benchmark(
        self,
        benchmark_name: str,
        agent_name: str = "MockAgent",
        limit: Optional[int] = None,
        use_wandb: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        벤치마크 실행 메인 루프
        """
        logger.info(f"Starting benchmark: {benchmark_name} with agent: {agent_name}")

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
                    **kwargs,
                },
            )

        # Pass kwargs (e.g., subset) to get_benchmark
        benchmark = self.get_benchmark(benchmark_name, **kwargs)
        agent = self.get_agent(agent_name)

        tasks = benchmark.load_tasks()
        if limit:
            tasks = tasks[:limit]
            logger.info(f"Limiting tasks to {limit}")

        logger.info(f"Loaded {len(tasks)} tasks.")

        task_results = []
        start_time = time.time()

        for task in tqdm(tasks, desc="Running tasks"):
            task_start = time.time()
            result_dict = benchmark.run_task(agent, task)
            duration = time.time() - task_start

            if use_wandb:
                wandb.log(
                    {
                        "task_id": result_dict.get("task_id"),
                        "status": result_dict.get("status"),
                        "duration": duration,
                    }
                )

            task_results.append(result_dict)

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
            res["execution_time"] = 0.0

        metrics = benchmark.evaluate(predictions=task_results, ground_truth=None)

        summary = {
            "benchmark": benchmark_name,
            "agent": agent_name,
            "total_tasks": len(tasks),
            "duration": total_duration,
            "metrics": metrics,
        }

        if use_wandb:
            wandb.log(metrics)
            wandb.finish()

        logger.info("Saving results to storage...")
        saved_path = self.saver.save_experiment(summary, task_results)

        # summary에 saved_path 추가
        summary["saved_path"] = saved_path

        logger.info(f"Benchmark Complete. Metrics: {metrics}")
        logger.info(f"Results saved at: {saved_path}")

        return summary
