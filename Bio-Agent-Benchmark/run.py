"""
프로젝트의 메인 실행 진입점입니다.
CLI 명령어를 통해 벤치마크를 실행합니다.
"""

import fire
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import ExperimentRunner
from evaluation.analyzer import Analyzer


class BioAgentBenchmarkCLI:
    """
    Bio-Agent-Benchmark CLI 인터페이스
    """

    def run(
        self,
        benchmark: str = "biomni",
        agent: str = "mock",
        limit: int = None,
        use_wandb: bool = False,
        subset: str = "all",
        parallel: int = 0,
        biomni_path: str = None,
        biomni_data_path: str = None,
        skip_datalake: bool = False,
    ):
        """
        벤치마크 실행
        Args:
            benchmark (str): 실행할 벤치마크 이름 (biomni, labbench 등)
            agent (str): 실행할 에이전트 이름 (mock, llm, biomni_a1, biomni_a1_multi)
            limit (int): 실행할 태스크 수 제한 (테스트용)
            use_wandb (bool): W&B 로깅 사용 여부
            subset (str): Lab-bench 실행 시 특정 서브셋 지정 (기본값: all)
            parallel (int): 병렬 워커 수 (0=순차, 4=4개 동시 실행)
            biomni_path (str): Biomni 레포 경로 (biomni_a1 에이전트 전용)
            biomni_data_path (str): Biomni 데이터 경로 (biomni_a1 에이전트 전용)
            skip_datalake (bool): data_lake 다운로드 건너뛰기 (이미 다운로드된 경우)
        """
        mode = f"parallel x{parallel}" if parallel > 0 else "sequential"
        print(f"\U0001f680 Initializing Benchmark: {benchmark} | Agent: {agent} | Subset: {subset} | Mode: {mode}")
        runner = ExperimentRunner()
        analyzer = Analyzer()
        # Build agent_kwargs for biomni_a1 agent
        agent_kwargs = {}
        if agent.lower() in ("biomni_a1", "biomni_a1_multi", "biomni_a1_exp5"):
            if biomni_path:
                agent_kwargs["biomni_path"] = biomni_path
            if biomni_data_path:
                agent_kwargs["data_path"] = biomni_data_path
            if skip_datalake:
                agent_kwargs["skip_datalake_download"] = True
            if parallel > 0:
                agent_kwargs["pool_size"] = parallel

        try:
            summary = runner.run_benchmark(
                benchmark_name=benchmark,
                agent_name=agent,
                limit=limit,
                use_wandb=use_wandb,
                parallel=parallel,
                agent_kwargs=agent_kwargs if agent_kwargs else None,
                subset=subset,
            )

            print("\n📊 Execution Finished. Generating Analysis Report...")
            saved_path = summary.get("saved_path")
            if saved_path:
                report = analyzer.analyze_experiment(saved_path)
                analyzer.print_report(report)
            else:
                print("Warning: Results were not saved, skipping detailed analysis.")

        except Exception as e:
            print(f"❌ Error during execution: {e}")
            raise e


if __name__ == "__main__":
    fire.Fire(BioAgentBenchmarkCLI)
