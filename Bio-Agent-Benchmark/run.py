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
        use_wandb: bool = True,
        subset: str = "all",
    ):
        """
        벤치마크 실행

        Args:
            benchmark (str): 실행할 벤치마크 이름 (biomni, labbench 등)
            agent (str): 실행할 에이전트 이름 (mock, llm 등)
            limit (int): 실행할 태스크 수 제한 (테스트용)
            use_wandb (bool): W&B 로깅 사용 여부
            subset (str): Lab-bench 실행 시 특정 서브셋 지정 (기본값: all)
        """
        print(
            f"🚀 Initializing Benchmark: {benchmark} | Agent: {agent} | Subset: {subset}"
        )

        runner = ExperimentRunner()
        analyzer = Analyzer()

        try:
            # Runner에 subset 인자 전달
            # ExperimentRunner.get_benchmark가 이를 처리함
            # run_benchmark 메서드 시그니처가 이를 직접 받지 않으므로,
            # runner.get_benchmark를 직접 호출하거나 run_benchmark를 수정해야 함.
            # 현재 구조상 run_benchmark 내부에서 get_benchmark를 호출하는데 인자가 고정되어 있음.
            # 따라서 ExperimentRunner.run_benchmark 메서드 수정이 필요함.

            # (runner.py 수정을 피하기 위해 여기서 kwargs를 넘기는 방식으로 변경)
            summary = runner.run_benchmark(
                benchmark_name=benchmark,
                agent_name=agent,
                limit=limit,
                use_wandb=use_wandb,
                subset=subset,  # runner.py 수정 필요
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
