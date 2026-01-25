import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import asdict

from storage.schemas import BenchmarkResult

logger = logging.getLogger(__name__)


class ResultSaver:
    def __init__(self, base_dir: str = "logs/experiments"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_experiment(self, summary: Dict[str, Any], results: List[BenchmarkResult]):
        """
        실험 결과를 디렉토리에 저장합니다.

        Args:
            summary (Dict): 실험 요약 정보 (metrics 포함)
            results (List[BenchmarkResult]): 개별 태스크 실행 결과 리스트
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_name = summary.get("benchmark", "unknown")
        agent_name = summary.get("agent", "unknown")

        exp_dir = os.path.join(
            self.base_dir, f"{timestamp}_{benchmark_name}_{agent_name}"
        )
        os.makedirs(exp_dir, exist_ok=True)

        # 1. Summary 저장
        summary_path = os.path.join(exp_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        # 2. Detailed Results 저장 (JSONL)
        results_path = os.path.join(exp_dir, "results.jsonl")
        failures = []

        with open(results_path, "w", encoding="utf-8") as f:
            for res in results:
                # Dataclass -> Dict 변환
                if isinstance(res, BenchmarkResult):
                    data = asdict(res)
                else:
                    data = res  # 이미 dict인 경우

                # Timestamp 객체 직렬화 처리
                if "timestamp" in data and isinstance(data["timestamp"], datetime):
                    data["timestamp"] = data["timestamp"].isoformat()

                f.write(json.dumps(data, ensure_ascii=False) + "\n")

                # 실패/오답 케이스 수집 (status가 error이거나, success여도 점수가 0인 경우 등은 별도 로직 필요)
                # 여기서는 명시적인 에러 상태나 score < 1.0 인 경우를 수집 가능
                # 일단은 status != success 인 것과 score < 1.0 인 것을 수집
                score = data.get("score", 0.0)
                if data.get("status") != "success" or score < 1.0:
                    failures.append(data)

        # 3. Failures 저장 (분석용)
        if failures:
            failures_path = os.path.join(exp_dir, "failures.json")
            with open(failures_path, "w", encoding="utf-8") as f:
                json.dump(failures, f, indent=4, ensure_ascii=False)

        logger.info(f"Experiment results saved to: {exp_dir}")
        return exp_dir
