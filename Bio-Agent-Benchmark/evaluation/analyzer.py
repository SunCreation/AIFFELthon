"""
실험 결과를 분석하는 Analyzer 모듈입니다.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class Analyzer:
    def __init__(self):
        pass

    def load_results(self, exp_dir: str) -> pd.DataFrame:
        """
        실험 결과 파일(results.jsonl)을 로드하여 DataFrame으로 반환합니다.
        """
        results_path = os.path.join(exp_dir, "results.jsonl")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"No results found at {results_path}")

        data = []
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        return pd.DataFrame(data)

    def analyze_experiment(self, exp_dir: str) -> Dict[str, Any]:
        """
        실험 결과를 상세 분석합니다.

        Returns:
            Dict: 분석 리포트
        """
        logger.info(f"Analyzing experiment at: {exp_dir}")
        df = self.load_results(exp_dir)

        # 1. Overall Metrics
        total = len(df)
        if "score" not in df.columns:
            # score가 없으면 success 상태로 추정 (mockup)
            success = len(df[df["status"] == "success"])
            accuracy = success / total if total > 0 else 0.0
        else:
            accuracy = df["score"].mean()

        # 2. Category-wise Metrics (Task Name 기반)
        # metadata 컬럼이 dict 형태이므로, 이를 풀어서 task_name 등을 추출
        if "metadata" in df.columns:
            # pandas json_normalize 등을 쓸 수도 있지만 간단히 apply 사용
            df["category"] = df["metadata"].apply(
                lambda x: x.get("task_name", "unknown")
                if isinstance(x, dict)
                else "unknown"
            )
        else:
            df["category"] = "unknown"

        category_metrics = {}
        if "score" in df.columns:
            category_metrics = df.groupby("category")["score"].mean().to_dict()

        # 3. Failures Summary
        failures = (
            df[df["score"] < 1.0]
            if "score" in df.columns
            else df[df["status"] != "success"]
        )
        failure_counts = len(failures)

        report = {
            "experiment_id": os.path.basename(exp_dir),
            "total_tasks": total,
            "overall_accuracy": accuracy,
            "failure_count": failure_counts,
            "category_performance": category_metrics,
            "top_failures": failures[["task_id", "prediction", "ground_truth"]]
            .head(5)
            .to_dict(orient="records"),
        }

        return report

    def print_report(self, report: Dict[str, Any]):
        """
        분석 리포트를 보기 좋게 출력합니다.
        """
        print("\n" + "=" * 50)
        print(f"📊 Analysis Report: {report['experiment_id']}")
        print("=" * 50)
        print(f"Total Tasks: {report['total_tasks']}")
        print(f"Overall Accuracy: {report['overall_accuracy']:.2%}")
        print(f"Failures: {report['failure_count']}")
        print("-" * 50)
        print("Category Performance:")
        for cat, score in report["category_performance"].items():
            print(f"  - {cat}: {score:.2%}")
        print("-" * 50)
        print("Sample Failures (Top 5):")
        for fail in report["top_failures"]:
            print(
                f"  [ID: {fail['task_id']}] Pred: '{fail['prediction']}' vs GT: '{fail['ground_truth']}'"
            )
        print("=" * 50 + "\n")
