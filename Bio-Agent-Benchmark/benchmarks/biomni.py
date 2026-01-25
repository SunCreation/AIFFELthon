"""
Biomni Eval1 벤치마크 구현체입니다.
Hugging Face 데이터셋을 로드하고 평가 로직을 수행합니다.
"""

import json
import ast
from typing import List, Dict, Any, Union, Optional
import pandas as pd
from .base import BaseBenchmark


class BiomniBenchmark(BaseBenchmark):
    DATASET_URL = "hf://datasets/biomni/Eval1/biomni_eval1_dataset.parquet"

    def __init__(self):
        self.df = None
        self.tasks = []

    def load_tasks(self) -> List[Dict[str, Any]]:
        if self.df is None:
            print(f"Loading Biomni dataset from {self.DATASET_URL}...")
            try:
                self.df = pd.read_parquet(self.DATASET_URL)
            except Exception:
                from datasets import load_dataset

                ds = load_dataset("biomni/Eval1", split="train")
                self.df = ds.to_pandas()

        self.tasks = []
        # Ensure df is a DataFrame before iterating
        if isinstance(self.df, pd.DataFrame):
            for idx, row in self.df.iterrows():
                task = {
                    "id": f"{row['task_name']}_{row['task_instance_id']}",
                    "task_name": row["task_name"],
                    "task_instance_id": row["task_instance_id"],
                    "prompt": row["prompt"],
                    "ground_truth": row["answer"],
                    "split": row.get("split", "unknown"),
                }
                self.tasks.append(task)

        print(f"Loaded {len(self.tasks)} tasks from Biomni Eval1.")
        return self.tasks

    def run_task(self, agent: Any, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = agent.predict(task["prompt"])
            return {
                "task_id": task["id"],
                "status": "success",
                "prediction": response,
                "ground_truth": task["ground_truth"],
                "metadata": {
                    "task_name": task["task_name"],
                    "task_instance_id": task["task_instance_id"],
                },
            }
        except Exception as e:
            return {
                "task_id": task["id"],
                "status": "error",
                "error_message": str(e),
                "metadata": {"task_name": task["task_name"]},
            }

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        total = 0
        correct = 0

        results_by_task = {}

        for pred in predictions:
            if pred["status"] != "success":
                continue

            task_name = pred["metadata"]["task_name"]
            user_answer = pred["prediction"]
            gt = pred["ground_truth"]

            score = self._compute_reward(task_name, user_answer, gt)

            total += 1
            correct += score

            if task_name not in results_by_task:
                results_by_task[task_name] = {"total": 0, "correct": 0}
            results_by_task[task_name]["total"] += 1
            results_by_task[task_name]["correct"] += score

        overall_accuracy = correct / total if total > 0 else 0.0

        metrics = {"overall_accuracy": overall_accuracy}

        for task, stats in results_by_task.items():
            metrics[f"accuracy_{task}"] = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            )

        return metrics

    def _compute_reward(
        self, task_name: str, user_answer: str, ground_truth: str
    ) -> float:
        try:
            if task_name == "crispr_delivery":
                return (
                    1.0
                    if user_answer.strip().lower() == ground_truth.strip().lower()
                    else 0.0
                )

            elif task_name.startswith("gwas_causal_gene"):
                return (
                    1.0
                    if user_answer.strip().upper() == ground_truth.strip().upper()
                    else 0.0
                )

            elif task_name == "gwas_variant_prioritization":
                return 1.0 if user_answer.strip() == ground_truth.strip() else 0.0

            elif task_name == "hle":
                return (
                    1.0
                    if user_answer.strip().upper() == ground_truth.strip().upper()
                    else 0.0
                )

            elif task_name.startswith("lab_bench"):
                return (
                    1.0
                    if user_answer.strip().upper() == ground_truth.strip().upper()
                    else 0.0
                )

            elif task_name == "rare_disease_diagnosis":
                return self._evaluate_json_match(user_answer, ground_truth, "OMIM_ID")

            elif task_name == "screen_gene_retrieval":
                return (
                    1.0
                    if user_answer.strip().upper() == ground_truth.strip().upper()
                    else 0.0
                )

            elif task_name == "patient_gene_detection":
                return self._evaluate_gene_detection(user_answer, ground_truth)

            else:
                return (
                    1.0
                    if str(user_answer).strip() == str(ground_truth).strip()
                    else 0.0
                )

        except Exception:
            return 0.0

    def _evaluate_json_match(
        self, user_answer: str, ground_truth: Union[str, Dict], key: str
    ) -> float:
        try:
            user_dict = self._parse_json_or_dict(user_answer)
            gt_dict = self._parse_json_or_dict(ground_truth)
            return 1.0 if str(user_dict.get(key)) == str(gt_dict.get(key)) else 0.0
        except Exception:
            return 0.0

    def _evaluate_gene_detection(self, user_answer: str, ground_truth: str) -> float:
        try:
            user_dict = self._parse_json_or_dict(user_answer)
            predicted_genes = user_dict.get("causal_gene", [])
            if not isinstance(predicted_genes, list):
                predicted_genes = [predicted_genes]

            if "," in str(ground_truth):
                true_genes = [g.strip() for g in ground_truth.split(",")]
            else:
                true_genes = [ground_truth]

            if predicted_genes and set(true_genes) & set(predicted_genes):
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def _parse_json_or_dict(self, data: Union[str, Dict]) -> Dict:
        if isinstance(data, dict):
            return data
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return ast.literal_eval(data)
