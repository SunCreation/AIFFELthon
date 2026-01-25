"""
Lab-Bench 벤치마크 구현체입니다.
Future-House/Lab-Bench 데이터셋을 사용합니다.
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import logging
from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class LabBenchBenchmark(BaseBenchmark):
    DATASET_ID = "futurehouse/lab-bench"
    # 실제 로드 가능한 서브셋 목록 (일부 서브셋은 구조가 다를 수 있으나, 일단 시도)
    AVAILABLE_SUBSETS = [
        "LitQA2",
        "DbQA",
        "SeqQA",
        "CloningScenarios",
        "ProtocolQA",
        "FigQA",
        "SuppQA",
        "TableQA",
    ]

    def __init__(self, subset: str = "all"):
        """
        Args:
            subset: 특정 서브셋 이름 또는 'all' (모든 서브셋 로드)
        """
        self.df = None  # 단일 DF로 관리하기 어려우므로 tasks 리스트 생성에 집중
        self.tasks = []
        self.target_subsets = self.AVAILABLE_SUBSETS if subset == "all" else [subset]

    def load_tasks(self) -> List[Dict[str, Any]]:
        if self.tasks:
            return self.tasks

        all_tasks = []

        for sub in self.target_subsets:
            logger.info(f"Loading Lab-Bench subset: {sub}...")
            try:
                from datasets import load_dataset

                # split='train'을 명시하여 Dataset 객체 반환 유도
                # trust_remote_code=True는 필요 시 추가, 현재 버전은 불필요 확인됨
                ds = load_dataset(self.DATASET_ID, sub, split="train")

                if hasattr(ds, "to_pandas"):
                    df = ds.to_pandas()
                else:
                    df = pd.DataFrame(ds)

                # 서브셋별 태스크 변환
                subset_tasks = self._convert_df_to_tasks(df, sub)
                all_tasks.extend(subset_tasks)
                logger.info(f"Loaded {len(subset_tasks)} tasks from {sub}")

            except Exception as e:
                logger.error(f"Failed to load subset {sub}: {e}")
                # Mock 데이터 추가 (테스트용)
                if sub == "LitQA2":  # 대표 서브셋 실패시에만 Mock 추가
                    logger.info("Adding MOCK data for verification.")
                    mock_df = pd.DataFrame(
                        [
                            {
                                "question": "MOCK: What is the capital of France?",
                                "distractors": ["London", "Berlin", "Madrid"],
                                "ideal": "Paris",
                                "id": "mock_litqa_1",
                            }
                        ]
                    )
                    all_tasks.extend(self._convert_df_to_tasks(mock_df, "LitQA2"))

        self.tasks = all_tasks
        logger.info(f"Total Lab-Bench tasks loaded: {len(self.tasks)}")
        return self.tasks

    def _convert_df_to_tasks(
        self, df: pd.DataFrame, subset_name: str
    ) -> List[Dict[str, Any]]:
        tasks = []
        if not isinstance(df, pd.DataFrame):
            return tasks

        for idx, row in df.iterrows():
            # 옵션 가져오기 및 None 처리
            distractors = row.get("distractors", [])
            if hasattr(distractors, "tolist"):
                options = distractors.tolist()
            elif isinstance(distractors, list):
                options = distractors.copy()
            else:
                options = []

            ideal = row.get("ideal", "")

            # 옵션 합치기
            all_options: List[str] = options + [str(ideal)]
            np.random.shuffle(all_options)

            # 정답 인덱스 찾기
            try:
                answer_idx = all_options.index(str(ideal))
                answer_letter = chr(ord("A") + answer_idx)
            except ValueError:
                continue

            # 프롬프트 구성
            options_text = "\n".join(
                [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(all_options)]
            )
            prompt = (
                f"Question: {row.get('question', '')}\n\n"
                f"Options:\n{options_text}\n\n"
                "Answer with the letter of the correct option (e.g., A, B, C, D)."
            )

            task_id = row.get("id", f"labbench_{subset_name}_{idx}")

            task = {
                "id": task_id,
                "task_name": f"lab_bench_{subset_name}",  # 카테고리 식별자
                "prompt": prompt,
                "ground_truth": answer_letter,
                "options": all_options,
                "metadata": {
                    "task_name": f"lab_bench_{subset_name}",
                    "subset": subset_name,
                    "ideal": str(ideal),
                },
            }
            tasks.append(task)
        return tasks

    def run_task(self, agent: Any, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = agent.predict(task["prompt"])
            return {
                "task_id": task["id"],
                "status": "success",
                "prediction": response,
                "ground_truth": task["ground_truth"],
                "metadata": task["metadata"],
            }
        except Exception as e:
            return {
                "task_id": task["id"],
                "status": "error",
                "error_message": str(e),
                "metadata": {},
            }

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        total = 0
        correct = 0

        # 전체 정확도 외에 서브셋별 정확도도 계산할 수 있지만,
        # BaseBenchmark 인터페이스는 Dict[str, float] 반환이므로
        # analyzer가 메타데이터 기반으로 상세 분석하도록 함.
        # 여기서는 전체 Accuracy만 반환.

        for pred in predictions:
            if pred["status"] != "success":
                continue

            prediction_text = str(pred["prediction"]).strip().upper()
            if "ANSWER:" in prediction_text:
                prediction_text = prediction_text.split("ANSWER:")[-1].strip()

            user_response = ""
            if len(prediction_text) > 0:
                first_char = prediction_text[0]
                if "A" <= first_char <= "Z":
                    user_response = first_char

            gt = str(pred["ground_truth"]).strip().upper()

            if user_response == gt:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}
