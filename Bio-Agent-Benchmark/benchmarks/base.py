"""
벤치마크의 기본 인터페이스를 정의하는 모듈입니다.
모든 구체적인 벤치마크 구현체는 이 BaseBenchmark를 상속받아야 합니다.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseBenchmark(ABC):
    """
    모든 벤치마크 클래스의 기본 추상 클래스입니다.
    """

    @abstractmethod
    def load_tasks(self) -> List[Dict[str, Any]]:
        """
        벤치마크 태스크 리스트를 로드합니다.

        Returns:
            List[Dict[str, Any]]: 태스크 정보를 담은 딕셔너리 리스트
        """
        pass

    @abstractmethod
    def run_task(self, agent: Any, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 태스크에 대해 에이전트를 실행합니다.

        Args:
            agent (Any): 테스트할 에이전트 객체
            task (Dict[str, Any]): 실행할 태스크 정보

        Returns:
            Dict[str, Any]: 실행 결과 (예: 에이전트의 응답, 실행 시간 등)
        """
        pass

    @abstractmethod
    def evaluate(
        self, predictions: List[Any], ground_truth: List[Any]
    ) -> Dict[str, float]:
        """
        모델의 예측값과 정답을 비교하여 평가 지표를 계산합니다.

        Args:
            predictions (List[Any]): 에이전트의 예측값 리스트
            ground_truth (List[Any]): 실제 정답 리스트

        Returns:
            Dict[str, float]: 평가 메트릭 (예: accuracy, f1 등)
        """
        pass
