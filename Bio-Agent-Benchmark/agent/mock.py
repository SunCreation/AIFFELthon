"""
테스트용 Mock Agent를 정의하는 모듈입니다.
실제 LLM 호출 없이 고정된 응답을 반환합니다.
"""


class MockAgent:
    """
    테스트를 위한 더미 에이전트 클래스입니다.
    """

    def __init__(self):
        pass

    def predict(self, input_text: str) -> str:
        """
        입력에 대해 더미 응답을 반환합니다.

        Args:
            input_text (str): 에이전트에게 전달될 입력

        Returns:
            str: 고정된 더미 응답 문자열
        """
        return "dummy response"
