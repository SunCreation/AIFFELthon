import os
import time
import threading
from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class LLMAgent:
    def __init__(self, model: str = None, timeout: float = 120.0):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")
        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=timeout,
            **({"base_url": base_url} if base_url else {}),
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        logger.info(
            f"LLMAgent initialized: model={self.model}, "
            f"base_url={base_url or 'https://api.openai.com/v1'}, "
            f"timeout={timeout}s"
        )

    def predict(self, prompt: str, task_id: str = "unknown") -> str:
        """
        LLM 추론 실행. 상세 모니터링 로그 포함.

        Args:
            prompt: 입력 프롬프트
            task_id: 모니터링용 태스크 ID
        Returns:
            예측 결과 문자열
        """
        prompt_chars = len(prompt)
        worker_id = threading.current_thread().name
        logger.info(
            f"[REQ] task={task_id} | worker={worker_id} | prompt_chars={prompt_chars}"
        )

        start = time.time()

        try:
            system_prompt = (
                "You are a biomedical AI assistant participating in a benchmark evaluation.\n"
                "Your task is to answer the question as accurately as possible.\n"
                "CRITICAL INSTRUCTION: Output ONLY the final answer.\n"
                "- Do NOT provide explanations, reasoning, or full sentences.\n"
                "- If the answer is a gene symbol, output ONLY the symbol (e.g., BRCA1).\n"
                "- If the answer is a variant ID, output ONLY the ID (e.g., rs12345).\n"
                "- If the answer is a choice, output ONLY the letter (e.g., A).\n"
                "- Do not add punctuation like periods at the end."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            latency = time.time() - start
            usage = response.usage
            content = response.choices[0].message.content
            answer = content.strip() if content else ""

            logger.info(
                f"[RES] task={task_id} | worker={worker_id} | "
                f"latency={latency:.1f}s | "
                f"prompt_tokens={usage.prompt_tokens} | "
                f"completion_tokens={usage.completion_tokens} | "
                f"answer={answer[:80]}"
            )

            return answer

        except Exception as e:
            latency = time.time() - start
            logger.error(
                f"[ERR] task={task_id} | worker={worker_id} | "
                f"latency={latency:.1f}s | error={type(e).__name__}: {e}"
            )
            return "Error: Could not generate response."
