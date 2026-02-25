import os
import time
import threading
import queue
from typing import Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class LLMAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        timeout: float = 120.0,
        use_streaming: bool = True,
        stream_stall_timeout: float = 60.0,
    ):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=timeout,
                base_url=base_url,
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=timeout,
            )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.use_streaming = use_streaming
        self.stream_stall_timeout = stream_stall_timeout
        logger.info(
            "LLMAgent initialized: model=%s, base_url=%s, timeout=%ss, "
            "streaming=%s, stall_timeout=%ss",
            self.model,
            base_url or "https://api.openai.com/v1",
            timeout,
            use_streaming,
            stream_stall_timeout,
        )

    def predict(self, prompt: str, task_id: str = "unknown") -> str:
        """
        LLM 추론 실행. Streaming/non-streaming 모드 지원.

        Args:
            prompt: 입력 프롬프트
            task_id: 모니터링용 태스크 ID
        Returns:
            예측 결과 문자열
        """
        if self.use_streaming:
            return self._predict_streaming(prompt, task_id)
        else:
            return self._predict_sync(prompt, task_id)

    def _build_messages(self, prompt: str) -> Any:
        """공통 메시지 구성."""
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
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _predict_streaming(self, prompt: str, task_id: str) -> str:
        """Streaming 모드 추론. TTFT 측정, stall 감지, 토큰별 진행 로그."""
        prompt_chars = len(prompt)
        worker_id = threading.current_thread().name
        logger.info(
            "[REQ] task=%s | worker=%s | prompt_chars=%d",
            task_id,
            worker_id,
            prompt_chars,
        )

        start = time.time()

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(prompt),
                temperature=0.0,
                stream=True,
            )

            content_parts = []
            reasoning_parts = []
            token_count = 0
            ttft = None  # type: Optional[float]
            last_token_time = start
            progress_interval = 10
            stream_events = queue.Queue()

            def _consume_stream() -> None:
                try:
                    for chunk_item in stream:
                        stream_events.put(("chunk", chunk_item))
                    stream_events.put(("done", None))
                except Exception as stream_error:
                    stream_events.put(("error", stream_error))

            consumer = threading.Thread(target=_consume_stream, daemon=True)
            consumer.start()

            while True:
                try:
                    event_name, payload = stream_events.get(timeout=0.5)
                except queue.Empty:
                    now = time.time()
                    if now - last_token_time >= self.stream_stall_timeout:
                        elapsed = now - start
                        logger.error(
                            "[STREAM_STALL] task=%s | worker=%s | idle=%.1fs | elapsed=%.1fs",
                            task_id,
                            worker_id,
                            now - last_token_time,
                            elapsed,
                        )
                        raise TimeoutError(
                            "No streaming token received for %.1fs"
                            % self.stream_stall_timeout
                        )
                    continue

                if event_name == "error":
                    raise payload

                if event_name == "done":
                    break

                chunk = payload
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                reasoning_content = getattr(delta, "reasoning_content", None)
                content = getattr(delta, "content", None)
                has_delta_text = bool(content) or bool(reasoning_content)

                if reasoning_content:
                    reasoning_parts.append(reasoning_content)

                if content:
                    content_parts.append(content)
                    token_count += 1

                if has_delta_text:
                    now = time.time()
                    last_token_time = now
                    if ttft is None:
                        ttft = now - start
                        logger.info(
                            "[STREAM_START] task=%s | worker=%s | ttft=%.2fs",
                            task_id,
                            worker_id,
                            ttft,
                        )

                if token_count > 0 and token_count % progress_interval == 0:
                    elapsed = time.time() - start
                    logger.info(
                        "[STREAM_PROGRESS] task=%s | worker=%s | "
                        "tokens_so_far=%d | elapsed=%.1fs",
                        task_id,
                        worker_id,
                        token_count,
                        elapsed,
                    )

            # Stream completed
            latency = time.time() - start
            answer = "".join(content_parts).strip()
            _ = "".join(reasoning_parts)

            logger.info(
                "[STREAM_END] task=%s | worker=%s | total_tokens=%d | "
                "ttft=%.2fs | latency=%.1fs | answer=%s",
                task_id,
                worker_id,
                token_count,
                ttft if ttft is not None else -1,
                latency,
                answer[:80],
            )

            return answer

        except Exception as e:
            latency = time.time() - start
            logger.error(
                "[ERR] task=%s | worker=%s | latency=%.1fs | error=%s: %s",
                task_id,
                worker_id,
                latency,
                type(e).__name__,
                e,
            )
            return "Error: Could not generate response."

    def _predict_sync(self, prompt: str, task_id: str) -> str:
        """Non-streaming 모드 추론 (기존 방식)."""
        prompt_chars = len(prompt)
        worker_id = threading.current_thread().name
        logger.info(
            "[REQ] task=%s | worker=%s | prompt_chars=%d",
            task_id,
            worker_id,
            prompt_chars,
        )

        start = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(prompt),
                temperature=0.0,
            )

            latency = time.time() - start
            usage = response.usage
            content = response.choices[0].message.content
            answer = content.strip() if content else ""
            completion_tokens = usage.completion_tokens if usage else 0
            prompt_tokens = usage.prompt_tokens if usage else "N/A"

            logger.info(
                "[RES] task=%s | worker=%s | latency=%.1fs | "
                "prompt_tokens=%s | completion_tokens=%d | answer=%s",
                task_id,
                worker_id,
                latency,
                str(prompt_tokens),
                completion_tokens,
                answer[:80],
            )

            return answer

        except Exception as e:
            latency = time.time() - start
            logger.error(
                "[ERR] task=%s | worker=%s | latency=%.1fs | error=%s: %s",
                task_id,
                worker_id,
                latency,
                type(e).__name__,
                e,
            )
            return "Error: Could not generate response."
