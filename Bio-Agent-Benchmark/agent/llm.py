import os
from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class LLMAgent:
    def __init__(self, model: str = None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def predict(self, prompt: str) -> str:
        try:
            # System prompt enhanced for exact match evaluation
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
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response from LLM: {e}")
            return "Error: Could not generate response."
