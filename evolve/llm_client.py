import re
import time
import openai


class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.7, max_tokens: int = 2048) -> str:
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except openai.APIError as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    raise

    def generate_code(self, system_prompt: str, user_prompt: str,
                      temperature: float = 0.7) -> str:
        raw = self.generate(system_prompt, user_prompt, temperature)
        return self._extract_code(raw)

    @staticmethod
    def _extract_code(text: str) -> str:
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        lines = text.strip().split("\n")
        code_lines = [l for l in lines if not l.startswith("#") or "import" in l or "def " in l]
        if any("def " in l or "return " in l or "if " in l for l in code_lines):
            return text.strip()
        return text.strip()
