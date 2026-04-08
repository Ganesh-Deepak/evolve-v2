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

        cleaned = text.strip().strip("`").strip()
        if not cleaned:
            return ""

        lines = cleaned.splitlines()
        start_idx = 0
        code_markers = (
            "def ", "return ", "if ", "for ", "while ", "try:", "except", "result",
            "legal", "closest_food", "best_action", "A[", "B["
        )

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("Here", "This", "Explanation", "Improved", "Updated")) and ":" in stripped:
                continue
            if stripped.startswith(code_markers) or stripped.endswith(":") or "=" in stripped:
                start_idx = i
                break

        candidate_lines = []
        for line in lines[start_idx:]:
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            if stripped and not candidate_lines and stripped.startswith(("Here", "This", "Explanation", "Improved", "Updated")):
                continue
            candidate_lines.append(line)

        return "\n".join(candidate_lines).strip()
