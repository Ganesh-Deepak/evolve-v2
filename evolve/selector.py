from evolve.models import Candidate
from evolve.vector_store import VectorStore


class Selector:
    def __init__(self, top_k: int, vector_store: VectorStore):
        self.top_k = top_k
        self.vector_store = vector_store
        self.global_best: Candidate | None = None

    def select(self, candidates: list[Candidate]) -> tuple[list[Candidate], list[str]]:
        logs = []
        ranked = sorted(candidates, key=lambda c: c.fitness or 0.0, reverse=True)

        selected = []
        selected_codes = []
        rejected_reasons = []

        for c in ranked:
            if len(selected) >= self.top_k:
                break

            is_dup = False
            for sc in selected_codes:
                if self._code_similarity(c.code, sc) > 0.95:
                    is_dup = True
                    break

            if is_dup:
                rejected_reasons.append(
                    f"  Rejected {c.code_hash[:8]} (fitness={c.fitness:.4f}) - too similar to selected candidate"
                )
                continue

            selected.append(c)
            selected_codes.append(c.code)

        if self.global_best is not None:
            best_in_selected = max(selected, key=lambda c: c.fitness or 0.0) if selected else None
            if best_in_selected is None or (self.global_best.fitness or 0) > (best_in_selected.fitness or 0):
                already_in = any(c.code_hash == self.global_best.code_hash for c in selected)
                if not already_in:
                    selected.insert(0, self.global_best)
                    logs.append(f"  Elitism: preserved global best {self.global_best.code_hash[:8]} (fitness={self.global_best.fitness:.4f})")

        best_candidate = max(candidates, key=lambda c: c.fitness or 0.0)
        if self.global_best is None or (best_candidate.fitness or 0) > (self.global_best.fitness or 0):
            self.global_best = best_candidate
            logs.append(f"  New global best: {best_candidate.code_hash[:8]} (fitness={best_candidate.fitness:.4f})")

        sel_str = ", ".join(f"{c.code_hash[:8]}({c.fitness:.4f})" for c in selected)
        logs.insert(0, f"  Selected: [{sel_str}]")
        logs.extend(rejected_reasons)

        return selected, logs

    def _code_similarity(self, code1: str, code2: str) -> float:
        if code1.strip() == code2.strip():
            return 1.0
        lines1 = set(code1.strip().split("\n"))
        lines2 = set(code2.strip().split("\n"))
        if not lines1 or not lines2:
            return 0.0
        intersection = lines1 & lines2
        union = lines1 | lines2
        return len(intersection) / len(union) if union else 0.0
