from evolve.models import Candidate
from evolve.vector_store import VectorStore


class Selector:
    def __init__(self, top_k: int, vector_store: VectorStore):
        self.top_k = top_k
        self.vector_store = vector_store
        self.global_best: Candidate | None = None

    def select(self, candidates: list[Candidate]) -> tuple[list[Candidate], list[str]]:
        logs = []

        # Separate valid and invalid candidates
        valid = [c for c in candidates if not c.fitness_breakdown.get("invalid_candidate") and c.fitness is not None]
        if not valid:
            # All invalid -- fall back to raw ranking
            valid = candidates

        # --- Fitness-distance balancing ---
        # Pick first by pure fitness, then balance fitness with diversity
        ranked_by_fitness = sorted(valid, key=lambda c: c.fitness or 0, reverse=True)

        selected = []
        selected_codes = []
        rejected_reasons = []

        # Always pick the top candidate first
        if ranked_by_fitness:
            best = ranked_by_fitness[0]
            selected.append(best)
            selected_codes.append(best.code)

        # Fill remaining slots using fitness-diversity score
        remaining = [c for c in ranked_by_fitness if c not in selected]
        while len(selected) < self.top_k and remaining:
            best_score = -float("inf")
            best_candidate = None

            # Normalize fitness values for scoring
            fit_vals = [c.fitness or 0 for c in remaining]
            max_fit = max(fit_vals) if fit_vals else 1
            min_fit = min(fit_vals) if fit_vals else 0
            fit_range = max_fit - min_fit if max_fit != min_fit else 1.0

            for c in remaining:
                # Fitness component (0-1)
                norm_fitness = ((c.fitness or 0) - min_fit) / fit_range

                # Diversity component: min distance to any already-selected candidate
                min_sim = min(
                    (self._code_similarity(c.code, sc) for sc in selected_codes),
                    default=0.0,
                )
                diversity = 1.0 - min_sim  # higher = more diverse

                # Reject near-duplicates
                if min_sim > 0.95:
                    rejected_reasons.append(
                        f"  Rejected {c.code_hash[:8]} (fitness={c.fitness:.4f}) - too similar to selected candidate"
                    )
                    remaining.remove(c)
                    continue

                # Combined score: 60% fitness, 40% diversity
                combined = 0.6 * norm_fitness + 0.4 * diversity
                if combined > best_score:
                    best_score = combined
                    best_candidate = c

            if best_candidate is None:
                break
            selected.append(best_candidate)
            selected_codes.append(best_candidate.code)
            remaining.remove(best_candidate)

        # Elitism: always keep global best in the parent pool
        if self.global_best is not None:
            best_in_selected = max(selected, key=self._rank_key) if selected else None
            if best_in_selected is None or self._rank_key(self.global_best) > self._rank_key(best_in_selected):
                already_in = any(c.code_hash == self.global_best.code_hash for c in selected)
                if not already_in:
                    selected.insert(0, self.global_best)
                    logs.append(f"  Elitism: preserved global best {self.global_best.code_hash[:8]} (fitness={self.global_best.fitness:.4f})")

        best_candidate = max(candidates, key=self._rank_key)
        if self.global_best is None or self._rank_key(best_candidate) > self._rank_key(self.global_best):
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

    def _rank_key(self, candidate: Candidate) -> tuple[int, float]:
        is_valid = 0 if candidate.fitness_breakdown.get("invalid_candidate") or candidate.fitness_breakdown.get("error") else 1
        return (is_valid, candidate.fitness if candidate.fitness is not None else float("-inf"))
