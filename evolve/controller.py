import time
from typing import Generator

from evolve.models import Candidate, RunConfig, GenerationResult, EvolutionLog
from evolve.candidate_generator import get_mutator
from evolve.evaluator import FitnessEvaluator
from evolve.selector import Selector
from evolve.llm_client import LLMClient
from evolve.vector_store import VectorStore


class EvolutionController:
    def __init__(self, config: RunConfig, vector_store: VectorStore | None = None):
        self.config = config
        self.vector_store = vector_store or VectorStore()
        self.llm_client = (
            LLMClient(config.openai_api_key)
            if config.mutation_strategy in ("llm_guided", "none") and config.openai_api_key
            else None
        )
        self.mutator = get_mutator(
            config.mutation_strategy, self.llm_client, self.vector_store
        )
        self.evaluator = FitnessEvaluator(config)
        self.selector = Selector(config.top_k, self.vector_store)
        self.log_entries: list[str] = []
        self.performance_history: list[dict] = []

    def run_evolution(self) -> Generator[GenerationResult, None, None]:
        start_time = time.time()

        initial = Candidate(
            code=self.config.initial_code,
            generation=0,
            mutation_type="initial",
            mutation_description="Initial seed code",
        )
        self.evaluator.evaluate(initial)
        self.vector_store.add_candidate(initial)

        self._log(f"[Init] Evaluated initial candidate: fitness={initial.fitness:.4f}")
        self.selector.global_best = initial

        parents = [initial]
        best_fitness_streak = 0
        last_best_fitness = initial.fitness

        all_generations = []

        for gen in range(1, self.config.num_generations + 1):
            gen_log = [f"\n--- Generation {gen}/{self.config.num_generations} ---"]

            candidates = self.mutator.generate(
                parents, gen, self.config, self.performance_history
            )
            gen_log.append(f"  Generated {len(candidates)} candidates via {self.config.mutation_strategy}")

            for i, c in enumerate(candidates):
                cached = self.vector_store.get_cached_fitness(c.code_hash)
                if cached is not None:
                    c.fitness = cached
                    c.fitness_breakdown = {"cached": True}
                    gen_log.append(f"  Candidate {i+1} ({c.code_hash[:8]}): fitness={c.fitness:.4f} [CACHED]")
                else:
                    self.evaluator.evaluate(c)
                    self.vector_store.add_candidate(c)
                    gen_log.append(
                        f"  Candidate {i+1} ({c.code_hash[:8]}): fitness={c.fitness:.4f} "
                        f"| {c.mutation_description}"
                    )

            selected, sel_logs = self.selector.select(candidates)
            gen_log.extend(sel_logs)

            parents = selected if selected else parents

            best_candidate = max(candidates, key=lambda c: c.fitness or 0.0)
            best_overall = self.selector.global_best or best_candidate

            fitnesses = [c.fitness or 0.0 for c in candidates]
            stats = {
                "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
                "max_fitness": max(fitnesses) if fitnesses else 0,
                "min_fitness": min(fitnesses) if fitnesses else 0,
            }

            self.performance_history.append({
                "gen": gen,
                "best": stats["max_fitness"],
                "avg": stats["avg_fitness"],
            })

            self.log_entries.extend(gen_log)

            result = GenerationResult(
                generation_num=gen,
                candidates=candidates,
                selected=selected,
                best_candidate=best_candidate,
                best_overall=best_overall,
                stats=stats,
                log_entries=gen_log,
            )
            all_generations.append(result)
            yield result

            if stats["max_fitness"] <= (last_best_fitness or 0):
                best_fitness_streak += 1
            else:
                best_fitness_streak = 0
                last_best_fitness = stats["max_fitness"]

            if best_fitness_streak >= self.config.early_stop_patience:
                self._log(f"\n[Early Stop] No improvement for {self.config.early_stop_patience} generations")
                break

        elapsed = time.time() - start_time
        self._log(f"\n[Done] Evolution completed in {elapsed:.1f}s")

    def _log(self, message: str) -> None:
        self.log_entries.append(message)
