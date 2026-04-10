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
        start_time = time.perf_counter()
        use_early_stopping = (
            self.config.mutation_strategy == "none"
            and self.config.early_stop_patience > 0
        )

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
            gen_start = time.perf_counter()
            gen_log = [f"\n--- Generation {gen}/{self.config.num_generations} ---"]

            candidates = self.mutator.generate(
                parents, gen, self.config, self.performance_history
            )
            gen_log.append(f"  Generated {len(candidates)} candidates via {self.config.mutation_strategy}")

            cached_count = 0
            evaluated_count = 0
            for i, c in enumerate(candidates):
                cached = self.vector_store.get_cached_result(c.code_hash)
                if cached is not None:
                    cached_count += 1
                    c.fitness = cached["fitness"]
                    c.fitness_breakdown = {
                        **cached.get("fitness_breakdown", {}),
                        "cached": True,
                    }
                    gen_log.append(
                        f"  Candidate {i+1} ({c.code_hash[:8]}): fitness={c.fitness:.4f} [CACHED]"
                        f" | {c.mutation_description}"
                    )
                else:
                    evaluated_count += 1
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

            gen_elapsed = time.perf_counter() - gen_start

            fitnesses = [c.fitness or 0.0 for c in candidates]
            eval_times = [c.fitness_breakdown.get("eval_time_ms", 0) for c in candidates]
            best_candidate_this_gen = max(candidates, key=lambda c: c.fitness or 0.0)
            best_eval_time = best_candidate_this_gen.fitness_breakdown.get("eval_time_ms", 0)
            best_exec_time = best_candidate_this_gen.fitness_breakdown.get("exec_time_ms", 0)
            best_complexity = best_candidate_this_gen.fitness_breakdown.get("estimated_time_complexity", "")
            best_generalized_complexity = best_candidate_this_gen.fitness_breakdown.get("generalized_time_complexity", "")

            stats = {
                "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
                "max_fitness": max(fitnesses) if fitnesses else 0,
                "min_fitness": min(fitnesses) if fitnesses else 0,
                "gen_time_sec": gen_elapsed,
                "gen_time_ms": gen_elapsed * 1000,
                "avg_eval_time_ms": round(sum(eval_times) / len(eval_times), 2) if eval_times else 0,
                "best_eval_time_ms": round(best_eval_time, 2),
                "best_exec_time_ms": round(best_exec_time, 2),
                "candidates_generated": len(candidates),
                "candidates_evaluated": evaluated_count,
                "candidates_cached": cached_count,
                "candidates_selected": len(selected),
                "generation_step_count": len(candidates),
                "best_estimated_time_complexity": best_complexity,
                "best_generalized_time_complexity": best_generalized_complexity,
            }

            gen_log.append(
                f"  Generation time: {gen_elapsed*1000:.1f}ms | "
                f"Best eval: {best_eval_time:.1f}ms | "
                f"Evaluated: {evaluated_count} | Cached: {cached_count}"
            )

            # Build per-candidate attempt summaries for the LLM prompt
            attempt_summaries = []
            for c in candidates:
                delta = (c.fitness or 0) - (parents[0].fitness or 0) if parents else 0
                direction = "improved" if delta > 0 else ("unchanged" if delta == 0 else "regressed")
                short_desc = c.mutation_description[:80]
                attempt_summaries.append(
                    f"  {c.code_hash[:8]}: fitness={c.fitness:.4f} ({direction}, delta={delta:+.4f}) | {short_desc}"
                )

            self.performance_history.append({
                "gen": gen,
                "best": stats["max_fitness"],
                "avg": stats["avg_fitness"],
                "gen_time_sec": stats["gen_time_sec"],
                "gen_time_ms": stats["gen_time_ms"],
                "best_eval_time_ms": stats["best_eval_time_ms"],
                "best_exec_time_ms": stats["best_exec_time_ms"],
                "attempt_summaries": attempt_summaries,
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

            if use_early_stopping:
                if stats["max_fitness"] <= (last_best_fitness or 0):
                    best_fitness_streak += 1
                else:
                    best_fitness_streak = 0
                    last_best_fitness = stats["max_fitness"]

                if best_fitness_streak >= self.config.early_stop_patience:
                    self._log(f"\n[Early Stop] No improvement for {self.config.early_stop_patience} generations")
                    break

        elapsed = time.perf_counter() - start_time
        self._log(f"\n[Done] Evolution completed in {elapsed:.3f}s")

    def _log(self, message: str) -> None:
        self.log_entries.append(message)
