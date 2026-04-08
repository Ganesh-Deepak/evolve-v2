import random
import re
import copy
from abc import ABC, abstractmethod

from evolve.models import Candidate, RunConfig, compute_code_hash
from evolve.llm_client import LLMClient
from evolve.vector_store import VectorStore
from evolve.prompts import (
    SYSTEM_PROMPT_PACMAN,
    SYSTEM_PROMPT_MATRIX,
    build_mutation_prompt,
)


class BaseMutator(ABC):
    @abstractmethod
    def generate(self, parents: list[Candidate], generation: int,
                 config: RunConfig, performance_history: list[dict]) -> list[Candidate]:
        pass


class NoEvolutionMutator(BaseMutator):
    """Single-shot LLM baseline: one LLM call with no evolutionary loop.
    If no LLM client is available, returns the original code unchanged."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client
        self._has_called = False

    def generate(self, parents: list[Candidate], generation: int,
                 config: RunConfig, performance_history: list[dict]) -> list[Candidate]:
        parent = parents[0]

        # Single-shot: only call LLM once (first generation), then return unchanged
        if self.llm_client and not self._has_called:
            self._has_called = True
            system_prompt = (SYSTEM_PROMPT_PACMAN if config.problem_type == "pacman"
                             else SYSTEM_PROMPT_MATRIX)
            user_prompt = (
                f"Improve the following code. Return only the improved code.\n\n"
                f"```python\n{parent.code}\n```"
            )
            try:
                new_code = self.llm_client.generate_code(
                    system_prompt, user_prompt, temperature=0.7
                )
                return [Candidate(
                    code=new_code,
                    generation=generation,
                    parent_hash=parent.code_hash,
                    mutation_type="single_shot_llm",
                    mutation_description="Single-shot LLM improvement (no evolution)",
                )]
            except Exception as e:
                return [Candidate(
                    code=parent.code,
                    generation=generation,
                    parent_hash=parent.code_hash,
                    mutation_type="none",
                    mutation_description=f"Single-shot LLM failed ({e}), code unchanged",
                )]

        return [
            Candidate(
                code=parent.code,
                generation=generation,
                parent_hash=parent.code_hash,
                mutation_type="none",
                mutation_description="No evolution - baseline (code unchanged)",
            )
            for _ in range(config.population_size)
        ]


class RandomMutator(BaseMutator):
    OPERATORS = ["parameter_perturbation", "operator_substitution", "block_swap",
                 "line_duplication", "constant_insertion"]

    def generate(self, parents: list[Candidate], generation: int,
                 config: RunConfig, performance_history: list[dict]) -> list[Candidate]:
        candidates = []
        for i in range(config.population_size):
            parent = parents[i % len(parents)]
            num_ops = random.randint(1, 2)
            ops_to_apply = random.sample(self.OPERATORS, min(num_ops, len(self.OPERATORS)))

            code = parent.code
            descriptions = []

            for op in ops_to_apply:
                code, desc = getattr(self, f"_apply_{op}")(code)
                descriptions.append(desc)

            candidates.append(Candidate(
                code=code,
                generation=generation,
                parent_hash=parent.code_hash,
                mutation_type=f"random_{'_'.join(ops_to_apply)}",
                mutation_description=" | ".join(descriptions),
            ))

        return candidates

    def _apply_parameter_perturbation(self, code: str) -> tuple[str, str]:
        numbers = list(re.finditer(r"(?<![a-zA-Z_])(\d+\.?\d*)", code))
        if not numbers:
            return code, "No numeric constants found"

        match = random.choice(numbers)
        old_val = float(match.group())
        if old_val == 0:
            new_val = random.uniform(0.1, 2.0)
        else:
            perturbation = random.uniform(-0.3, 0.3)
            new_val = old_val * (1 + perturbation)

        if "." not in match.group():
            new_val = int(round(new_val))

        new_code = code[:match.start()] + str(new_val) + code[match.end():]
        return new_code, f"Perturbed constant {old_val} -> {new_val}"

    def _apply_operator_substitution(self, code: str) -> tuple[str, str]:
        swaps = {
            "+": "-", "-": "+",
            "*": "+", "//": "/",
            ">": ">=", "<": "<=",
            ">=": ">", "<=": "<",
            "==": "!=", "!=": "==",
        }

        ops_found = []
        for op in swaps:
            for match in re.finditer(re.escape(op), code):
                if not (op in "+-*" and code[match.start()-1:match.start()] in "+-*=<>!"):
                    ops_found.append((match.start(), match.end(), op))

        if not ops_found:
            return code, "No operators found to substitute"

        start, end, old_op = random.choice(ops_found)
        new_op = swaps[old_op]
        new_code = code[:start] + new_op + code[end:]
        return new_code, f"Swapped operator '{old_op}' -> '{new_op}'"

    def _apply_block_swap(self, code: str) -> tuple[str, str]:
        lines = code.split("\n")
        non_empty = [(i, l) for i, l in enumerate(lines) if l.strip()]
        if len(non_empty) < 3:
            return code, "Too few lines to swap"

        idx1, idx2 = random.sample(range(len(non_empty)), 2)
        i1, l1 = non_empty[idx1]
        i2, l2 = non_empty[idx2]

        indent1 = len(l1) - len(l1.lstrip())
        indent2 = len(l2) - len(l2.lstrip())

        lines[i1] = " " * indent1 + l2.strip()
        lines[i2] = " " * indent2 + l1.strip()

        return "\n".join(lines), f"Swapped lines {i1+1} and {i2+1}"

    def _apply_line_duplication(self, code: str) -> tuple[str, str]:
        lines = code.split("\n")
        non_empty = [(i, l) for i, l in enumerate(lines) if l.strip() and not l.strip().startswith("#")]
        if not non_empty:
            return code, "No lines to duplicate"

        idx, line = random.choice(non_empty)
        lines.insert(idx + 1, line)
        return "\n".join(lines), f"Duplicated line {idx+1}: '{line.strip()[:40]}...'"

    def _apply_constant_insertion(self, code: str) -> tuple[str, str]:
        lines = code.split("\n")
        non_empty_indices = [i for i, l in enumerate(lines) if l.strip()]
        if not non_empty_indices:
            return code, "No lines found"

        idx = random.choice(non_empty_indices)
        indent = len(lines[idx]) - len(lines[idx].lstrip())
        var_name = random.choice(["threshold", "weight", "factor", "bonus", "penalty"])
        value = round(random.uniform(0.1, 10.0), 2)
        new_line = " " * indent + f"{var_name} = {value}"
        lines.insert(idx, new_line)
        return "\n".join(lines), f"Inserted constant: {var_name} = {value}"


class LLMGuidedMutator(BaseMutator):
    MATRIX_FOCUSES = [
        "Produce a fully correct baseline but reduce Python overhead by caching rows, columns, or scalar entries in local variables.",
        "Favor a mostly unrolled implementation with direct expressions for each output cell when it improves runtime.",
        "Reduce repeated indexing and temporary allocations while keeping the arithmetic explicit and easy for the evaluator to measure.",
    ]

    def __init__(self, llm_client: LLMClient, vector_store: VectorStore):
        self.llm_client = llm_client
        self.vector_store = vector_store

    def generate(self, parents: list[Candidate], generation: int,
                 config: RunConfig, performance_history: list[dict]) -> list[Candidate]:
        temperature = max(0.3, 0.8 - 0.5 * (generation / max(config.num_generations, 1)))

        system_prompt = (SYSTEM_PROMPT_PACMAN if config.problem_type == "pacman"
                         else SYSTEM_PROMPT_MATRIX)

        if config.problem_type == "pacman":
            fitness_desc = f"{config.fitness_weights[0]}*avg_score + {config.fitness_weights[1]}*max_score + {config.fitness_weights[2]}*survival"
        else:
            fitness_desc = (f"{config.fitness_weights[0]}*correctness + "
                           f"{config.fitness_weights[1]}*(1/(num_operations+1)) + "
                           f"{config.fitness_weights[2]}*(1/(exec_time_ms+1))")

        candidates = []
        for i in range(config.population_size):
            parent = parents[i % len(parents)]

            rag_examples = self.vector_store.get_similar(
                parent.code, n=3,
                min_fitness=(parent.fitness or 0) * 0.5
            )

            user_prompt = build_mutation_prompt(
                current_code=parent.code,
                fitness=parent.fitness or 0.0,
                rag_examples=rag_examples,
                performance_history=performance_history,
                gen=generation,
                max_gen=config.num_generations,
                temperature=temperature,
                fitness_description=fitness_desc,
                strategy_focus=self._strategy_focus(config.problem_type, i),
            )

            try:
                new_code = self.llm_client.generate_code(
                    system_prompt, user_prompt, temperature
                )
                desc = f"LLM-guided mutation (temp={temperature:.2f}, {len(rag_examples)} RAG examples)"
            except Exception as e:
                new_code = parent.code
                desc = f"LLM mutation failed ({e}), kept parent code"

            candidates.append(Candidate(
                code=new_code,
                generation=generation,
                parent_hash=parent.code_hash,
                mutation_type="llm_guided",
                mutation_description=desc,
            ))

        return candidates

    def _strategy_focus(self, problem_type: str, candidate_idx: int) -> str:
        if problem_type != "matrix":
            return ""
        return self.MATRIX_FOCUSES[candidate_idx % len(self.MATRIX_FOCUSES)]


def get_mutator(strategy: str, llm_client: LLMClient | None = None,
                vector_store: VectorStore | None = None) -> BaseMutator:
    if strategy == "none":
        return NoEvolutionMutator(llm_client=llm_client)
    elif strategy == "random":
        return RandomMutator()
    elif strategy == "llm_guided":
        if not llm_client or not vector_store:
            raise ValueError("LLM-guided mutation requires llm_client and vector_store")
        return LLMGuidedMutator(llm_client, vector_store)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
