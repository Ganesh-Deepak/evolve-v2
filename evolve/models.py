from dataclasses import dataclass, field
import hashlib
import textwrap


def compute_code_hash(code: str) -> str:
    normalized = textwrap.dedent(code).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class RunConfig:
    problem_type: str  # "pacman" | "matrix"
    problem_description: str
    initial_code: str
    mutation_strategy: str  # "none" | "random" | "llm_guided"
    num_generations: int = 10
    population_size: int = 5
    top_k: int = 3
    early_stop_patience: int = 3
    timeout_per_candidate: int = 30
    fitness_weights: tuple = (0.5, 0.3, 0.2)
    openai_api_key: str = ""
    pacman_layouts: tuple[str, ...] = ("mediumClassic", "smallClassic")


@dataclass
class Candidate:
    code: str
    code_hash: str = ""
    generation: int = 0
    parent_hash: str | None = None
    mutation_type: str = "initial"
    mutation_description: str = ""
    fitness: float | None = None
    fitness_breakdown: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.code_hash:
            self.code_hash = compute_code_hash(self.code)


@dataclass
class GenerationResult:
    generation_num: int
    candidates: list[Candidate]
    selected: list[Candidate]
    best_candidate: Candidate
    best_overall: Candidate
    stats: dict = field(default_factory=dict)
    log_entries: list[str] = field(default_factory=list)


@dataclass
class EvolutionLog:
    config: RunConfig
    generations: list[GenerationResult]
    elapsed_seconds: float = 0.0
