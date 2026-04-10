import ast
import re
import subprocess
import sys
import time
from pathlib import Path

from evolve.models import Candidate, RunConfig

DANGEROUS_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+shutil\b",
    r"\b__import__\b",
    r"\bopen\s*\(",
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\bcompile\s*\(",
]

INVALID_CANDIDATE_FITNESS = -1_000_000.0


def is_safe_code(code: str) -> bool:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            return False
    return True


class FitnessEvaluator:
    def __init__(self, config: RunConfig):
        self.config = config
        self.w1, self.w2, self.w3 = config.fitness_weights

    def evaluate(self, candidate: Candidate) -> Candidate:
        if not is_safe_code(candidate.code):
            candidate.fitness = INVALID_CANDIDATE_FITNESS
            candidate.fitness_breakdown = {
                "error": "Unsafe code detected",
                "invalid_candidate": True,
                "eval_time_ms": 0.0,
            }
            candidate.mutation_description += " [REJECTED: unsafe code]"
            return candidate

        eval_start = time.perf_counter()
        try:
            if self.config.problem_type == "pacman":
                fitness, breakdown = self._evaluate_pacman(candidate.code)
            elif self.config.problem_type == "matrix":
                fitness, breakdown = self._evaluate_matrix(candidate.code)
            else:
                fitness, breakdown = INVALID_CANDIDATE_FITNESS, {
                    "error": f"Unknown problem type: {self.config.problem_type}",
                    "invalid_candidate": True,
                }
        except Exception as e:
            fitness = INVALID_CANDIDATE_FITNESS
            breakdown = {"error": str(e), "invalid_candidate": True}

        eval_time_ms = (time.perf_counter() - eval_start) * 1000
        breakdown["eval_time_ms"] = round(eval_time_ms, 2)

        candidate.fitness = fitness
        candidate.fitness_breakdown = breakdown
        return candidate

    def _evaluate_pacman(self, code: str) -> tuple[float, dict]:
        pacman_dir = Path("./pacman").resolve()
        if not pacman_dir.exists():
            return INVALID_CANDIDATE_FITNESS, {
                "error": "Pac-Man framework not found in ./pacman/",
                "invalid_candidate": True,
            }

        agent_code = f'''from game import Agent
from game import Directions
import random
import math

class EvolvedAgent(Agent):
    def get_action(self, state):
{_indent(code, 8)}
'''
        # File must end in "gents.py" for the framework's load_agent to find it
        agent_path = pacman_dir / "evolvedAgents.py"
        agent_path.write_text(agent_code, encoding="utf-8")

        layouts = list(self.config.pacman_layouts)
        all_scores = []
        total_games = 0
        failed_layouts = []

        for layout in layouts:
            try:
                result = subprocess.run(
                    [sys.executable, "pacman.py",
                     "-p", "EvolvedAgent",
                     "-l", layout,
                     "-n", "5",
                     "-q", "--frame_time", "0"],
                    capture_output=True, text=True,
                    timeout=self.config.timeout_per_candidate * 3,
                    cwd=str(pacman_dir),
                )
                if result.returncode != 0:
                    failed_layouts.append(layout)
                    continue
                scores = _parse_pacman_output(result.stdout)
                if not scores:
                    failed_layouts.append(layout)
                    continue
                all_scores.extend(scores)
                total_games += len(scores)
            except subprocess.TimeoutExpired:
                failed_layouts.append(layout)
            except Exception:
                failed_layouts.append(layout)

        if not all_scores:
            return INVALID_CANDIDATE_FITNESS, {
                "error": "No Pac-Man games completed successfully",
                "invalid_candidate": True,
                "failed_layouts": ", ".join(failed_layouts),
            }

        avg_score = sum(all_scores) / len(all_scores)
        max_score = max(all_scores)
        min_score = min(all_scores)
        wins = sum(1 for s in all_scores if s > 0)
        win_rate = wins / len(all_scores) if all_scores else 0.0

        # Fitness = w1*avg_score + w2*max_score + w3*win_rate
        # w3 rewards survival (games where Pac-Man wins / scores positive)
        fitness = self.w1 * avg_score + self.w2 * max_score + self.w3 * (win_rate * abs(avg_score) if avg_score != 0 else 0)
        breakdown = {
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "win_rate": win_rate,
            "num_games": total_games,
            "layouts_tested": ", ".join(layouts),
            "failed_layouts": ", ".join(failed_layouts),
            "all_scores": all_scores,
        }
        breakdown.update(_estimate_algorithmic_complexity(code, "pacman"))
        return fitness, breakdown

    def _evaluate_matrix(self, code: str) -> tuple[float, dict]:
        import numpy as np

        func_code = f"def matrix_multiply(A, B):\n{_indent(code, 4)}"

        try:
            namespace = {"__builtins__": {"range": range, "len": len, "sum": sum,
                                          "int": int, "float": float, "list": list,
                                          "abs": abs, "min": min, "max": max,
                                          "enumerate": enumerate, "zip": zip,
                                          "round": round, "True": True, "False": False}}
            exec(func_code, namespace)
            matrix_multiply = namespace["matrix_multiply"]
        except Exception as e:
            return INVALID_CANDIDATE_FITNESS, {
                "error": f"Code compilation failed: {e}",
                "invalid_candidate": True,
            }

        np.random.seed(42)
        test_cases = [(np.random.randint(-10, 11, (3, 3)).tolist(),
                       np.random.randint(-10, 11, (3, 3)).tolist())
                      for _ in range(100)]

        correct = 0
        runtime_failures = 0
        # Time the actual function execution across all test cases
        exec_start = time.perf_counter()
        for A, B in test_cases:
            try:
                result = matrix_multiply(A, B)
                expected = (np.array(A) @ np.array(B)).tolist()
                if _matrices_equal(result, expected):
                    correct += 1
            except Exception:
                runtime_failures += 1
        exec_time_ms = (time.perf_counter() - exec_start) * 1000

        correctness = correct / len(test_cases)

        num_ops = _count_matrix_operations(matrix_multiply)

        if runtime_failures == len(test_cases):
            return INVALID_CANDIDATE_FITNESS, {
                "error": "Matrix candidate failed on every test case",
                "invalid_candidate": True,
                "correct_count": correct,
                "total_tests": len(test_cases),
            }

        fitness = (self.w1 * correctness
                   + self.w2 * (1.0 / (num_ops + 1))
                   + self.w3 * (1.0 / (exec_time_ms + 1)))
        breakdown = {
            "correctness": correctness,
            "correct_count": correct,
            "total_tests": len(test_cases),
            "runtime_failures": runtime_failures,
            "num_operations": num_ops,
            "exec_time_ms": round(exec_time_ms, 2),
        }
        breakdown.update(_estimate_algorithmic_complexity(code, "matrix"))
        return fitness, breakdown


def _indent(code: str, spaces: int) -> str:
    prefix = " " * spaces
    lines = code.split("\n")
    return "\n".join(prefix + line if line.strip() else line for line in lines)


def _parse_pacman_output(output: str) -> list[float]:
    scores = []
    for line in output.split("\n"):
        match = re.search(r"Score:\s*([-\d.]+)", line)
        if match:
            scores.append(float(match.group(1)))
        scores_match = re.search(r"Scores:\s*(.*)", line)
        if scores_match:
            try:
                scores = [float(s.strip()) for s in scores_match.group(1).split(",") if s.strip()]
            except ValueError:
                pass
    if not scores:
        avg_match = re.search(r"Average Score:\s*([-\d.]+)", output)
        if avg_match:
            scores = [float(avg_match.group(1))]
    return scores


def _matrices_equal(a, b, tol=1e-6) -> bool:
    if not a or not b:
        return False
    try:
        for i in range(3):
            for j in range(3):
                if abs(a[i][j] - b[i][j]) > tol:
                    return False
        return True
    except (IndexError, TypeError):
        return False


def _estimate_algorithmic_complexity(code: str, problem_type: str) -> dict:
    wrapped = f"def _candidate():\n{_indent(code, 4)}"
    try:
        tree = ast.parse(wrapped)
    except SyntaxError:
        return {
            "estimated_time_complexity": "Unknown",
            "generalized_time_complexity": "Unknown",
            "complexity_note": "Could not parse code for static complexity estimation.",
        }

    visitor = _ComplexityVisitor()
    visitor.visit(tree)

    if problem_type == "matrix":
        actual = "O(1) for fixed 3x3 input"
        if visitor.max_any_depth == 0:
            generalized = "O(1) specialized constant-size implementation"
        else:
            generalized = f"{_depth_to_big_o(visitor.max_any_depth)} generalized loop pattern"
        note = (
            "Static estimate. Fixed 3x3 matrix multiplication is constant-size at runtime, "
            "but the loop structure can still suggest how the approach would scale if generalized."
        )
    else:
        depth = max(visitor.max_dynamic_depth, visitor.max_any_depth)
        generalized = _depth_to_big_o(depth)
        actual = f"{generalized} heuristic per action decision"
        note = (
            "Heuristic static estimate based on loop/comprehension nesting inside the agent logic. "
            "Built-ins like min/max can still hide work, so treat this as an estimate rather than a proof."
        )

    return {
        "estimated_time_complexity": actual,
        "generalized_time_complexity": generalized,
        "complexity_note": note,
        "max_loop_depth": visitor.max_any_depth,
    }


class _OperationCounter:
    def __init__(self):
        self.adds = 0
        self.subs = 0
        self.mults = 0

    @property
    def total(self) -> int:
        return self.adds + self.subs + self.mults


class _TrackedScalar:
    def __init__(self, value: float, counter: _OperationCounter):
        self.value = value
        self.counter = counter

    def _coerce(self, other):
        if isinstance(other, _TrackedScalar):
            return other.value
        return other

    def __add__(self, other):
        self.counter.adds += 1
        return _TrackedScalar(self.value + self._coerce(other), self.counter)

    def __radd__(self, other):
        self.counter.adds += 1
        return _TrackedScalar(self._coerce(other) + self.value, self.counter)

    def __sub__(self, other):
        self.counter.subs += 1
        return _TrackedScalar(self.value - self._coerce(other), self.counter)

    def __rsub__(self, other):
        self.counter.subs += 1
        return _TrackedScalar(self._coerce(other) - self.value, self.counter)

    def __mul__(self, other):
        self.counter.mults += 1
        return _TrackedScalar(self.value * self._coerce(other), self.counter)

    def __rmul__(self, other):
        self.counter.mults += 1
        return _TrackedScalar(self._coerce(other) * self.value, self.counter)

    def __neg__(self):
        return _TrackedScalar(-self.value, self.counter)

    def __pos__(self):
        return _TrackedScalar(+self.value, self.counter)

    def __abs__(self):
        return abs(self.value)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __lt__(self, other):
        return self.value < self._coerce(other)

    def __le__(self, other):
        return self.value <= self._coerce(other)

    def __gt__(self, other):
        return self.value > self._coerce(other)

    def __ge__(self, other):
        return self.value >= self._coerce(other)

    def __eq__(self, other):
        return self.value == self._coerce(other)

    def __repr__(self):
        return repr(self.value)


def _count_matrix_operations(matrix_multiply) -> int:
    counter = _OperationCounter()
    A = [
        [_TrackedScalar(1, counter), _TrackedScalar(2, counter), _TrackedScalar(3, counter)],
        [_TrackedScalar(4, counter), _TrackedScalar(5, counter), _TrackedScalar(6, counter)],
        [_TrackedScalar(7, counter), _TrackedScalar(8, counter), _TrackedScalar(9, counter)],
    ]
    B = [
        [_TrackedScalar(9, counter), _TrackedScalar(8, counter), _TrackedScalar(7, counter)],
        [_TrackedScalar(6, counter), _TrackedScalar(5, counter), _TrackedScalar(4, counter)],
        [_TrackedScalar(3, counter), _TrackedScalar(2, counter), _TrackedScalar(1, counter)],
    ]

    try:
        matrix_multiply(A, B)
        return counter.total
    except Exception:
        return 999


class _ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.current_any_depth = 0
        self.current_dynamic_depth = 0
        self.max_any_depth = 0
        self.max_dynamic_depth = 0

    def _enter_loop(self, dynamic: bool):
        self.current_any_depth += 1
        self.max_any_depth = max(self.max_any_depth, self.current_any_depth)
        if dynamic:
            self.current_dynamic_depth += 1
            self.max_dynamic_depth = max(self.max_dynamic_depth, self.current_dynamic_depth)

    def _exit_loop(self, dynamic: bool):
        if dynamic:
            self.current_dynamic_depth -= 1
        self.current_any_depth -= 1

    def visit_For(self, node):
        dynamic = not _is_constant_iter(node.iter)
        self._enter_loop(dynamic)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        self._exit_loop(dynamic)

    def visit_While(self, node):
        self._enter_loop(True)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        self._exit_loop(True)

    def visit_ListComp(self, node):
        self._visit_comprehension(node.generators)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self._visit_comprehension(node.generators)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self._visit_comprehension(node.generators)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node.generators)
        self.generic_visit(node)

    def _visit_comprehension(self, generators):
        dynamic_flags = [not _is_constant_iter(gen.iter) for gen in generators]
        for dynamic in dynamic_flags:
            self._enter_loop(dynamic)
        for dynamic in reversed(dynamic_flags):
            self._exit_loop(dynamic)


def _is_constant_iter(node) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        return all(isinstance(arg, ast.Constant) for arg in node.args)
    return False


def _depth_to_big_o(depth: int) -> str:
    return {
        0: "O(1)",
        1: "O(n)",
        2: "O(n^2)",
        3: "O(n^3)",
    }.get(depth, f"O(n^{depth})")
