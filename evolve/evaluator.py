import ast
import os
import re
import subprocess
import sys
import tempfile
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


def is_safe_code(code: str) -> bool:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            return False
    return True


class FitnessEvaluator:
    def __init__(self, config: RunConfig):
        self.config = config
        self.w1, self.w2, self.w3 = config.fitness_weights
        self.pacman_dir = Path(config.initial_code).parent if False else Path("./pacman")

    def evaluate(self, candidate: Candidate) -> Candidate:
        if not is_safe_code(candidate.code):
            candidate.fitness = 0.0
            candidate.fitness_breakdown = {"error": "Unsafe code detected"}
            candidate.mutation_description += " [REJECTED: unsafe code]"
            return candidate

        try:
            if self.config.problem_type == "pacman":
                fitness, breakdown = self._evaluate_pacman(candidate.code)
            elif self.config.problem_type == "matrix":
                fitness, breakdown = self._evaluate_matrix(candidate.code)
            else:
                fitness, breakdown = 0.0, {"error": f"Unknown problem type: {self.config.problem_type}"}
        except Exception as e:
            fitness = 0.0
            breakdown = {"error": str(e)}

        candidate.fitness = fitness
        candidate.fitness_breakdown = breakdown
        return candidate

    def _evaluate_pacman(self, code: str) -> tuple[float, dict]:
        pacman_dir = Path("./pacman").resolve()
        if not pacman_dir.exists():
            return 0.0, {"error": "Pac-Man framework not found in ./pacman/"}

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

        layouts = ["mediumClassic", "smallClassic"]
        all_scores = []
        total_games = 0

        for layout in layouts:
            try:
                result = subprocess.run(
                    [sys.executable, "pacman.py",
                     "-p", "EvolvedAgent",
                     "-l", layout,
                     "-n", "3",
                     "-q", "--frame_time", "0"],
                    capture_output=True, text=True,
                    timeout=self.config.timeout_per_candidate * 3,
                    cwd=str(pacman_dir),
                )
                scores = _parse_pacman_output(result.stdout)
                all_scores.extend(scores)
                total_games += len(scores)
            except subprocess.TimeoutExpired:
                all_scores.append(0)
            except Exception as e:
                all_scores.append(0)

        if not all_scores:
            return 0.0, {"error": "No games completed"}

        avg_score = sum(all_scores) / len(all_scores)
        max_score = max(all_scores)

        fitness = self.w1 * avg_score + self.w2 * max_score
        breakdown = {
            "avg_score": avg_score,
            "max_score": max_score,
            "num_games": total_games,
            "all_scores": all_scores,
        }
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
            return 0.0, {"error": f"Code compilation failed: {e}"}

        np.random.seed(42)
        test_cases = [(np.random.randint(-10, 11, (3, 3)).tolist(),
                       np.random.randint(-10, 11, (3, 3)).tolist())
                      for _ in range(100)]

        correct = 0
        for A, B in test_cases:
            try:
                result = matrix_multiply(A, B)
                expected = (np.array(A) @ np.array(B)).tolist()
                if _matrices_equal(result, expected):
                    correct += 1
            except Exception:
                pass

        correctness = correct / len(test_cases)

        num_ops = _count_operations(func_code)

        fitness = self.w1 * correctness + self.w2 * (1.0 / (num_ops + 1))
        breakdown = {
            "correctness": correctness,
            "correct_count": correct,
            "total_tests": len(test_cases),
            "num_operations": num_ops,
        }
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


def _count_operations(func_code: str) -> int:
    try:
        tree = ast.parse(func_code)
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
                    count += 1
        return count
    except SyntaxError:
        return 999
