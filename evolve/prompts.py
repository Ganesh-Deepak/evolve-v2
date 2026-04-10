SYSTEM_PROMPT_PACMAN = """You are an expert Python programmer improving a Pac-Man agent for the UC Berkeley CS188 Python 3 framework.
You must output only executable Python statements for the body of get_action(self, state).

Available GameState APIs (snake_case):
- state.get_legal_actions()
- state.get_pacman_position()
- state.get_food().as_list()
- state.get_ghost_positions()
- state.get_ghost_states()  # each ghost has scared_timer
- state.get_capsules()
- state.generate_pacman_successor(action)
- state.get_score()
- state.get_num_food()
- state.get_walls()

Hard constraints:
- Return raw Python code only. No markdown fences. No explanation.
- Do not output a class definition or a def line.
- Keep the code self-contained inside the method body.
- Use only supported GameState APIs plus builtins, math, and random.
- Always return a legal action from state.get_legal_actions(); if no legal actions exist, return 'Stop'.
- Prefer a correct, runnable heuristic over a clever but brittle rewrite.

Optimization priorities:
- maximize score
- avoid immediate ghost danger unless ghosts are scared
- reduce wasted steps, especially unnecessary Stop actions
- keep the decision logic fast enough to run every turn"""

SYSTEM_PROMPT_MATRIX = """You are an expert Python programmer optimizing 3x3 matrix multiplication.
You must output only executable Python statements for the body of matrix_multiply(A, B).

Function contract:
- A and B are 3x3 lists of lists containing numbers
- return a 3x3 list of lists with the exact mathematical product

Hard constraints:
- Return raw Python code only. No markdown fences. No explanation.
- Do not output a def line.
- Do not import anything and do not use numpy.
- Prefer a correct, runnable solution over a risky algebraic trick.
- Keep arithmetic explicit and readable for the evaluator.

Optimization priorities:
- correctness first
- minimize scalar +, -, and * operations
- minimize Python overhead such as repeated indexing, temporary allocations, and unnecessary loops
- prefer local scalar caching and simple return expressions

Evaluator notes:
- hidden work inside comprehensions or sum-like patterns is still counted
- fully correct explicit code is better than speculative low-operation code that may fail"""

MUTATION_USER_TEMPLATE = """[Current Candidate]
fitness = {fitness:.4f}
```python
{current_code}
```

[Observed Metrics]
{observed_metrics}

[Recent Search History]
{performance_history}

[Previously Attempted Mutations & Their Outcomes]
{attempt_history}

[Similar High-Performing Examples]
{rag_examples}

[Generation Context]
generation = {gen}/{max_gen}
temperature = {temperature:.2f}
fitness formula = {fitness_description}

[Task]
Produce one improved replacement body for this candidate.
- Preserve syntax correctness and framework compatibility.
- Prefer targeted, high-confidence improvements over noisy rewrites.
- If the current code is invalid or weak, repair correctness before optimizing further.
- IMPORTANT: Do NOT repeat mutations that already regressed or produced no improvement (see attempted mutations above). Try a genuinely different approach.
{strategy_focus}
Return only the replacement code body."""

SINGLE_SHOT_USER_TEMPLATE = """[Current Candidate]
```python
{current_code}
```

[Observed Metrics]
{observed_metrics}

[Objective]
This is a single-shot baseline improvement pass, not an iterative search.
fitness formula = {fitness_description}

[Task]
Make one strong, high-confidence improvement to the code.
- Preserve correctness first.
- Prefer a robust improvement over an ambitious rewrite.
{strategy_focus}
Return only the replacement code body."""

CROSSOVER_USER_TEMPLATE = """[Parent A] (fitness={fitness_a:.4f})
```python
{code_a}
```

[Parent B] (fitness={fitness_b:.4f})
```python
{code_b}
```

[Fitness Formula]
{fitness_description}

[Task]
Combine the best ideas from both parents into a single improved child.
- Parent A's strengths: higher-fitness components, structural patterns that score well.
- Parent B's strengths: different approach or technique that may complement Parent A.
- Merge complementary logic rather than just picking one parent.
- The result must be syntactically valid and self-contained.
- Do NOT just return one parent unchanged.
Return only the replacement code body."""


PACMAN_DESCRIPTION_USER_TEMPLATE = """Convert the following algorithm description into a complete Python method body for get_action(self, state).

[Description]
{description}

[Requirements]
- Use only the supported GameState APIs from the system prompt.
- Return a legal action string and return 'Stop' if no legal actions exist.
- If the description is underspecified, produce a simple, reliable heuristic rather than inventing unsupported APIs.
- Avoid markdown, explanations, and placeholder comments.

Return only the raw Python method body."""

MATRIX_DESCRIPTION_USER_TEMPLATE = """Convert the following algorithm description into a complete Python function body for matrix_multiply(A, B).

[Description]
{description}

[Requirements]
- The result must be the exact product of two 3x3 matrices.
- If the description is underspecified, produce a fully correct baseline implementation specialized to 3x3 input.
- Do not use imports, numpy, or helper text.
- Keep the arithmetic explicit and the return value as a 3x3 list of lists.

Return only the raw Python function body."""


def build_fitness_description(problem_type: str, fitness_weights: tuple[float, float, float]) -> str:
    if problem_type == "pacman":
        return (
            f"{fitness_weights[0]}*avg_score + "
            f"{fitness_weights[1]}*max_score + "
            f"{fitness_weights[2]}*survival"
        )
    return (
        f"{fitness_weights[0]}*correctness + "
        f"{fitness_weights[1]}*(1/(num_operations+1)) + "
        f"{fitness_weights[2]}*(1/(exec_time_ms+1))"
    )


def build_description_to_code_prompt(problem_type: str, description: str) -> tuple[str, str]:
    system_prompt = SYSTEM_PROMPT_PACMAN if problem_type == "pacman" else SYSTEM_PROMPT_MATRIX
    user_prompt = (
        PACMAN_DESCRIPTION_USER_TEMPLATE.format(description=description)
        if problem_type == "pacman"
        else MATRIX_DESCRIPTION_USER_TEMPLATE.format(description=description)
    )
    return system_prompt, user_prompt


def build_single_shot_prompt(
    problem_type: str,
    current_code: str,
    fitness_description: str,
    fitness_breakdown: dict | None = None,
) -> str:
    return SINGLE_SHOT_USER_TEMPLATE.format(
        current_code=current_code,
        observed_metrics=_format_observed_metrics(problem_type, fitness_breakdown),
        fitness_description=fitness_description,
        strategy_focus=f"\n- Specific focus: {_baseline_focus(problem_type)}" if _baseline_focus(problem_type) else "",
    )


def build_mutation_prompt(
    problem_type: str,
    current_code: str,
    fitness: float,
    rag_examples: list[tuple[str, float]],
    performance_history: list[dict],
    gen: int,
    max_gen: int,
    temperature: float,
    fitness_description: str,
    fitness_breakdown: dict | None = None,
    strategy_focus: str = "",
) -> str:
    history_str = ""
    for h in performance_history[-5:]:
        history_str += (
            f"- Gen {h['gen']}: best={h['best']:.4f}, avg={h['avg']:.4f}, "
            f"gen_time_ms={h.get('gen_time_ms', 0):.2f}\n"
        )
    if not history_str:
        history_str = "- No history yet (first generation)\n"

    # Build attempt history from recent generations
    attempt_str = ""
    for h in performance_history[-3:]:
        summaries = h.get("attempt_summaries", [])
        if summaries:
            attempt_str += f"Gen {h['gen']}:\n"
            for s in summaries:
                attempt_str += f"{s}\n"
    if not attempt_str:
        attempt_str = "- No previous attempts yet.\n"

    examples_str = ""
    for i, (code, fit) in enumerate(rag_examples, 1):
        examples_str += f"\nExample {i} (fitness={fit:.4f}):\n```python\n{code}\n```\n"
    if not examples_str:
        examples_str = "- No similar examples available yet.\n"

    return MUTATION_USER_TEMPLATE.format(
        current_code=current_code,
        fitness=fitness,
        observed_metrics=_format_observed_metrics(problem_type, fitness_breakdown),
        performance_history=history_str,
        attempt_history=attempt_str,
        rag_examples=examples_str,
        gen=gen,
        max_gen=max_gen,
        temperature=temperature,
        fitness_description=fitness_description,
        strategy_focus=f"- Specific focus: {strategy_focus}" if strategy_focus else "",
    )


def build_crossover_prompt(
    code_a: str,
    fitness_a: float,
    code_b: str,
    fitness_b: float,
    fitness_description: str,
) -> str:
    return CROSSOVER_USER_TEMPLATE.format(
        code_a=code_a,
        fitness_a=fitness_a,
        code_b=code_b,
        fitness_b=fitness_b,
        fitness_description=fitness_description,
    )


def _baseline_focus(problem_type: str) -> str:
    if problem_type == "pacman":
        return "use a reliable successor-based heuristic that balances food progress with ghost safety"
    return "keep the implementation fully correct while reducing repeated indexing and Python overhead"


def _format_observed_metrics(problem_type: str, fitness_breakdown: dict | None) -> str:
    if not fitness_breakdown:
        return "- No detailed metrics available yet."

    if fitness_breakdown.get("invalid_candidate"):
        error = fitness_breakdown.get("error", "Unknown evaluation failure")
        return (
            f"- Candidate was invalid during evaluation.\n"
            f"- Error: {error}\n"
            f"- Repair syntax/runtime correctness before further optimization."
        )

    if problem_type == "pacman":
        lines = [
            f"- avg_score = {fitness_breakdown.get('avg_score', 0):.4f}",
            f"- max_score = {fitness_breakdown.get('max_score', 0):.4f}",
            f"- min_score = {fitness_breakdown.get('min_score', 0):.4f}",
            f"- win_rate = {fitness_breakdown.get('win_rate', 0):.4f}",
            f"- layouts_tested = {fitness_breakdown.get('layouts_tested', '') or 'n/a'}",
        ]
        failed_layouts = fitness_breakdown.get("failed_layouts")
        if failed_layouts:
            lines.append(f"- failed_layouts = {failed_layouts}")
        complexity = fitness_breakdown.get("estimated_time_complexity")
        if complexity:
            lines.append(f"- estimated_time_complexity = {complexity}")
        return "\n".join(lines)

    lines = [
        f"- correctness = {fitness_breakdown.get('correctness', 0):.4f}",
        f"- num_operations = {fitness_breakdown.get('num_operations', 0)}",
        f"- exec_time_ms = {fitness_breakdown.get('exec_time_ms', 0)}",
    ]
    estimated = fitness_breakdown.get("estimated_time_complexity")
    generalized = fitness_breakdown.get("generalized_time_complexity")
    if estimated:
        lines.append(f"- estimated_time_complexity = {estimated}")
    if generalized:
        lines.append(f"- generalized_time_complexity = {generalized}")
    runtime_failures = fitness_breakdown.get("runtime_failures")
    if runtime_failures:
        lines.append(f"- runtime_failures = {runtime_failures}")
    return "\n".join(lines)
