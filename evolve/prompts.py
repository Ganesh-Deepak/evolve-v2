SYSTEM_PROMPT_PACMAN = """You are an expert AI programmer specializing in game-playing agents.
You are improving a Pac-Man agent that plays in the UC Berkeley CS188 framework (Python 3 version).

The agent must implement a get_action(self, state) method that returns one of:
'North', 'South', 'East', 'West', 'Stop'

Available state methods (all snake_case):
- state.get_legal_actions(): list of legal actions (strings)
- state.get_pacman_position(): (x, y) tuple
- state.get_food(): Grid of booleans (use .as_list() to get list of (x,y))
- state.get_ghost_positions(): list of (x, y) tuples
- state.get_ghost_states(): list of GhostState objects (each has .scared_timer attribute)
- state.get_capsules(): list of power pellet positions (x, y)
- state.generate_pacman_successor(action): returns next GameState after taking action
- state.get_score(): current game score
- state.get_num_food(): number of remaining food dots
- state.get_walls(): Grid of booleans for wall positions

Rules:
- Output ONLY the complete get_action method body (the code inside the method)
- Do NOT include the class definition or method signature
- Do NOT import anything outside the standard library and math
- The function must return a valid action string from get_legal_actions()
- Optimize for: high game score, long survival time, fewer wasted steps"""

SYSTEM_PROMPT_MATRIX = """You are an expert algorithm designer optimizing matrix multiplication.
You are improving a function that multiplies two 3x3 matrices.

The function signature is:
def matrix_multiply(A, B):
    # A and B are 3x3 lists of lists (e.g., [[1,2,3],[4,5,6],[7,8,9]])
    # Must return a 3x3 list of lists with the correct product

Rules:
- Output ONLY the complete function body (code inside matrix_multiply)
- Do NOT include the function signature line
- Minimize the number of scalar multiplications and additions
- Minimize execution time (avoid unnecessary loops, redundant computations)
- The result MUST be mathematically correct for all inputs
- Standard approach uses 27 multiplications - try to find approaches with fewer
- You may use temporary variables and clever algebraic rearrangements"""

MUTATION_USER_TEMPLATE = """## Current Code (fitness={fitness:.4f}):
```python
{current_code}
```

## Performance History (last {history_len} generations):
{performance_history}

## Similar High-Performing Solutions (from database):
{rag_examples}

## Generation {gen}/{max_gen} | Temperature: {temperature:.2f}
## Fitness Formula: {fitness_description}

Analyze the current solution and improve it. Focus on areas where performance is weakest.
Output the complete improved code."""


def build_mutation_prompt(
    current_code: str,
    fitness: float,
    rag_examples: list[tuple[str, float]],
    performance_history: list[dict],
    gen: int,
    max_gen: int,
    temperature: float,
    fitness_description: str,
) -> str:
    history_str = ""
    for h in performance_history[-5:]:
        history_str += f"  Gen {h['gen']}: best={h['best']:.4f}, avg={h['avg']:.4f}\n"
    if not history_str:
        history_str = "  No history yet (first generation)\n"

    examples_str = ""
    for i, (code, fit) in enumerate(rag_examples, 1):
        examples_str += f"\n### Example {i} (fitness={fit:.4f}):\n```python\n{code}\n```\n"
    if not examples_str:
        examples_str = "  No similar examples available yet.\n"

    return MUTATION_USER_TEMPLATE.format(
        current_code=current_code,
        fitness=fitness,
        history_len=len(performance_history),
        performance_history=history_str,
        rag_examples=examples_str,
        gen=gen,
        max_gen=max_gen,
        temperature=temperature,
        fitness_description=fitness_description,
    )
