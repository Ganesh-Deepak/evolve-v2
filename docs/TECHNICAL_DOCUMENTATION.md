# Evolve -- Technical Documentation

This document explains the complete engineering design of the Evolve system. It is written for someone who may be new to evolutionary algorithms, LLMs, or software architecture. Every component is explained from the ground up.

---

## Table of Contents

1. [What This System Does (Big Picture)](#1-what-this-system-does)
2. [Core Concepts Explained](#2-core-concepts-explained)
3. [System Architecture](#3-system-architecture)
4. [Data Flow: How a Single Evolution Run Works](#4-data-flow)
5. [Module-by-Module Technical Reference](#5-module-reference)
6. [The Pac-Man Integration](#6-pacman-integration)
7. [The Matrix Multiplication Problem](#7-matrix-multiplication)
8. [The Vector Database (ChromaDB)](#8-vector-database)
9. [LLM Integration and Prompt Engineering](#9-llm-integration)
10. [The Streamlit Web Interface](#10-streamlit-ui)
11. [Safety and Sandboxing](#11-safety)
12. [How to Extend the System](#12-extending)

---

## 1. What This System Does

Imagine you have a piece of code that plays a Pac-Man game, but it's not very good. Instead of manually rewriting it over and over, this system automates the improvement process using ideas from biological evolution:

1. **Start** with your initial code (the "parent")
2. **Mutate** it to create several slightly different versions (the "children" or "candidates")
3. **Test** each candidate by actually running it (play Pac-Man games, check matrix multiplication results)
4. **Score** each candidate with a fitness function (how well did it perform?)
5. **Select** the best-performing candidates
6. **Repeat** from step 2 using the winners as the new parents

This is called an **evolutionary algorithm**. After many generations, the code tends to get better and better, just like organisms evolve over time through natural selection.

What makes this system special is that it can use an AI language model (GPT-4o-mini) to make *intelligent* mutations rather than random ones. The AI reads the current code, understands what it does, and suggests specific improvements.

---

## 2. Core Concepts Explained

### 2.1 Evolutionary Algorithms

An evolutionary algorithm mimics natural selection:

- **Population**: A group of candidate solutions (in our case, different versions of code)
- **Fitness**: A numerical score measuring how good each candidate is
- **Selection**: Picking the best candidates to "survive" to the next generation
- **Mutation**: Making small changes to create new candidates from surviving ones
- **Generation**: One full cycle of mutate -> evaluate -> select
- **Elitism**: Always keeping the single best solution ever found, so you never lose progress

### 2.2 Fitness Function

A fitness function takes a candidate solution and returns a number. Higher = better. For Pac-Man:

```
Fitness = w1 * average_game_score + w2 * maximum_game_score
```

Where `w1` and `w2` are configurable weights that let you control what matters more. For matrix multiplication:

```
Fitness = w1 * correctness + w2 * (1 / (num_operations + 1))
```

This rewards both correctness (getting the right answer) and efficiency (using fewer arithmetic operations).

### 2.3 Mutation Strategies

The system supports three ways to create new candidates:

1. **No Evolution**: Don't change anything. This is the control group for comparison.
2. **Random Mutation**: Make random programmatic changes (swap operators, tweak numbers, rearrange lines). No AI involved.
3. **LLM-Guided Mutation**: Ask GPT-4o-mini to read the code and suggest intelligent improvements. Uses retrieval-augmented generation (RAG) to show the AI examples of good solutions from previous runs.

### 2.4 Vector Database and Embeddings

A **vector database** stores data as high-dimensional numerical vectors. When you "embed" code into a vector, similar code produces similar vectors. This lets us:

- Find code that is similar to a given piece of code (for RAG -- giving the AI examples)
- Detect duplicate solutions (so we don't waste time re-evaluating the same code)
- Cache fitness scores (if we've seen this exact code before, skip evaluation)

We use **ChromaDB** with the **all-MiniLM-L6-v2** embedding model, which converts text into 384-dimensional vectors.

### 2.5 Retrieval-Augmented Generation (RAG)

RAG is a technique where, before asking the AI to generate something, you first *retrieve* relevant examples from a database and include them in the prompt. In our case:

1. Take the current code
2. Search the vector database for similar code that scored well
3. Include those examples in the prompt to GPT-4o-mini
4. The AI can learn from these examples and make better suggestions

---

## 3. System Architecture

### 3.1 High-Level Architecture Diagram

```
+------------------+
|   Streamlit UI   |  <-- User interacts here (browser)
|    (app.py)      |
+--------+---------+
         |
         v
+--------+---------+
| EvolutionController |  <-- Orchestrates everything
|  (controller.py)    |
+--------+---------+
         |
    +----+----+----+
    |         |    |
    v         v    v
+-------+ +------+ +--------+
|Mutator| |Eval  | |Selector|
|       | |      | |        |
+---+---+ +--+---+ +---+----+
    |        |          |
    v        v          v
+-------+ +------+ +--------+
|LLM    | |Pac-Man| |Vector  |
|Client | |subprocess| |Store|
+-------+ +--------+ +------+
```

### 3.2 How Components Connect

The system follows a layered architecture:

- **Presentation Layer**: `app.py` (Streamlit) -- handles all user interaction
- **Orchestration Layer**: `controller.py` -- manages the evolution loop
- **Logic Layer**: `candidate_generator.py`, `evaluator.py`, `selector.py` -- core algorithms
- **Integration Layer**: `llm_client.py`, `vector_store.py` -- external services (OpenAI, ChromaDB)
- **Data Layer**: `models.py` -- data structures shared across all layers

Each layer only talks to the layer directly below it. The UI never directly calls the evaluator; it goes through the controller.

---

## 4. Data Flow: How a Single Evolution Run Works

Here's exactly what happens when you click "Start Evolution" with Random Mutation, 3 generations, and population size 3:

### Step 1: Configuration

The UI collects all your settings into a `RunConfig` object:

```python
config = RunConfig(
    problem_type="matrix",
    initial_code="result = [[0]*3 ...]",
    mutation_strategy="random",
    num_generations=3,
    population_size=3,
    top_k=2,
    fitness_weights=(0.5, 0.3, 0.2),
)
```

### Step 2: Initialization

The controller creates all required components:

```
EvolutionController.__init__(config)
  -> Creates VectorStore (ChromaDB connection)
  -> Creates RandomMutator (since strategy="random")
  -> Creates FitnessEvaluator (matrix mode)
  -> Creates Selector (top_k=2)
  -> Seeds vector store with template code from templates/ directory
```

### Step 3: Evaluate Initial Code

Before any evolution, the system scores the starting code:

```
Initial code -> FitnessEvaluator.evaluate()
  -> _evaluate_matrix() executes the code against 100 random test cases
  -> Returns fitness=0.575 (100% correct, 3 operations detected)
  -> Stores result in ChromaDB for future reference
```

### Step 4: Generation 1

```
RandomMutator.generate(parents=[initial], generation=1)
  -> For each of 3 candidates:
     -> Pick random parent (we only have 1)
     -> Apply 1-2 random operators:
        - parameter_perturbation: change "3" to "4"
        - operator_substitution: change "+" to "-"
        - block_swap: swap two lines
     -> Return 3 mutated Candidate objects

FitnessEvaluator.evaluate(candidate_1)
  -> _evaluate_matrix() runs the modified code
  -> Maybe correctness dropped to 0.80 because the mutation broke something
  -> fitness = 0.5 * 0.80 + 0.3 * (1/4) = 0.475

(repeat for candidates 2 and 3)

Selector.select([candidate_1, candidate_2, candidate_3])
  -> Sort by fitness: [0.575, 0.475, 0.075]
  -> Pick top 2 (removing duplicates if any)
  -> Check elitism: is global_best in the selected? Yes
  -> Return [candidate_A, candidate_B]

yield GenerationResult to UI
  -> UI updates fitness chart, operation log, generation table
```

### Step 5: Generations 2 and 3

Same process, but now parents are the 2 selected candidates from the previous generation. The system tracks whether fitness is improving. If no improvement for `early_stop_patience` generations, it stops early.

### Step 6: Complete

The generator finishes. The UI shows:
- Final best solution code
- Complete fitness progression chart
- Download buttons for CSV and PNG

---

## 5. Module-by-Module Technical Reference

### 5.1 models.py -- Data Structures

This file defines the data containers that all other modules use. Think of these as the "vocabulary" of the system.

**`compute_code_hash(code: str) -> str`**

Creates a unique fingerprint for a piece of code. Two pieces of code that are identical (ignoring leading whitespace) will always produce the same hash. Uses SHA-256 cryptographic hash, truncated to 16 characters.

```python
compute_code_hash("return 42")  # -> "e115c9b995f74018"
```

**Why this matters**: We use this hash to detect if we've seen the exact same code before, so we can skip re-evaluation and reuse cached fitness scores.

**`RunConfig`** -- Holds all user-configurable settings for a run. Created once from the UI inputs and passed to the controller.

**`Candidate`** -- Represents a single piece of code being evaluated. Contains the code itself, its hash, which generation it belongs to, what mutation created it, and its fitness score (initially None, filled in after evaluation).

**`GenerationResult`** -- The output of one generation. Contains all candidates that were created, which ones were selected, the best candidate, and aggregate statistics. This is what gets yielded to the UI after each generation.

---

### 5.2 controller.py -- The Evolution Loop

This is the "brain" that runs the entire evolution. Its key method is `run_evolution()`, which is a Python **generator**.

**What is a generator?** A generator is a function that uses `yield` instead of `return`. Each time you call `next()` on it (or iterate over it with `for`), it runs until it hits `yield`, returns that value, then *pauses*. Next time you call it, it resumes from where it paused.

```python
def run_evolution(self) -> Generator[GenerationResult, None, None]:
    # ... setup ...
    for gen in range(1, num_generations + 1):
        # ... generate, evaluate, select ...
        yield gen_result  # Pause here, UI updates, then resume
```

This design lets the Streamlit UI update the display after each generation without needing threads or async code. The loop runs synchronously -- one generation at a time, with the UI refreshing between each.

**Early stopping**: If the best fitness hasn't improved for `early_stop_patience` consecutive generations, the loop breaks early. This saves time when evolution has converged.

**Fitness caching**: Before evaluating a candidate, the controller checks ChromaDB for a cached fitness score with the same code hash. If found, it skips evaluation entirely.

---

### 5.3 candidate_generator.py -- Mutation Strategies

This module implements three classes, all inheriting from `BaseMutator`. A factory function `get_mutator()` creates the right one based on the strategy name.

**`NoEvolutionMutator`**: Simply returns the original code unchanged. Every generation produces the same single candidate. This serves as the baseline -- if evolution is working, the other strategies should outperform this one.

**`RandomMutator`**: Applies random programmatic transformations to the code text. Each candidate gets 1-2 randomly chosen operators:

| Operator | What It Does | Example |
|----------|-------------|---------|
| `parameter_perturbation` | Finds numeric constants via regex and changes them by up to 30% | `5` becomes `6.2` |
| `operator_substitution` | Swaps mathematical or comparison operators | `+` becomes `-`, `>` becomes `>=` |
| `block_swap` | Picks two non-empty lines and swaps them (preserving indentation) | Lines 3 and 7 switch places |
| `line_duplication` | Copies a random line and inserts it right after | Line 5 appears twice |
| `constant_insertion` | Adds a new variable assignment at a random position | `threshold = 3.47` is inserted |

Most random mutations will make the code *worse* (just like most biological mutations are harmful). But occasionally one will accidentally improve something. Over many generations, selection pressure keeps the improvements and discards the rest.

**`LLMGuidedMutator`**: Uses GPT-4o-mini to make intelligent changes. The process:

1. Compute adaptive temperature: starts at 0.8 (creative/exploratory), decreases to 0.3 (focused/conservative) over generations
2. Query ChromaDB for the top 3 most similar high-performing candidates (RAG)
3. Build a detailed prompt with: current code, fitness score, performance history, and RAG examples
4. Call GPT-4o-mini via the OpenAI API
5. Extract the code from the response (stripping markdown formatting)

---

### 5.4 evaluator.py -- Fitness Scoring

This module runs candidate code and computes fitness scores. It has two evaluation paths.

**Safety check**: Before running any code, `is_safe_code()` scans for dangerous patterns using regex:
- `import os`, `import sys`, `import subprocess` (could access the filesystem)
- `__import__`, `exec()`, `eval()`, `open()`, `compile()` (could run arbitrary code)

Any candidate containing these patterns gets fitness=0 immediately.

**Pac-Man evaluation** (`_evaluate_pacman`):

1. Wraps the candidate code in an `EvolvedAgent` class definition
2. Writes this to `pacman/evolvedAgents.py`
3. Runs the Pac-Man game as a subprocess: `python pacman.py -p EvolvedAgent -l mediumClassic -n 3 -q`
4. Parses the printed scores from stdout
5. Computes fitness from average and maximum scores

The `-q` flag runs in quiet mode (no graphical display), making it much faster. The `-n 3` runs 3 games and averages the results to reduce variance.

**Matrix evaluation** (`_evaluate_matrix`):

1. Wraps the candidate code in a `matrix_multiply(A, B)` function
2. Executes it in a restricted Python namespace (limited builtins -- no file I/O, no imports)
3. Tests against 100 randomly generated 3x3 matrix pairs (with a fixed random seed for reproducibility)
4. Counts arithmetic operations by analyzing the code's Abstract Syntax Tree (AST)
5. Computes fitness from correctness percentage and operation count

---

### 5.5 selector.py -- Survival of the Fittest

After all candidates in a generation are evaluated, the selector decides which ones survive.

**Algorithm:**

1. Sort all candidates by fitness (highest first)
2. Iterate through the sorted list, adding each to the "selected" set if it's sufficiently different from those already selected (Jaccard similarity < 0.95 on the set of code lines)
3. Stop when we have `top_k` selected candidates
4. **Elitism check**: If the global best candidate (best across ALL generations) isn't in the selected set, add it. This ensures the best solution is never lost.
5. Update the global best if any candidate in this generation is better

**Why diversity matters**: Without the similarity check, evolution could get "stuck" with several nearly-identical candidates. By rejecting duplicates, we maintain diversity in the population, which helps explore more of the solution space.

---

### 5.6 llm_client.py -- OpenAI API Wrapper

A thin wrapper around the OpenAI Python SDK.

**`generate()`**: Makes a chat completion API call with a system prompt and user prompt. Includes retry logic: if the API returns a rate limit error or connection error, it waits and retries up to 3 times with exponential backoff (1s, 2s, 4s).

**`generate_code()`**: Calls `generate()` and then extracts just the Python code from the response. LLMs often wrap code in markdown blocks like:

````
Here's the improved code:
```python
# actual code here
```
````

The `_extract_code()` static method uses regex to find the last code block in the response and return only its contents.

---

### 5.7 vector_store.py -- ChromaDB Vector Database

ChromaDB is an open-source vector database. It stores text along with numerical "embeddings" (vectors) and lets you search by similarity.

**How embeddings work**: The `all-MiniLM-L6-v2` model converts any text into a 384-dimensional vector. Similar text produces vectors that point in similar directions. "Cosine similarity" measures this -- 1.0 means identical, 0.0 means completely unrelated.

**Collection schema**: One collection called `evolve_candidates`:
- **document**: The code text (ChromaDB automatically embeds this)
- **id**: The code hash (unique identifier)
- **metadata**: `{fitness, generation, mutation_type}`

**Key operations:**

| Method | Purpose | When Used |
|--------|---------|-----------|
| `add_candidate()` | Store a candidate and its fitness | After evaluation |
| `get_similar()` | Find similar high-performing code | Before LLM-guided mutation (RAG) |
| `is_duplicate()` | Check if nearly-identical code exists | During selection (diversity) |
| `get_cached_fitness()` | Look up a previously computed score | Before evaluation (optimization) |
| `seed_templates()` | Pre-populate with starter code | At initialization |
| `clear()` | Wipe all data for a fresh run | At the start of each evolution |

---

### 5.8 prompts.py -- LLM Prompt Engineering

This module contains the exact text prompts sent to GPT-4o-mini. Prompt engineering is critical for getting good results from the LLM.

**System prompt** (SYSTEM_PROMPT_PACMAN): Sets the context and rules. Tells the AI it's an expert programmer improving a Pac-Man agent. Lists all available API methods (get_legal_actions, get_food, etc.) so the AI knows what tools it has. Specifies output format rules (just the code, no class definition).

**User prompt** (built by `build_mutation_prompt()`): Contains:
- The current code with its fitness score
- Performance history (last 5 generations' best and average scores)
- RAG examples: 1-3 similar high-performing solutions from the vector database
- Current generation number and temperature
- The fitness formula (so the AI knows what to optimize)

The prompt asks the AI to analyze weaknesses and provide complete improved code.

---

## 6. Pac-Man Integration

### 6.1 The Framework

We use the UC Berkeley CS188 Pac-Man project (Python 3 version from github.com/aig-upf/pacman-projects). This is a well-known educational framework where you implement AI agents that play Pac-Man.

### 6.2 Agent Interface

Every agent must be a class with a `get_action(self, state)` method:

```python
class EvolvedAgent(Agent):
    def get_action(self, state):
        # state is a GameState object with methods like:
        # state.get_legal_actions() -> ['North', 'South', 'East', 'West']
        # state.get_pacman_position() -> (x, y)
        # state.get_food() -> Grid object (.as_list() for coordinates)
        # state.get_ghost_positions() -> [(x1,y1), (x2,y2)]
        # state.generate_pacman_successor(action) -> next GameState
        return 'North'  # Must return a legal action
```

**Important**: This Python 3 version uses **snake_case** method names (e.g., `get_legal_actions` not `getLegalActions`).

### 6.3 How Evaluation Works

1. The evaluator writes the candidate code into `pacman/evolvedAgents.py`
2. The file name ends in `gents.py` because the framework's `load_agent()` function only looks at files matching `*gents.py`
3. A subprocess runs: `python pacman.py -p EvolvedAgent -l mediumClassic -n 3 -q`
4. The framework imports `evolvedAgents.py`, finds `EvolvedAgent`, and runs 3 games
5. Game scores are printed to stdout and parsed by our evaluator
6. Each candidate is tested on 2 layouts (mediumClassic, smallClassic) with 3 games each = 6 total games

### 6.4 Timeout and Error Handling

- Each evaluation has a 90-second timeout (30s per candidate x 3 games)
- If the code has a syntax error, it gets fitness=0
- If the code runs forever (infinite loop), the subprocess is killed after timeout, fitness=0
- All errors are logged in the operation log so you can see what went wrong

---

## 7. Matrix Multiplication Problem

### 7.1 Goal

Standard 3x3 matrix multiplication uses 27 multiplications and 18 additions. The goal is to discover algorithms that use fewer operations (like Strassen's algorithm uses only 7 multiplications for 2x2 matrices).

### 7.2 How Evaluation Works

1. The candidate code is wrapped in `def matrix_multiply(A, B):` and executed with `exec()` in a sandboxed namespace
2. The sandbox only provides safe builtins (range, len, sum, etc.) -- no file I/O or imports
3. 100 random 3x3 matrix pairs are generated (with fixed seed for reproducibility)
4. Each pair is tested: does the candidate produce the correct result?
5. Operations are counted by parsing the code's AST (Abstract Syntax Tree) and counting `*`, `+`, and `-` nodes

### 7.3 Fitness Formula

```
Fitness = w1 * (correct/100) + w2 * (1 / (num_operations + 1))
```

A perfectly correct naive solution scores: `0.5 * 1.0 + 0.3 * (1/4) = 0.575`

---

## 8. Vector Database (ChromaDB)

### 8.1 Why a Vector Database?

Three reasons:

1. **Retrieval-Augmented Generation (RAG)**: Before the LLM mutates code, we retrieve similar high-performing code from the database to include as examples in the prompt. This gives the LLM "inspiration" from what has worked before.

2. **Fitness Caching**: If evolution produces the exact same code twice (same hash), we skip evaluation and return the cached score. This saves significant time, especially with Pac-Man evaluations that take several seconds each.

3. **Duplicate Detection**: During selection, we check if a new candidate is too similar to one already selected (cosine similarity > 0.95). This maintains population diversity.

### 8.2 How Embedding Works

The `all-MiniLM-L6-v2` model converts code text into a 384-dimensional vector. For example:

```
"for i in range(3):" -> [0.12, -0.34, 0.56, ..., 0.78]  (384 numbers)
```

Similar code produces similar vectors. ChromaDB stores these vectors and can quickly find the nearest neighbors using approximate nearest neighbor search (HNSW algorithm).

### 8.3 Persistence

ChromaDB stores data on disk in `data/chromadb/`. This means data survives between Streamlit page refreshes. At the start of each new evolution run, we call `clear()` to start fresh.

---

## 9. LLM Integration and Prompt Engineering

### 9.1 Adaptive Temperature

Temperature controls how "creative" vs "focused" the LLM is:

```python
temperature = max(0.3, 0.8 - 0.5 * (generation / max_generations))
```

| Generation | Temperature | Behavior |
|-----------|------------|----------|
| 1 (early) | 0.75 | Exploratory -- tries creative, diverse approaches |
| 5 (mid) | 0.55 | Balanced -- mix of creativity and focus |
| 10 (late) | 0.30 | Exploitative -- makes careful, targeted improvements |

This mimics a common strategy in optimization: explore broadly at first, then exploit the best areas found.

### 9.2 RAG Pipeline

```
Current code -> embed with all-MiniLM-L6-v2
                    |
                    v
            ChromaDB.query(top 3 similar with high fitness)
                    |
                    v
            Include in prompt as "Similar High-Performing Solutions"
                    |
                    v
            GPT-4o-mini generates improved code
```

### 9.3 Code Extraction

LLM responses often contain explanatory text alongside the code. The `_extract_code()` method handles this:

1. First tries to find a markdown code block (```python ... ```)
2. If found, returns the contents of the *last* code block (the final/complete version)
3. If no code block found, returns the raw text (fallback for simple responses)

---

## 10. The Streamlit Web Interface

### 10.1 Why Streamlit?

Streamlit is a Python library that turns Python scripts into web applications. You write normal Python code, and Streamlit automatically creates the web UI. No HTML, CSS, or JavaScript needed. This makes it ideal for a course project where the focus is on algorithms, not web development.

### 10.2 How Streamlit Works

Streamlit re-runs your entire Python script from top to bottom every time the user interacts with a widget. This seems inefficient, but it works because:

- `st.empty()` creates placeholder elements that can be updated in-place
- `st.session_state` persists data between reruns
- Widgets like `st.slider()` return their current value on each rerun

### 10.3 Live Updates During Evolution

The evolution loop uses Python generators to provide live updates:

```python
for gen_result in controller.run_evolution():
    # This code runs once per generation
    chart_placeholder.plotly_chart(updated_chart)  # Replace chart in-place
    gen_counter.markdown("Generation 3 / 10")      # Update counter
    best_score.metric("Best Fitness", "0.575")      # Update score
```

Each `yield` in `run_evolution()` causes the loop body to execute, updating all UI elements. The user sees the chart grow generation by generation.

### 10.4 Comparison Experiment Mode

When the comparison checkbox is selected:

1. Three sequential evolution runs execute (one per strategy)
2. Each gets its own ChromaDB instance (so they don't share cached data)
3. Results are collected into a single DataFrame
4. A Plotly chart overlays all three fitness curves
5. Export buttons provide CSV and PNG downloads

---

## 11. Safety and Sandboxing

### 11.1 Why Sandboxing Matters

The system generates and executes code. If a mutation produces malicious code (e.g., `import os; os.remove("important_file")`), it could damage your system. We prevent this at multiple levels:

### 11.2 Safety Layers

**Layer 1 -- Code Scanning** (`is_safe_code()`): Before execution, regex patterns check for dangerous imports and functions. Code containing `import os`, `eval()`, `exec()`, `open()`, etc. is immediately rejected with fitness=0.

**Layer 2 -- Sandboxed Execution** (Matrix): The `exec()` call uses a restricted namespace with only safe builtins (range, len, sum, etc.). Even if code somehow bypasses the scanner, it can't access the filesystem.

**Layer 3 -- Subprocess Isolation** (Pac-Man): Each Pac-Man evaluation runs in a completely separate Python process. If it crashes or hangs, only that subprocess is affected.

**Layer 4 -- Timeouts**: Every evaluation has a timeout. Infinite loops are killed after 30-90 seconds.

---

## 12. How to Extend the System

### 12.1 Adding a New Problem Type

1. Add a new `_evaluate_<problem>()` method in `evaluator.py`
2. Add a system prompt in `prompts.py`
3. Add default code and description in `app.py`
4. Add template files in `templates/`
5. Update the `problem_type` checks in `evaluator.py` and `candidate_generator.py`

### 12.2 Adding a New Mutation Operator

In `candidate_generator.py`:

1. Add the operator name to `RandomMutator.OPERATORS` list
2. Create a method `_apply_<operator_name>(self, code: str) -> tuple[str, str]`
3. It should return `(modified_code, description_string)`

### 12.3 Changing the LLM Provider

In `llm_client.py`, replace the OpenAI client with another provider's SDK. The interface is simple: takes a system prompt and user prompt, returns text. The rest of the system doesn't need to change.

### 12.4 Adding New Fitness Metrics

In `evaluator.py`, modify the `_evaluate_*` methods to compute additional metrics. Add them to the `breakdown` dict. Update the fitness formula to include the new metric with a weight from `config.fitness_weights`.
