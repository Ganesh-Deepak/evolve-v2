# Technical Documentation

This document walks through the engineering behind Evolve. It covers how each piece works, why we made certain design decisions, and how the pieces fit together. Written assuming you might not have a background in evolutionary algorithms or LLMs.

---

## The big picture

The core idea is borrowed from biology: natural selection. In nature, organisms with beneficial mutations survive and reproduce, while harmful mutations get weeded out. We apply the same principle to code:

1. Take some initial code (the "parent")
2. Create slightly different versions ("children" or "candidates")
3. Test each candidate by actually running it
4. Keep the ones that perform best
5. Use those as parents for the next round
6. Repeat

After enough rounds (we call them "generations"), the code tends to improve. What makes our approach interesting is that we added an LLM (GPT-4o-mini) as one of the mutation strategies -- instead of making random changes, the LLM reads the code and tries to suggest meaningful improvements. We also use a vector database to feed the LLM examples of previously successful code (RAG).

---

## Architecture overview

```
                    +------------------+
                    |   Streamlit UI   |   <- user interacts here
                    |    (app.py)      |
                    +--------+---------+
                             |
                             v
                    +--------+---------+
                    | EvolutionController |  <- runs the loop
                    |  (controller.py)    |
                    +--------+---------+
                             |
                    +--------+--------+--------+
                    |                 |         |
                    v                 v         v
               +---------+    +-----------+  +---------+
               | Mutator |    | Evaluator |  | Selector|
               +---------+    +-----------+  +---------+
                    |              |              |
                    v              v              v
               +---------+    +----------+   +----------+
               |LLM      |    |Subprocess|   |ChromaDB  |
               |Client   |    |(Pac-Man) |   |VectorStore|
               +---------+    +----------+   +----------+
```

The layers:
- **UI layer** (`app.py`) -- Streamlit handles user input, displays charts, manages downloads
- **Controller** (`controller.py`) -- orchestrates one full evolution run
- **Core logic** (`candidate_generator.py`, `evaluator.py`, `selector.py`) -- mutation, testing, and selection
- **External services** (`llm_client.py`, `vector_store.py`) -- OpenAI API and ChromaDB
- **Shared data** (`models.py`) -- dataclasses that everything else uses

Each layer only talks to the one below it. The UI never calls the evaluator directly -- it goes through the controller.

---

## Key concepts

### Fitness function

A fitness function takes a candidate and returns a number. Higher is better.

For Pac-Man:
```
fitness = w1 * avg_score + w2 * max_score + w3 * survival_metric
```

The survival metric rewards games where Pac-Man achieves a positive score (i.e., wins). This uses all three configurable weights per the assignment spec.

For matrix multiplication:
```
fitness = w1 * correctness + w2 * (1 / (num_operations + 1)) + w3 * (1 / (exec_time_ms + 1))
```

The weights (w1, w2, w3) are configurable through the UI, so you can experiment with different priorities.

### The three mutation strategies

**No Evolution (Single-Shot LLM):** Makes a single LLM call to improve the code in the first generation, then returns unchanged code for subsequent generations. When no API key is available, it just returns the original code as-is. This is the control group -- it shows what a one-off LLM improvement looks like without iterative evolution. If evolution works, the other strategies should outperform this baseline over multiple generations.

**Random Mutation:** Makes random programmatic changes to the source code. The five operators are:
- Parameter perturbation -- finds numeric constants and tweaks them by up to 30%
- Operator substitution -- swaps operators (e.g., `+` to `-`, `>` to `>=`)
- Block swap -- switches two lines around (preserving indentation)
- Line duplication -- copies a random line
- Constant insertion -- adds a new variable assignment

Most random mutations make the code worse (just like in biology). But once in a while, one accidentally helps, and selection keeps it around.

**LLM-Guided:** Asks GPT-4o-mini to read the current code and write an improved version. This strategy uses several techniques inspired by AlphaEvolve to drive consistent improvement:

- **Crossover:** ~30% of each generation's candidates are produced by combining the best parts of two different parents, not just mutating one. The LLM receives both parents' code and fitness and is asked to merge their complementary strengths.
- **Mutation with attempt memory:** The remaining candidates are mutations of a single parent. The prompt includes a history of previously attempted mutations and whether they improved, regressed, or had no effect -- so the LLM avoids repeating failed approaches.
- **RAG (Retrieval-Augmented Generation):** Before each mutation, the system retrieves similar code from the vector database regardless of minimum fitness, ensuring diverse examples rather than only self-similar high-scoring code.
- **Slow temperature decay:** Temperature starts at 0.9 and decays to 0.4 over the run (formula: `max(0.4, 0.9 - 0.4 * progress)`), keeping exploration alive much longer than a fast decay schedule would.

### Vector database (ChromaDB)

We use ChromaDB with the `all-MiniLM-L6-v2` embedding model, which converts code text into 384-dimensional vectors. Similar code produces similar vectors. We use this for three things:

1. **RAG** -- find similar successful code to show the LLM as examples
2. **Fitness caching** -- if we've already evaluated identical code, skip the expensive evaluation
3. **Duplicate detection** -- during selection, reject candidates that are too similar to ones already picked (keeps the population diverse)

---

## Module-by-module breakdown

### models.py

Defines the data structures everything else uses.

`compute_code_hash(code)` creates a SHA-256 fingerprint of the code (first 16 hex chars). Two identical pieces of code always produce the same hash, which we use for caching and deduplication.

`RunConfig` holds all user settings -- problem type, mutation strategy, generation count, population size, fitness weights, etc. Created once from the UI and passed around.

`Candidate` represents a single piece of code being evaluated. Tracks the code, its hash, what generation and mutation produced it, and its fitness score.

`GenerationResult` bundles all the output from one generation -- the candidates, which ones were selected, the best one, and aggregate stats. This is what gets yielded to the UI for live updates.

### controller.py

The main orchestrator. The `run_evolution()` method is a Python generator -- it uses `yield` instead of `return`, which lets the Streamlit UI update the display after each generation without needing threads or async.

The loop:
1. Evaluate the initial code
2. For each generation:
   a. Generate candidates using the mutator
   b. Check ChromaDB cache for each -- skip evaluation if we've seen it before
   c. Evaluate uncached candidates
   d. Store results in ChromaDB
   e. Select top-K with diversity filtering
   f. Yield results to UI
   g. Continue until the configured generation count is reached, then finalize the run

### candidate_generator.py

Three classes that all inherit from `BaseMutator`:

- `NoEvolutionMutator` -- returns the input unchanged
- `RandomMutator` -- applies 1-2 random operators per candidate
- `LLMGuidedMutator` -- combines crossover and mutation. For each generation, ~30% of the population is produced by LLM-based crossover (combining two parents), and the rest by single-parent mutation with attempt-history awareness. Queries RAG for diverse examples and calls GPT-4o-mini.

The factory function `get_mutator(strategy)` creates the right one based on the config.

### evaluator.py

Runs candidate code and computes fitness. Has two evaluation paths.

**Safety first:** Before running anything, `is_safe_code()` scans for dangerous patterns with regex -- `import os`, `eval()`, `exec()`, `open()`, etc. Anything flagged gets fitness=0 immediately.

**Pac-Man evaluation:**
1. Wraps the candidate code in an `EvolvedAgent` class
2. Writes it to `pacman/evolvedAgents.py`
3. Runs the game as a subprocess: `python pacman.py -p EvolvedAgent -l mediumClassic -n 3 -q`
4. Parses scores from stdout
5. Computes weighted fitness from average and max scores

**Matrix evaluation:**
1. Wraps code in a `matrix_multiply(A, B)` function
2. Runs it in a sandboxed namespace (restricted builtins -- no file I/O, no imports)
3. Tests against 100 random matrix pairs (fixed seed for reproducibility)
4. Counts arithmetic operations by running the function with tracked scalar values
5. Measures execution time on the fixed test set
6. Computes fitness from correctness, operation count, and runtime

### selector.py

Decides which candidates survive each generation using fitness-distance balancing (inspired by MAP-Elites and AlphaEvolve's island model):

1. Pick the highest-fitness valid candidate first (guaranteed)
2. For remaining slots, score each candidate as `0.6 * normalized_fitness + 0.4 * diversity`, where diversity is `1.0 - min_similarity` to already-selected candidates (Jaccard on code lines)
3. Reject candidates with > 95% similarity to any already-selected candidate
4. Stop after picking top-K
5. Elitism: if the all-time best candidate isn't in the selected set, force-add it so we never lose our best result
6. Update the global best if anything in this generation is better

The fitness-distance balance is critical. Pure fitness ranking causes convergence within 2-3 generations -- all parents become near-identical and the LLM keeps producing the same mutations. By reserving 40% of the selection score for diversity, we ensure the parent pool contains genuinely different approaches that can be combined via crossover.

### llm_client.py

Thin wrapper around the OpenAI SDK. Two main methods:

- `generate()` -- makes a chat completion call with retry logic (exponential backoff on rate limits and connection errors, up to 3 attempts)
- `generate_code()` -- calls generate() and then strips out markdown formatting to get just the Python code

The code extraction handles the common LLM pattern of wrapping code in ` ```python ... ``` ` blocks.

### vector_store.py

Manages the ChromaDB connection. The collection stores each candidate's code as a document (auto-embedded), with metadata for fitness, generation, and mutation type.

Key methods:
- `add_candidate()` -- upserts: if the same code hash exists and new fitness is higher, update it
- `get_similar()` -- cosine similarity search, filtered by minimum fitness threshold
- `is_duplicate()` -- checks if code is > 98% similar to something already stored
- `get_cached_fitness()` -- looks up a score by code hash
- `seed_templates()` -- pre-populates the DB with starter code from `templates/`

### prompts.py

The prompt templates for GPT-4o-mini. There's a system prompt for each problem type (explaining the available APIs, output rules, what to optimize) and several user prompt templates:

**Mutation prompt** -- filled in with:
- Current code and fitness score
- Last 5 generations' performance history (fitness trends)
- Previously attempted mutations and their outcomes (improved/regressed/unchanged) from the last 3 generations, so the LLM avoids repeating failed approaches
- RAG examples from the vector DB (retrieved without a min-fitness filter for diversity)
- Current generation number and temperature
- The fitness formula

**Crossover prompt** -- given two parent candidates with their code and fitness scores, asks the LLM to merge their complementary strengths into a single improved child.

**Single-shot prompt** -- used by the no-evolution baseline for a one-off improvement.

---

## Pac-Man integration

We use the UC Berkeley CS188 Pac-Man project (Python 3 port). The framework provides the game engine, display, ghost AI, and layouts -- we just need to supply an agent class with a `get_action(state)` method.

Important detail: this is the Python 3 version, so all methods use **snake_case** (`get_legal_actions`, not `getLegalActions`). This tripped us up initially and is worth mentioning because a lot of online resources reference the older Python 2 version.

The agent file must end in `gents.py` (like `evolvedAgents.py`) because the framework's module loader only looks for that suffix.

Each candidate is tested on two layouts (mediumClassic, smallClassic) with 5 games each (10 games total). Using multiple layouts and more games per layout reduces stochastic noise in the fitness signal -- Pac-Man scores vary significantly between runs, so a larger sample gives more reliable fitness comparisons.

---

## Safety and sandboxing

Since we're generating and running arbitrary code, safety matters. We have four layers:

1. **Code scanning** -- regex check for dangerous imports and functions before execution
2. **Sandboxed execution** -- for matrix eval, we run in a restricted namespace with only safe builtins
3. **Subprocess isolation** -- Pac-Man eval runs in a separate process, so crashes don't affect the main app
4. **Timeouts** -- every evaluation has a time limit (90s for Pac-Man, 30s for matrix). Infinite loops get killed.

---

## Design decisions worth noting

**Why generators for the evolution loop?** Streamlit reruns the entire script on every interaction. By making `run_evolution()` a generator that yields after each generation, we get live UI updates without needing threads or async -- the UI just iterates over the generator in a for loop.

**Why ChromaDB specifically?** It's the simplest vector DB to set up -- runs embedded in the process, no external server needed, and persists to disk automatically. For a project of this scale, it's more than sufficient.

**Why GPT-4o-mini instead of GPT-4?** Cost. A full evolution run with GPT-4 would cost $1-2, while GPT-4o-mini does the same thing for pennies. For code mutations on relatively small functions, the quality difference is negligible.

**Why fixed random seeds for matrix testing?** Reproducibility. Every candidate gets tested against the exact same 100 matrix pairs, so fitness comparisons are fair.

---

## AlphaEvolve-inspired improvements

Several techniques from Google DeepMind's AlphaEvolve (2025) were adapted for this project to improve convergence:

| Technique | AlphaEvolve | Our Implementation |
|-----------|------------|-------------------|
| Crossover | Multi-parent diff-based recombination | LLM-based two-parent crossover (~30% of each generation) |
| Diversity | Island model with separate populations | Fitness-distance balancing in selection (60/40 fitness/diversity) |
| Attempt memory | Full evolutionary tree with diffs | Last 3 generations' per-candidate mutation outcomes in the prompt |
| Temperature | Adaptive per-island | Slow decay: 0.9 -> 0.4 over the run |
| Eval reliability | Large-scale distributed evaluation | 5 games per layout (up from 3) for Pac-Man |
| RAG retrieval | N/A (uses evolutionary tree) | Removed min-fitness filter to surface diverse examples |

These changes address the main failure modes observed in the original design:
1. **Convergence stall** -- without diversity pressure, all parents become near-identical within 2-3 generations. Fitness-distance balancing keeps the parent pool diverse.
2. **Repeated mutations** -- without attempt memory, the LLM proposes similar changes every generation. The attempt history tells it what already failed.
3. **No recombination** -- mutation alone can only make local improvements. Crossover lets the system combine independently-discovered strategies from different parents.
4. **Noisy signal** -- 3 Pac-Man games per layout made fitness comparisons unreliable. 5 games reduces variance enough to distinguish genuinely better code.
