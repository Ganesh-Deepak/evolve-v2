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
fitness = w1 * average_game_score + w2 * max_game_score
```

For matrix multiplication:
```
fitness = w1 * correctness + w2 * (1 / (num_operations + 1))
```

The weights (w1, w2, w3) are configurable through the UI, so you can experiment with different priorities.

### The three mutation strategies

**No Evolution:** Does nothing. Returns the original code unchanged every generation. This is the control group -- if evolution works, the other strategies should beat this flat line.

**Random Mutation:** Makes random programmatic changes to the source code. The five operators are:
- Parameter perturbation -- finds numeric constants and tweaks them by up to 30%
- Operator substitution -- swaps operators (e.g., `+` to `-`, `>` to `>=`)
- Block swap -- switches two lines around (preserving indentation)
- Line duplication -- copies a random line
- Constant insertion -- adds a new variable assignment

Most random mutations make the code worse (just like in biology). But once in a while, one accidentally helps, and selection keeps it around.

**LLM-Guided:** Asks GPT-4o-mini to read the current code and write an improved version. Before calling the LLM, the system retrieves similar high-performing code from the vector database and includes it in the prompt (RAG). The temperature starts high (0.8 = more creative) and decreases over generations (0.3 = more focused), mimicking the explore-then-exploit pattern common in optimization.

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
   g. Check early stopping (if no improvement for N generations, stop)

### candidate_generator.py

Three classes that all inherit from `BaseMutator`:

- `NoEvolutionMutator` -- returns the input unchanged
- `RandomMutator` -- applies 1-2 random operators per candidate
- `LLMGuidedMutator` -- queries RAG, builds prompt, calls GPT-4o-mini, extracts code

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
4. Counts arithmetic operations via AST analysis
5. Computes fitness from correctness and operation count

### selector.py

Decides which candidates survive each generation.

1. Sort by fitness (highest first)
2. Walk through the sorted list, adding each candidate if it's different enough from those already picked (Jaccard similarity < 0.95 on code lines)
3. Stop after picking top-K
4. Elitism: if the all-time best candidate isn't in the selected set, force-add it so we never lose our best result
5. Update the global best if anything in this generation is better

The diversity check matters a lot. Without it, you can end up with K nearly-identical candidates, and evolution stalls. Forcing diversity keeps the population exploring different approaches.

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

The prompt templates for GPT-4o-mini. There's a system prompt for each problem type (explaining the available APIs, output rules, what to optimize) and a user prompt template that gets filled in with:
- Current code and fitness score
- Last 5 generations' performance history
- RAG examples from the vector DB
- Current generation number and temperature
- The fitness formula

---

## Pac-Man integration

We use the UC Berkeley CS188 Pac-Man project (Python 3 port). The framework provides the game engine, display, ghost AI, and layouts -- we just need to supply an agent class with a `get_action(state)` method.

Important detail: this is the Python 3 version, so all methods use **snake_case** (`get_legal_actions`, not `getLegalActions`). This tripped us up initially and is worth mentioning because a lot of online resources reference the older Python 2 version.

The agent file must end in `gents.py` (like `evolvedAgents.py`) because the framework's module loader only looks for that suffix.

Each candidate is tested on two layouts (mediumClassic, smallClassic) with 3 games each. Using multiple layouts reduces the chance of a solution that's accidentally good on one specific map.

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
