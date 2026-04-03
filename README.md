# Evolve -- Evolutionary Code Improvement Using LLMs

**CS5381 - Analysis of Algorithms | Term Project | Spring 2026**

## Project Description

This project implements a simplified evolutionary agent system inspired by AlphaEvolve [1]. The system takes an initial piece of code (like a Pac-Man game agent), generates mutated candidate versions using evolutionary strategies, evaluates each candidate with a fitness function, selects the best performers, and iterates. Over multiple generations, the code improves measurably.

What makes this different from plain evolutionary algorithms is the integration of an LLM (GPT-4o-mini) for intelligent code mutations, plus a vector database (ChromaDB) that caches high-performing candidates and provides retrieval-augmented generation (RAG) to speed up convergence. The whole thing runs through a Streamlit web interface where you can watch evolution happen in real time.

We tested on two problems:
- **Pac-Man Agent** -- evolving an agent that plays Pac-Man using the UC Berkeley CS188 framework
- **3x3 Matrix Multiplication** (Bonus) -- finding algorithms that use fewer arithmetic operations than the standard 27-multiplication approach

## System Architecture

```
                    +------------------+
                    |   Streamlit UI   |   <- user configures & monitors
                    |    (app.py)      |
                    +--------+---------+
                             |
                    +--------+---------+
                    | EvolutionController |  <- orchestrates the loop
                    |  (controller.py)    |
                    +--------+---------+
                             |
                    +--------+--------+--------+
                    |                 |         |
                    v                 v         v
               +---------+    +-----------+  +---------+
               |Candidate|    | Fitness   |  | Top-K   |
               |Generator|    | Evaluator |  | Selector|
               +---------+    +-----------+  +---------+
                    |              |              |
                    v              v              v
               +---------+    +----------+   +----------+
               | OpenAI  |    |Subprocess|   | ChromaDB |
               | GPT-4o  |    |(Pac-Man) |   | VectorDB |
               +---------+    +----------+   +----------+
```

**Flow of execution:**
1. User configures parameters in the sidebar (problem type, generations, population size, mutation strategy, fitness weights)
2. User clicks "Start Evolution"
3. The `EvolutionController` initializes all components and seeds the vector database with templates
4. For each generation:
   - The **Candidate Generator** creates mutated versions of the parent code (via random operators, LLM-guided mutation, or single-shot LLM)
   - The **Evaluator** runs each candidate (Pac-Man subprocess or sandboxed matrix exec) and computes fitness
   - The **Selector** picks the top-K diverse candidates using Jaccard similarity filtering + elitism
   - Results are yielded to the UI, which updates the fitness chart and operation log in real time
5. After all generations (or early stopping), the best solution is displayed with download options

## Features

- Three mutation strategies: No Evolution (single-shot LLM baseline), Random Mutation, LLM-Guided Mutation
- Configurable fitness function with 3 adjustable weights (w1, w2, w3) constrained to sum to 1.0
- Real-time fitness visualization with Plotly charts across generations
- 3-strategy comparison experiment mode with overlay chart
- Vector database (ChromaDB) for caching evaluated candidates and RAG retrieval
- Sentence-transformer embeddings (all-MiniLM-L6-v2) for code similarity and duplicate detection
- Top-K selection with diversity filtering and elitism (global best is never lost)
- Safety sandboxing: regex code scanning, restricted exec namespaces, subprocess isolation, timeouts
- CSV and PNG export for analysis and reporting
- Operation log showing exactly what mutations and selections occurred each generation

## Prerequisites

- **Python 3.10+**
- **pip** (comes with Python)
- **~2 GB disk space** (for ML model downloads)
- **OpenAI API Key** (only for LLM-Guided and single-shot LLM strategies; get one at https://platform.openai.com/api-keys)

## Requirements / Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | >= 1.32.0 | Web UI framework |
| openai | >= 1.12.0 | GPT-4o-mini API calls |
| chromadb | >= 0.4.22 | Vector database for caching and RAG |
| plotly | >= 5.18.0 | Interactive fitness charts |
| pandas | >= 2.2.0 | Data manipulation and CSV export |
| numpy | >= 1.26.0 | Matrix operations for evaluation |
| sentence-transformers | >= 2.5.0 | Code embedding (all-MiniLM-L6-v2) |
| kaleido | >= 0.2.1 | PNG chart export |
| python-dotenv | >= 1.0.0 | Environment variable loading |

## Step-by-Step Instructions for Execution

```bash
# 1. Clone the repository
git clone https://github.com/Ganesh-Deepak/evolve-v2.git
cd evolve-v2

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set your OpenAI API key
# You can also enter it directly in the app sidebar
cp .env.example .env
# Edit .env and add your key

# 5. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501.

### Quick test run

1. Select **Matrix Multiplication (3x3)** as problem type
2. Set Generations to 5, Population to 3
3. Choose **Random Mutation** (no API key needed)
4. Click **Start Evolution**
5. Watch the fitness chart update in real time

## Data Formats

### Input
- **Initial Code**: Plain Python code entered as text in the UI sidebar
- **Problem Description**: Free-text description of the optimization goal
- **Parameters**: Generations (int), Population Size (int), Top-K (int), Fitness Weights (3 floats summing to 1.0)

### Output (CSV)
The exported CSV contains one row per candidate per generation:

| Column | Type | Description |
|--------|------|-------------|
| `generation` | int | Generation number (1-indexed) |
| `candidate_id` | str | First 8 chars of SHA-256 code hash |
| `fitness_score` | float | Computed fitness value |
| `mutation_type` | str | Strategy used (none, random_*, llm_guided, single_shot_llm) |
| `mutation_description` | str | Human-readable description of what changed |
| `correctness` | float | (Matrix only) Fraction of test cases passed |
| `num_operations` | int | (Matrix only) Count of arithmetic operations |
| `avg_score` | float | (Pac-Man only) Average game score |
| `max_score` | float | (Pac-Man only) Maximum game score |
| `win_rate` | float | (Pac-Man only) Fraction of games with positive score |

### Output (PNG)
Plotly fitness progression chart showing Best, Average, and Worst fitness per generation.

## Comparing Strategies

The project includes a comparison experiment mode (checkbox in sidebar) that runs all three strategies back-to-back:

1. **No Evolution (Single-Shot LLM)** -- one LLM call to improve the code, then evaluate; no iterative evolution
2. **Random Mutation** -- random programmatic code changes (parameter perturbation, operator swaps, line swaps, etc.) with evolutionary selection
3. **LLM-Guided Mutation** -- GPT-4o-mini analyzes code + RAG examples from vector DB and suggests targeted improvements each generation

Results are overlaid on a single comparison chart for direct visual analysis.

## Fitness Functions

**Pac-Man Agent:**
```
Fitness = w1 * avg_score + w2 * max_score + w3 * survival_metric
```
Where survival_metric rewards games where Pac-Man achieves a positive score (wins).

**Matrix Multiplication (Bonus):**
```
Fitness = w1 * correctness + w2 * (1 / (num_operations + 1))
```
Where correctness = fraction of 100 random test cases producing correct results, and num_operations is counted via AST analysis (multiplications + additions + subtractions).

All weights are configurable via the UI and must satisfy w1 + w2 + w3 = 1.0.

## Caching and Performance Optimization

To address execution time concerns, we implemented several acceleration mechanisms:

- **Fitness caching via ChromaDB**: Each candidate's code hash is stored with its fitness score. If identical code appears again (same SHA-256 hash), the cached score is returned immediately without re-evaluation.
- **Template seeding**: Starter code templates are pre-loaded into the vector database at initialization, giving the RAG system useful examples from the start.
- **Duplicate detection**: During selection, candidates with >95% Jaccard similarity to already-selected code are rejected, preventing wasted evaluations on near-identical solutions.
- **Early stopping**: If fitness doesn't improve for N consecutive generations (configurable patience), evolution terminates early.

## Known Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| First run takes several minutes | Downloads the sentence-transformers embedding model (~80 MB) | One-time download; subsequent runs are fast |
| Pac-Man evaluation is slow | Each candidate plays 6 games (3 games x 2 layouts) via subprocess | Reduce population size for faster iteration; use Matrix problem for quick testing |
| ChromaDB "collection already exists" error | Stale database from a previous crash | Delete the `data/chromadb/` folder and restart |
| Random mutations rarely improve Pac-Man agents | Random code changes almost always break complex game logic | Use LLM-Guided strategy for Pac-Man; Random works better for Matrix |
| Fitness weights don't sum to 1.0 | UI slider precision | Adjust values carefully; the app blocks execution until the sum is valid |
| LLM returns non-Python text occasionally | GPT-4o-mini sometimes includes explanations | Code extraction regex handles markdown blocks; falls back to raw text |
| `kaleido` error on PNG download | Missing optional dependency | Run `pip install kaleido` |

## Suggestions and Feedback

- For best results on Pac-Man, use LLM-Guided mutation with at least 10 generations and population size 5
- Matrix multiplication converges faster -- good for quick experimentation and debugging
- Start with small runs (3-5 generations, population 3) to verify setup before longer experiments
- The comparison experiment is the most useful output for analysis -- always include it in reports
- Each group member should use distinct parameter configurations (different weights, generations, or population sizes) to produce unique datasets

## Project Structure

```
evolve-v2/
|-- app.py                      # Streamlit web interface
|-- requirements.txt            # Python dependencies
|-- .env.example                # API key template
|
|-- evolve/                     # Core evolution engine
|   |-- models.py               # Data classes (RunConfig, Candidate, GenerationResult)
|   |-- controller.py           # Main evolution loop orchestrator
|   |-- candidate_generator.py  # 3 mutation strategies (none/random/LLM-guided)
|   |-- evaluator.py            # Fitness functions (Pac-Man + Matrix)
|   |-- selector.py             # Top-K selection with diversity + elitism
|   |-- llm_client.py           # OpenAI API wrapper with retry logic
|   |-- vector_store.py         # ChromaDB vector database for RAG + caching
|   |-- prompts.py              # LLM prompt templates
|
|-- pacman/                     # UC Berkeley CS188 Pac-Man framework (Python 3)
|   |-- pacman.py               # Game engine
|   |-- game.py                 # Core game logic and Agent base class
|   |-- layouts/                # Game maps (mediumClassic, smallClassic, etc.)
|
|-- templates/                  # Starter code templates seeded into vector DB
|   |-- pacman_greedy.py        # Greedy food-chasing agent
|   |-- pacman_scared.py        # Ghost-avoiding agent
|   |-- pacman_random.py        # Random action agent
|   |-- matrix_naive.py         # Standard O(n^3) triple-loop multiply
|   |-- matrix_optimized.py     # Partially unrolled multiply
|
|-- docs/
|   |-- TECHNICAL_DOCUMENTATION.md   # Detailed engineering documentation
|   |-- HOW_TO_USE.md                # Step-by-step usage guide
|
|-- data/                       # Created at runtime (gitignored)
    |-- chromadb/               # Vector database persistent storage
```

## References

[1] A. Novikov et al., "AlphaEvolve: A coding agent for scientific and algorithmic discovery," arXiv:2506.13131, Jun. 2025.

[2] S. Tamilselvi, "Introduction to Evolutionary Algorithms," in *Genetic Algorithms*, IntechOpen, 2022.

[3] H. Amit, "An Overview of Evolutionary Algorithms," We Talk Data, Medium, 2022.

[4] UC Berkeley CS188: Introduction to Artificial Intelligence -- Pac-Man Projects. https://inst.eecs.berkeley.edu/~cs188

[5] OpenAI API Documentation -- Chat Completions. https://platform.openai.com/docs/guides/chat

[6] ChromaDB Documentation. https://docs.trychroma.com/

[7] Sentence-Transformers: all-MiniLM-L6-v2. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Further Reading

See `docs/TECHNICAL_DOCUMENTATION.md` for a deep dive into how each module works, and `docs/HOW_TO_USE.md` for a step-by-step walkthrough of running experiments.
