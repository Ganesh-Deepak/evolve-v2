# Evolve -- Evolutionary Code Improvement Using LLMs

**CS5381 - Analysis of Algorithms | Term Project | Spring 2026**

## What is this?

This project explores whether we can use evolutionary algorithms to automatically improve code. The idea is pretty straightforward -- take a piece of code (say, a simple Pac-Man agent), create a bunch of slightly modified versions of it, test which ones work best, keep those, and repeat. After enough rounds, the code should get noticeably better.

We went a step further and added an LLM (GPT-4o-mini) into the mutation step. Instead of just randomly changing numbers and swapping lines, the LLM actually reads the code and tries to make smart improvements. We also hooked up a vector database (ChromaDB) so the LLM gets to see examples of what worked well in previous runs -- this is basically RAG (Retrieval-Augmented Generation) applied to code evolution.

The whole thing runs through a Streamlit web app where you can configure parameters, watch evolution happen in real time, and compare how different mutation strategies stack up.

## Two problems we tested on

- **Pac-Man Agent** -- evolving an agent that plays Pac-Man using the UC Berkeley CS188 framework. The agent decides which direction to move each turn based on food positions, ghost locations, etc.
- **3x3 Matrix Multiplication** -- trying to find algorithms that multiply matrices using fewer arithmetic operations than the standard approach (which needs 27 multiplications).

## Getting started

**You need:** Python 3.10+, pip, and optionally an OpenAI API key (only for the LLM-guided strategy).

```bash
# clone and set up
git clone https://github.com/Ganesh-Deepak/evolve-v2.git
cd evolve-v2
python -m venv venv

# activate virtual environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# install everything
pip install -r requirements.txt
```

Then just run:
```bash
streamlit run app.py
```

The app opens at http://localhost:8501. Configure your experiment in the sidebar and hit "Start Evolution."

> If you want to use LLM-Guided Mutation, you'll need an OpenAI API key. You can get one at https://platform.openai.com/api-keys. The cost is minimal -- a typical run uses about $0.01-0.05 worth of API calls with GPT-4o-mini.

## How it works (short version)

1. Start with some initial code (a basic Pac-Man agent or a naive matrix multiply)
2. Generate mutations -- either random code changes or LLM-suggested improvements
3. Run each candidate through a fitness function (play Pac-Man games or check matrix correctness + operation count)
4. Keep the top-K performers, throw away the rest
5. Repeat for N generations
6. The best solution found across all generations is your output

There are three mutation strategies you can compare:
- **No Evolution** -- just evaluates the original code (baseline/control)
- **Random Mutation** -- programmatic tweaks like changing numbers, swapping operators, rearranging lines
- **LLM-Guided** -- GPT-4o-mini analyzes the code and suggests targeted improvements, with RAG pulling in examples from a vector DB

The app has a comparison mode that runs all three back-to-back and plots the results on one chart.

## Project layout

```
evolve-v2/
|-- app.py                      # Streamlit web interface
|-- requirements.txt            # dependencies
|
|-- evolve/                     # core evolution engine
|   |-- models.py               # data classes (RunConfig, Candidate, etc.)
|   |-- controller.py           # main evolution loop
|   |-- candidate_generator.py  # mutation strategies (none, random, LLM)
|   |-- evaluator.py            # fitness functions for both problems
|   |-- selector.py             # top-k selection with elitism
|   |-- llm_client.py           # OpenAI API wrapper
|   |-- vector_store.py         # ChromaDB for RAG and caching
|   |-- prompts.py              # prompt templates for the LLM
|
|-- pacman/                     # UC Berkeley CS188 Pac-Man framework
|   |-- pacman.py, game.py      # game engine
|   |-- layouts/                # game maps
|
|-- templates/                  # starter code templates
|   |-- pacman_greedy.py        # greedy food-chaser
|   |-- pacman_scared.py        # ghost-avoider
|   |-- matrix_naive.py         # standard triple loop
|   |-- matrix_optimized.py     # partially unrolled
|
|-- docs/
|   |-- TECHNICAL_DOCUMENTATION.md
|   |-- HOW_TO_USE.md
|
|-- data/                       # created at runtime
    |-- chromadb/               # vector database storage
```

## Tech stack

| What | Why |
|------|-----|
| Python 3.10+ | main language |
| Streamlit | web UI (required by assignment spec) |
| OpenAI GPT-4o-mini | LLM for intelligent mutations |
| ChromaDB + sentence-transformers | vector DB for storing/retrieving candidates |
| Plotly | interactive charts |
| UC Berkeley CS188 | Pac-Man game engine |

## Troubleshooting

- **"Weights must sum to 1.0"** -- the three fitness weight sliders need to add up to exactly 1.00
- **Pac-Man fitness stuck at 0** -- the mutated code probably has a syntax error or references a method that doesn't exist. Check the operation log
- **LLM-Guided not working** -- make sure you entered a valid OpenAI API key
- **First run is slow** -- it downloads the sentence-transformers embedding model (~80 MB) on the first run
- **ChromaDB errors** -- delete the `data/chromadb/` folder and restart

## Further reading

See `docs/TECHNICAL_DOCUMENTATION.md` for a deep dive into how each module works, and `docs/HOW_TO_USE.md` for a step-by-step walkthrough of using the app.
