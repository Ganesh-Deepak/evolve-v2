# Evolve: AI-Powered Evolutionary Code Improvement System

**CS5381 -- Analysis of Algorithms | Group Project -- Spring 2026**

Evolve is a web-based system that uses evolutionary algorithms combined with Large Language Models (LLMs) to automatically improve code. Given a starting piece of code (like a Pac-Man game-playing agent), the system generates mutated versions, tests them, keeps the best ones, and repeats -- gradually discovering better-performing solutions.

---

## Quick Start

### Prerequisites

- **Python 3.10+** installed on your system
- **Git** (to clone the repository)
- **OpenAI API Key** (only needed for LLM-Guided Mutation strategy; get one at https://platform.openai.com/api-keys)

### Setup (One-Time)

```bash
# 1. Clone the repository
git clone <repo-url>
cd evolve

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
# Make sure your virtual environment is activated, then:
streamlit run app.py
```

This opens the web interface in your browser (usually at http://localhost:8501).

---

## How to Use

### Step 1: Configure in the Sidebar

| Setting | What It Does | Recommended Value |
|---------|-------------|-------------------|
| **OpenAI API Key** | Your key for GPT-4o-mini (only needed for LLM-Guided strategy) | Your key |
| **Problem Type** | Choose Pac-Man Agent or Matrix Multiplication | Start with Matrix Multiplication (faster) |
| **Initial Code** | The starting code that will be evolved | Use the default provided |
| **Number of Generations** | How many rounds of evolution to run | 10 |
| **Population Size** | How many candidate solutions per generation | 5 |
| **Top-K Selection** | How many best candidates survive each generation | 3 |
| **Mutation Strategy** | How new candidates are created (see below) | Random Mutation to start |
| **Fitness Weights** | How important each scoring metric is (must sum to 1.0) | Use defaults (0.5, 0.3, 0.2) |

### Step 2: Choose a Mutation Strategy

- **No Evolution (Baseline)**: No changes are made. This is your control/reference point.
- **Random Mutation**: The system automatically tweaks numbers, swaps code lines, and changes operators. No API key needed.
- **LLM-Guided Mutation**: GPT-4o-mini analyzes the code and makes intelligent improvements. Requires an OpenAI API key.

### Step 3: Click "Start Evolution"

Watch in real-time as:
- The **generation counter** ticks up
- The **fitness chart** shows improvement over generations
- The **operation log** explains what mutations happened and why
- The **generation details table** shows all candidates and their scores

### Step 4: Review Results

After evolution completes:
- The **best solution** is displayed as a code block
- **Download CSV** to get raw data for your report
- **Download PNG** to get the fitness chart for your presentation

### Step 5 (Optional): Run Comparison Experiment

Check the **"Run 3-Strategy Comparison Experiment"** checkbox before starting. This runs all three strategies back-to-back and produces a comparison chart showing which strategy performed best -- exactly what you need for the assignment report.

---

## Supported Problems

### Pac-Man Agent
Evolves a game-playing agent for the UC Berkeley CS188 Pac-Man framework. The agent decides which direction Pac-Man should move each turn. Fitness is based on game score and survival.

### Matrix Multiplication (3x3) -- Bonus
Evolves a function that multiplies two 3x3 matrices, trying to minimize the number of arithmetic operations while maintaining correctness.

---

## Project Structure

```
evolve/
├── app.py                          # Web UI (Streamlit)
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── .gitignore
│
├── evolve/                         # Core engine
│   ├── models.py                   # Data structures
│   ├── controller.py               # Evolution loop orchestrator
│   ├── candidate_generator.py      # 3 mutation strategies
│   ├── evaluator.py                # Fitness scoring
│   ├── selector.py                 # Selection with elitism
│   ├── llm_client.py               # OpenAI API wrapper
│   ├── vector_store.py             # ChromaDB vector database
│   └── prompts.py                  # LLM prompt templates
│
├── pacman/                         # UC Berkeley CS188 Pac-Man framework
│   ├── pacman.py                   # Main game engine
│   ├── game.py                     # Game logic and Agent class
│   └── layouts/                    # Game maps
│
├── templates/                      # Starter code templates
│   ├── pacman_greedy.py            # Greedy food-chasing agent
│   ├── pacman_scared.py            # Ghost-avoiding agent
│   ├── pacman_random.py            # Random action agent
│   ├── matrix_naive.py             # Standard triple-loop multiply
│   └── matrix_optimized.py         # Unrolled multiply
│
└── data/                           # Generated at runtime
    ├── chromadb/                   # Vector database storage
    ├── generations/                # JSON logs per run
    └── plots/                      # Exported charts
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | Python 3.10+ with FastAPI patterns | Core evolution engine |
| Frontend | Streamlit | Web-based UI (mandatory, not CLI) |
| LLM | OpenAI GPT-4o-mini | Intelligent code mutations |
| Vector DB | ChromaDB | Cache candidates, enable retrieval-augmented evolution |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Code similarity detection |
| Charts | Plotly | Interactive fitness progression charts |
| Pac-Man | UC Berkeley CS188 framework | Game engine for agent evaluation |

---

## For Team Members: Generating Your Individual Data

Each team member must produce their own data files. Here's how:

1. Open the app and configure a **unique parameter set** (e.g., different weights, generation count, or population size)
2. Run evolution with your parameters
3. Click **"Download Results CSV"** -- save as `yourname_data.csv`
4. Click **"Download Chart PNG"** -- include in your `yourname_data.docx` report
5. Take screenshots of the UI showing your configuration and results
6. Write a brief analysis of what you observed

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Weights must sum to 1.0" | Adjust w1, w2, w3 so they add up to exactly 1.00 |
| "API key required" | Enter your OpenAI API key, or switch to Random Mutation (no key needed) |
| Pac-Man fitness is 0.0 | The generated code likely has a bug. Check the operation log for error details |
| App won't start | Make sure you activated the virtual environment first |
| ChromaDB errors | Delete the `data/chromadb/` folder and restart |
| Import errors | Run `pip install -r requirements.txt` again |
