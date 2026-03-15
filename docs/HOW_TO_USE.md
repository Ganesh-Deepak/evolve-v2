# How to Use Evolve -- Step-by-Step Guide

This guide walks you through everything from installation to generating your assignment deliverables. No prior experience with AI, vector databases, or evolutionary algorithms is assumed.

---

## Part 1: Installation

### What You Need Before Starting

- **Python 3.10 or higher**: Check by running `python --version` in your terminal
- **pip**: Python's package manager (comes with Python)
- **An internet connection**: For downloading packages and (optionally) calling the OpenAI API
- **~2 GB of free disk space**: The machine learning models and dependencies are fairly large

### Step-by-Step Installation

**1. Open a terminal/command prompt and navigate to the project folder:**

```bash
cd D:\evolve
```

**2. Create a virtual environment:**

A virtual environment is an isolated Python installation just for this project. It prevents conflicts with other Python projects on your computer.

```bash
python -m venv venv
```

This creates a `venv/` folder containing a fresh Python installation.

**3. Activate the virtual environment:**

```bash
# Windows (Command Prompt):
venv\Scripts\activate

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your terminal prompt. This means the virtual environment is active.

**4. Install all required packages:**

```bash
pip install -r requirements.txt
```

This downloads and installs all dependencies. It may take 5-10 minutes on the first run because it downloads the PyTorch machine learning library (~800 MB) and the sentence-transformers embedding model.

**5. Verify the installation:**

```bash
python -c "from evolve.models import RunConfig; print('Installation successful!')"
```

If you see "Installation successful!" you're ready to go.

---

## Part 2: Running the App

**1. Make sure your virtual environment is activated** (you should see `(venv)` in your terminal)

**2. Start the Streamlit app:**

```bash
streamlit run app.py
```

**3. Your browser should automatically open** to http://localhost:8501. If not, open that URL manually.

You'll see the Evolve welcome page with a sidebar on the left containing all configuration options.

---

## Part 3: Your First Run (Matrix Multiplication -- Easiest)

Let's start with matrix multiplication because it's fast (no Pac-Man games to run) and doesn't need an API key.

**1. In the sidebar, set these values:**

| Setting | Value |
|---------|-------|
| Problem Type | Matrix Multiplication (3x3) |
| Number of Generations | 5 |
| Population Size | 3 |
| Top-K Selection | 2 |
| Mutation Strategy | Random Mutation |
| w1 | 0.50 |
| w2 | 0.30 |
| w3 | 0.20 |

**2. Click "Start Evolution"**

**3. Watch what happens:**
- The generation counter updates: "Generation 1 / 5", "Generation 2 / 5", etc.
- The fitness chart grows as new data points are added
- The operation log shows what mutations were applied

**4. When it finishes:**
- Look at the "Best Solution Found" section to see the evolved code
- Click "Download Results CSV" to save your data
- Click "Download Chart PNG" to save the fitness chart

That's it! You've successfully run an evolution experiment.

---

## Part 4: Running with Pac-Man

### What Makes Pac-Man Different

- Each game takes a few seconds to run (vs. milliseconds for matrix multiplication)
- A full 10-generation run with population size 5 might take 5-15 minutes
- The initial code is a simple greedy food-chasing agent

### Steps

**1. Set Problem Type to "Pac-Man Agent"**

The initial code will auto-populate with a greedy agent that chases the nearest food pellet.

**2. Use these starter settings:**

| Setting | Value |
|---------|-------|
| Number of Generations | 5 (start small) |
| Population Size | 3 |
| Mutation Strategy | Random Mutation |

**3. Click "Start Evolution" and wait**

Each generation takes 30-60 seconds. The fitness chart will update after each one.

**4. Understanding Pac-Man fitness:**

- Positive fitness = the agent scores well in the game
- Negative fitness = the agent dies quickly (loses 500 points per death)
- The default greedy agent typically scores around -200 to -400

---

## Part 5: Using LLM-Guided Mutation

This is the most powerful strategy but requires an OpenAI API key.

### Getting an API Key

1. Go to https://platform.openai.com/signup and create an account
2. Go to https://platform.openai.com/api-keys
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)
5. **Important**: You need to add credits to your account. Go to https://platform.openai.com/settings/organization/billing and add at least $5. GPT-4o-mini is very cheap (~$0.01-0.05 per evolution run).

### Using It in the App

1. Paste your API key in the "OpenAI API Key" field in the sidebar
2. Set Mutation Strategy to "LLM-Guided Mutation"
3. Click "Start Evolution"

### What Happens Behind the Scenes

For each candidate:
1. The system queries the vector database for similar high-performing code
2. It builds a prompt with the current code, fitness history, and those examples
3. GPT-4o-mini reads the prompt and writes improved code
4. The improved code is evaluated against the fitness function

You'll see in the operation log entries like: "LLM-guided mutation (temp=0.65, 3 RAG examples)"

---

## Part 6: Running the 3-Strategy Comparison Experiment

This is required for your assignment -- a chart comparing all three strategies.

**1. Check the "Run 3-Strategy Comparison Experiment" checkbox** in the sidebar

**2. Make sure you have an API key entered** (otherwise LLM-Guided will be skipped)

**3. Set your parameters:**

| Setting | Recommended |
|---------|------------|
| Number of Generations | 10 |
| Population Size | 5 |
| Problem Type | Whichever you want to compare |

**4. Click "Start Evolution"**

The system will run three separate experiments back-to-back:
- First: No Evolution (baseline) -- takes seconds
- Second: Random Mutation -- takes a few minutes
- Third: LLM-Guided Mutation -- takes a few minutes

**5. Results:**
- A comparison chart appears showing all three strategies on one plot
- The No Evolution line is flat (no improvement over generations)
- Random Mutation should show some improvement
- LLM-Guided should show the most improvement
- Download the chart as PNG and CSV for your report

---

## Part 7: Generating Your Individual Data Files

Each team member needs their own data. Here's how:

**1. Choose a unique parameter configuration:**

Member 1 might use: generations=10, population=5, w1=0.5, w2=0.3, w3=0.2
Member 2 might use: generations=15, population=3, w1=0.6, w2=0.2, w3=0.2
Member 3 might use: generations=10, population=8, w1=0.4, w2=0.4, w3=0.2

**2. Run the experiment with your configuration**

**3. Download outputs:**
- CSV file -> rename to `yourname_data.csv`
- PNG chart -> include in `yourname_data.docx`

**4. The CSV file contains these columns:**
- `generation`: Which generation (1, 2, 3, ...)
- `candidate_id`: Unique 8-character hash
- `fitness_score`: The fitness value
- `mutation_type`: What strategy was used
- `mutation_description`: Human-readable description of what changed
- Plus problem-specific metrics (correctness, game scores, etc.)

**5. Write your analysis** in the .docx file:
- What parameters did you use and why?
- Did fitness improve over generations?
- Which strategy worked best?
- Include screenshots of the UI

---

## Part 8: Troubleshooting Guide

### "ModuleNotFoundError: No module named 'evolve'"

You're running the app from the wrong directory. Make sure you're in the `D:\evolve` folder:
```bash
cd D:\evolve
streamlit run app.py
```

### "Weights must sum to 1.0"

The three fitness weight inputs (w1, w2, w3) must add up to exactly 1.00. Adjust them so they sum correctly.

### "API key required for LLM-Guided strategy"

You selected "LLM-Guided Mutation" but didn't enter an API key. Either:
- Paste your OpenAI API key in the field, or
- Switch to "Random Mutation" (doesn't need a key)

### The app is slow / seems stuck

- Pac-Man evaluation takes time. Each generation with 5 candidates = ~15 games = ~30-60 seconds
- The first run is slowest because it downloads the embedding model (~80 MB)
- Check the operation log for progress updates

### Pac-Man fitness is always 0.0

Look at the operation log for error messages. Common causes:
- The mutation produced invalid Python syntax
- The code references a method that doesn't exist (check for snake_case: `get_legal_actions` not `getLegalActions`)
- The code goes into an infinite loop (killed by timeout)

### "ChromaDB error" or "Collection already exists"

Delete the database and restart:
```bash
# Stop the app (Ctrl+C in terminal)
# Delete the database folder:
rm -rf data/chromadb
# Restart:
streamlit run app.py
```

### The chart doesn't show any data

Make sure at least one generation has completed. The chart only appears after the first generation yields results.

### "kaleido" error when downloading PNG

Install the missing package:
```bash
pip install kaleido
```

---

## Part 9: Tips for Getting Good Results

1. **Start small**: Use 3-5 generations and population 3 to test your setup before running longer experiments
2. **Matrix first**: Debug with matrix multiplication (fast evaluation) before switching to Pac-Man
3. **LLM-Guided is best for Pac-Man**: Random mutations rarely improve complex game logic, but the LLM can make strategic improvements
4. **Watch the operation log**: It tells you exactly what mutations were applied and why candidates were selected/rejected
5. **Run the comparison experiment last**: Once everything works, run the full 3-strategy comparison for your report
6. **Keep your API costs low**: GPT-4o-mini is cheap (~$0.15 per million input tokens). A typical run costs $0.01-0.05.
