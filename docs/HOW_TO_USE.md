# How to Use Evolve

This is a walkthrough for getting the app running and producing results. If you've never used virtual environments or Streamlit before, don't worry -- it's all covered here.

---

## Setting up

### Prerequisites

- Python 3.10 or newer (check with `python --version`)
- About 2 GB of free disk space (the ML models are pretty large)
- An internet connection for installing packages

### Installation

Open a terminal and navigate to the project folder:

```bash
cd evolve-v2
```

Create a virtual environment -- this keeps all the project dependencies separate from your system Python:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows (Command Prompt):
venv\Scripts\activate

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt. Now install the dependencies:

```bash
pip install -r requirements.txt
```

This takes a while the first time because it pulls in PyTorch and the sentence-transformers library. Go grab a coffee.

To check everything installed correctly:

```bash
python -c "from evolve.models import RunConfig; print('All good!')"
```

---

## Starting the app

With the virtual environment active:

```bash
streamlit run app.py
```

Your browser should open automatically to http://localhost:8501. If it doesn't, just open that URL manually.

---

## Your first experiment (Matrix Multiplication)

Matrix multiplication is the faster problem to run, so it's a good place to start.

In the sidebar, set:

| Setting | Value |
|---------|-------|
| Problem Type | Matrix Multiplication (3x3) |
| Generations | 5 |
| Population Size | 3 |
| Top-K | 2 |
| Strategy | Random Mutation |
| Weights | 0.50, 0.30, 0.20 (defaults) |

Click **Start Evolution** and watch it go. You'll see:
- A generation counter ticking up
- A fitness chart that updates after each generation
- An operation log showing what mutations were applied
- A table with all candidates and their scores

When it finishes, you can see the best code it found and download the results as CSV or the chart as PNG.
You can also download a generation-level captured-data CSV with runtime, steps per generation, generation count, and fitness trends.

---

## Running Pac-Man experiments

Pac-Man takes longer per generation because it actually runs Pac-Man games for each candidate. A few things to know:

- Each candidate plays 5 games on 2 different maps = 10 games per candidate
- With population size 5, that's 50 games per generation
- A generation takes roughly 45-90 seconds
- Start with 5 generations and population 3 to test things out before doing longer runs

The default starting code is a greedy agent that chases the nearest food pellet. It usually scores around -200 to -400 (negative because Pac-Man loses 500 points when it dies).

---

## Using LLM-Guided Mutation

This is the most interesting strategy, but it needs an OpenAI API key.

### Getting an API key

1. Sign up at https://platform.openai.com/signup
2. Go to https://platform.openai.com/api-keys and create a new key
3. Add some credits at https://platform.openai.com/settings/organization/billing (even $5 is more than enough -- GPT-4o-mini is very cheap)

### Using it

1. Paste your API key in the sidebar
2. Switch strategy to "LLM-Guided Mutation"
3. Click Start Evolution

Behind the scenes, for each generation the system:
- Produces ~30% of candidates via **crossover** (combining two parents' code into a new child)
- Produces the rest via **mutation** (improving a single parent)
- For mutations, includes a history of previously attempted mutations and their outcomes, so the LLM avoids repeating failed approaches
- Searches the vector database for diverse code examples (RAG)
- Uses **fitness-distance balancing** in selection to keep the parent pool diverse across generations
- Sends prompts to GPT-4o-mini with a slowly-decaying temperature (0.9 -> 0.4)

The operation log will show entries like "LLM-guided mutation (temp=0.75, 3 RAG examples)" and "LLM crossover of abc123+def456."

---

## Running the 3-strategy comparison

This is probably the most useful feature for generating results. Check the "Run Comparison Experiment" box in the sidebar, make sure you have an API key entered, and click Start. It runs all three strategies one after another and produces a chart overlaying the results.

Recommended settings for a solid comparison:
- 10 generations
- Population size 5
- Make sure you have an API key (otherwise LLM-Guided gets skipped)

The comparison chart and raw data are downloadable.

---

## Generating data for your report

Each person should use slightly different parameters to get unique data. For example:

- Person A: 10 generations, population 5, weights 0.5/0.3/0.2
- Person B: 15 generations, population 3, weights 0.6/0.2/0.2
- Person C: 10 generations, population 8, weights 0.4/0.4/0.2

After running, download:
- The **candidate CSV** with per-candidate data (fitness scores, mutation types, etc.)
- The **captured-data CSV** with runtime performance, steps per generation, generation count, and fitness across iterations
- The **chart PNG** for your report
- Take screenshots of the app showing your configuration

The candidate CSV has columns for generation number, candidate hash, fitness score, mutation type, complexity estimate, and problem-specific metrics (game scores for Pac-Man, correctness/operations for matrix). The captured-data CSV contains the generation-by-generation runtime and search-step summary needed for analysis.

---

## Common issues

**"ModuleNotFoundError"** -- you're probably not in the right directory or the virtual environment isn't activated. Run `cd evolve-v2` and activate the venv.

**"Weights must sum to 1.0"** -- adjust w1, w2, w3 so they add up to 1.00.

**App seems frozen** -- Pac-Man evaluation takes time. Check the operation log for progress. If it's truly stuck, the 90-second timeout will eventually kick in.

**Fitness always 0 for Pac-Man** -- the mutation probably broke the code syntax. Look at the log for error details.

**ChromaDB errors** -- delete `data/chromadb/` and restart the app.

**"kaleido" error on PNG download** -- run `pip install kaleido`.
