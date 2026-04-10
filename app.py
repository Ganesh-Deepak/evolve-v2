import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
from pathlib import Path

from evolve.models import RunConfig, Candidate
from evolve.controller import EvolutionController
from evolve.vector_store import VectorStore
from evolve.llm_client import LLMClient
from evolve.prompts import build_description_to_code_prompt

st.set_page_config(
    page_title="Evolve",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* overall background and font tweaks */
    .stApp {
        background: linear-gradient(175deg, #0f0f1a 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* header area */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
    }
    .hero-header h1 {
        color: white;
        font-size: 2.2rem;
        margin: 0 0 0.3rem 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0;
    }

    /* stat cards */
    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .stat-card .label {
        color: rgba(255,255,255,0.6);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .stat-card .value {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stat-card .value.green { color: #2ecc71; }
    .stat-card .value.orange { color: #f39c12; }
    .stat-card .value.red { color: #e74c3c; }

    /* strategy badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-none { background: rgba(231,76,60,0.2); color: #e74c3c; }
    .badge-random { background: rgba(243,156,18,0.2); color: #f39c12; }
    .badge-llm { background: rgba(46,204,113,0.2); color: #2ecc71; }

    /* welcome cards */
    .feature-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        transition: border-color 0.3s ease;
    }
    .feature-card:hover {
        border-color: rgba(102,126,234,0.4);
    }
    .feature-card h3 {
        color: #667eea;
        margin-top: 0;
        font-size: 1.1rem;
    }
    .feature-card p {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* sidebar styling */
    section[data-testid="stSidebar"] {
        background: #12121f;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #667eea;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* plotly chart container */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    /* nicer dividers */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* results section */
    .result-box {
        background: rgba(46,204,113,0.08);
        border: 1px solid rgba(46,204,113,0.25);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .result-box h3 {
        color: #2ecc71;
        margin-top: 0;
    }

    /* ── Section headers ───────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid rgba(102,126,234,0.25);
    }
    .section-header .icon {
        font-size: 1.4rem;
        width: 2.2rem;
        height: 2.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(102,126,234,0.15);
        border-radius: 8px;
    }
    .section-header h3 {
        color: #c5cee0;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.2px;
    }
    .section-header .subtitle {
        color: rgba(255,255,255,0.45);
        font-size: 0.82rem;
        margin-left: auto;
    }

    /* ── Log panel ──────────────────────────────────────────── */
    .log-panel {
        background: rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        font-size: 0.82rem;
        line-height: 1.6;
    }
    .log-panel .log-gen-header {
        color: #8fa4f0;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.5rem 0 0.4rem 0;
        margin-bottom: 0.3rem;
        border-bottom: 1px solid rgba(102,126,234,0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .log-panel .log-gen-header .gen-badge {
        background: rgba(102,126,234,0.2);
        color: #8fa4f0;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
        font-size: 0.78rem;
    }
    .log-panel .log-row {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
        padding: 0.2rem 0;
        color: rgba(255,255,255,0.7);
    }
    .log-panel .log-row .log-icon {
        flex-shrink: 0;
        width: 1.1rem;
        text-align: center;
        font-size: 0.75rem;
    }
    .log-panel .log-row .log-text {
        flex: 1;
    }
    .log-panel .log-row.row-candidate {
        font-family: 'JetBrains Mono', 'Cascadia Code', monospace;
        font-size: 0.78rem;
        padding: 0.25rem 0;
    }
    .log-panel .log-row.row-candidate .cand-hash {
        color: rgba(255,255,255,0.45);
    }
    .log-panel .log-row.row-candidate .cand-fitness {
        font-weight: 600;
        color: #5de0a0;
    }
    .log-panel .log-row.row-candidate .cand-fitness.neg {
        color: #ef8b82;
    }
    .log-panel .log-row.row-candidate .cand-cached {
        color: #f5b74a;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .log-panel .log-row.row-candidate .cand-desc {
        color: rgba(255,255,255,0.45);
        font-style: italic;
    }
    .log-panel .log-row.row-selected {
        color: #5de0a0;
        font-weight: 500;
        padding-top: 0.35rem;
        margin-top: 0.2rem;
        border-top: 1px solid rgba(46,204,113,0.15);
    }
    .log-panel .log-row.row-best {
        color: #2ecc71;
        font-weight: 600;
    }
    .log-panel .log-row.row-warn {
        color: #ef8b82;
    }
    .log-panel .log-row.row-timing {
        color: rgba(255,255,255,0.4);
        font-size: 0.75rem;
        padding-top: 0.3rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 0.2rem;
    }

    /* ── Generation summary banner ─────────────────────────── */
    .gen-summary-banner {
        background: linear-gradient(135deg, rgba(46,204,113,0.08) 0%, rgba(102,126,234,0.08) 100%);
        border: 1px solid rgba(46,204,113,0.15);
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .gen-summary-banner .summary-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.1rem;
    }
    .gen-summary-banner .summary-item .s-label {
        font-size: 0.65rem;
        color: rgba(255,255,255,0.4);
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .gen-summary-banner .summary-item .s-value {
        font-size: 1rem;
        font-weight: 700;
        color: #c5cee0;
    }
    .gen-summary-banner .summary-item .s-value.green { color: #2ecc71; }
    .gen-summary-banner .summary-item .s-value.purple { color: #667eea; }
    .gen-summary-banner .summary-sep {
        width: 1px;
        height: 2rem;
        background: rgba(255,255,255,0.1);
    }

    /* ── Candidate card grid ───────────────────────────────── */
    .candidate-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 0.8rem;
        margin: 0.8rem 0;
    }
    .candidate-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        transition: border-color 0.2s ease, transform 0.15s ease;
    }
    .candidate-card:hover {
        border-color: rgba(102,126,234,0.35);
        transform: translateY(-1px);
    }
    .candidate-card.selected {
        border-color: rgba(46,204,113,0.4);
        background: rgba(46,204,113,0.04);
    }
    .candidate-card .card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .candidate-card .card-hash {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: rgba(255,255,255,0.5);
        background: rgba(255,255,255,0.06);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
    }
    .candidate-card .card-fitness {
        font-size: 1.15rem;
        font-weight: 700;
        color: #667eea;
    }
    .candidate-card .card-fitness.high { color: #2ecc71; }
    .candidate-card .card-fitness.mid  { color: #f39c12; }
    .candidate-card .card-fitness.low  { color: #e74c3c; }
    .candidate-card .card-fitness.invalid { color: #e74c3c; font-size: 0.85rem; font-weight: 600; }
    .candidate-card .card-meta {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
    }
    .candidate-card .meta-tag {
        font-size: 0.72rem;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.06);
        color: rgba(255,255,255,0.6);
    }
    .candidate-card .meta-tag.mutation { background: rgba(102,126,234,0.15); color: #8fa4f0; }
    .candidate-card .meta-tag.status-valid { background: rgba(46,204,113,0.15); color: #5de0a0; }
    .candidate-card .meta-tag.status-cached { background: rgba(243,156,18,0.15); color: #f5b74a; }
    .candidate-card .meta-tag.status-invalid { background: rgba(231,76,60,0.15); color: #ef8b82; }
    .candidate-card .meta-tag.selected-tag { background: rgba(46,204,113,0.2); color: #2ecc71; font-weight: 600; }
    .candidate-card .card-desc {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        margin-top: 0.5rem;
        line-height: 1.4;
        font-style: italic;
    }

    /* ── Summary table styling ─────────────────────────────── */
    .styled-table-wrap {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0 1.5rem 0;
    }
    .styled-table-wrap .table-title {
        color: #c5cee0;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.7rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Expander overrides ────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #c5cee0 !important;
        font-size: 0.95rem !important;
    }

    /* ── Analysis bullets ──────────────────────────────────── */
    .analysis-card {
        background: rgba(102,126,234,0.06);
        border: 1px solid rgba(102,126,234,0.15);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .analysis-card li {
        color: rgba(255,255,255,0.8);
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }

    /* ── Best solution box ─────────────────────────────────── */
    .best-solution-box {
        background: linear-gradient(135deg, rgba(46,204,113,0.06) 0%, rgba(102,126,234,0.06) 100%);
        border: 1px solid rgba(46,204,113,0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem 0.6rem 1.5rem;
        margin: 1rem 0;
    }
    .best-solution-box .sol-label {
        color: #2ecc71;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    /* ── Comparison strategy expanders ──────────────────────── */
    .strategy-header {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>&#x1F9EC; Evolve</h1>
    <p>Evolutionary code improvement powered by LLMs and retrieval-augmented generation</p>
</div>
""", unsafe_allow_html=True)

# ── Default code templates ──────────────────────────────────────────────────
DEFAULT_PACMAN_CODE = """legal = state.get_legal_actions()
if 'Stop' in legal:
    legal.remove('Stop')
food = state.get_food().as_list()
pos = state.get_pacman_position()

if not food:
    return legal[0] if legal else 'Stop'

closest_food = min(food, key=lambda f: abs(f[0] - pos[0]) + abs(f[1] - pos[1]))

best_action = legal[0]
best_dist = float('inf')
for action in legal:
    successor = state.generate_pacman_successor(action)
    new_pos = successor.get_pacman_position()
    dist = abs(new_pos[0] - closest_food[0]) + abs(new_pos[1] - closest_food[1])
    if dist < best_dist:
        best_dist = dist
        best_action = action

return best_action"""

DEFAULT_MATRIX_CODE = """result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
for i in range(3):
    for j in range(3):
        for k in range(3):
            result[i][j] = result[i][j] + A[i][k] * B[k][j]
return result"""

DEFAULT_DESCRIPTIONS = {
    "Pac-Man Agent": "Improve a Pac-Man game-playing agent to maximize score, survive longer, and move efficiently.",
    "Matrix Multiplication (3x3)": "Optimize 3x3 matrix multiplication to minimize the number of arithmetic operations while maintaining correctness.",
}

PACMAN_LAYOUT_PRESETS = {
    "Small + Medium (Default)": ("smallClassic", "mediumClassic"),
    "Medium + Open": ("mediumClassic", "openClassic"),
    "Open + Original": ("openClassic", "originalClassic"),
}

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")

    api_key = st.text_input("OpenAI API Key", type="password",
                            help="Required only for LLM-Guided Mutation")

    problem_type_display = st.selectbox(
        "Problem Type",
        ["Pac-Man Agent", "Matrix Multiplication (3x3)"]
    )
    problem_type = "pacman" if "Pac-Man" in problem_type_display else "matrix"

    problem_desc = st.text_area(
        "Problem Description",
        value=DEFAULT_DESCRIPTIONS[problem_type_display],
        height=80,
    )

    pacman_layouts = PACMAN_LAYOUT_PRESETS["Small + Medium (Default)"]
    if problem_type == "pacman":
        layout_preset = st.selectbox(
            "Pac-Man Layout Set",
            list(PACMAN_LAYOUT_PRESETS.keys()),
            help="Larger layouts increase evaluation time and can make behavior differences more visible.",
        )
        pacman_layouts = PACMAN_LAYOUT_PRESETS[layout_preset]
        st.caption(f"Layouts used for evaluation: {', '.join(pacman_layouts)}")

    input_type = st.selectbox(
        "Input Type",
        ["Python Code", "Pseudocode / Description"],
        help="You can provide Python code directly, or describe the algorithm in pseudocode or plain text. "
             "If you choose Pseudocode/Description, the LLM will convert it to Python before evolution starts.",
    )

    if input_type == "Python Code":
        initial_code = st.text_area(
            "Initial Code",
            value=DEFAULT_PACMAN_CODE if problem_type == "pacman" else DEFAULT_MATRIX_CODE,
            height=220,
            help="Starting Python code to evolve",
        )
    else:
        pseudocode_input = st.text_area(
            "Pseudocode / Algorithm Description",
            value="A greedy agent that always moves toward the nearest food pellet, avoiding ghosts when they are close."
                  if problem_type == "pacman" else
                  "Multiply two 3x3 matrices using the standard triple-loop approach.",
            height=220,
            help="Describe the algorithm in pseudocode or plain English. The LLM will generate Python code from this.",
        )
        initial_code = None  # will be generated before evolution starts

    st.markdown("## Evolution Parameters")

    num_generations = st.slider("Generations", 1, 50, 10)
    population_size = st.slider("Population Size", 2, 20, 5)
    top_k = st.slider("Top-K Selection", 1, 10, 3)

    strategy_display = st.selectbox(
        "Mutation Strategy",
        ["No Evolution (Single-Shot LLM)", "Random Mutation", "LLM-Guided Mutation"]
    )
    strategy_map = {
        "No Evolution (Single-Shot LLM)": "none",
        "Random Mutation": "random",
        "LLM-Guided Mutation": "llm_guided",
    }
    strategy = strategy_map[strategy_display]

    st.markdown("## Fitness Weights")
    if problem_type == "pacman":
        st.caption("fitness = w1 * avg_score + w2 * max_score + w3 * survival")
    else:
        st.caption("fitness = w1 * correctness + w2 * (1/(ops+1)) + w3 * (1/(exec_ms+1))")

    col1, col2, col3 = st.columns(3)
    w1 = col1.number_input("w1", 0.0, 1.0, 0.5, 0.05)
    w2 = col2.number_input("w2", 0.0, 1.0, 0.3, 0.05)
    w3 = col3.number_input("w3", 0.0, 1.0, 0.2, 0.05)

    weights_valid = abs(w1 + w2 + w3 - 1.0) < 0.01
    if not weights_valid:
        st.error(f"Weights must sum to 1.0 (currently {w1+w2+w3:.2f})")

    api_key_needed = strategy == "llm_guided" and not api_key
    pseudocode_key_needed = input_type == "Pseudocode / Description" and not api_key
    if api_key_needed:
        st.warning("API key required for LLM-Guided strategy")
    if pseudocode_key_needed:
        st.warning("API key required to convert pseudocode/description into Python code")

    st.divider()
    run_comparison = st.checkbox("Run Comparison Experiment",
                                 help="Run all three strategies and compare results")

    start_disabled = not weights_valid or api_key_needed or pseudocode_key_needed
    start_button = st.button("Start Evolution", type="primary",
                             disabled=start_disabled, width="stretch")


# ── Helper functions ────────────────────────────────────────────────────────

def load_templates(problem_type: str) -> list[tuple[str, str]]:
    templates = []
    template_dir = Path("./templates")
    prefix = "pacman_" if problem_type == "pacman" else "matrix_"
    for f in template_dir.glob(f"{prefix}*.py"):
        code = f.read_text(encoding="utf-8")
        templates.append((code, f.stem))
    return templates


CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.8)", size=12),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.1)",
    ),
    margin=dict(l=50, r=30, t=50, b=40),
)


def build_fitness_chart(history: list[dict], title: str = "Fitness Progression") -> go.Figure:
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["best"],
        mode="lines+markers", name="Best",
        line=dict(color="#2ecc71", width=2.5),
        marker=dict(size=7, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["avg"],
        mode="lines+markers", name="Average",
        line=dict(color="#667eea", width=2),
        marker=dict(size=5),
    ))
    if "worst" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["generation"], y=df["worst"],
            mode="lines", name="Worst",
            line=dict(color="#e74c3c", width=1.2, dash="dot"),
        ))
    fig.update_layout(title=title, xaxis_title="Generation", yaxis_title="Fitness",
                      height=420, **CHART_LAYOUT)
    return fig


def build_comparison_chart(all_results: dict[str, list[dict]]) -> go.Figure:
    fig = go.Figure()
    colors = {"none": "#e74c3c", "random": "#f39c12", "llm_guided": "#2ecc71"}
    names = {"none": "No Evolution", "random": "Random Mutation", "llm_guided": "LLM-Guided"}
    for strat, history in all_results.items():
        if not history:
            continue
        df = pd.DataFrame(history)
        fig.add_trace(go.Scatter(
            x=df["generation"], y=df["best"],
            mode="lines+markers", name=names.get(strat, strat),
            line=dict(color=colors.get(strat, "#999"), width=2.5),
            marker=dict(size=6),
        ))
    fig.update_layout(
        title="Strategy Comparison -- Best Fitness per Generation",
        xaxis_title="Generation", yaxis_title="Best Fitness",
        height=480, **CHART_LAYOUT,
    )
    return fig


def build_runtime_chart(history: list[dict], title: str = "Candidate Execution Time per Generation") -> go.Figure:
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    fig = go.Figure()

    # Primary: best candidate eval time (the actual code run time)
    if "best_eval_time" in df.columns:
        fig.add_trace(go.Bar(
            x=df["generation"], y=df["best_eval_time"],
            name="Best Candidate Eval (ms)",
            marker_color="#2ecc71",
            opacity=0.85,
        ))
    if "avg_eval_time" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["generation"], y=df["avg_eval_time"],
            mode="lines+markers", name="Avg Eval Time (ms)",
            line=dict(color="#667eea", width=2),
            marker=dict(size=5),
        ))
    if "best_exec_time" in df.columns and df["best_exec_time"].sum() > 0:
        fig.add_trace(go.Scatter(
            x=df["generation"], y=df["best_exec_time"],
            mode="lines+markers", name="Best Exec Time (ms)",
            line=dict(color="#f39c12", width=2, dash="dot"),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Generation",
        yaxis_title="Time (ms)",
        height=380,
        **CHART_LAYOUT,
    )
    return fig


def build_steps_chart(history: list[dict], title: str = "Steps per Generation") -> go.Figure:
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["generation"], y=df["candidates_evaluated"],
        name="Candidates Evaluated",
        marker_color="#2ecc71",
    ))
    fig.add_trace(go.Bar(
        x=df["generation"], y=df["candidates_cached"],
        name="Candidates Cached",
        marker_color="#f39c12",
    ))
    fig.add_trace(go.Scatter(
        x=df["generation"], y=df["candidates_selected"],
        mode="lines+markers", name="Selected",
        line=dict(color="#e74c3c", width=2.5),
        marker=dict(size=7),
        yaxis="y2",
    ))
    steps_layout = {k: v for k, v in CHART_LAYOUT.items() if k != "yaxis"}
    fig.update_layout(
        title=title,
        xaxis_title="Generation",
        yaxis=dict(title="Candidate Steps", side="left"),
        yaxis2=dict(title="Selected", side="right", overlaying="y",
                    gridcolor="rgba(255,255,255,0.03)"),
        barmode="stack",
        height=380,
        **steps_layout,
    )
    return fig


def format_duration_ms(duration_ms: float) -> str:
    if duration_ms < 1000:
        return f"{duration_ms:.1f} ms"
    return f"{duration_ms / 1000:.3f} s"


def build_generation_summary_df(fitness_history: list[dict], runtime_history: list[dict]) -> pd.DataFrame:
    fitness_df = pd.DataFrame(fitness_history)
    runtime_df = pd.DataFrame(runtime_history)
    if fitness_df.empty:
        return runtime_df
    if runtime_df.empty:
        return fitness_df
    return fitness_df.merge(runtime_df, on="generation", how="left")


def build_single_run_analysis(summary_df: pd.DataFrame) -> list[str]:
    if summary_df.empty:
        return []

    fastest = summary_df.loc[summary_df["gen_time_ms"].idxmin()]
    best = summary_df.loc[summary_df["best"].idxmax()]
    slowest = summary_df.loc[summary_df["gen_time_ms"].idxmax()]

    return [
        f"Best fitness reached {best['best']:.4f} in generation {int(best['generation'])}.",
        f"Fastest generation was {int(fastest['generation'])} at {format_duration_ms(fastest['gen_time_ms'])}.",
        f"Slowest generation was {int(slowest['generation'])} at {format_duration_ms(slowest['gen_time_ms'])}.",
    ]


def build_comparison_analysis(comp_df: pd.DataFrame) -> list[str]:
    if comp_df.empty:
        return []

    analyses = []
    grouped = comp_df.groupby("strategy", dropna=False)
    best_row = comp_df.loc[comp_df["best_fitness"].idxmax()]
    analyses.append(
        f"Highest best-fitness run was {best_row['strategy']} in generation {int(best_row['generation'])} with fitness {best_row['best_fitness']:.4f}."
    )
    avg_runtime = grouped["gen_time_ms"].mean(numeric_only=True).sort_values()
    if not avg_runtime.empty:
        analyses.append(
            f"Fastest strategy on average was {avg_runtime.index[0]} at {format_duration_ms(avg_runtime.iloc[0])} per generation."
        )
    return analyses


def render_stat_card(label: str, value: str, color: str = ""):
    color_class = f" {color}" if color else ""
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">{label}</div>
        <div class="value{color_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def get_candidate_status(candidate: Candidate) -> str:
    if candidate.fitness_breakdown.get("invalid_candidate"):
        return "Invalid"
    if candidate.fitness_breakdown.get("cached"):
        return "Cached"
    return "Valid"


def format_candidate_fitness(candidate: Candidate) -> str:
    if candidate.fitness is None:
        return "N/A"
    if candidate.fitness_breakdown.get("invalid_candidate"):
        return "Invalid"
    return f"{candidate.fitness:.4f}"


def render_section_header(title: str, icon: str = "", subtitle: str = ""):
    sub_html = f'<span class="subtitle">{subtitle}</span>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        <span class="icon">{icon}</span>
        <h3>{title}</h3>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def _safe(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_log_entries(entries: list[str]):
    import re as _re
    html = '<div class="log-panel">'

    for entry in entries:
        line = entry.strip()
        if not line:
            continue
        lower = line.lower()

        # Generation header
        if "---" in line and "generation" in lower:
            clean = line.strip("-").strip()
            html += (
                f'<div class="log-gen-header">'
                f'<span class="gen-badge">{_safe(clean)}</span>'
                f'</div>'
            )
            continue

        # Candidate line: "  Candidate N (hash): fitness=X | desc"
        cand_match = _re.match(
            r"\s*Candidate\s+(\d+)\s+\(([a-f0-9]+)\):\s*fitness=([\-\d.]+)\s*(.*)",
            line,
        )
        if cand_match:
            num, hash_str, fitness_str, rest = cand_match.groups()
            is_cached = "[CACHED]" in rest
            rest_clean = rest.lstrip("| ").replace("[CACHED]", "").strip().lstrip("| ").strip()
            fitness_val = float(fitness_str) if fitness_str else 0
            fit_cls = "neg" if fitness_val < 0 else ""
            cached_tag = ' <span class="cand-cached">CACHED</span>' if is_cached else ""
            html += (
                f'<div class="log-row row-candidate">'
                f'<span class="log-icon">&#x25B8;</span>'
                f'<span class="log-text">'
                f'<span class="cand-hash">#{num} {hash_str}</span> '
                f'<span class="cand-fitness {fit_cls}">{fitness_str}</span>'
                f'{cached_tag}'
                f' <span class="cand-desc">{_safe(rest_clean[:60])}</span>'
                f'</span>'
                f'</div>'
            )
            continue

        # Selected line
        if "selected:" in lower:
            html += (
                f'<div class="log-row row-selected">'
                f'<span class="log-icon">&#x2714;</span>'
                f'<span class="log-text">{_safe(line.strip())}</span>'
                f'</div>'
            )
            continue

        # New global best
        if "new global best" in lower:
            html += (
                f'<div class="log-row row-best">'
                f'<span class="log-icon">&#x2B50;</span>'
                f'<span class="log-text">{_safe(line.strip())}</span>'
                f'</div>'
            )
            continue

        # Timing / stats line
        if "generation time:" in lower or "gen_time" in lower:
            html += (
                f'<div class="log-row row-timing">'
                f'<span class="log-icon">&#x23F1;</span>'
                f'<span class="log-text">{_safe(line.strip())}</span>'
                f'</div>'
            )
            continue

        # Warning
        if "invalid" in lower or "error" in lower or "fail" in lower or "rejected" in lower:
            html += (
                f'<div class="log-row row-warn">'
                f'<span class="log-icon">&#x26A0;</span>'
                f'<span class="log-text">{_safe(line.strip())}</span>'
                f'</div>'
            )
            continue

        # Elitism / preserved
        if "elitism" in lower or "preserved" in lower:
            html += (
                f'<div class="log-row row-best">'
                f'<span class="log-icon">&#x1F6E1;</span>'
                f'<span class="log-text">{_safe(line.strip())}</span>'
                f'</div>'
            )
            continue

        # Default
        html += (
            f'<div class="log-row">'
            f'<span class="log-icon">&#x2022;</span>'
            f'<span class="log-text">{_safe(line.strip())}</span>'
            f'</div>'
        )

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_candidate_cards(candidates: list, selected: list, generation_num: int, best_hash: str = ""):
    cards_html = '<div class="candidate-grid">'
    for idx, c in enumerate(candidates, 1):
        status = get_candidate_status(c)
        is_selected = c in selected
        is_best = best_hash and c.code_hash[:8] == best_hash
        card_cls = "selected" if is_best else ""

        status_cls = f"status-{status.lower()}"

        # Fitness display
        if c.fitness is None or status == "Invalid":
            fitness_html = '<span class="card-fitness invalid">INVALID</span>'
        else:
            val = c.fitness
            color_cls = "high" if val >= 0.7 else ("mid" if val >= 0.3 else "low")
            fitness_html = f'<span class="card-fitness {color_cls}">{val:.4f}</span>'

        # Eval time
        eval_ms = c.fitness_breakdown.get("eval_time_ms", "")
        eval_str = f"{eval_ms:.1f}ms" if isinstance(eval_ms, (int, float)) else ""

        # Complexity
        complexity = c.fitness_breakdown.get("estimated_time_complexity", "")

        # Meta tags
        tags_html = f'<span class="meta-tag mutation">{c.mutation_type}</span>'
        tags_html += f'<span class="meta-tag {status_cls}">{status}</span>'
        if eval_str:
            tags_html += f'<span class="meta-tag">{eval_str}</span>'
        if complexity:
            tags_html += f'<span class="meta-tag">{complexity}</span>'
        if is_selected:
            tags_html += '<span class="meta-tag selected-tag">SELECTED</span>'
        if is_best:
            tags_html += '<span class="meta-tag selected-tag">BEST</span>'

        desc = c.mutation_description[:90] if c.mutation_description else ""
        safe_desc = _safe(desc)

        candidate_label = f"Candidate {idx} &mdash; {c.code_hash[:8]}"

        cards_html += f"""
        <div class="candidate-card {card_cls}">
            <div class="card-top">
                <span class="card-hash">{candidate_label}</span>
                {fitness_html}
            </div>
            <div class="card-meta">{tags_html}</div>
            <div class="card-desc">{safe_desc}</div>
        </div>"""

    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


def run_single_evolution(config: RunConfig, vector_store: VectorStore,
                         chart_placeholder, status_placeholder,
                         gen_counter, best_score, log_container, details_container,
                         runtime_chart_placeholder=None, steps_chart_placeholder=None,
                         chart_key_prefix: str = "run"):
    controller = EvolutionController(config, vector_store)

    templates = load_templates(config.problem_type)
    vector_store.seed_templates(templates)

    fitness_history = []
    runtime_history = []
    all_candidates = []
    final_result = None

    status_placeholder.info("Evolution in progress...")

    for gen_result in controller.run_evolution():
        final_result = gen_result

        gen_counter.markdown(
            f'<div style="font-size:1.5rem;font-weight:700;color:#c5cee0;letter-spacing:-0.3px;">'
            f'Generation {gen_result.generation_num} / {config.num_generations}</div>',
            unsafe_allow_html=True,
        )
        best_score.metric("Best Fitness", f"{gen_result.best_overall.fitness:.4f}",
                          delta=f"{gen_result.stats.get('max_fitness', 0) - gen_result.stats.get('avg_fitness', 0):.4f}")

        # Filter out invalid candidates (fitness = -1_000_000) for chart stats
        valid_fitnesses = [
            c.fitness for c in gen_result.candidates
            if c.fitness is not None and not c.fitness_breakdown.get("invalid_candidate")
        ]
        if valid_fitnesses:
            chart_best = max(valid_fitnesses)
            chart_avg = sum(valid_fitnesses) / len(valid_fitnesses)
            chart_worst = min(valid_fitnesses)
        else:
            chart_best = gen_result.stats["max_fitness"]
            chart_avg = gen_result.stats["avg_fitness"]
            chart_worst = gen_result.stats["min_fitness"]

        fitness_history.append({
            "generation": gen_result.generation_num,
            "best": chart_best,
            "avg": chart_avg,
            "worst": chart_worst,
        })

        runtime_history.append({
            "generation": gen_result.generation_num,
            "gen_time_sec": gen_result.stats.get("gen_time_sec", 0),
            "gen_time_ms": gen_result.stats.get("gen_time_ms", 0),
            "gen_time": gen_result.stats.get("gen_time_sec", 0),
            "best_eval_time": gen_result.stats.get("best_eval_time_ms", 0),
            "best_exec_time": gen_result.stats.get("best_exec_time_ms", 0),
            "avg_eval_time": gen_result.stats.get("avg_eval_time_ms", 0),
            "candidates_generated": gen_result.stats.get("candidates_generated", 0),
            "candidates_evaluated": gen_result.stats.get("candidates_evaluated", 0),
            "candidates_cached": gen_result.stats.get("candidates_cached", 0),
            "candidates_selected": gen_result.stats.get("candidates_selected", 0),
            "best_estimated_time_complexity": gen_result.stats.get("best_estimated_time_complexity", ""),
            "best_generalized_time_complexity": gen_result.stats.get("best_generalized_time_complexity", ""),
        })

        chart_placeholder.plotly_chart(
            build_fitness_chart(fitness_history),
            width="stretch",
            key=f"{chart_key_prefix}_fitness_gen_{gen_result.generation_num}",
        )

        if runtime_chart_placeholder:
            runtime_chart_placeholder.plotly_chart(
                build_runtime_chart(runtime_history),
                width="stretch",
                key=f"{chart_key_prefix}_runtime_gen_{gen_result.generation_num}",
            )

        if steps_chart_placeholder:
            steps_chart_placeholder.plotly_chart(
                build_steps_chart(runtime_history),
                width="stretch",
                key=f"{chart_key_prefix}_steps_gen_{gen_result.generation_num}",
            )

        with log_container:
            render_log_entries(gen_result.log_entries)

        with details_container:
            valid_cands = [c for c in gen_result.candidates if c.fitness is not None and not c.fitness_breakdown.get("invalid_candidate")]
            valid_count = len(valid_cands)
            best_cand = max(valid_cands, key=lambda c: c.fitness) if valid_cands else None
            best_this_gen = best_cand.fitness if best_cand else 0
            best_hash = best_cand.code_hash[:8] if best_cand else "N/A"

            with st.expander(
                f"Generation {gen_result.generation_num}  |  {len(gen_result.candidates)} candidates  |  Best: {best_this_gen:.4f} ({best_hash})",
                expanded=False,
            ):
                # Summary banner at top
                avg_fit = sum(c.fitness for c in valid_cands) / len(valid_cands) if valid_cands else 0
                selected_count = len(gen_result.selected)
                st.markdown(
                    f'<div class="gen-summary-banner">'
                    f'<div class="summary-item"><span class="s-label">Best Fitness</span><span class="s-value green">{best_this_gen:.4f}</span></div>'
                    f'<div class="summary-sep"></div>'
                    f'<div class="summary-item"><span class="s-label">Best Candidate</span><span class="s-value">{best_hash}</span></div>'
                    f'<div class="summary-sep"></div>'
                    f'<div class="summary-item"><span class="s-label">Avg Fitness</span><span class="s-value purple">{avg_fit:.4f}</span></div>'
                    f'<div class="summary-sep"></div>'
                    f'<div class="summary-item"><span class="s-label">Valid</span><span class="s-value">{valid_count}/{len(gen_result.candidates)}</span></div>'
                    f'<div class="summary-sep"></div>'
                    f'<div class="summary-item"><span class="s-label">Selected</span><span class="s-value">{selected_count}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                render_candidate_cards(gen_result.candidates, gen_result.selected, gen_result.generation_num, best_hash=best_hash)

        all_candidates.extend(gen_result.candidates)

    status_placeholder.success("Complete!")
    return final_result, fitness_history, runtime_history, all_candidates, controller.log_entries


# ── Pseudocode-to-Python conversion helper ──────────────────────────────────

def convert_pseudocode_to_python(description: str, problem_type: str, api_key: str) -> str:
    """Use the LLM to convert pseudocode/description into Python code."""
    client = LLMClient(api_key)
    system, user_prompt = build_description_to_code_prompt(problem_type, description)
    return client.generate_code(system, user_prompt)


# ── Main content area ───────────────────────────────────────────────────────

if start_button:
    # Handle pseudocode conversion if needed
    if input_type == "Pseudocode / Description" and initial_code is None:
        if not api_key:
            st.error("An API key is required to convert pseudocode/description to Python code.")
            st.stop()
        with st.spinner("Converting your description to Python code..."):
            initial_code = convert_pseudocode_to_python(pseudocode_input, problem_type, api_key)
        st.info("Generated initial Python code from your description:")
        st.code(initial_code, language="python")

    effective_num_generations = 1 if strategy == "none" else num_generations

    if run_comparison:
        # ── Comparison experiment ───────────────────────────────────────
        render_section_header("Comparison Experiment", "&#x2694;&#xFE0F;", "Running all three strategies side-by-side")

        strategies_to_run = ["none", "random", "llm_guided"]
        strategy_names = {"none": "No Evolution (Single-Shot LLM)", "random": "Random Mutation", "llm_guided": "LLM-Guided"}

        all_histories = {}
        all_runtime_histories = {}
        all_best_codes = {}

        progress_bar = st.progress(0)
        comparison_status = st.empty()

        for idx, strat in enumerate(strategies_to_run):
            if strat == "llm_guided" and not api_key:
                comparison_status.warning("Skipping LLM-Guided (no API key provided)")
                continue

            badge_class = {"none": "badge-none", "random": "badge-random", "llm_guided": "badge-llm"}
            comparison_status.markdown(
                f'Running: <span class="badge {badge_class.get(strat, "")}">{strategy_names[strat]}</span> ({idx+1}/3)',
                unsafe_allow_html=True,
            )
            progress_bar.progress(idx / 3)

            config = RunConfig(
                problem_type=problem_type,
                problem_description=problem_desc,
                initial_code=initial_code,
                mutation_strategy=strat,
                num_generations=1 if strat == "none" else num_generations,
                population_size=population_size,
                top_k=top_k,
                fitness_weights=(w1, w2, w3),
                openai_api_key=api_key,
                pacman_layouts=pacman_layouts,
            )

            vs = VectorStore(persist_dir=f"./data/chromadb_{strat}")
            vs.clear()

            strat_icon = {"none": ":material/block:", "random": ":material/casino:", "llm_guided": ":material/psychology:"}
            with st.expander(f"{strategy_names[strat]}", expanded=False, icon=strat_icon.get(strat, None)):
                chart_ph = st.empty()
                runtime_ph = st.empty()
                status_ph = st.empty()
                gen_ph = st.empty()
                best_ph = st.empty()
                steps_ph = st.empty()
                log_ph = st.container()
                det_ph = st.container()

                result, history, runtime_hist, candidates, logs = run_single_evolution(
                    config, vs, chart_ph, status_ph, gen_ph, best_ph, log_ph, det_ph, runtime_ph, steps_ph,
                    chart_key_prefix=f"comparison_{strat}"
                )

            all_histories[strat] = history
            all_runtime_histories[strat] = runtime_hist
            if result:
                all_best_codes[strat] = result.best_overall.code

        progress_bar.progress(1.0)
        comparison_status.success("All strategies complete!")

        st.markdown("---")
        render_section_header("Results", "&#x1F3C6;", "Strategy comparison overview")
        comp_fig = build_comparison_chart(all_histories)
        st.plotly_chart(comp_fig, width="stretch")

        # summary cards
        summary_cols = st.columns(len(all_histories))
        for col, (strat, hist) in zip(summary_cols, all_histories.items()):
            if hist:
                best_val = max(h["best"] for h in hist)
                with col:
                    render_stat_card(
                        strategy_names.get(strat, strat),
                        f"{best_val:.4f}",
                        "green" if strat == "llm_guided" else ("orange" if strat == "random" else "red"),
                    )

        # Runtime comparison chart - show eval time (actual code execution)
        render_section_header("Candidate Execution Time", "&#x23F1;&#xFE0F;", "Best candidate eval time per generation")
        rt_fig = go.Figure()
        rt_colors = {"none": "#e74c3c", "random": "#f39c12", "llm_guided": "#2ecc71"}
        for strat, rt_hist in all_runtime_histories.items():
            if rt_hist:
                rt_df = pd.DataFrame(rt_hist)
                rt_fig.add_trace(go.Scatter(
                    x=rt_df["generation"], y=rt_df["best_eval_time"],
                    mode="lines+markers", name=strategy_names.get(strat, strat),
                    line=dict(color=rt_colors.get(strat, "#999"), width=2.5),
                    marker=dict(size=6),
                ))
        rt_fig.update_layout(
            title="Execution Time Comparison -- Best Candidate Eval (ms)",
            xaxis_title="Generation", yaxis_title="Eval Time (ms)",
            height=400, **CHART_LAYOUT,
        )
        st.plotly_chart(rt_fig, width="stretch")

        render_section_header("Search Effort per Generation", "&#x1F4CA;")
        step_fig = go.Figure()
        bar_colors_eval = {"none": "#e74c3c", "random": "#f39c12", "llm_guided": "#2ecc71"}
        bar_colors_cache = {"none": "rgba(192,57,43,0.5)", "random": "rgba(230,126,34,0.5)", "llm_guided": "rgba(39,174,96,0.5)"}
        for strat, rt_hist in all_runtime_histories.items():
            if rt_hist:
                rt_df = pd.DataFrame(rt_hist)
                sname = strategy_names.get(strat, strat)
                # Cumulative evaluated (running total of fresh evaluations)
                cumulative_eval = rt_df["candidates_evaluated"].cumsum()
                step_fig.add_trace(go.Scatter(
                    x=rt_df["generation"], y=cumulative_eval,
                    mode="lines+markers", name=f"{sname} (cumulative evaluated)",
                    line=dict(color=bar_colors_eval.get(strat, "#999"), width=2.5),
                    marker=dict(size=7),
                ))
                # Per-generation cache hits as bars
                step_fig.add_trace(go.Bar(
                    x=rt_df["generation"], y=rt_df["candidates_cached"],
                    name=f"{sname} (cached)",
                    marker_color=bar_colors_cache.get(strat, "#99999980"),
                    opacity=0.6,
                ))
        steps_layout = {k: v for k, v in CHART_LAYOUT.items() if k != "yaxis"}
        step_fig.update_layout(
            title="Cumulative Candidates Evaluated & Cache Hits per Generation",
            xaxis_title="Generation",
            yaxis=dict(title="Candidates"),
            barmode="group",
            height=420,
            **steps_layout,
        )
        st.plotly_chart(step_fig, width="stretch")

        render_section_header("Export", "&#x1F4E5;", "Download results and charts")
        col1, col2 = st.columns(2)

        # Build CSV with runtime data included
        comp_rows = []
        for s, hist in all_histories.items():
            rt_hist = all_runtime_histories.get(s, [])
            for i, h in enumerate(hist):
                row = {
                    "generation": h["generation"],
                    "strategy": strategy_names.get(s, s),
                    "best_fitness": h["best"],
                    "avg_fitness": h["avg"],
                }
                if i < len(rt_hist):
                    row["gen_time_sec"] = rt_hist[i].get("gen_time_sec", "")
                    row["gen_time_ms"] = rt_hist[i].get("gen_time_ms", "")
                    row["best_eval_time_ms"] = rt_hist[i].get("best_eval_time", "")
                    row["best_exec_time_ms"] = rt_hist[i].get("best_exec_time", "")
                    row["candidates_generated"] = rt_hist[i].get("candidates_generated", "")
                    row["candidates_evaluated"] = rt_hist[i].get("candidates_evaluated", "")
                    row["candidates_cached"] = rt_hist[i].get("candidates_cached", "")
                    row["candidates_selected"] = rt_hist[i].get("candidates_selected", "")
                    row["best_estimated_time_complexity"] = rt_hist[i].get("best_estimated_time_complexity", "")
                    row["best_generalized_time_complexity"] = rt_hist[i].get("best_generalized_time_complexity", "")
                comp_rows.append(row)
        comp_df = pd.DataFrame(comp_rows)
        col1.download_button(
            "Download CSV",
            comp_df.to_csv(index=False),
            "comparison_results.csv",
            "text/csv",
            width="stretch",
        )

        try:
            png_bytes = comp_fig.to_image(format="png", width=1400, height=700)
            col2.download_button("Download Chart", png_bytes,
                                 "comparison_chart.png", "image/png",
                                 width="stretch")
        except Exception:
            col2.info("Install kaleido for PNG export: `pip install kaleido`")

        if all_best_codes:
            render_section_header("Best Solutions per Strategy", "&#x1F4BB;")
            for strat, code in all_best_codes.items():
                badge_cls = {"none": "badge-none", "random": "badge-random", "llm_guided": "badge-llm"}
                with st.expander(f"{strategy_names.get(strat, strat)}"):
                    st.code(code, language="python")

        comparison_analysis = build_comparison_analysis(comp_df)
        if comparison_analysis:
            render_section_header("Comparison Analysis", "&#x1F50D;")
            bullets = "".join(f"<li>{line}</li>" for line in comparison_analysis)
            st.markdown(f'<div class="analysis-card"><ul>{bullets}</ul></div>', unsafe_allow_html=True)

    else:
        # ── Single strategy run ─────────────────────────────────────────
        config = RunConfig(
            problem_type=problem_type,
            problem_description=problem_desc,
            initial_code=initial_code,
            mutation_strategy=strategy,
            num_generations=effective_num_generations,
            population_size=population_size,
            top_k=top_k,
            fitness_weights=(w1, w2, w3),
            openai_api_key=api_key,
            pacman_layouts=pacman_layouts,
        )

        # status row
        c1, c2, c3 = st.columns(3)
        status_placeholder = c1.empty()
        gen_counter = c2.empty()
        best_score = c3.empty()

        chart_placeholder = st.empty()
        runtime_chart_placeholder = st.empty()
        steps_chart_placeholder = st.empty()

        log_expander = st.expander("Operation Log", expanded=False, icon=":material/terminal:")
        details_expander = st.expander("Generation Details", expanded=False, icon=":material/groups:")

        vs = VectorStore()
        vs.clear()

        result, history, runtime_history, candidates, logs = run_single_evolution(
            config, vs, chart_placeholder, status_placeholder,
            gen_counter, best_score, log_expander, details_expander,
            runtime_chart_placeholder, steps_chart_placeholder,
            chart_key_prefix=f"single_{config.problem_type}_{config.mutation_strategy}"
        )

        if result:
            st.markdown("---")

            # result summary
            total_time_ms = sum(r.get("gen_time_ms", 0) for r in runtime_history) if runtime_history else 0
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                render_stat_card("Best Fitness", f"{result.best_overall.fitness:.4f}", "green")
            with r2:
                render_stat_card("Generations", str(len(history)), "")
            with r3:
                render_stat_card("Candidates Tested", str(len(candidates)), "orange")
            with r4:
                render_stat_card("Total Runtime", format_duration_ms(total_time_ms), "")

            render_section_header("Best Solution Found", "&#x1F31F;", f"Fitness: {result.best_overall.fitness:.4f}")
            st.markdown('<div class="best-solution-box"><div class="sol-label">Champion Code</div></div>', unsafe_allow_html=True)
            st.code(result.best_overall.code, language="python")

            best_breakdown = result.best_overall.fitness_breakdown
            complexity_lines = []
            if best_breakdown.get("estimated_time_complexity"):
                complexity_lines.append(f"Estimated complexity: `{best_breakdown['estimated_time_complexity']}`")
            if best_breakdown.get("generalized_time_complexity"):
                complexity_lines.append(f"Generalized pattern: `{best_breakdown['generalized_time_complexity']}`")
            if complexity_lines:
                st.caption(" | ".join(complexity_lines))

            render_section_header("Export", "&#x1F4E5;", "Download results and charts")
            col1, col2, col3 = st.columns(3)

            results_df = pd.DataFrame([{
                "generation": c.generation,
                "candidate_id": c.code_hash[:8],
                "fitness_score": c.fitness,
                "mutation_type": c.mutation_type,
                "mutation_description": c.mutation_description,
                **{k: v for k, v in c.fitness_breakdown.items() if not isinstance(v, list)},
            } for c in candidates])

            generation_summary_df = build_generation_summary_df(history, runtime_history)

            col1.download_button(
                "Download Candidate CSV",
                results_df.to_csv(index=False),
                "evolution_results.csv",
                "text/csv",
                width="stretch",
            )

            col2.download_button(
                "Download Captured Data CSV",
                generation_summary_df.to_csv(index=False),
                "generation_summary.csv",
                "text/csv",
                width="stretch",
            )

            try:
                fig = build_fitness_chart(history)
                png_bytes = fig.to_image(format="png", width=1400, height=700)
                col3.download_button("Download Chart", png_bytes,
                                     "fitness_chart.png", "image/png",
                                     width="stretch")
            except Exception:
                col3.info("Install kaleido for PNG export: `pip install kaleido`")

            render_section_header("Captured Data", "&#x1F4CB;", "Generation-by-generation breakdown")
            st.markdown('<div class="styled-table-wrap"><div class="table-title">Generation Summary</div></div>', unsafe_allow_html=True)
            st.dataframe(generation_summary_df, width="stretch", hide_index=True)

            single_run_analysis = build_single_run_analysis(generation_summary_df)
            if single_run_analysis:
                render_section_header("Run Analysis", "&#x1F50D;")
                bullets = "".join(f"<li>{line}</li>" for line in single_run_analysis)
                st.markdown(f'<div class="analysis-card"><ul>{bullets}</ul></div>', unsafe_allow_html=True)

else:
    # ── Welcome screen ──────────────────────────────────────────────────
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3>&#x1F9EC; Evolutionary Search</h3>
            <p>Generate candidate solutions through mutation, evaluate them with a fitness function, and let natural selection do its thing over multiple generations.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3>&#x1F9E0; LLM-Guided Mutations</h3>
            <p>Instead of random changes, GPT-4o-mini reads the code, understands it, and suggests targeted improvements. Combined with RAG for even better results.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <h3>&#x2696;&#xFE0F; Compare Strategies</h3>
            <p>Run all three mutation strategies side by side and see which one wins. Export charts and data for your analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    render_section_header("Available Problems", "&#x1F3AF;")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="feature-card">
            <h3>&#x1F47E; Pac-Man Agent</h3>
            <p>Evolve a game-playing agent for the UC Berkeley CS188 framework. Start with a simple greedy agent and watch it improve over generations.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="feature-card">
            <h3>&#x1F522; Matrix Multiplication</h3>
            <p>Optimize 3x3 matrix multiplication to use fewer arithmetic operations. The standard approach uses 27 multiplications &mdash; can evolution do better?</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    st.markdown("""
    <div style="text-align:center; color:rgba(255,255,255,0.4); font-size:0.88rem; padding:1rem 0;">
        Configure parameters in the sidebar and click <strong style="color:#667eea;">Start Evolution</strong> to begin.
    </div>
    """, unsafe_allow_html=True)
