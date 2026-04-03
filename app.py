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
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>Evolve</h1>
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
        st.caption("fitness = w1 * correctness + w2 * (1/(ops+1))")

    col1, col2, col3 = st.columns(3)
    w1 = col1.number_input("w1", 0.0, 1.0, 0.5, 0.05)
    w2 = col2.number_input("w2", 0.0, 1.0, 0.3, 0.05)
    w3 = col3.number_input("w3", 0.0, 1.0, 0.2, 0.05)

    weights_valid = abs(w1 + w2 + w3 - 1.0) < 0.01
    if not weights_valid:
        st.error(f"Weights must sum to 1.0 (currently {w1+w2+w3:.2f})")

    api_key_needed = strategy == "llm_guided" and not api_key
    if api_key_needed:
        st.warning("API key required for LLM-Guided strategy")

    st.divider()
    run_comparison = st.checkbox("Run Comparison Experiment",
                                 help="Run all three strategies and compare results")

    start_disabled = not weights_valid or api_key_needed
    start_button = st.button("Start Evolution", type="primary",
                             disabled=start_disabled, use_container_width=True)


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


def build_runtime_chart(history: list[dict], title: str = "Runtime per Generation") -> go.Figure:
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["generation"], y=df["gen_time"],
        name="Generation Time (s)",
        marker_color="#667eea",
        opacity=0.7,
    ))
    if "best_eval_time" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["generation"], y=df["best_eval_time"],
            mode="lines+markers", name="Best Candidate Eval (ms)",
            line=dict(color="#2ecc71", width=2.5),
            marker=dict(size=6),
            yaxis="y2",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Generation",
        yaxis=dict(title="Generation Time (s)", side="left"),
        yaxis2=dict(title="Eval Time (ms)", side="right", overlaying="y",
                    gridcolor="rgba(255,255,255,0.03)"),
        height=380,
        **CHART_LAYOUT,
    )
    return fig


def render_stat_card(label: str, value: str, color: str = ""):
    color_class = f" {color}" if color else ""
    st.markdown(f"""
    <div class="stat-card">
        <div class="label">{label}</div>
        <div class="value{color_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def run_single_evolution(config: RunConfig, vector_store: VectorStore,
                         chart_placeholder, status_placeholder,
                         gen_counter, best_score, log_container, details_container,
                         runtime_chart_placeholder=None):
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

        gen_counter.markdown(f"**Generation {gen_result.generation_num} / {config.num_generations}**")
        best_score.metric("Best Fitness", f"{gen_result.best_overall.fitness:.4f}",
                          delta=f"{gen_result.stats.get('max_fitness', 0) - gen_result.stats.get('avg_fitness', 0):.4f}")

        fitness_history.append({
            "generation": gen_result.generation_num,
            "best": gen_result.stats["max_fitness"],
            "avg": gen_result.stats["avg_fitness"],
            "worst": gen_result.stats["min_fitness"],
        })

        runtime_history.append({
            "generation": gen_result.generation_num,
            "gen_time": gen_result.stats.get("gen_time_sec", 0),
            "best_eval_time": gen_result.stats.get("best_eval_time_ms", 0),
            "best_exec_time": gen_result.stats.get("best_exec_time_ms", 0),
            "avg_eval_time": gen_result.stats.get("avg_eval_time_ms", 0),
        })

        chart_placeholder.plotly_chart(
            build_fitness_chart(fitness_history),
            use_container_width=True,
        )

        if runtime_chart_placeholder:
            runtime_chart_placeholder.plotly_chart(
                build_runtime_chart(runtime_history),
                use_container_width=True,
            )

        with log_container:
            for entry in gen_result.log_entries:
                st.text(entry)

        with details_container:
            rows = []
            for c in gen_result.candidates:
                eval_ms = c.fitness_breakdown.get("eval_time_ms", "")
                rows.append({
                    "Hash": c.code_hash[:8],
                    "Mutation": c.mutation_type,
                    "Fitness": f"{c.fitness:.4f}" if c.fitness is not None else "N/A",
                    "Eval (ms)": f"{eval_ms:.1f}" if isinstance(eval_ms, (int, float)) else "",
                    "Description": c.mutation_description[:70],
                    "Selected": "Yes" if c in gen_result.selected else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        all_candidates.extend(gen_result.candidates)

    status_placeholder.success("Complete!")
    return final_result, fitness_history, runtime_history, all_candidates, controller.log_entries


# ── Pseudocode-to-Python conversion helper ──────────────────────────────────

def convert_pseudocode_to_python(description: str, problem_type: str, api_key: str) -> str:
    """Use the LLM to convert pseudocode/description into Python code."""
    client = LLMClient(api_key)
    if problem_type == "pacman":
        system = (
            "You are a Python programmer. Convert the following algorithm description "
            "into Python code for a Pac-Man agent's get_action method body. "
            "Available: state.get_legal_actions(), state.get_pacman_position(), "
            "state.get_food().as_list(), state.get_ghost_positions(), "
            "state.generate_pacman_successor(action). Return ONLY the function body code."
        )
    else:
        system = (
            "You are a Python programmer. Convert the following algorithm description "
            "into Python code for a matrix_multiply(A, B) function body that multiplies "
            "two 3x3 matrices (lists of lists). Return ONLY the function body code."
        )
    return client.generate_code(system, f"Algorithm description:\n{description}")


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

    if run_comparison:
        # ── Comparison experiment ───────────────────────────────────────
        st.markdown("### Comparison Experiment")
        st.caption("Running all three mutation strategies back-to-back")

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
                num_generations=num_generations if strat != "none" else 1,
                population_size=population_size,
                top_k=top_k,
                fitness_weights=(w1, w2, w3),
                openai_api_key=api_key,
            )

            vs = VectorStore(persist_dir=f"./data/chromadb_{strat}")
            vs.clear()

            with st.expander(f"{strategy_names[strat]}", expanded=False):
                chart_ph = st.empty()
                runtime_ph = st.empty()
                status_ph = st.empty()
                gen_ph = st.empty()
                best_ph = st.empty()
                log_ph = st.container()
                det_ph = st.container()

                result, history, runtime_hist, candidates, logs = run_single_evolution(
                    config, vs, chart_ph, status_ph, gen_ph, best_ph, log_ph, det_ph, runtime_ph
                )

            all_histories[strat] = history
            all_runtime_histories[strat] = runtime_hist
            if result:
                all_best_codes[strat] = result.best_overall.code

        progress_bar.progress(1.0)
        comparison_status.success("All strategies complete!")

        st.markdown("---")
        st.markdown("### Results")
        comp_fig = build_comparison_chart(all_histories)
        st.plotly_chart(comp_fig, use_container_width=True)

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

        # Runtime comparison chart
        st.markdown("### Runtime per Generation")
        rt_fig = go.Figure()
        rt_colors = {"none": "#e74c3c", "random": "#f39c12", "llm_guided": "#2ecc71"}
        for strat, rt_hist in all_runtime_histories.items():
            if rt_hist:
                rt_df = pd.DataFrame(rt_hist)
                rt_fig.add_trace(go.Scatter(
                    x=rt_df["generation"], y=rt_df["gen_time"],
                    mode="lines+markers", name=strategy_names.get(strat, strat),
                    line=dict(color=rt_colors.get(strat, "#999"), width=2.5),
                    marker=dict(size=6),
                ))
        rt_fig.update_layout(
            title="Runtime Comparison -- Generation Time (seconds)",
            xaxis_title="Generation", yaxis_title="Time (s)",
            height=400, **CHART_LAYOUT,
        )
        st.plotly_chart(rt_fig, use_container_width=True)

        st.markdown("")
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
                    row["gen_time_sec"] = rt_hist[i].get("gen_time", "")
                    row["best_eval_time_ms"] = rt_hist[i].get("best_eval_time", "")
                    row["best_exec_time_ms"] = rt_hist[i].get("best_exec_time", "")
                comp_rows.append(row)
        comp_df = pd.DataFrame(comp_rows)
        col1.download_button(
            "Download CSV",
            comp_df.to_csv(index=False),
            "comparison_results.csv",
            "text/csv",
            use_container_width=True,
        )

        try:
            png_bytes = comp_fig.to_image(format="png", width=1400, height=700)
            col2.download_button("Download Chart", png_bytes,
                                 "comparison_chart.png", "image/png",
                                 use_container_width=True)
        except Exception:
            col2.info("Install kaleido for PNG export: `pip install kaleido`")

        if all_best_codes:
            st.markdown("### Best solutions per strategy")
            for strat, code in all_best_codes.items():
                with st.expander(f"{strategy_names.get(strat, strat)}"):
                    st.code(code, language="python")

    else:
        # ── Single strategy run ─────────────────────────────────────────
        config = RunConfig(
            problem_type=problem_type,
            problem_description=problem_desc,
            initial_code=initial_code,
            mutation_strategy=strategy,
            num_generations=num_generations,
            population_size=population_size,
            top_k=top_k,
            fitness_weights=(w1, w2, w3),
            openai_api_key=api_key,
        )

        # status row
        c1, c2, c3 = st.columns(3)
        status_placeholder = c1.empty()
        gen_counter = c2.empty()
        best_score = c3.empty()

        chart_placeholder = st.empty()
        runtime_chart_placeholder = st.empty()

        log_expander = st.expander("Operation Log", expanded=False)
        details_expander = st.expander("Generation Details", expanded=False)

        vs = VectorStore()
        vs.clear()

        result, history, runtime_history, candidates, logs = run_single_evolution(
            config, vs, chart_placeholder, status_placeholder,
            gen_counter, best_score, log_expander, details_expander,
            runtime_chart_placeholder,
        )

        if result:
            st.markdown("---")

            # result summary
            total_time = sum(r.get("gen_time", 0) for r in runtime_history) if runtime_history else 0
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                render_stat_card("Best Fitness", f"{result.best_overall.fitness:.4f}", "green")
            with r2:
                render_stat_card("Generations", str(len(history)), "")
            with r3:
                render_stat_card("Candidates Tested", str(len(candidates)), "orange")
            with r4:
                render_stat_card("Total Runtime", f"{total_time:.1f}s", "")

            st.markdown("")
            st.markdown("**Best solution found:**")
            st.code(result.best_overall.code, language="python")

            st.markdown("")
            col1, col2 = st.columns(2)

            results_df = pd.DataFrame([{
                "generation": c.generation,
                "candidate_id": c.code_hash[:8],
                "fitness_score": c.fitness,
                "mutation_type": c.mutation_type,
                "mutation_description": c.mutation_description,
                **{k: v for k, v in c.fitness_breakdown.items() if not isinstance(v, list)},
            } for c in candidates])

            col1.download_button(
                "Download CSV",
                results_df.to_csv(index=False),
                "evolution_results.csv",
                "text/csv",
                use_container_width=True,
            )

            try:
                fig = build_fitness_chart(history)
                png_bytes = fig.to_image(format="png", width=1400, height=700)
                col2.download_button("Download Chart", png_bytes,
                                     "fitness_chart.png", "image/png",
                                     use_container_width=True)
            except Exception:
                col2.info("Install kaleido for PNG export: `pip install kaleido`")

else:
    # ── Welcome screen ──────────────────────────────────────────────────
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3>Evolutionary Search</h3>
            <p>Generate candidate solutions through mutation, evaluate them with a fitness function, and let natural selection do its thing over multiple generations.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3>LLM-Guided Mutations</h3>
            <p>Instead of random changes, GPT-4o-mini reads the code, understands it, and suggests targeted improvements. Combined with RAG for even better results.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <h3>Compare Strategies</h3>
            <p>Run all three mutation strategies side by side and see which one wins. Export charts and data for your analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("""
        <div class="feature-card">
            <h3>Pac-Man Agent</h3>
            <p>Evolve a game-playing agent for the UC Berkeley CS188 framework. Start with a simple greedy agent and watch it improve over generations.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="feature-card">
            <h3>Matrix Multiplication</h3>
            <p>Optimize 3x3 matrix multiplication to use fewer arithmetic operations. The standard approach uses 27 multiplications -- can evolution do better?</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    st.caption("Configure parameters in the sidebar and click Start Evolution to begin.")
