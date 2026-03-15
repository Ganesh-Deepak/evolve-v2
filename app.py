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

st.set_page_config(
    page_title="Evolve - AI-Powered Evolutionary Code Improvement",
    page_icon="🧬",
    layout="wide",
)

st.title("Evolve: AI-Powered Evolutionary Code Improvement")
st.caption("CS5381 – Analysis of Algorithms | Evolutionary Agent for Algorithm Discovery")

# --- Default code templates ---
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

# --- Sidebar: Input Panel ---
with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input("OpenAI API Key", type="password",
                            help="Required for LLM-Guided Mutation strategy")

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

    initial_code = st.text_area(
        "Initial Code",
        value=DEFAULT_PACMAN_CODE if problem_type == "pacman" else DEFAULT_MATRIX_CODE,
        height=250,
        help="The starting code that will be evolved",
    )

    st.subheader("Evolution Parameters")

    num_generations = st.slider("Number of Generations", 1, 50, 10)
    population_size = st.slider("Population Size", 2, 20, 5)
    top_k = st.slider("Top-K Selection", 1, 10, 3)

    strategy_display = st.selectbox(
        "Mutation Strategy",
        ["No Evolution (Baseline)", "Random Mutation", "LLM-Guided Mutation"]
    )
    strategy_map = {
        "No Evolution (Baseline)": "none",
        "Random Mutation": "random",
        "LLM-Guided Mutation": "llm_guided",
    }
    strategy = strategy_map[strategy_display]

    st.subheader("Fitness Weights")
    if problem_type == "pacman":
        st.caption("Fitness = w1 × avg_score + w2 × max_score")
    else:
        st.caption("Fitness = w1 × correctness + w2 × (1/(ops+1))")

    col1, col2, col3 = st.columns(3)
    w1 = col1.number_input("w1", 0.0, 1.0, 0.5, 0.05)
    w2 = col2.number_input("w2", 0.0, 1.0, 0.3, 0.05)
    w3 = col3.number_input("w3", 0.0, 1.0, 0.2, 0.05)

    weights_valid = abs(w1 + w2 + w3 - 1.0) < 0.01
    if not weights_valid:
        st.error(f"Weights must sum to 1.0 (current: {w1+w2+w3:.2f})")

    api_key_needed = strategy == "llm_guided" and not api_key
    if api_key_needed:
        st.warning("API key required for LLM-Guided strategy")

    st.divider()
    run_comparison = st.checkbox("Run 3-Strategy Comparison Experiment",
                                 help="Runs all three strategies and produces a comparison chart")

    start_disabled = not weights_valid or api_key_needed
    start_button = st.button("Start Evolution", type="primary",
                             disabled=start_disabled, use_container_width=True)


def load_templates(problem_type: str) -> list[tuple[str, str]]:
    templates = []
    template_dir = Path("./templates")
    prefix = "pacman_" if problem_type == "pacman" else "matrix_"
    for f in template_dir.glob(f"{prefix}*.py"):
        code = f.read_text(encoding="utf-8")
        templates.append((code, f.stem))
    return templates


def build_fitness_chart(history: list[dict], title: str = "Fitness Progression") -> go.Figure:
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["generation"], y=df["best"],
                             mode="lines+markers", name="Best", line=dict(color="#2ecc71", width=2)))
    fig.add_trace(go.Scatter(x=df["generation"], y=df["avg"],
                             mode="lines+markers", name="Average", line=dict(color="#3498db", width=2)))
    if "worst" in df.columns:
        fig.add_trace(go.Scatter(x=df["generation"], y=df["worst"],
                                 mode="lines", name="Worst",
                                 line=dict(color="#e74c3c", width=1, dash="dash")))
    fig.update_layout(
        title=title, xaxis_title="Generation", yaxis_title="Fitness Score",
        template="plotly_white", height=400,
    )
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
            line=dict(color=colors.get(strat, "#999"), width=2),
        ))
    fig.update_layout(
        title="Strategy Comparison: Best Fitness per Generation",
        xaxis_title="Generation", yaxis_title="Best Fitness Score",
        template="plotly_white", height=450,
    )
    return fig


def run_single_evolution(config: RunConfig, vector_store: VectorStore,
                         chart_placeholder, status_placeholder,
                         gen_counter, best_score, log_container, details_container):
    controller = EvolutionController(config, vector_store)

    templates = load_templates(config.problem_type)
    vector_store.seed_templates(templates)

    fitness_history = []
    all_candidates = []
    final_result = None

    status_placeholder.info("Running...")

    for gen_result in controller.run_evolution():
        final_result = gen_result

        gen_counter.markdown(f"### Generation {gen_result.generation_num} / {config.num_generations}")
        best_score.metric("Best Fitness", f"{gen_result.best_overall.fitness:.4f}",
                          delta=f"{gen_result.stats.get('max_fitness', 0) - gen_result.stats.get('avg_fitness', 0):.4f}")

        fitness_history.append({
            "generation": gen_result.generation_num,
            "best": gen_result.stats["max_fitness"],
            "avg": gen_result.stats["avg_fitness"],
            "worst": gen_result.stats["min_fitness"],
        })

        chart_placeholder.plotly_chart(
            build_fitness_chart(fitness_history),
            use_container_width=True,
        )

        with log_container:
            for entry in gen_result.log_entries:
                st.text(entry)

        with details_container:
            rows = []
            for c in gen_result.candidates:
                rows.append({
                    "Hash": c.code_hash[:8],
                    "Mutation": c.mutation_type,
                    "Fitness": f"{c.fitness:.4f}" if c.fitness is not None else "N/A",
                    "Description": c.mutation_description[:80],
                    "Selected": "Yes" if c in gen_result.selected else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        all_candidates.extend(gen_result.candidates)

    status_placeholder.success("Evolution Complete!")
    return final_result, fitness_history, all_candidates, controller.log_entries


# --- Main Content Area ---
if start_button:
    if run_comparison:
        st.header("3-Strategy Comparison Experiment")

        strategies_to_run = ["none", "random", "llm_guided"]
        strategy_names = {"none": "No Evolution", "random": "Random Mutation", "llm_guided": "LLM-Guided"}

        all_histories = {}
        all_best_codes = {}

        progress_bar = st.progress(0)
        comparison_status = st.empty()

        for idx, strat in enumerate(strategies_to_run):
            if strat == "llm_guided" and not api_key:
                comparison_status.warning("Skipping LLM-Guided (no API key)")
                continue

            comparison_status.info(f"Running strategy {idx+1}/3: {strategy_names[strat]}...")
            progress_bar.progress((idx) / 3)

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

            with st.expander(f"{strategy_names[strat]} - Details", expanded=False):
                chart_ph = st.empty()
                status_ph = st.empty()
                gen_ph = st.empty()
                best_ph = st.empty()
                log_ph = st.container()
                det_ph = st.container()

                result, history, candidates, logs = run_single_evolution(
                    config, vs, chart_ph, status_ph, gen_ph, best_ph, log_ph, det_ph
                )

            all_histories[strat] = history
            if result:
                all_best_codes[strat] = result.best_overall.code

        progress_bar.progress(1.0)
        comparison_status.success("Comparison complete!")

        st.subheader("Comparison Chart")
        comp_fig = build_comparison_chart(all_histories)
        st.plotly_chart(comp_fig, use_container_width=True)

        col1, col2 = st.columns(2)
        comp_df = pd.DataFrame([
            {"generation": h["generation"], "strategy": strategy_names.get(s, s), "best_fitness": h["best"], "avg_fitness": h["avg"]}
            for s, hist in all_histories.items() for h in hist
        ])
        col1.download_button(
            "Download Comparison CSV",
            comp_df.to_csv(index=False),
            "comparison_results.csv",
            "text/csv",
        )

        try:
            png_bytes = comp_fig.to_image(format="png", width=1200, height=600)
            col2.download_button("Download Comparison PNG", png_bytes,
                                 "comparison_chart.png", "image/png")
        except Exception:
            col2.info("Install kaleido for PNG export: pip install kaleido")

        if all_best_codes:
            st.subheader("Best Solutions per Strategy")
            for strat, code in all_best_codes.items():
                with st.expander(f"Best from {strategy_names.get(strat, strat)}"):
                    st.code(code, language="python")

    else:
        # --- Single Strategy Run ---
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

        col_status, col_gen, col_best = st.columns(3)
        status_placeholder = col_status.empty()
        gen_counter = col_gen.empty()
        best_score = col_best.empty()

        chart_placeholder = st.empty()

        log_expander = st.expander("Operation Log", expanded=False)
        details_expander = st.expander("Generation Details", expanded=False)

        vs = VectorStore()
        vs.clear()

        result, history, candidates, logs = run_single_evolution(
            config, vs, chart_placeholder, status_placeholder,
            gen_counter, best_score, log_expander, details_expander,
        )

        if result:
            st.divider()
            st.subheader("Best Solution Found")
            st.metric("Final Best Fitness", f"{result.best_overall.fitness:.4f}")
            st.code(result.best_overall.code, language="python")

            st.divider()
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
                "Download Results CSV",
                results_df.to_csv(index=False),
                "evolution_results.csv",
                "text/csv",
            )

            try:
                fig = build_fitness_chart(history)
                png_bytes = fig.to_image(format="png", width=1200, height=600)
                col2.download_button("Download Chart PNG", png_bytes,
                                     "fitness_chart.png", "image/png")
            except Exception:
                col2.info("Install kaleido for PNG export: pip install kaleido")

else:
    # --- Welcome screen ---
    st.markdown("""
    ### How It Works
    1. **Configure** your evolution parameters in the sidebar
    2. **Choose** a mutation strategy:
       - **No Evolution**: Single LLM call baseline (control)
       - **Random Mutation**: Programmatic code mutations (no LLM)
       - **LLM-Guided Mutation**: GPT-4o-mini with retrieval-augmented prompting
    3. **Start** the evolution and watch fitness improve in real-time
    4. **Compare** all three strategies with the comparison experiment mode
    5. **Export** results as CSV and charts as PNG for your report

    ### Supported Problems
    - **Pac-Man Agent**: Evolve a game-playing agent for the UC Berkeley CS188 framework
    - **Matrix Multiplication**: Optimize 3x3 matrix multiplication to minimize operations

    ---
    *Configure parameters in the sidebar and click "Start Evolution" to begin.*
    """)
