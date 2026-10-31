"""
Indosat Ooredoo Hutchison (IOH) — Customer Churn, Survival Hazard & Agentic Retention Intelligence
Interactive Streamlit Application (`streamlit_app.py`)

Powered by:
  - LangChain `ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=1.0)` (`langchain-google-genai`)
  - LangGraph 6-Node `StateGraph(RetentionAgentState)` with Deterministic Policy Guardrails
  - Dual-Horizon Calibrated Survival & Acute Ticket Hazard Models (`StratifiedGroupKFold(id % 200)`)

SECURITY & ACCESS CONTROL:
  This application can ONLY be accessed by supplying the valid `API_KEY` (stored in `.env`)
  via the `apikey` URL query parameter:
      https://example-url.streamlit.app?apikey=API_KEY
      http://localhost:8501/?apikey=API_KEY
"""

from __future__ import annotations

import hmac
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.data_pipeline import (
    build_point_in_time_feature_matrix,
    compute_modulo200_duplication_audit,
    compute_tenure_hazard_and_survival_table,
    load_and_clean_tables,
)
from src.decision_engine import (
    compute_customer_economics_and_decisions,
    summarize_actionable_active_vs_retrospective,
)
from src.models import train_and_evaluate_dual_horizon_models
from src.retention_agent import (
    compute_experiment_power_analysis,
    get_prioritized_active_watchlist,
    invoke_interactive_retention_chat,
    run_langgraph_for_customer_row,
    run_live_gemini_agent_for_customer,
    summarize_cohort_rct_holdout,
)

# ==============================================================================
# 0. Page Configuration & Strict URL Query-Parameter API Key Authentication
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "figures"

load_dotenv(dotenv_path=ENV_PATH, override=True)

st.set_page_config(
    page_title="IOH Retention Intelligence & Gemini 3.8 Flash Co-Pilot",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_url_apikey_param() -> str:
    """Extract the `apikey` query parameter from the request URL (`?apikey=API_KEY`)."""
    try:
        qp = st.query_params
        val = qp.get("apikey", "")
        if isinstance(val, list):
            return str(val[0]).strip() if val else ""
        return str(val).strip()
    except Exception:
        return ""


def _get_secret_or_env(key: str, default: str = "") -> str:
    """Read a configuration key from `.env` / `os.environ` or Streamlit Cloud `st.secrets`."""
    val = os.getenv(key, "").strip()
    if val:
        return val
    try:
        if key in st.secrets:
            s_val = st.secrets[key]
            os.environ[key] = str(s_val)
            return str(s_val).strip()
    except Exception:
        pass
    return default


def enforce_url_api_key_authentication() -> None:
    """
    Strictly enforce that the Streamlit app can ONLY be accessed when `?apikey=API_KEY`
    is present in the URL and matches `API_KEY` in `.env` or `st.secrets` using constant-time comparison.
    """
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    # Sync top-level Streamlit Cloud secrets into os.environ for LangChain / Google GenAI
    for env_key in (
        "API_KEY",
        "GEMINI_MODEL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        _get_secret_or_env(env_key)

    expected_key = _get_secret_or_env("API_KEY")
    provided_key = _get_url_apikey_param()

    if not expected_key:
        st.error(
            "🔒 **Server Configuration Error**: `API_KEY` is not configured in `.env`. "
            "Please set `API_KEY=...` in `.env` before accessing the application."
        )
        st.stop()

    is_valid = bool(provided_key) and hmac.compare_digest(
        provided_key.encode("utf-8"),
        expected_key.encode("utf-8"),
    )

    if not is_valid:
        st.error(
            "🔒 **Unauthorized Access (401)**: Valid `apikey` URL query parameter is required."
        )
        st.markdown(
            """
            ### Access Restricted to URL Query-Parameter Authentication
            This application can **only** be accessed by appending the authorized `API_KEY`
            directly to the application URL query string:

            ```text
            https://example-url.streamlit.app?apikey=API_KEY
            http://localhost:8501/?apikey=API_KEY
            ```

            *Verify the `API_KEY` value stored in your project `.env` file and reload the URL with `?apikey=<API_KEY>`.*
            """
        )
        st.stop()


# Enforce URL `?apikey=API_KEY` gate immediately before loading any data or UI components
enforce_url_api_key_authentication()


# ==============================================================================
# 1. Cached Data & Artifact Loaders
# ==============================================================================


@st.cache_data(show_spinner="Loading IOH Customer Retention Intelligence artifacts...")
def load_platform_data() -> Dict[str, Any]:
    """Load precomputed CSV/JSON artifacts from `outputs/` or run pipeline on-the-fly if absent."""
    decisions_path = OUTPUTS_DIR / "retention_decisions_all_719.csv"
    watchlist_path = OUTPUTS_DIR / "active_watchlist_prioritized.csv"
    scope_path = OUTPUTS_DIR / "active_vs_retrospective_campaign_summary.csv"

    if not (decisions_path.exists() and watchlist_path.exists()):
        cb_clean, su_clean, si_clean, _, _ = load_and_clean_tables(BASE_DIR)
        pit_df = build_point_in_time_feature_matrix(cb_clean, su_clean, si_clean)
        scored_df, eval_df, _ = train_and_evaluate_dual_horizon_models(pit_df)
        df_scored = compute_customer_economics_and_decisions(scored_df)
        watchlist_df = get_prioritized_active_watchlist(df_scored, top_n=25)
        scope_df = summarize_actionable_active_vs_retrospective(df_scored)
        rct_summary_df = summarize_cohort_rct_holdout(df_scored)
        power_df = compute_experiment_power_analysis()
        survival_df = compute_tenure_hazard_and_survival_table(pit_df)
        mod200_df = compute_modulo200_duplication_audit(pit_df)
    else:
        df_scored = pd.read_csv(decisions_path)
        watchlist_df = pd.read_csv(watchlist_path)
        scope_df = (
            pd.read_csv(scope_path)
            if scope_path.exists()
            else summarize_actionable_active_vs_retrospective(df_scored)
        )
        rct_path = OUTPUTS_DIR / "rct_holdout_summary.csv"
        power_path = OUTPUTS_DIR / "rct_power_analysis.csv"
        surv_path = OUTPUTS_DIR / "tenure_survival_hazard_table.csv"
        mod_path = OUTPUTS_DIR / "modulo200_duplication_audit.csv"
        eval_path = OUTPUTS_DIR / "model_evaluation_metrics.csv"

        rct_summary_df = (
            pd.read_csv(rct_path)
            if rct_path.exists()
            else summarize_cohort_rct_holdout(df_scored)
        )
        power_df = (
            pd.read_csv(power_path)
            if power_path.exists()
            else compute_experiment_power_analysis()
        )
        survival_df = pd.read_csv(surv_path) if surv_path.exists() else pd.DataFrame()
        mod200_df = pd.read_csv(mod_path) if mod_path.exists() else pd.DataFrame()
        eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()

    def _read_optional_csv(filename: str) -> pd.DataFrame:
        p = OUTPUTS_DIR / filename
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    return {
        "df": df_scored,
        "watchlist": watchlist_df,
        "scope_summary": scope_df,
        "rct_summary": rct_summary_df,
        "power_analysis": power_df,
        "survival_table": survival_df,
        "mod200_audit": mod200_df,
        "model_metrics": eval_df,
        "policy_comparison": _read_optional_csv("economic_policy_comparison.csv"),
        "sensitivity_grid": _read_optional_csv("uplift_ltv_sensitivity_matrix.csv"),
        "pareto_frontier": _read_optional_csv("budget_pareto_frontier.csv"),
        "fairness_audit": _read_optional_csv("fairness_and_parity_audit.csv"),
    }


data_bundle = load_platform_data()
df_all: pd.DataFrame = data_bundle["df"]
df_watchlist: pd.DataFrame = data_bundle["watchlist"]


# ==============================================================================
# 2. Sidebar — Gemini 3.8 Flash Configuration & Platform Telemetry
# ==============================================================================

with st.sidebar:
    st.markdown("## 📡 IOH Retention Intelligence")
    st.caption(
        "Dual-Horizon Survival Hazard + Economic Uplift + LangGraph Agentic Co-Pilot"
    )
    st.success("🔐 Authenticated via URL `?apikey=...`")

    st.markdown("---")
    st.markdown("### 🤖 `ChatGoogleGenerativeAI` Settings")
    default_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    selected_model = st.text_input(
        "Gemini Model ID",
        value=default_model,
        help="Default: gemini-3.8-flash via langchain-google-genai",
    )
    temperature_val = st.slider(
        "Temperature (Gemini 3.0+ Best Practice = 1.0)",
        min_value=0.0,
        max_value=1.5,
        value=1.0,
        step=0.1,
        help="Per Google GenAI docs, Gemini 3.0+ defaults to temperature=1.0 to avoid reasoning degradation.",
    )
    thinking_level_val = st.selectbox(
        "Thinking Level (`thinking_level`)",
        options=["low", "minimal", "medium", "high"],
        index=0,
        help="Controls internal reasoning budget for Gemini 3+ models.",
    )

    vertex_flag = _get_secret_or_env("GOOGLE_GENAI_USE_VERTEXAI", "true")
    gcp_project = _get_secret_or_env("GOOGLE_CLOUD_PROJECT", "not-configured")
    gcp_location = _get_secret_or_env("GOOGLE_CLOUD_LOCATION", "global")
    st.markdown(
        f"""
        **Backend Configuration (`.env`)**
        - **Package**: `langchain-google-genai`
        - **Class**: `ChatGoogleGenerativeAI`
        - **Vertex AI Backend**: `{vertex_flag}`
        - **GCP Project**: `{gcp_project}`
        - **Location**: `{gcp_location}`
        """
    )

    st.markdown("---")
    st.markdown("### 📊 Cohort Snapshot")
    n_total = len(df_all)
    n_active = int((df_all["is_churn"] == 0).sum())
    n_churned = int((df_all["is_churn"] == 1).sum())
    st.metric("Total Audited Customers", f"{n_total:,}", "519 base + 200 mod-200 clones")
    st.metric("Currently Active (Live Targetable)", f"{n_active:,}", "80.0% of cohort")
    st.metric("Historical Churned (Backtest)", f"{n_churned:,}", "20.0% overall churn")


# ==============================================================================
# 3. Main Header & Navigation Tabs
# ==============================================================================

st.title(
    "📡 Indosat Ooredoo Hutchison — Customer Churn, Survival Hazard & Agentic Retention Platform"
)
st.markdown(
    "**Dual-Horizon Leak-Free Survival Architecture** (`StratifiedGroupKFold(id % 200)`) • "
    "**ARPU-Scaled Economic Optimizer** • **LangGraph + `ChatGoogleGenerativeAI(model=\"gemini-3.8-flash\")` Co-Pilot**"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🎯 1. Executive Command Center & Watchlist",
        "🔍 2. Forensic Hygiene & Survival Hazard",
        "📈 3. Model Validation & Economic Simulator",
        "🤖 4. Live Gemini 3.8 Flash + LangGraph Co-Pilot",
        "⚖️ 5. Fairness Audit & RCT Power Calculator",
    ]
)


# ==============================================================================
# TAB 1: Monday Executive Command Center & Active Watchlist
# ==============================================================================
with tab1:
    st.subheader(
        "Monday Morning Executive Command Center: Live Active Campaign vs. Retrospective Backtest"
    )
    st.info(
        "**Scope Separation Guarantee**: Historical churned customers (`is_churn == 1`, $N = 145$) "
        "cannot be saved retroactively. All operational budgets, Monday-morning specialist queues, "
        "and live ROI projections below are strictly separated between the **Live Active Campaign ($N = 574$)** "
        "and the **Retrospective Full-Cohort Backtest ($N = 719$)**."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric(
        "Live Active Actioned Accounts",
        "22 Accounts",
        "21 Commercial + 1 Billing Audit (IOH-553)",
    )
    col_b.metric(
        "Active Commercial Subsidy Spend",
        "Rp 9,150,000",
        "20 Premium (Rp 450k) + 1 Basic (Rp 150k)",
    )
    col_c.metric(
        "Live Dynamic Net Economic Value",
        "+Rp 18,883,200",
        "206.4% ROI (21 Commercial Saves)",
    )
    col_d.metric(
        "Stage-0 Month-1 Onboarding Nudges",
        "34 Active Accounts",
        "Rp 0 Subsidy (Pre-2nd-Bill Lock-In)",
    )

    if not data_bundle["scope_summary"].empty:
        st.markdown("#### Side-by-Side Campaign Scope Comparison (`N = 574` Live Active vs. `N = 719` Backtest)")
        st.dataframe(
            data_bundle["scope_summary"], width="stretch", hide_index=True
        )

    st.markdown("#### 🚨 Priority Active Retention Watchlist (`is_churn == 0`, Ordered by SLA Urgency & ENEV)")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        regions = ["All"] + sorted(df_watchlist["region"].dropna().unique().tolist())
        sel_region = st.selectbox("Filter Watchlist by Region", regions, index=0)
    with filter_col2:
        tiers = ["All"] + sorted(
            df_watchlist["action_tier_dynamic"].dropna().unique().tolist()
        )
        sel_tier = st.selectbox("Filter by Action Tier", tiers, index=0)

    filtered_wl = df_watchlist.copy()
    if sel_region != "All":
        filtered_wl = filtered_wl[filtered_wl["region"] == sel_region]
    if sel_tier != "All":
        filtered_wl = filtered_wl[filtered_wl["action_tier_dynamic"] == sel_tier]

    display_cols = [
        c
        for c in [
            "customerID",
            "region",
            "contractType",
            "tenureMonths",
            "hasInternetService",
            "hasPhoneService",
            "monthlyCharges",
            "n_pending",
            "has_billing_discrepancy",
            "p_structural",
            "p_acute",
            "p_composite",
            "action_tier_dynamic",
            "action_cost_dynamic_idr",
            "optimal_enev_dynamic_idr",
            "action_category",
            "urgency_sla_hours",
        ]
        if c in filtered_wl.columns
    ]
    st.dataframe(filtered_wl[display_cols], width="stretch", hide_index=True)


# ==============================================================================
# TAB 2: Forensic Data Hygiene, Survival Hazard & 72-Hour SLA Clock
# ==============================================================================
with tab2:
    st.subheader("Forensic Data Hygiene, Discrete-Time Survival Hazard & The 72-Hour Ticket Clock")

    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.metric(
        "Modulo-200 Template Sibling Pairs",
        "519 Pairs Audited",
        "90.4% Churn Agreement / 23.1% Exact Clones",
    )
    fcol2.metric(
        "Month-2 Invoice Churn Cliff",
        "46.7% MtM Hazard (126/270)",
        "70.2% on MtM Fiber vs 0.65% at Month 1",
    )
    fcol3.metric(
        "Post-Ticket Median Time-to-Churn",
        "3.0 Days (72h Clock)",
        "150 Post-Churn Tickets Filtered Out",
    )

    if not data_bundle["survival_table"].empty:
        st.markdown("#### Discrete-Time Tenure Hazard & Kaplan-Meier Survival Table")
        st.dataframe(
            data_bundle["survival_table"], width="stretch", hide_index=True
        )

    if not data_bundle["mod200_audit"].empty:
        st.markdown("#### Modulo-200 Near-Duplicate Template Generation Audit")
        st.dataframe(
            data_bundle["mod200_audit"], width="stretch", hide_index=True
        )

    img_col1, img_col2 = st.columns(2)
    fig1_path = FIGURES_DIR / "fig1_data_hygiene_and_leakage.png"
    fig2_path = FIGURES_DIR / "fig2_structural_vs_acute_anatomy.png"
    fig3_path = FIGURES_DIR / "fig3_72h_clock_and_billing_trap.png"

    if fig1_path.exists():
        with img_col1:
            st.image(
                str(fig1_path),
                caption="Figure 1: Forensic Data Hygiene, Modulo-200 Clone Audit & Tenure Hazard Cliff",
                width="stretch",
            )
    if fig2_path.exists():
        with img_col2:
            st.image(
                str(fig2_path),
                caption="Figure 2: Structural Onboarding Vulnerability vs. Acute Support Ticket Anatomy",
                width="stretch",
            )
    if fig3_path.exists():
        st.image(
            str(fig3_path),
            caption="Figure 3: The 72-Hour Post-Ticket Churn Clock & Billing Overcharge Trap (IOH-553)",
            width="stretch",
        )


# ==============================================================================
# TAB 3: Dual-Horizon Model Validation & Economic Simulator
# ==============================================================================
with tab3:
    st.subheader(
        "Dual-Horizon Leak-Free Model Validation & ARPU-Scaled Economic Optimizer"
    )

    if not data_bundle["model_metrics"].empty:
        st.markdown("#### 4-Scheme Cross-Validation Audit (`StratifiedGroupKFold(id % 200)` Primary Benchmark)")
        st.dataframe(
            data_bundle["model_metrics"], width="stretch", hide_index=True
        )

    st.markdown("---")
    st.markdown("#### 🧮 Interactive Expected Net Economic Value (ENEV) & LTV Simulator")
    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    with sim_c1:
        sim_arpu = st.number_input(
            "Customer Monthly ARPU (IDR)",
            min_value=150_000,
            max_value=1_000_000,
            value=450_000,
            step=25_000,
        )
    with sim_c2:
        sim_p_churn = st.slider(
            "Calibrated Churn Probability (p_i)",
            min_value=0.01,
            max_value=0.99,
            value=0.55,
            step=0.01,
        )
    with sim_c3:
        sim_margin = st.slider(
            "Contribution Margin (m)",
            min_value=0.40,
            max_value=0.85,
            value=0.72,
            step=0.02,
        )
    with sim_c4:
        sim_uplift_prem = st.slider(
            "Premium Relative Uplift (tau_prem)",
            min_value=0.15,
            max_value=0.70,
            value=0.50,
            step=0.05,
        )

    sim_ltv_dynamic = sim_arpu * sim_margin * 20.0
    sim_enev_basic = (sim_p_churn * 0.20 * sim_ltv_dynamic) - 150_000.0
    sim_enev_prem = (sim_p_churn * sim_uplift_prem * sim_ltv_dynamic) - 450_000.0
    best_tier = (
        "PREMIUM (Rp 450,000)"
        if sim_enev_prem >= max(sim_enev_basic, 0.0) and sim_enev_prem > 0
        else ("BASIC (Rp 150,000)" if sim_enev_basic > 0 else "DO_NOT_ACT (Rp 0)")
    )

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Dynamic ARPU-Scaled LTV (V_i)", f"Rp {sim_ltv_dynamic:,.0f}")
    scol2.metric("ENEV Basic (Rp 150k Cost, 20% Uplift)", f"Rp {sim_enev_basic:,.0f}")
    scol3.metric(
        f"ENEV Premium (Rp 450k Cost, {int(sim_uplift_prem*100)}% Uplift)",
        f"Rp {sim_enev_prem:,.0f}",
    )
    scol4.metric("Optimal Decision Policy", best_tier)

    if not data_bundle["policy_comparison"].empty:
        with st.expander("📊 View Full 8-Policy Economic Comparison Table"):
            st.dataframe(
                data_bundle["policy_comparison"],
                width="stretch",
                hide_index=True,
            )

    if not data_bundle["pareto_frontier"].empty:
        with st.expander("📊 View Budget-Constrained Pareto Frontier Table"):
            st.dataframe(
                data_bundle["pareto_frontier"],
                width="stretch",
                hide_index=True,
            )

    vcol1, vcol2 = st.columns(2)
    fig4_path = FIGURES_DIR / "fig4_model_validation_and_calibration.png"
    fig5_path = FIGURES_DIR / "fig5_economic_optimization_and_pareto.png"
    fig6_path = FIGURES_DIR / "fig6_quadrant_decision_matrix.png"
    if fig4_path.exists():
        with vcol1:
            st.image(
                str(fig4_path),
                caption="Figure 4: Leak-Free Out-of-Fold ROC/PR Curves & Reliability Calibration",
                width="stretch",
            )
    if fig5_path.exists():
        with vcol2:
            st.image(
                str(fig5_path),
                caption="Figure 5: Economic Optimization & Budget-Constrained Pareto Frontier",
                width="stretch",
            )
    if fig6_path.exists():
        st.image(
            str(fig6_path),
            caption="Figure 6: Dual-Horizon Risk Quadrant Decision Matrix (Structural vs. Acute Hazard)",
            width="stretch",
        )


# ==============================================================================
# TAB 4: Live Gemini 3.8 Flash (`ChatGoogleGenerativeAI`) + LangGraph Co-Pilot
# ==============================================================================
with tab4:
    st.subheader(
        "🤖 Live LangGraph + `ChatGoogleGenerativeAI(model=\"gemini-3.8-flash\")` Retention Co-Pilot"
    )
    st.markdown(
        """
        Select any customer below to inspect their **6-node LangGraph `StateGraph` execution**,
        deterministic policy guardrails (`commercial_upsell_allowed`, `max_authorized_incentive_idr`),
        executed `@tool` payloads, and invoke **`ChatGoogleGenerativeAI(model="gemini-3.8-flash")`**
        for live structured battlecard synthesis and interactive tool-calling consultation.
        """
    )

    showcase_presets = {
        "IOH-198 — Active Bandung Fiber | Open Pending Billing Ticket (97.6% Risk, Tier-2 <24h Save)": "IOH-198",
        "IOH-553 — Active Jakarta DSL | +Rp 410,000 Billing Ledger Discrepancy (Mandatory Refund)": "IOH-553",
        "IOH-664 — Active Bandung Fiber (Tenure 1m) | Resolved Tech Ticket (95.3% Structural Risk)": "IOH-664",
        "IOH-051 — Historical Churned Bandung Fiber | 2 Repeat Tech Tickets in 3 Days (<12h NOC Tier-3)": "IOH-051",
        "IOH-002 — Healthy Active Surabaya 1-Year Voice | Low Risk (0.1% Risk, Silent Monitoring)": "IOH-002",
        "Custom Selection (Choose from all 719 customers)": "CUSTOM",
    }

    pcol1, pcol2 = st.columns([2, 1])
    with pcol1:
        preset_choice = st.selectbox(
            "Quick-Select Showcase Archetype or Custom Customer",
            options=list(showcase_presets.keys()),
            index=0,
        )
    with pcol2:
        all_cids = df_all["customerID"].astype(str).tolist()
        default_cid = (
            showcase_presets[preset_choice]
            if showcase_presets[preset_choice] != "CUSTOM"
            else "IOH-198"
        )
        default_idx = all_cids.index(default_cid) if default_cid in all_cids else 0
        selected_cid = st.selectbox(
            "Customer ID (`customerID`)",
            options=all_cids,
            index=default_idx,
        )

    cust_row = df_all.loc[df_all["customerID"] == selected_cid].iloc[0]

    # Run deterministic LangGraph state or retrieve live Gemini synthesis if triggered
    live_state_key = f"live_ctx_{selected_cid}"
    live_meta_key = f"live_meta_{selected_cid}"

    btn_col1, btn_col2 = st.columns([1, 2])
    with btn_col1:
        run_live_btn = st.button(
            f"⚡ Run Live `{selected_model}` LangGraph Synthesis for {selected_cid}",
            type="primary",
            width="stretch",
        )
    with btn_col2:
        st.caption(
            f"Invokes `ChatGoogleGenerativeAI(model=\"{selected_model}\", temperature={temperature_val}, "
            f"thinking_level=\"{thinking_level_val}\")` inside `llm_copilot_synthesis_node` with JSON Schema structured output."
        )

    if run_live_btn:
        with st.spinner(
            f"Executing 6-node LangGraph StateGraph with live ChatGoogleGenerativeAI ({selected_model})..."
        ):
            live_ctx, live_telemetry = run_live_gemini_agent_for_customer(
                cust_row,
                model_name=selected_model,
                temperature=temperature_val,
                thinking_level=thinking_level_val,
            )
            st.session_state[live_state_key] = live_ctx
            st.session_state[live_meta_key] = live_telemetry

    if live_state_key in st.session_state:
        ctx = st.session_state[live_state_key]
        telemetry = st.session_state.get(live_meta_key, {})
        if telemetry.get("live_llm_used"):
            st.success(
                f"✅ Live `ChatGoogleGenerativeAI(model=\"{telemetry.get('model')}\")` synthesis completed!"
            )
        elif telemetry.get("error"):
            st.warning(
                f"Used Policy-Grounded Fallback (`{telemetry.get('error')}`)"
            )
    else:
        ctx = run_langgraph_for_customer_row(cust_row)

    # Display Customer & Policy Gate KPI Strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Composite / Structural Risk",
        f"{ctx.p_composite_risk:.1%}",
        f"Struct: {ctx.p_structural_risk:.1%} | Acute: {ctx.p_acute_risk:.1%}",
    )
    k2.metric(
        "Action Tier & Incentive Cap",
        ctx.recommended_action_tier,
        f"Cap: Rp {ctx.max_authorized_incentive_idr:,.0f}",
    )
    k3.metric(
        "Expected Net Value (ENEV)",
        f"Rp {ctx.expected_net_value_idr:,.0f}",
        f"Dynamic LTV: Rp {ctx.customer_ltv_dynamic_idr:,.0f}",
    )
    k4.metric(
        "Commercial Upsell Allowed",
        "✅ ALLOWED" if ctx.commercial_upsell_allowed else "🛑 BLOCKED (SLA First)",
        ctx.guardrail_policy_triggered,
    )
    k5.metric(
        "Autonomy & Target SLA",
        f"<{ctx.target_sla_hours} Hours",
        ctx.autonomy_tier.split("(")[0].strip(),
    )

    # Battlecard & Bahasa Indonesia Outreach Draft
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("#### 🛡️ Specialist Retention Battlecard")
        st.info(ctx.agent_synthesized_battlecard)
        st.markdown(
            f"**LangGraph Route**: `{ctx.langgraph_route}`  \n"
            f"**Nodes Visited**: `{' -> '.join(ctx.langgraph_nodes_visited)}`  \n"
            f"**HITL Governance Gate**: `{ctx.human_approval_status}`  \n"
            f"**RCT Counterfactual Bucket**: `{ctx.exploration_bucket}` (`pi = {ctx.action_propensity:.2f}`)"
        )
    with bcol2:
        st.markdown("#### 💬 Customer Outreach Draft (Bahasa Indonesia)")
        st.success(ctx.customer_outreach_draft)
        st.markdown("#### 🛠️ Pre-Authorized `@tool` Executions")
        if ctx.authorized_tool_executions:
            st.json(ctx.authorized_tool_executions)
        else:
            st.caption(
                "Zero tools executed (`silent_monitoring_node` short-circuit — saves 100% LLM token cost)."
            )

    with st.expander("📄 View Full Serializable `CustomerRetentionContext` JSON Contract"):
        st.json(asdict(ctx))

    st.markdown("---")
    st.markdown(
        f"#### 💬 Interactive Specialist Chat with `ChatGoogleGenerativeAI(model=\"{selected_model}\")` + Bound `@tool` Functions"
    )
    st.caption(
        "Ask the Gemini 3.8 Flash Retention Co-Pilot to explain the root cause, customize the Bahasa Indonesia WhatsApp script, "
        "or invoke `@tool` functions (`audit_and_refund_billing_discrepancy`, `escalate_noc_priority_ticket`, "
        "`resolve_pending_billing_inquiry`, `stage_or_execute_retention_incentive`) within hard policy guardrails."
    )

    chat_state_key = f"chat_history_{selected_cid}"
    if chat_state_key not in st.session_state:
        st.session_state[chat_state_key] = []

    for msg in st.session_state[chat_state_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander("🛠️ Executed LangChain `@tool` Calls"):
                    st.json(msg["tool_calls"])
            if msg.get("usage_metadata"):
                st.caption(f"Token Usage Metadata: `{msg['usage_metadata']}`")

    user_prompt = st.chat_input(
        f"Ask Gemini 3.8 Flash about {selected_cid} (e.g., 'Execute the appropriate tool and draft a WhatsApp follow-up')..."
    )
    if user_prompt:
        st.session_state[chat_state_key].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Querying ChatGoogleGenerativeAI ({selected_model})..."):
                chat_res = invoke_interactive_retention_chat(
                    user_query=user_prompt,
                    customer_ctx=ctx,
                    chat_history=st.session_state[chat_state_key][:-1],
                    model_name=selected_model,
                    temperature=temperature_val,
                    thinking_level=thinking_level_val,
                )
                st.markdown(chat_res["answer"])
                if chat_res.get("tool_calls"):
                    with st.expander("🛠️ Executed LangChain `@tool` Calls"):
                        st.json(chat_res["tool_calls"])
                if chat_res.get("usage_metadata"):
                    st.caption(f"Token Usage Metadata: `{chat_res['usage_metadata']}`")

                st.session_state[chat_state_key].append(
                    {
                        "role": "assistant",
                        "content": chat_res["answer"],
                        "tool_calls": chat_res.get("tool_calls", []),
                        "usage_metadata": chat_res.get("usage_metadata", {}),
                    }
                )

    fig7_path = FIGURES_DIR / "fig7_langgraph_agentic_architecture.png"
    if fig7_path.exists():
        with st.expander("🗺️ View LangGraph StateGraph Architecture Diagram (Figure 7)"):
            st.image(
                str(fig7_path),
                caption="Figure 7: LangGraph 6-Node Agentic Retention Architecture & Guardrail Topology",
                width="stretch",
            )


# ==============================================================================
# TAB 5: Fairness Audit & RCT Experiment Power Calculator
# ==============================================================================
with tab5:
    st.subheader(
        "⚖️ Demographic & Regional Fairness Audit + Counterfactual RCT Power Analysis"
    )

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown("#### Cohort RCT Holdout & Border Probe Allocation (`N = 719`)")
        st.dataframe(
            data_bundle["rct_summary"], width="stretch", hide_index=True
        )
    with rcol2:
        st.markdown("#### Two-Proportion Z-Test Statistical Power Table (80% Power, alpha=0.05)")
        st.dataframe(
            data_bundle["power_analysis"], width="stretch", hide_index=True
        )

    if not data_bundle["fairness_audit"].empty:
        st.markdown("#### Equal-Opportunity Demographic & Regional Fairness Audit")
        st.dataframe(
            data_bundle["fairness_audit"], width="stretch", hide_index=True
        )

    fig8_path = FIGURES_DIR / "fig8_demographic_fairness_audit.png"
    if fig8_path.exists():
        st.image(
            str(fig8_path),
            caption="Figure 8: Regional, Contract & Service-Tier Fairness Audit (True Positive Save Coverage & Calibration)",
            width="stretch",
        )
