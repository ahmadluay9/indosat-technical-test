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
LOGO_PATH = BASE_DIR / "img" / "Indosat_Ooredoo_Hutchison.png"

load_dotenv(dotenv_path=ENV_PATH, override=True)

st.set_page_config(
    page_title="IOH Retention Intelligence & Gemini 3.8 Flash Co-Pilot",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

if LOGO_PATH.exists() and hasattr(st, "logo"):
    try:
        st.logo(str(LOGO_PATH))
    except Exception:
        pass


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
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=240)
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

    sample_payloads_path = OUTPUTS_DIR / "sample_langgraph_agent_payloads.json"
    sample_payloads = (
        json.loads(sample_payloads_path.read_text(encoding="utf-8"))
        if sample_payloads_path.exists()
        else []
    )
    mmd_path = OUTPUTS_DIR / "langgraph_retention_workflow.mmd"
    mermaid_workflow = (
        mmd_path.read_text(encoding="utf-8") if mmd_path.exists() else ""
    )
    nb_path = BASE_DIR / "IOH_Customer_Retention_Intelligence.ipynb"
    notebook_cells = (
        json.loads(nb_path.read_text(encoding="utf-8")).get("cells", [])
        if nb_path.exists()
        else []
    )

    return {
        "df": df_scored,
        "watchlist": watchlist_df,
        "scope_summary": scope_df,
        "rct_summary": rct_summary_df,
        "power_analysis": power_df,
        "survival_table": survival_df,
        "mod200_audit": mod200_df,
        "quarantined": _read_optional_csv("quarantined_records.csv"),
        "model_metrics": eval_df,
        "feature_importances": _read_optional_csv("model_feature_importances.csv"),
        "rule3_ablation": _read_optional_csv("active_rule3_feature_ablation.csv"),
        "policy_comparison": _read_optional_csv("economic_policy_comparison.csv"),
        "sensitivity_grid": _read_optional_csv("uplift_ltv_sensitivity_matrix.csv"),
        "pareto_frontier": _read_optional_csv("budget_pareto_frontier.csv"),
        "fairness_audit": _read_optional_csv("fairness_and_parity_audit.csv"),
        "sample_payloads": sample_payloads,
        "mermaid_workflow": mermaid_workflow,
        "notebook_cells": notebook_cells,
    }


data_bundle = load_platform_data()
df_all: pd.DataFrame = data_bundle["df"]
df_watchlist: pd.DataFrame = data_bundle["watchlist"]


# ==============================================================================
# 2. Sidebar — Gemini 3.8 Flash Configuration & Platform Telemetry
# ==============================================================================

with st.sidebar:
    st.markdown("## 📡 Retention Intelligence")
    st.caption(
        "Dual-Horizon Survival Hazard + Economic Uplift + LangGraph Agentic Co-Pilot"
    )
    st.success("🔐 Authenticated via URL `?apikey=...`")

    st.markdown("---")
    st.markdown("### 🤖 `ChatGoogleGenerativeAI` Settings")
    ALLOWED_GEMINI_MODEL = "gemini-3.8-flash"
    st.selectbox(
        "Gemini Model ID",
        options=[ALLOWED_GEMINI_MODEL],
        index=0,
        disabled=True,
        help="Strictly locked to gemini-3.8-flash via langchain-google-genai",
    )
    selected_model = ALLOWED_GEMINI_MODEL
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

if LOGO_PATH.exists():
    hdr_logo_col, hdr_title_col = st.columns([1.1, 4.9], vertical_alignment="center")
    with hdr_logo_col:
        st.image(str(LOGO_PATH), width=240)
    with hdr_title_col:
        st.title(
            "Indosat Ooredoo Hutchison — Customer Churn, Survival Hazard & Agentic Retention Platform"
        )
        st.markdown(
            "**Dual-Horizon Leak-Free Survival Architecture** (`StratifiedGroupKFold(id % 200)`) • "
            "**ARPU-Scaled Economic Optimizer** • **LangGraph + `ChatGoogleGenerativeAI(model=\"gemini-3.8-flash\")` Co-Pilot**"
        )
else:
    st.title(
        "📡 Indosat Ooredoo Hutchison — Customer Churn, Survival Hazard & Agentic Retention Platform"
    )
    st.markdown(
        "**Dual-Horizon Leak-Free Survival Architecture** (`StratifiedGroupKFold(id % 200)`) • "
        "**ARPU-Scaled Economic Optimizer** • **LangGraph + `ChatGoogleGenerativeAI(model=\"gemini-3.8-flash\")` Co-Pilot**"
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🎯 1. Executive Command Center & Watchlist",
        "🔍 2. Forensic Hygiene & Survival Hazard",
        "📈 3. Model Validation & Economic Simulator",
        "🤖 4. Live Gemini 3.8 Flash + LangGraph Co-Pilot",
        "⚖️ 5. Fairness Audit, RCT Power & Roadmap",
        "📓 6. Complete Notebook Report",
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
        "206.4% ROI (Rp 18.35M / 189.5% incl. ops)",
    )
    col_d.metric(
        "Stage-0 Month-1 Onboarding Nudges",
        "34 Active Accounts",
        "Rp 0 Subsidy (+Rp 0.51M Digital Ops)",
    )

    with st.expander(
        "📋 Executive Summary & Monday Morning Operational Playbook (4 Priority Decisions)",
        expanded=True,
    ):
        st.markdown(
            """
            A forensic audit and survival analysis of IOH's **719 valid customer accounts** (**574 active**, **145 historical churned**)
            reveals that **86.9% of all churn (126 of 145 events) occurs at Tenure Month 2**—immediately after the second monthly invoice
            arrives for Month-to-month Fiber optic subscribers—and **91.7% of ticketed churners exit within 72 hours of an unresolved support ticket**.

            #### Monday Morning Operational Playbook
            1. **Dispatch Tier-2/Tier-3 Retention Specialists to 12 Acute Active Accounts Within `<24 Hours` (Rp 5.38M Spend):**
               Resolve open technical or billing discrepancies *before* presenting commercial retention offers. Includes **11 positive-ENEV commercial saves**
               and **1 mandatory billing compliance remediation** (`IOH-553`, overcharged by Rp 410,000 at `tenureMonths = 17`, booked honestly as a
               `-Rp 124,613` compliance cost [`-Rp 159,613` incl. Rp 35k specialist handling cost] rather than a synthetic positive return).
            2. **Trigger Automated Digital Onboarding for 10 High-Risk Ticketless Active Accounts (Rp 3.92M Spend):**
               Target Month-to-month Fiber optic accounts in high-risk cohorts with automated 1-Year contract migration and autopay incentives.
               Across the **21 commercial active interventions** (`Rp 9.15M` subsidy spend + `Rp 0.54M` contact handling cost), expected net value is
               **Rp 18.88M (206.4% ROI)** under dynamic ARPU-scaled LTV (`14.4x ARPU`, or **Rp 18.35M / 189.5% ROI** net of operational contact costs)
               and **Rp 11.89M (129.9% ROI)** under conservative static `Rp 6.0M` LTV.
            3. **Launch a Stage-0 Pre-Second-Bill Autopay Nudge for All 34 Month-1 Month-to-Month Customers (Rp 0.51M Digital Ops Cost):**
               Right-censored survival analysis shows discrete-time churn hazard spikes from **0.65% at Month 1** to **46.67% at Month 2** across
               Month-to-month customers (**70.18% on Month-to-month Fiber optic**). Intervening on the **34 active Month-1 Month-to-month subscribers**
               (6 Fiber, 22 DSL, 6 Voice-only; 27 currently unactioned + 7 already in the Top-22 queue) *before* their second bill arrives addresses
               the primary structural driver of early-life churn.
            4. **Enforce a 10% Above-Threshold RCT Holdout + 5% Below-Threshold Diagnostic Probe:**
               Statistical power analysis ($\\alpha = 0.05$, 80% power) proves that detecting the +10pp Basic tier uplift (50% to 40% churn) requires
               $n_{\\text{control}} \\ge 220$ controls—achieved via a **10% holdout** across ~2,195 cumulative interventions (whereas a 5% holdout would
               require 4,183 interventions). A **5% diagnostic probe** in the below-threshold border zone ($3\\% \\le p < 12.5\\%$) tests threshold
               calibration and sleeping-dog assumptions.

            #### Headline Metrics at a Glance
            | Metric Category | Live Active Campaign ($N = 574$ Active) | Retrospective Policy Backtest ($N = 719$ Full Cohort) |
            | :--- | :--- | :--- |
            | **Target Scope & Purpose** | **Operational Budget & Monday Dispatch Queue** | **Illustrative Upper-Bound Recognition Benchmark Only** |
            | **Actioned Accounts** | **22 active accounts** (21 commercial + 1 compliance `IOH-553`) + **34 Stage-0 Month-1 MtM nudges** | **165 accounts** (21 commercial active + 144 historical churned) |
            | **Subsidy & Ops Investment** | **Rp 9.15M** commercial subsidy (+ **Rp 0.15M** `IOH-553` credit + **Rp 0.57M** contact ops) | **Rp 73.65M** total subsidy |
            | **Expected Net Value & ROI** | **Rp 18.88M net (206.4% ROI)** on 21 commercial saves (**Rp 18.35M / 189.5% ROI** incl. ops) | **Rp 355.35M net (482.5% ROI)** under root-cause Lens 3 (**+Rp 5.55M / +1.6%** for Policy 6 over Top-25% in Lens 1) |
            | **Model Out-of-Fold ROC-AUC** | **Model A (Structural): 0.9837** (`id % 200` GroupKFold) / **0.9948** (Stratified `id % 200`) / **0.9911** (ID-Block) | **Model B (Acute PIT): 0.9915** (`id % 200` GroupKFold) / **0.9968** (Stratified `id % 200`) / **0.9970** (ID-Block) |
            """
        )

    if not data_bundle["scope_summary"].empty:
        st.markdown(
            "#### Table 0.1 — Live Campaign Scope (`N = 574`) vs. Retrospective Backtest (`N = 719`)"
        )
        st.dataframe(
            data_bundle["scope_summary"], width="stretch", hide_index=True
        )

    st.markdown(
        "#### 🚨 Table 5.3 — Monday Morning Live Active Watchlist (22 Actioned Active Accounts, Ranked by Priority)"
    )
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
            "enev_rank_active",
            "customerID",
            "region",
            "contractType",
            "hasInternetService",
            "hasPhoneService",
            "tenureMonths",
            "monthlyCharges",
            "n_pending",
            "has_billing_discrepancy",
            "p_structural",
            "p_acute",
            "p_composite",
            "primary_driver",
            "action_category",
            "action_tier_dynamic",
            "action_cost_dynamic_idr",
            "ops_execution_cost_idr",
            "optimal_enev_dynamic_idr",
            "autonomy_tier",
            "urgency_sla_hours",
        ]
        if c in filtered_wl.columns
    ]
    st.dataframe(filtered_wl[display_cols], width="stretch", hide_index=True)

    with st.expander(
        "🌱 View Stage-0 Pre-Second-Bill Prevention Cohort (34 Active Month-1 Month-to-Month Subscribers)"
    ):
        stage0_df = df_all.loc[
            (df_all["is_churn"] == 0)
            & (df_all["contractType"] == "Month-to-month")
            & (df_all["tenureMonths"] == 1)
        ].sort_values("p_composite", ascending=False)
        st.caption(
            f"Showing all {len(stage0_df)} active Month-to-month customers at `tenureMonths == 1` "
            "(6 Fiber optic, 22 DSL, 6 Voice-only) eligible for the pre-second-bill autopay/onboarding nudge."
        )
        st.dataframe(stage0_df, width="stretch", hide_index=True)

    with st.expander(
        "📊 View Full 719-Customer Scored Decision Matrix (`outputs/retention_decisions_all_719.csv`)"
    ):
        st.dataframe(df_all, width="stretch", hide_index=True)


# ==============================================================================
# TAB 2: Forensic Data Hygiene, Survival Hazard & 72-Hour SLA Clock
# ==============================================================================
with tab2:
    st.subheader(
        "Forensic Data Hygiene, Discrete-Time Survival Hazard & The 72-Hour Ticket Clock"
    )

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

    with st.expander(
        "🔍 Section 1 — Nine Forensic Data Hygiene & Leakage Findings (`src/data_pipeline.py`)",
        expanded=True,
    ):
        st.markdown(
            """
            Before feature engineering or modeling, our deterministic ingestion pipeline (`src/data_pipeline.py`) resolves
            **nine structural anomalies and leakage channels** verified directly against the raw CSV extracts:

            | # | Forensic Finding | Empirical Evidence in Raw Tables | Remediation in `src/data_pipeline.py` |
            | :-: | :--- | :--- | :--- |
            | **1** | **Corrupted Primary Key** | `customer_base.csv` row 720 has `customerID = "IOH-1pictures"` with no matching usage or ticket rows. | Quarantined to `outputs/quarantined_records.csv`; valid cohort $N = 719$. |
            | **2** | **Region Casing Target Leakage** | 10 raw region strings map to 5 cities. Lowercase `"bandung"` has a **97.6% churn rate (40/41)** vs **7.7% (5/65)** for shorthand `"bdg"`. | Canonicalized to 5 clean city names (`Jakarta`, `Surabaya`, `Bandung`, `Medan`, `Semarang`). |
            | **3** | **Multi-Row Extraction Artifact** | **134 of 145 churned accounts (92.4%)** have 2 monthly usage rows (`tenureMonths` 1 and 2), whereas **0 of 574 active accounts** have $>1$ row (`diagnostic_n_usage_rows > 1` is a 100%-precision churn leak). | Aggregated to a single point-in-time snapshot per customer; row-count and delta columns excluded from model features. |
            | **4** | **Two-Clock Observation Structure** | Active customers are observed across `yearMonth` **2025-09 ($n=6$), 2025-10 ($n=167$), and 2025-11 ($n=401$)**. Churned customers are frozen at exit (`yearMonth` **2025-04 to 2025-11**, `churnDate` **2025-04-18 to 2025-11-28**). | Explicitly modeled via right-censored Kaplan-Meier / discrete-time hazard analysis and dual-horizon scoring. |
            | **5** | **Post-Churn Support Tickets** | **150 of 748 tickets (20.1%)** across **113 of 145 churned customers** occur *after* `churnDate` (`IOH-003`, `IOH-351`, `IOH-551` have *only* post-churn tickets). | Enforced strict point-in-time filter $\\text{interactionDate} \\le \\text{churnDate}$ (`is_post_churn_leakage == 0`). |
            | **6** | **Modulo-200 Template Duplication** | Customer pairs 200 IDs apart (`(id - 1) % 200`, $n=519$ pairs) share a **63.2% 5-column profile match rate** (vs **4.8% random**), **90.37% churn label agreement** (vs **68.35% random**), and **23.12% exact 8-column clones** (e.g., `IOH-488`/`IOH-688`, `IOH-464`/`IOH-664`, `IOH-457`/`IOH-657`). | Added `template_group = (cust_num - 1) % 200` and evaluated all models under strict Modulo-200 `GroupKFold` and `StratifiedGroupKFold`. |
            | **7** | **Mixed Date Formats & Blank Charges** | Dates mix `YYYY-MM-DD`, `DD/MM/YYYY`, and `YYYY/MM/DD`. 34 rows have blank `totalCharges` (all `tenureMonths == 1`). | Deterministic multi-pass date parser; exact imputation $\\text{totalCharges} = \\text{monthlyCharges} \\times \\text{tenureMonths}$. |
            | **8** | **Billing Ledger Overcharge (`IOH-553`)** | Active customer `IOH-553` (`tenureMonths = 17`, `monthlyCharges = Rp 410,000`) is billed `totalCharges = Rp 7,380,000` (+Rp 410,000 discrepancy over expected Rp 6,970,000) with 0 tickets. | Preserved raw ledger values, flagged `has_billing_discrepancy = 1`, and routed to mandatory billing compliance remediation. |
            | **9** | **Missing Demographics** | `gender` missing in 46 rows (6.4%); `partner` missing in 38 rows (5.3%). | Mapped to `"Unknown"` for fairness auditing; excluded demographics from predictive model inputs. |
            """
        )
        if not data_bundle["quarantined"].empty:
            st.markdown("**Quarantined Primary Key Record (`outputs/quarantined_records.csv`)**")
            st.dataframe(
                data_bundle["quarantined"], width="stretch", hide_index=True
            )

    if not data_bundle["mod200_audit"].empty:
        st.markdown(
            "#### Table 1.1 — Modulo-200 Template Duplication Audit (`(id - 1) % 200` vs. Random Pairs)"
        )
        st.dataframe(
            data_bundle["mod200_audit"], width="stretch", hide_index=True
        )

    sibling_cols = [
        c
        for c in [
            "customerID",
            "template_group",
            "gender",
            "region",
            "contractType",
            "paymentMethod",
            "hasInternetService",
            "hasPhoneService",
            "tenureMonths",
            "monthlyCharges",
            "dataUsageGB",
            "is_churn",
        ]
        if c in df_all.columns
    ]
    sibling_examples = df_all.loc[
        df_all["customerID"].isin(
            ["IOH-488", "IOH-688", "IOH-464", "IOH-664", "IOH-457", "IOH-657"]
        ),
        sibling_cols,
    ]
    if "template_group" in sibling_examples.columns:
        sibling_examples = sibling_examples.sort_values(
            ["template_group", "customerID"]
        )
    st.markdown(
        "#### Table 1.2 — Concrete Sibling Clones 200 IDs Apart (`template_group = (id - 1) % 200`)"
    )
    st.dataframe(sibling_examples, width="stretch", hide_index=True)

    with st.expander(
        "📉 Section 2 — Right-Censored Survival Analysis, The Month-2 Churn Cliff & 72-Hour Ticket Window",
        expanded=True,
    ):
        st.markdown(
            """
            #### 2.1 Resolving the Two-Clock Problem via Discrete-Time Survival Analysis
            Because churned accounts are recorded at failure tenure $T_i$ (`churnDate` between `2025-04-18` and `2025-11-28`) while active
            accounts are right-censored at their current tenure $C_i$ (`yearMonth` `2025-09` to `2025-11`), treating tenure as a static
            cross-sectional feature conflates event time with censoring time. Each customer's exact cohort entry month is recoverable via:
            $$\\text{activation\\_month} = \\text{yearMonth} - (\\text{tenureMonths} - 1)$$

            Constructing the discrete-time life table $h(t) = d_t / n_t$ and Kaplan-Meier survival function $S(t) = \\prod_{j \\le t} (1 - h(j))$
            across $t = 1 \\dots 12$ exposes the central commercial insight in the dataset:
            - **The Month-2 Churn Cliff:** **126 of all 145 churn events (86.9%) occur at exactly `tenureMonths == 2`**, compared with only 2 events at Month 1, 7 at Month 3, 5 at Month 4, and 3 at Month 5.
            - **Extreme Concentration in Month-to-Month Fiber Optic:** Across all Month-to-month subscribers ($n = 306$), discrete-time hazard jumps from **0.65% (2/306) at Month 1** to **46.67% (126/270) at Month 2**, then drops to **5.11% (7/137) at Month 3**. Within **Month-to-month Fiber optic ($n = 178$)**, Month-2 conditional hazard reaches **70.18% (120/171)**, versus **6.06% (6/99)** on Month-to-month DSL and **0.0%** across all 413 One-Year and Two-Year contracts.
            - **Why Month 2? Second-Bill Shock + Early Technical Escalation:** Among the 134 multi-row churners tracked from Month 1 to Month 2, data consumption *increased* by **+2.27 GB on average** (+7.24 GB among technical-ticket churners; +4.62 GB among billing-ticket churners). Customers do not quietly disengage before cancelling; they actively use the service in Month 1, experience an unresolved technical fault or second-invoice billing shock in Month 2, and cancel within 72 hours.
            - **Immediate Stage-0 Prevention Cohort (34 Active Month-1 MtM Subscribers):** Exactly **34 active Month-to-month accounts** (6 Fiber optic, 22 DSL, 6 Voice-only) are currently right-censored at `tenureMonths == 1`.

            #### 2.2 The 3-Rule Structural Baseline & The 72-Hour Pre-Churn Ticket Window
            - **3-Rule Structural Heuristic (`Month-to-month` + `Electronic check` + `tenureMonths <= 12`):** Captures **144 of 145 historical churners (99.3% recall)** across 266 matching accounts (54.1% precision). Within those 266 rule-matching accounts, **Fiber optic churn is 70.3% (135/192)** while **DSL churn is only 9.5% (7/74)**. Of the 51 ticketless active accounts inside the 3-rule cohort, **49 (96.1%) are lower-risk DSL subscribers in Semarang and Medan**, explaining why our structural model suppresses them below the Premium subsidy threshold without relying on ticket status.
            - **Pre-Churn Ticket Timing (Extraction-Window Caveat):** After removing post-churn leakage, **132 of 145 churners (91.0%)** logged at least one pre-churn ticket (all 132 within **1 to 5 days before `churnDate`**, median **3.0 days**, **91.7% within $\\le 3$ days**), while **13 churners (9.0%) had zero pre-churn tickets** (12 of whom exited at Month 2 on Month-to-month Fiber optic). In the raw interaction extract (142 ticketed churners prior to post-churn filtering), 119/142 (83.8%) had their last ticket 2–5 days before churn (`days_before_churn` spanning -5 to +5 days). Because the raw extract captures a narrow pre-churn interaction window, we treat `<72 hours` (`<12h` Tier-3 / `<24h` Tier-2) as a conservative operational SLA rather than claiming monthly scoring is universally obsolete.
            """
        )

    if not data_bundle["survival_table"].empty:
        st.markdown(
            "#### Table 2.1 — Discrete-Time Tenure Hazard $h(t)$ & Kaplan-Meier Survival $S(t)$"
        )
        st.dataframe(
            data_bundle["survival_table"], width="stretch", hide_index=True
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

    with st.expander(
        "📐 Section 3 — Dual-Horizon Architecture, 4-Scheme Modulo-200 CV & Time-Horizon Reconciliation",
        expanded=True,
    ):
        st.markdown(
            """
            #### 3.1 Dual-Horizon Architecture & Strict Modulo-200 Template Cross-Validation
            To separate baseline contract/tariff vulnerability from acute support-ticket escalation, we train two distinct model horizons
            using L2-regularized Logistic Regression, HistGradientBoosting, and Random Forest (`src/models.py`):
            - **Model A (`p_structural` — Baseline Structural Horizon):** Uses only contract, payment method, service tier, tenure, region, and spend/usage ratios (**zero support-ticket features**).
            - **Model B (`p_acute` — Acute Point-in-Time Horizon):** Combines structural features with strictly pre-churn support ticket counts, topics, repeat-ticket indicators, and resolution status.

            Because the forensic audit identified Modulo-200 template duplication (`template_group = (id - 1) % 200`), we evaluate all models across **four cross-validation schemes**:
            1. **`StratifiedGroupKFold` on `template_group = (id - 1) % 200`:** Keeps every `(id, id+200, id+400, id+600)` sibling quartet in the same fold while balancing fold base rates (**Model A Ensemble ROC-AUC = 0.9948, PR-AUC = 0.9812**; **Model B Ensemble ROC-AUC = 0.9968, PR-AUC = 0.9893**).
            2. **Unstratified `GroupKFold` on `template_group = (id - 1) % 200`:** Stress-tests sensitivity to severe fold base-rate variance (`4.9%` to `43.4%` churn across folds), yielding **Model A Ensemble ROC-AUC = 0.9837 (PR-AUC = 0.9417, Brier = 0.0383)** and **Model B Ensemble ROC-AUC = 0.9915 (PR-AUC = 0.9660, Brier = 0.0230)**.
            3. **Contiguous ID-Block `GroupKFold` (`(id - 1) // 144`):** **Model A ROC-AUC = 0.9911**, **Model B ROC-AUC = 0.9970**.
            4. **Random `StratifiedKFold`:** **Model A ROC-AUC = 0.9958**, **Model B ROC-AUC = 0.9974**.

            Crucially, the **live actioned cohort of 22 active customers (`p_composite >= 0.125`) is 100% invariant** across Contiguous ID-Block and Modulo-200 `StratifiedGroupKFold` scoring (`21/22` overlap under unstratified Modulo-200 `GroupKFold`).

            #### 3.2 Ensemble Design, Calibration, Consensus Index & Deterministic Root-Cause Rules
            - **Ensemble & Composite Weighting Rationale:** Out-of-fold Platt-calibrated probabilities are combined via fixed prior weights (`0.20 LR + 0.40 GBDT + 0.40 RF`) so linear log-odds regularize tree step-functions on an $N=719$ sample without overfitting a meta-stacker. `p_composite = 0.30 * p_structural + 0.70 * p_acute` acts as a **dual-horizon decision index**.
            - **Calibration & Time-Horizon Reconciliation:** Across active customers ($N = 574$), $\\sum p_{\\text{composite}} = 20.13$ expected churn events, closely matching the historical run-rate of **~18.1 churns/month** (145 churns over the 8-month April–November 2025 exit window) plus the 34 Month-1 Month-to-month cohorts approaching the Month-2 cliff ($\\sum p_{\\text{next\\_cycle\\_hazard}} = 14.57$).
            - **Inter-Model Consensus Index & Deterministic Root-Cause Rules:** We report `model_spread = max(p) - min(p)` and `prediction_confidence = clip(1 - 2 * std(p), 0, 1)` strictly as an **inter-model consensus routing index**. Primary/secondary root-cause labels (`ONBOARDING_EARLY_LIFE`, `BILLING_PRICE_FRICTION`, `NETWORK_TECH_FAILURE`, `CONTRACT_PAYMENT_LOCKIN`) are computed via **transparent, deterministic domain-weighted rules**.
            """
        )

    if not data_bundle["model_metrics"].empty:
        st.markdown(
            "#### Table 3.1 — Out-of-Fold Model Performance Across All 4 Cross-Validation Schemes (`outputs/model_evaluation_metrics.csv`)"
        )
        st.dataframe(
            data_bundle["model_metrics"], width="stretch", hide_index=True
        )

    if not data_bundle["rule3_ablation"].empty:
        st.markdown(
            "#### Table 3.2 — Feature & CV Ablation on the 3-Rule Cohort ($N=266$: 51 Ticketless Active, 71 Ticketed Active, 144 Churned)"
        )
        st.dataframe(
            data_bundle["rule3_ablation"], width="stretch", hide_index=True
        )

    if not data_bundle["feature_importances"].empty:
        with st.expander(
            "📊 View Model A & Model B Feature Importances (`outputs/model_feature_importances.csv`)"
        ):
            st.dataframe(
                data_bundle["feature_importances"],
                width="stretch",
                hide_index=True,
            )

    st.markdown("---")
    with st.expander(
        "💰 Section 4 — Expected Net Economic Value (ENEV) Optimization, Closed-Form Break-Even Thresholds & Cost Accounting",
        expanded=True,
    ):
        st.markdown(
            """
            #### 4.1 Closed-Form Break-Even Thresholds & Dynamic LTV Scaling
            For each customer $i$ with churn probability $p_i$, intervention cost $C_k$, and causal save uplift $\\tau_k$, Expected Net Economic Value is:
            $$\\text{ENEV}(i, k) = p_i \\cdot \\tau_k \\cdot \\text{LTV}_i - C_k$$

            Under the benchmark Rp 6,000,000 LTV (`Basic` $C_1 = \\text{Rp } 150\\text{k}, \\tau_1 = 10\\%$; `Premium` $C_2 = \\text{Rp } 500\\text{k}, \\tau_2 = 25\\%$):
            - **Do Not Act ($k=0$):** Optimal when $p_i < 12.50\\%$ (or $\\max_k \\text{ENEV}(i, k) \\le 0$).
            - **Basic Tier ($k=1$):** Optimal when $12.50\\% \\le p_i < 38.89\\%$.
            - **Premium Tier ($k=2$):** Dominates Basic when incremental benefit $(0.25 - 0.10) p_i \\cdot \\text{LTV}_i > \\text{Rp } 350\\text{k}$ ($p_i \\ge 38.89\\%$ at static Rp 6.0M LTV; $p_i \\ge 16.67\\%$ vs Do-Not-Act).

            To prevent over-subsidizing low-ARPU accounts, our primary production policy scales LTV dynamically with monthly ARPU
            ($\\text{LTV}_{\\text{dynamic}, i} = 18 \\text{ months} \\times 80\\% \\text{ gross margin} \\times \\text{ARPU}_i = 14.4 \\times \\text{ARPU}_i$).
            Across the 719 valid customers, `monthlyCharges` spans **Rp 150,000 to Rp 1,000,000** (83 accounts exceed Rp 850,000), yielding a dynamic LTV range
            of **Rp 2.16M to Rp 14.40M** (or **Rp 2.59M to Rp 17.28M** under a 24-month contract-migration sensitivity at $17.28 \\times \\text{ARPU}_i$).

            #### 4.2 Full Cost Accounting: Compliance vs. Commercial Subsidies & Operational Contact Costs
            1. **No Synthetic ENEV Floors for Compliance Remediation:** Refunding or crediting `IOH-553`'s Rp 410,000 billing ledger overcharge (`tenureMonths = 17`, `monthlyCharges = Rp 410,000`, `totalCharges = Rp 7,380,000`) is a mandatory regulatory/billing obligation, booked at its true unadjusted formula value (**-Rp 124,613** dynamic ENEV; **-Rp 159,613** including Rp 35,000 specialist handling cost) and separated from the **21 positive-ENEV commercial interventions**.
            2. **Explicit Contact & Call-Center Handling Costs:** Every Tier-2/Tier-3 specialist outreach incurs **Rp 35,000** in agent handling cost (12 accounts = Rp 420,000), and every Tier-1 digital outreach incurs **Rp 15,000** (10 accounts = Rp 150,000; plus Rp 510,000 across the 34 Stage-0 Month-1 MtM nudges).
            3. **Live Campaign First; Candid Retrospective Policy Comparison:**
               - **Live Active Campaign ($N = 574$):** **21 commercial interventions** require **Rp 9.15M** in subsidies (+ **Rp 0.54M** ops cost) and generate **Rp 18.88M expected net value (206.4% ROI)** before ops (**Rp 18.35M / 189.5% ROI** after ops; **Rp 11.89M / 129.9% ROI** under static Rp 6.0M LTV). Including mandatory compliance account `IOH-553` (22 total active actions, Rp 9.30M subsidy + Rp 0.57M ops), total live net value is **Rp 18.76M (201.7% ROI)** before ops (**Rp 18.19M / 184.3% ROI** incl. ops).
               - **Retrospective Full-Cohort Backtest ($N = 719$):** Under **Lens 1 (Uniform Uplift + Static Rp 6.0M LTV)**, Policy 6 beats naive Top-25% Risk Targeting by **Rp 5.55M (+1.6% net value: Rp 356.55M vs. Rp 351.00M)** while reducing subsidy spend by **Rp 8.55M (-10.6%: Rp 72.45M vs. Rp 81.00M)** (and saving **Rp 47.25M / -39.5%** vs. the 3-Rule Heuristic's Rp 119.70M spend). Under **Lens 3 (Mismatch-Adjusted Static LTV)**, Policy 7's **+Rp 159.30M** advantage over Policy 6 (**Rp 355.35M vs. Rp 196.05M**) arises from our explicit domain assumption penalizing non-root-cause-matched offers by **0.50x** when an acute ticket is open.
            """
        )

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
            "Gross Margin (m)",
            min_value=0.40,
            max_value=0.90,
            value=0.80,
            step=0.02,
        )
    with sim_c4:
        sim_horizon_m = st.slider(
            "Expected Tenure Horizon (Months)",
            min_value=12,
            max_value=24,
            value=18,
            step=1,
        )

    sim_ltv_dynamic = sim_arpu * sim_margin * float(sim_horizon_m)
    sim_enev_basic = (sim_p_churn * 0.10 * sim_ltv_dynamic) - 150_000.0
    sim_enev_prem = (sim_p_churn * 0.25 * sim_ltv_dynamic) - 450_000.0
    best_tier = (
        "PREMIUM (Rp 450,000)"
        if sim_enev_prem >= max(sim_enev_basic, 0.0) and sim_enev_prem > 0
        else ("BASIC (Rp 150,000)" if sim_enev_basic > 0 else "DO_NOT_ACT (Rp 0)")
    )

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Dynamic ARPU-Scaled LTV (V_i)", f"Rp {sim_ltv_dynamic:,.0f}")
    scol2.metric("ENEV Basic (Rp 150k Cost, 10% Absolute Save)", f"Rp {sim_enev_basic:,.0f}")
    scol3.metric("ENEV Premium (Rp 450k Cost, 25% Absolute Save)", f"Rp {sim_enev_prem:,.0f}")
    scol4.metric("Optimal Decision Policy", best_tier)

    if not data_bundle["sensitivity_grid"].empty:
        st.markdown(
            "#### Table 4.1 — Live Active Campaign ($N=574$) & Retrospective Sensitivity Across Uplift & LTV Scenarios (`outputs/uplift_ltv_sensitivity_matrix.csv`)"
        )
        st.dataframe(
            data_bundle["sensitivity_grid"],
            width="stretch",
            hide_index=True,
        )

    if not data_bundle["policy_comparison"].empty:
        st.markdown(
            "#### Table 4.2 — Policy Comparison: Live Active Campaign ($N=574$) First, Retrospective Backtest ($N=719$) Second (`outputs/economic_policy_comparison.csv`)"
        )
        st.dataframe(
            data_bundle["policy_comparison"],
            width="stretch",
            hide_index=True,
        )

    if not data_bundle["pareto_frontier"].empty:
        with st.expander(
            "📊 View Budget-Constrained Pareto Frontier Table (`outputs/budget_pareto_frontier.csv`)"
        ):
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
    with st.expander(
        f"💬 Interactive Specialist Chat with `ChatGoogleGenerativeAI(model=\"{selected_model}\")` + Bound `@tool` Functions",
        expanded=True,
    ):
        st.caption(
            "Ask the Gemini 3.8 Flash Retention Co-Pilot to explain the root cause, customize the Bahasa Indonesia WhatsApp script, "
            "or invoke `@tool` functions (`audit_and_refund_billing_discrepancy`, `escalate_noc_priority_ticket`, "
            "`resolve_pending_billing_inquiry`, `stage_or_execute_retention_incentive`) within hard policy guardrails."
        )

        chat_state_key = f"chat_history_{selected_cid}"
        if chat_state_key not in st.session_state:
            st.session_state[chat_state_key] = []

        # Quick-action prompt buttons for 1-click specialist testing
        qcol1, qcol2, qcol3, qcol4 = st.columns([2, 2, 2, 1])
        quick_prompt: str | None = None
        with qcol1:
            if st.button(
                "🛠️ Execute Authorized `@tool` & Explain",
                key=f"qbtn_tool_{selected_cid}",
                width="stretch",
            ):
                quick_prompt = (
                    f"Execute the appropriate retention tool for customer {selected_cid} "
                    f"within the authorized policy guardrails and summarize the action taken."
                )
        with qcol2:
            if st.button(
                "📱 Customize Bahasa Indonesia WhatsApp",
                key=f"qbtn_wa_{selected_cid}",
                width="stretch",
            ):
                quick_prompt = (
                    f"Write a personalized, empathetic Bahasa Indonesia WhatsApp message for {selected_cid} "
                    f"that strictly respects commercial_upsell_allowed={ctx.commercial_upsell_allowed} "
                    f"and max_authorized_incentive_idr=Rp {ctx.max_authorized_incentive_idr:,.0f}."
                )
        with qcol3:
            if st.button(
                "🔍 Explain Root Cause & Guardrails",
                key=f"qbtn_why_{selected_cid}",
                width="stretch",
            ):
                quick_prompt = (
                    f"Why is {selected_cid} at {ctx.p_composite_risk:.1%} composite churn risk, "
                    f"and why was '{ctx.guardrail_policy_triggered}' triggered?"
                )
        with qcol4:
            if st.button(
                "🗑️ Clear Chat",
                key=f"qbtn_clear_{selected_cid}",
                width="stretch",
            ):
                st.session_state[chat_state_key] = []
                st.rerun()

        # Inline message input form (works reliably inside st.tabs and Streamlit Cloud iframes)
        with st.form(key=f"specialist_chat_form_{selected_cid}", clear_on_submit=True):
            fcol1, fcol2 = st.columns([5, 1])
            with fcol1:
                typed_prompt = st.text_input(
                    f"Message Gemini 3.8 Flash Co-Pilot ({selected_cid})",
                    placeholder=f"Ask about {selected_cid} (e.g., 'Execute the appropriate tool and draft a WhatsApp follow-up')...",
                    label_visibility="collapsed",
                )
            with fcol2:
                send_clicked = st.form_submit_button(
                    "🚀 Send",
                    type="primary",
                    width="stretch",
                )

        # Also support bottom/tab chat_input
        chat_bar_prompt = st.chat_input(
            f"Or type here to ask Gemini 3.8 Flash about {selected_cid}...",
            key=f"chat_bar_{selected_cid}",
        )

        active_prompt = (
            quick_prompt
            or (typed_prompt.strip() if send_clicked and typed_prompt.strip() else None)
            or chat_bar_prompt
        )

        chat_container = st.container(border=True)
        with chat_container:
            # Always render an initial assistant greeting so the chat panel looks visibly open
            with st.chat_message("assistant"):
                st.markdown(
                    f"**Halo! I am the IOH Gemini 3.8 Flash Retention Co-Pilot for `{selected_cid}`.**  \n"
                    f"- **Composite Risk**: `{ctx.p_composite_risk:.1%}` (`{ctx.primary_root_cause}`)  \n"
                    f"- **Policy Gate**: `{ctx.guardrail_policy_triggered}` (`commercial_upsell_allowed={ctx.commercial_upsell_allowed}`)  \n"
                    f"- **Authorized Cap**: `Rp {ctx.max_authorized_incentive_idr:,.0f}` (`{ctx.recommended_action_tier}`, SLA `<{ctx.target_sla_hours}h`)  \n\n"
                    f"*Click one of the quick-action buttons above or type a question in the box to begin.*"
                )

            for msg in st.session_state[chat_state_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("tool_calls"):
                        with st.expander("🛠️ Executed LangChain `@tool` Calls"):
                            st.json(msg["tool_calls"])
                    if msg.get("usage_metadata"):
                        st.caption(f"Token Usage Metadata: `{msg['usage_metadata']}`")

            if active_prompt:
                st.session_state[chat_state_key].append(
                    {"role": "user", "content": active_prompt}
                )
                with st.chat_message("user"):
                    st.markdown(active_prompt)

                with st.chat_message("assistant"):
                    with st.spinner(
                        f"Querying ChatGoogleGenerativeAI ({selected_model})..."
                    ):
                        chat_res = invoke_interactive_retention_chat(
                            user_query=active_prompt,
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
                            st.caption(
                                f"Token Usage Metadata: `{chat_res['usage_metadata']}`"
                            )

                        st.session_state[chat_state_key].append(
                            {
                                "role": "assistant",
                                "content": chat_res["answer"],
                                "tool_calls": chat_res.get("tool_calls", []),
                                "usage_metadata": chat_res.get("usage_metadata", {}),
                            }
                        )

    with st.expander(
        "🛡️ Section 5.1 — Deterministic Guardrails First, Structured LLM Synthesis Second & Tiered Autonomy Matrix",
        expanded=False,
    ):
        st.markdown(
            """
            Hard commercial rules execute deterministically in Python (`src/retention_agent.py`) before any LLM or tool invocation:
            - **Anti-Payoff Guardrail:** Blocks commercial discounts/lock-in pitches from being presented before an active technical outage (`n_pending_tech > 0`) or billing discrepancy (`IOH-553`) is resolved.
            - **Tiered Autonomy Matrix (Live Active Campaign, $N = 22$ Actioned Accounts):**
              - **Tier 1 (`AUTO_EXECUTE_DIGITAL`, $n = 10$ active):** Ticketless onboarding/contract-lockin accounts with high inter-model consensus.
              - **Tier 2 (`AGENT_ASSISTED_WITH_24H_SLA`, $n = 9$ active):** Moderate-risk or resolved-ticket accounts requiring specialist follow-up within 24 hours.
              - **Tier 3 (`MANDATORY_HUMAN_ESCALATION_12H_SLA`, $n = 3$ active):** High-value acute technical/billing escalations (`IOH-553`, `IOH-198`, `IOH-098`) routed to senior specialists within 12 hours.
            """
        )

    if data_bundle["sample_payloads"]:
        with st.expander(
            "📦 View Precomputed Sample LangGraph Agent JSON Payloads (`outputs/sample_langgraph_agent_payloads.json`, incl. `IOH-198`)"
        ):
            st.json(data_bundle["sample_payloads"])

    if data_bundle["mermaid_workflow"]:
        with st.expander(
            "🔗 View LangGraph Workflow Mermaid Definition (`outputs/langgraph_retention_workflow.mmd`)"
        ):
            st.code(data_bundle["mermaid_workflow"], language="mermaid")

    fig7_path = FIGURES_DIR / "fig7_langgraph_agentic_architecture.png"
    if fig7_path.exists():
        with st.expander(
            "🗺️ View LangGraph StateGraph Architecture Diagram (Figure 7)",
            expanded=True,
        ):
            st.image(
                str(fig7_path),
                caption="Figure 7: LangGraph 6-Node Agentic Retention Architecture & Guardrail Topology",
                width="stretch",
            )


# ==============================================================================
# TAB 5: Fairness Audit, RCT Experiment Power & 90-Day Roadmap
# ==============================================================================
with tab5:
    st.subheader(
        "⚖️ Demographic & Regional Fairness Audit, Counterfactual RCT Power & 90-Day Production Roadmap"
    )

    with st.expander(
        "🧪 Section 5.2 — Statistical Power Analysis, 10% Above-Threshold Holdout & 5% Below-Threshold Probe",
        expanded=True,
    ):
        st.markdown(
            """
            A rigorous learning loop requires sufficient sample size to detect both Premium (+25pp) and Basic (+10pp) uplift effects
            using a two-proportion $z$-test ($\\alpha = 0.05$ two-sided, $1 - \\beta = 0.80$ power):
            - **Why 10% Holdout Is Required:** Detecting a 25pp reduction ($50\\% \\rightarrow 25\\%$ churn) requires only $n_{\\text{control}} = 32$ controls (~315 total interventions at 10% holdout). However, detecting the **10pp Basic tier reduction ($50\\% \\rightarrow 40\\%$ churn)** under 90/10 allocation requires **$n_{\\text{control}} = 220$ controls ($N_{\\text{total}} = 2,195$ cumulative actioned accounts)** (or $n_{\\text{per-arm}} = 388$ in a 50/50 split). At a 5% holdout, detecting +10pp would require ~4,183 interventions (~16 months of volume). Raising the stratified holdout to **10%** (`HOLDOUT_CONTROL_10PCT`, $n=2$ in the current 21-account commercial batch; mandatory compliance account `IOH-553` is exempt) achieves full statistical power in ~8 months of national rollout.
            - **5% Below-Threshold Diagnostic Probe:** To test whether the 12.5% break-even threshold or sleeping-dog assumption is miscalibrated, we allocate a **5% randomized diagnostic probe (`BELOW_THRESHOLD_DIAGNOSTIC_PROBE_5PCT`)** across the 36 active accounts in the border zone ($3\\% \\le p_{\\text{composite}} < 12.5\\%$ or Month-1 MtM cohort, yielding $n=2$ low-cost digital probe accounts in the current snapshot). All propensity scores $\\pi(x) \\in \\{0.90, 0.05\\}$ are logged for doubly-robust AIPW uplift estimation.
            """
        )

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown(
            "#### Table 5.1 — Statistical Power & Minimum Sample Size Analysis ($\\alpha = 0.05$, 80% Power)"
        )
        st.dataframe(
            data_bundle["power_analysis"], width="stretch", hide_index=True
        )
    with rcol2:
        st.markdown(
            "#### Table 5.2 — Stratified 10% Holdout, Mandatory Compliance Exemption & 5% Border-Zone Probe"
        )
        st.dataframe(
            data_bundle["rct_summary"], width="stretch", hide_index=True
        )

    st.markdown("---")
    with st.expander(
        "⚖️ Section 6.1 — Candid Fairness Assessment: Active Gender DI (0.665) vs. Calibration & Equal Opportunity",
        expanded=True,
    ):
        st.markdown(
            """
            A transparent fairness audit must evaluate the **live active population ($N = 574$)** separately from the full retrospective dataset ($N = 719$):
            - **Active-Customer Gender Disparate Impact Is 0.665 (Below the 0.80 4/5ths Threshold):** Among active customers, the commercial action rate is **3.20% for Male (11/344)** vs. **4.81% for Female (9/187)**, giving an active Disparate Impact ratio of **0.665** (whereas across all 719 customers including historical churners, the ratio is **0.828**).
            - **Why Active DI Is 0.665 Without Model Bias:** `gender` and `partner` are completely excluded from all model pipelines (`ablation_no_demo_corr > 0.999`). The 1.61pp active action-rate gap occurs because among the 574 active customers, **10 Female accounts (5.35%)** vs. **7 Male accounts (2.03%)** currently have unresolved pre-churn support tickets (`n_tickets > 0` inside the Month-to-month cohort).
            - **Why Calibration Parity & Equal Opportunity Are the Correct Metrics Here:** When allocating a *protective retention benefit* (resolving technical outages and lowering tariffs), artificially capping Female interventions or inflating ticketless Male subsidies to force a 0.80 demographic quota would withhold support from customers with open service failures. Instead, fairness requires:
              1. **Within-Group Calibration Parity:** Mean predicted risk matches observed churn within **$\\le 0.6\\text{pp}$** across both genders (`Female` predicted 21.7% vs observed 21.9%; `Male` predicted 19.6% vs observed 19.0%).
              2. **Equal Opportunity (True Positive Rate Parity):** Among true churners, the policy captures **98.0% of Female churners** and **99.0% of Male churners** (**Equal Opportunity Ratio = 0.990** across Gender, and **$\\ge 0.974$** across all 5 Regions).
            """
        )

    if not data_bundle["fairness_audit"].empty:
        st.markdown(
            "#### Table 6.1 — Subgroup Calibration, Equal Opportunity & Disparate Impact Audit (`outputs/fairness_and_parity_audit.csv`)"
        )
        st.dataframe(
            data_bundle["fairness_audit"], width="stretch", hide_index=True
        )

    with st.expander(
        "🗺️ Section 6.2 — Key Analytical Trade-Offs & 90-Day Production Roadmap",
        expanded=True,
    ):
        st.markdown(
            """
            - **Why Deploy Dual-Horizon ML Over the 3-Rule Heuristic + Simple ARPU Ranking?** In retrospective Lens 1, Policy 6 beats naive Top-25% ranking by **Rp 5.55M (+1.6%)** while saving **Rp 8.55M (-10.6%)** vs Top-25% and **Rp 47.25M (-39.5%)** vs the 3-Rule Heuristic in subsidy spend. More importantly, on the **122 active accounts inside the 3-rule cohort**, the heuristic treats all 122 identically, whereas the dual-horizon engine separates the **51 ticketless Semarang/Medan DSL accounts** (0% Fiber, mean risk 5.0%, `DO_NOT_ACT`) from the **22 acute Fiber/ticketed accounts** requiring immediate dispatch.
            - **90-Day Production Rollout:**
              1. **Weeks 1–2:** Execute Monday dispatch queue (12 `<24h` specialist calls + 10 automated onboarding offers + 34 Stage-0 Month-1 MtM pre-second-bill nudges).
              2. **Weeks 3–6:** Connect event-driven Kafka/webhook triggers on support ticket creation (`interactionDate`) and Month-1 invoice generation (`activation_month + 1`).
              3. **Weeks 7–12:** Accumulate stratified 10% holdout ($n_{\\text{control}} \\ge 220$) and 5% border-zone probe logs to refit empirical CATE uplift models ($\\hat{\\tau}(x)$) via doubly-robust AIPW.
            """
        )

    fig8_path = FIGURES_DIR / "fig8_demographic_fairness_audit.png"
    if fig8_path.exists():
        st.image(
            str(fig8_path),
            caption="Figure 8: Regional, Contract & Service-Tier Fairness Audit (True Positive Save Coverage & Calibration)",
            width="stretch",
        )


# ==============================================================================
# TAB 6: Complete Notebook Report (`IOH_Customer_Retention_Intelligence.ipynb`)
# ==============================================================================
with tab6:
    st.subheader(
        "📓 Complete Technical Assessment Notebook (`IOH_Customer_Retention_Intelligence.ipynb`)"
    )
    st.caption(
        "Full end-to-end rendering of all narrative sections, mathematical formulations, Python code cells, "
        "executed tables, and figures from `IOH_Customer_Retention_Intelligence.ipynb`."
    )
    show_nb_code = st.checkbox(
        "Show Notebook Python Code Cells", value=False, key="toggle_nb_code"
    )

    nb_cells = data_bundle.get("notebook_cells", [])
    for idx, cell in enumerate(nb_cells):
        ctype = cell.get("cell_type", "")
        src_text = "".join(cell.get("source", []))
        if ctype == "markdown":
            st.markdown(src_text)
        elif ctype == "code":
            if show_nb_code:
                with st.expander(f"🐍 Code Cell [{idx}]", expanded=True):
                    st.code(src_text, language="python")
            # Render the exact tables and figures produced by each code cell in the notebook
            if idx == 1:
                st.markdown(
                    "**Table 0.1 — Live Campaign Scope ($N=574$) vs. Retrospective Backtest ($N=719$)**"
                )
                st.dataframe(
                    data_bundle["scope_summary"],
                    width="stretch",
                    hide_index=True,
                )
            elif idx == 3:
                st.markdown(
                    "**Table 1.1 — Modulo-200 Template Duplication Audit (`(id - 1) % 200` vs. Random Pairs)**"
                )
                st.dataframe(
                    data_bundle["mod200_audit"],
                    width="stretch",
                    hide_index=True,
                )
                st.markdown(
                    "**Table 1.2 — Concrete Sibling Clones 200 IDs Apart (`template_group = (id - 1) % 200`)**"
                )
                st.dataframe(sibling_examples, width="stretch", hide_index=True)
                if (FIGURES_DIR / "fig1_data_hygiene_and_leakage.png").exists():
                    st.image(
                        str(FIGURES_DIR / "fig1_data_hygiene_and_leakage.png"),
                        width="stretch",
                    )
            elif idx == 5:
                st.markdown(
                    "**Table 2.1 — Discrete-Time Tenure Hazard $h(t)$ & Kaplan-Meier Survival $S(t)$ (Months 1–6)**"
                )
                surv_df = data_bundle["survival_table"]
                if not surv_df.empty and "tenure_month" in surv_df.columns:
                    st.dataframe(
                        surv_df.loc[surv_df["tenure_month"] <= 6],
                        width="stretch",
                        hide_index=True,
                    )
                if (FIGURES_DIR / "fig2_structural_vs_acute_anatomy.png").exists():
                    st.image(
                        str(FIGURES_DIR / "fig2_structural_vs_acute_anatomy.png"),
                        width="stretch",
                    )
                if (FIGURES_DIR / "fig3_72h_clock_and_billing_trap.png").exists():
                    st.image(
                        str(FIGURES_DIR / "fig3_72h_clock_and_billing_trap.png"),
                        width="stretch",
                    )
            elif idx == 7:
                st.markdown(
                    "**Table 3.1 — Out-of-Fold Model Performance Across All 4 Cross-Validation Schemes**"
                )
                st.dataframe(
                    data_bundle["model_metrics"],
                    width="stretch",
                    hide_index=True,
                )
                st.markdown(
                    "**Table 3.2 — Feature & CV Ablation on the 3-Rule Cohort ($N=266$: 51 Ticketless Active, 71 Ticketed Active, 144 Churned)**"
                )
                st.dataframe(
                    data_bundle["rule3_ablation"],
                    width="stretch",
                    hide_index=True,
                )
                if (
                    FIGURES_DIR / "fig4_model_validation_and_calibration.png"
                ).exists():
                    st.image(
                        str(
                            FIGURES_DIR
                            / "fig4_model_validation_and_calibration.png"
                        ),
                        width="stretch",
                    )
            elif idx == 9:
                st.markdown(
                    "**Table 4.1 — Live Active Campaign ($N=574$) & Retrospective Sensitivity Across Uplift & LTV Scenarios**"
                )
                st.dataframe(
                    data_bundle["sensitivity_grid"],
                    width="stretch",
                    hide_index=True,
                )
                st.markdown(
                    "**Table 4.2 — Policy Comparison: Live Active Campaign ($N=574$) First, Retrospective Backtest ($N=719$) Second**"
                )
                st.dataframe(
                    data_bundle["policy_comparison"],
                    width="stretch",
                    hide_index=True,
                )
                if (
                    FIGURES_DIR / "fig5_economic_optimization_and_pareto.png"
                ).exists():
                    st.image(
                        str(
                            FIGURES_DIR
                            / "fig5_economic_optimization_and_pareto.png"
                        ),
                        width="stretch",
                    )
                if (FIGURES_DIR / "fig6_quadrant_decision_matrix.png").exists():
                    st.image(
                        str(FIGURES_DIR / "fig6_quadrant_decision_matrix.png"),
                        width="stretch",
                    )
            elif idx == 11:
                st.markdown(
                    "**Table 5.1 — Statistical Power & Minimum Sample Size Analysis ($\\alpha = 0.05$, 80% Power)**"
                )
                st.dataframe(
                    data_bundle["power_analysis"],
                    width="stretch",
                    hide_index=True,
                )
                st.markdown(
                    "**Table 5.2 — Stratified 10% Holdout, Mandatory Compliance Exemption & 5% Border-Zone Probe**"
                )
                st.dataframe(
                    data_bundle["rct_summary"],
                    width="stretch",
                    hide_index=True,
                )
                st.markdown(
                    "**Table 5.3 — Monday Morning Live Active Watchlist (22 Actioned Active Accounts, Ranked by Priority)**"
                )
                st.dataframe(
                    df_watchlist[display_cols],
                    width="stretch",
                    hide_index=True,
                )
                if data_bundle["sample_payloads"]:
                    ioh198 = next(
                        (
                            p
                            for p in data_bundle["sample_payloads"]
                            if p.get("customer_id") == "IOH-198"
                        ),
                        data_bundle["sample_payloads"][0],
                    )
                    st.markdown(
                        "**Sample LangGraph Agent JSON Payload (`IOH-198` — Acute Fiber Tech Escalation)**"
                    )
                    st.json(ioh198)
                if (
                    FIGURES_DIR / "fig7_langgraph_agentic_architecture.png"
                ).exists():
                    st.image(
                        str(
                            FIGURES_DIR
                            / "fig7_langgraph_agentic_architecture.png"
                        ),
                        width="stretch",
                    )
            elif idx == 13:
                st.markdown(
                    "**Table 6.1 — Subgroup Calibration, Equal Opportunity & Disparate Impact Audit (`outputs/fairness_and_parity_audit.csv`)**"
                )
                st.dataframe(
                    data_bundle["fairness_audit"],
                    width="stretch",
                    hide_index=True,
                )
                if (FIGURES_DIR / "fig8_demographic_fairness_audit.png").exists():
                    st.image(
                        str(FIGURES_DIR / "fig8_demographic_fairness_audit.png"),
                        width="stretch",
                    )

