"""
End-to-End Runner for IOH Customer Churn & Agentic Retention Intelligence.

Executes:
1. Forensic Data Hygiene, Point-in-Time Feature Engineering, Survival/Hazard & Modulo-200 Audits (`src.data_pipeline`).
2. Dual-Horizon Churn Risk Modeling, 4-Scheme Cross-Validation & Root-Cause Attribution (`src.models`).
3. Economic & Causal Uplift Decision Optimization (`src.decision_engine`).
4. LangGraph & LangChain Agentic Retention Workflow & Statistical Power Analysis (`src.retention_agent`).
5. Exports structured CSVs/JSONs to `outputs/` and all 8 Executive Figures to `figures/`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.data_pipeline import (
    build_point_in_time_feature_matrix,
    compute_modulo200_duplication_audit,
    compute_tenure_hazard_and_survival_table,
    load_and_clean_tables,
)
from src.decision_engine import (
    compute_customer_economics_and_decisions,
    simulate_policy_comparison,
    simulate_uplift_ltv_sensitivity,
    summarize_actionable_active_vs_retrospective,
)
from src.figures import generate_all_figures
from src.models import (
    compute_fairness_and_parity_audit,
    compute_feature_ablation_on_active_rule3,
    train_and_evaluate_dual_horizon_models,
)
from src.retention_agent import (
    build_retention_langgraph,
    compute_experiment_power_analysis,
    generate_sample_agent_payloads,
    get_prioritized_active_watchlist,
    summarize_cohort_rct_holdout,
)


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    out_dir = root_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("IOH CUSTOMER CHURN & LANGGRAPH AGENTIC RETENTION INTELLIGENCE PIPELINE")
    print("=" * 80)

    # 1. Load & Clean Data
    cb_clean, su_clean, si_clean, quarantine_df, audit_summary = (
        load_and_clean_tables(root_dir)
    )
    print("\n[1] FORENSIC DATA HYGIENE SUMMARY:")
    for k, v in asdict(audit_summary).items():
        print(f"  - {k}: {v}")

    # 2. Build Point-in-Time Feature Matrix + Survival Hazard & Modulo-200 Audits
    pit_df = build_point_in_time_feature_matrix(cb_clean, su_clean, si_clean)
    hazard_df = compute_tenure_hazard_and_survival_table(pit_df)
    mod200_df = compute_modulo200_duplication_audit(pit_df)
    print(f"\n[2] POINT-IN-TIME FEATURE MATRIX BUILT: shape={pit_df.shape}")
    print("\n[2a] DISCRETE-TIME TENURE HAZARD & MONTH-2 CHURN CLIFF AUDIT (MONTHS 1-6):")
    print(hazard_df.loc[hazard_df["tenure_month"] <= 6].to_string(index=False))
    print("\n[2b] MODULO-200 NEAR-DUPLICATE GENERATION AUDIT:")
    print(mod200_df.to_string(index=False))

    # 3. Train & Evaluate Dual-Horizon Risk Models (4 CV Schemes incl. Modulo-200 Holdout)
    scored_df, eval_df, feature_imp_df = train_and_evaluate_dual_horizon_models(
        pit_df
    )
    ablation_df = compute_feature_ablation_on_active_rule3(scored_df)
    print("\n[3] DUAL-HORIZON MODEL EVALUATION (4 CROSS-VALIDATION SCHEMES):")
    print(
        eval_df[
            [
                "horizon",
                "model_name",
                "validation_scheme",
                "roc_auc",
                "pr_auc",
                "brier_score",
                "f1_at_05",
            ]
        ].to_string(index=False)
    )

    # 4. Economic & Causal Decisioning Engine
    decision_df = compute_customer_economics_and_decisions(scored_df)
    scope_summary_df = summarize_actionable_active_vs_retrospective(decision_df)
    policy_df, frontier_df = simulate_policy_comparison(decision_df)
    sensitivity_df = simulate_uplift_ltv_sensitivity(decision_df)
    fairness_df = compute_fairness_and_parity_audit(decision_df)
    rct_summary_df = summarize_cohort_rct_holdout(decision_df)
    power_df = compute_experiment_power_analysis()

    print("\n[4a] ACTIONABLE ACTIVE POPULATION (N=574) vs. RETROSPECTIVE BACKTEST SCOPE (N=719):")
    print(scope_summary_df.to_string(index=False))
    print("\n[4b] RETROSPECTIVE FULL-COHORT POLICY BACKTEST (N=719, HISTORICAL EVALUATION ONLY):")
    print(
        policy_df[
            [
                "policy_name",
                "n_targeted",
                "n_basic",
                "n_premium",
                "total_cost_idr",
                "expected_saves",
                "net_economic_value_idr",
                "roi_pct",
                "net_value_dynamic_ltv_idr",
                "net_value_mismatch_adjusted_idr",
            ]
        ].to_string(index=False)
    )
    print("\n[4c] UPLIFT x LTV SENSITIVITY MATRIX (LIVE COMMERCIAL vs. RETROSPECTIVE):")
    print(
        sensitivity_df[
            [
                "uplift_scenario",
                "ltv_scenario",
                "uplift_multiplier",
                "live_expected_saves",
                "live_program_cost_idr",
                "live_net_economic_value_idr",
                "live_roi_pct",
                "expected_saves",
                "program_cost_idr",
                "net_economic_value_idr",
                "roi_pct",
            ]
        ].to_string(index=False)
    )
    print("\n[4d] DEMOGRAPHIC & REGIONAL FAIRNESS / PARITY AUDIT:")
    print(fairness_df.to_string(index=False))
    print("\n[4e] RCT STATISTICAL POWER & MINIMUM DETECTABLE EFFECT (MDE) ANALYSIS:")
    print(power_df.to_string(index=False))

    # 5. LangGraph Agentic Retention Workflow & Active Watchlist
    compiled_graph = build_retention_langgraph(use_memory_checkpointer=True)
    watchlist_df = get_prioritized_active_watchlist(decision_df, top_n=25)
    sample_payloads = generate_sample_agent_payloads(decision_df)

    print(
        "\n[5] TOP 12 ACTIVE CUSTOMERS REQUIRING IMMEDIATE ACTION (<24H SLA, is_churn == 0):"
    )
    cols_watch = [
        "customerID",
        "region",
        "contractType",
        "tenureMonths",
        "hasInternetService",
        "hasPhoneService",
        "n_pending",
        "has_billing_discrepancy",
        "p_structural",
        "p_acute",
        "p_composite",
        "primary_driver",
        "action_tier_dynamic",
        "action_category",
        "urgency_sla_hours",
    ]
    print(watchlist_df[cols_watch].head(12).to_string(index=False))

    print("\n[6] LANGGRAPH STATEGRAPH SHOWCASE EXECUTION TRACES:")
    for payload in sample_payloads:
        print(
            f"  * {payload['customer_id']} ({payload['account_status']}) | "
            f"Route={payload['langgraph_route']} | "
            f"Nodes={' -> '.join(payload['langgraph_nodes_visited'])} | "
            f"Tools={[t['tool_name'] for t in payload['authorized_tool_executions']]} | "
            f"Approval={payload['human_approval_status']}"
        )

    # Export artifacts to outputs/
    decision_df.to_csv(out_dir / "retention_decisions_all_719.csv", index=False)
    scope_summary_df.to_csv(
        out_dir / "active_vs_retrospective_campaign_summary.csv", index=False
    )
    watchlist_df.to_csv(out_dir / "active_watchlist_prioritized.csv", index=False)
    eval_df.to_csv(out_dir / "model_evaluation_metrics.csv", index=False)
    feature_imp_df.to_csv(out_dir / "model_feature_importances.csv", index=False)
    ablation_df.to_csv(out_dir / "active_rule3_feature_ablation.csv", index=False)
    hazard_df.to_csv(out_dir / "tenure_survival_hazard_table.csv", index=False)
    mod200_df.to_csv(out_dir / "modulo200_duplication_audit.csv", index=False)
    policy_df.to_csv(out_dir / "economic_policy_comparison.csv", index=False)
    frontier_df.to_csv(out_dir / "budget_pareto_frontier.csv", index=False)
    sensitivity_df.to_csv(out_dir / "uplift_ltv_sensitivity_matrix.csv", index=False)
    fairness_df.to_csv(out_dir / "fairness_and_parity_audit.csv", index=False)
    rct_summary_df.to_csv(out_dir / "rct_holdout_summary.csv", index=False)
    power_df.to_csv(out_dir / "rct_power_analysis.csv", index=False)
    quarantine_df.to_csv(out_dir / "quarantined_records.csv", index=False)

    with open(
        out_dir / "sample_langgraph_agent_payloads.json", "w", encoding="utf-8"
    ) as f:
        json.dump(sample_payloads, f, indent=2, ensure_ascii=False)

    # Export Mermaid diagram of the compiled LangGraph StateGraph
    mermaid_str = compiled_graph.get_graph().draw_mermaid()
    with open(
        out_dir / "langgraph_retention_workflow.mmd", "w", encoding="utf-8"
    ) as f:
        f.write(mermaid_str)

    # Generate all 8 executive figures
    fig_paths = generate_all_figures(
        pit_df=pit_df,
        scored_df=decision_df,
        eval_df=eval_df,
        feature_imp_df=feature_imp_df,
        policy_df=policy_df,
        frontier_df=frontier_df,
        fairness_df=fairness_df,
        survival_df=hazard_df,
        mod200_df=mod200_df,
        figures_dir=root_dir / "figures",
    )
    print(f"\n[7] GENERATED {len(fig_paths)} EXECUTIVE FIGURES IN: {root_dir / 'figures'}")
    print(f"[8] ALL OUTPUTS EXPORTED TO: {out_dir}")


if __name__ == "__main__":
    main()
