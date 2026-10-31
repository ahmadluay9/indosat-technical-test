"""
Expected Net Economic Value (ENEV) & Root-Cause Policy Decisioning Engine.

Methodological Scope & Assumptions:
1. Separates Churn Risk Prediction (`p_churn`) from Action Selection (`optimal_action`)
   by optimizing Expected Net Economic Value (ENEV).
2. Explicitly distinguishes:
   - **Live Deployable Campaign (`is_churn == 0`, `N = 574` Active Customers, `22` Actioned)**:
     Evaluated prospectively using model-estimated churn probability (`p_composite`) and the
     brief's treatment uplift parameters (`20%` Basic / `50%` Premium), yielding `3.51` expected
     saves on `Rp 9.30M` spend (`Rp 11.76M` net value / `126.5%` ROI under Static LTV `Rp 6.0M`;
     `Rp 18.76M` net value / `201.7%` ROI under Dynamic ARPU-scaled LTV).
   - **Retrospective Full-Cohort Backtest (`N = 719`, including `145` already-churned accounts)**:
     Evaluated retrospectively as an assumed-uplift upper-bound benchmark (`Rp 355.35M` static
     net value / `482.5%` ROI; `Rp 373.03M` dynamic net value / `506.5%` ROI) with customer-level
     non-parametric bootstrap `95% Confidence Intervals` (`B = 1,000`).
3. Because historical campaign treatment/control logs do not exist in the provided dataset,
   treatment uplift (`20%` relative churn reduction for Basic `Rp 150k`, `50%` for Premium `Rp 450k`)
   is treated as an explicit scenario parameter from the problem brief rather than a causally
   estimated CATE parameter. We therefore provide a full Uplift x LTV sensitivity matrix
   (`simulate_uplift_ltv_sensitivity`) for both the Live Active Campaign and Retrospective Backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


STATIC_LTV_IDR: float = 6_000_000.0
COST_BASIC_IDR: float = 150_000.0
UPLIFT_BASIC: float = 0.20

COST_PREMIUM_IDR: float = 450_000.0
UPLIFT_PREMIUM: float = 0.50

# Operational execution & contact handling cost assumptions (sensitivity line)
OPS_CONTACT_COST_DIGITAL_IDR: float = 15_000.0
OPS_CONTACT_COST_SPECIALIST_IDR: float = 35_000.0
DAILY_SPECIALIST_QUEUE_CAPACITY: int = 15  # Max <24h SLA specialist cases per day (live queue = 12)

# Exact analytical break-even thresholds under static V = Rp 6,000,000
BREAKEVEN_BASIC_VS_NONE: float = COST_BASIC_IDR / (
    UPLIFT_BASIC * STATIC_LTV_IDR
)  # 0.1250 (12.50%)
BREAKEVEN_PREMIUM_VS_NONE: float = COST_PREMIUM_IDR / (
    UPLIFT_PREMIUM * STATIC_LTV_IDR
)  # 0.1500 (15.00%)
BREAKEVEN_PREMIUM_VS_BASIC: float = (COST_PREMIUM_IDR - COST_BASIC_IDR) / (
    (UPLIFT_PREMIUM - UPLIFT_BASIC) * STATIC_LTV_IDR
)  # 0.1667 (16.67%)


@dataclass
class PolicySimulationResult:
    """Summary metrics for a retention targeting policy across Live Active (N=574) and Retrospective (N=719) cohorts."""

    policy_name: str
    # Live Active Campaign Metrics (is_churn == 0, N = 574)
    n_active_targeted: int
    n_active_basic: int
    n_active_premium: int
    live_active_cost_idr: float
    live_expected_saves: float
    live_net_value_static_idr: float
    live_roi_static_pct: float
    live_net_value_dynamic_idr: float
    live_roi_dynamic_pct: float
    # Retrospective Full-Cohort Backtest Metrics (N = 719) + Bootstrap 95% CIs
    n_targeted: int
    n_basic: int
    n_premium: int
    total_cost_idr: float
    expected_saves: float
    gross_value_saved_idr: float
    net_economic_value_idr: float
    net_value_ci95_low_idr: float
    net_value_ci95_high_idr: float
    roi_pct: float
    roi_ci95_low_pct: float
    roi_ci95_high_pct: float
    cost_per_save_idr: float
    net_value_dynamic_ltv_idr: float
    net_value_dynamic_ci95_low_idr: float
    net_value_dynamic_ci95_high_idr: float
    roi_dynamic_pct: float
    net_value_mismatch_adjusted_idr: float
    roi_mismatch_adjusted_pct: float


def compute_customer_economics_and_decisions(
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich the scored customer DataFrame with:
    1. Static LTV (`ltv_static_idr = 6,000,000`) and Dynamic ARPU-based LTV (`ltv_dynamic_idr`).
    2. Expected Net Economic Value (ENEV) for `DO_NOT_ACT`, `BASIC`, and `PREMIUM`
       under both Static and Dynamic LTV, without artificial floors.
    3. Optimal action tier under Static Reference rules (`action_tier_static`) and
       Dynamic Root-Cause Matched rules (`action_tier_dynamic`).
    4. Unified ENEV ranking (`enev_rank_active`, `1..22` on active actioned customers sorted
       strictly by `optimal_enev_dynamic_idr` descending) and Operational Queue assignment
       (`operational_queue`).
    """
    df = scored_df.copy()
    p = df["p_composite"].values

    # 1. Static vs Dynamic Customer Lifetime Value (V_i)
    df["ltv_static_idr"] = STATIC_LTV_IDR
    # Dynamic LTV parameterization: 20 months post-save horizon * 72% gross margin (14.4x monthly ARPU).
    # In service_usage.csv, monthlyCharges spans Rp 150,000/mo (Voice min -> Rp 2,160,000 LTV, Basic break-even p=34.72%)
    # to Rp 1,000,000/mo (Fiber max, with 83 customers > Rp 850k -> Rp 14,400,000 LTV, Basic break-even p=5.21%),
    # with Rp 416.7k/mo matching the static Rp 6.0M LTV.
    df["ltv_dynamic_idr"] = np.round(df["monthlyCharges"] * 20.0 * 0.72, 0)
    # Contract-Migration LTV Sensitivity (24-month post-save horizon * 72% gross margin = 17.28x monthly ARPU)
    df["ltv_contract_migration_24m_idr"] = np.round(
        df["monthlyCharges"] * 24.0 * 0.72, 0
    )

    # 2. Static ENEV Calculation (using composite risk probability p)
    df["enev_static_none"] = 0.0
    df["enev_static_basic"] = np.round(
        p * UPLIFT_BASIC * STATIC_LTV_IDR - COST_BASIC_IDR, 0
    )
    df["enev_static_premium"] = np.round(
        p * UPLIFT_PREMIUM * STATIC_LTV_IDR - COST_PREMIUM_IDR, 0
    )

    static_tier = np.where(
        p >= BREAKEVEN_PREMIUM_VS_BASIC,
        "PREMIUM",
        np.where(p >= BREAKEVEN_BASIC_VS_NONE, "BASIC", "DO_NOT_ACT"),
    )
    df["action_tier_static"] = static_tier
    df["optimal_enev_static_idr"] = np.maximum(
        0.0, np.maximum(df["enev_static_basic"], df["enev_static_premium"])
    )
    df["action_cost_static_idr"] = np.where(
        static_tier == "PREMIUM",
        COST_PREMIUM_IDR,
        np.where(static_tier == "BASIC", COST_BASIC_IDR, 0.0),
    )

    # 3. Dynamic ENEV Calculation (Heterogeneous V_i, un-floored true commercial ENEV)
    v_dyn = df["ltv_dynamic_idr"].values
    df["enev_dynamic_basic"] = np.round(
        p * UPLIFT_BASIC * v_dyn - COST_BASIC_IDR, 0
    )
    df["enev_dynamic_premium"] = np.round(
        p * UPLIFT_PREMIUM * v_dyn - COST_PREMIUM_IDR, 0
    )

    # Customer-specific break-even thresholds under Dynamic LTV
    df["breakeven_basic_dynamic"] = np.round(
        COST_BASIC_IDR / (UPLIFT_BASIC * v_dyn), 4
    )
    df["breakeven_premium_vs_basic_dynamic"] = np.round(
        (COST_PREMIUM_IDR - COST_BASIC_IDR)
        / ((UPLIFT_PREMIUM - UPLIFT_BASIC) * v_dyn),
        4,
    )

    dyn_tier: List[str] = []
    playbooks: List[str] = []
    guardrails_triggered: List[str] = []
    autonomy_tiers: List[str] = []
    urgency_sla_hours: List[int] = []
    action_costs_dyn: List[float] = []
    expected_net_benefit_dyn: List[float] = []
    expected_net_benefit_static_chosen: List[float] = []
    expected_saves_contrib: List[float] = []

    for i in range(len(df)):
        row = df.iloc[i]
        pi = float(row["p_composite"])
        eb = float(row["enev_dynamic_basic"])
        ep = float(row["enev_dynamic_premium"])
        eb_stat = float(row["enev_static_basic"])
        ep_stat = float(row["enev_static_premium"])
        driver = str(row["primary_driver"])
        has_disc = int(row["has_billing_discrepancy"]) == 1
        has_pend_tech = int(row["n_pending_tech"]) > 0
        has_pend_bill = int(row["n_pending_billing"]) > 0
        has_repeat = int(row["has_repeat_ticket"]) == 1
        conf = float(row["prediction_confidence"])

        # Deterministic Guardrail 1: Billing Ledger Overcharge Anomaly (e.g. IOH-553)
        # Note: True commercial ENEV is recorded honestly without an artificial floor (-Rp 124,613 dynamic / -Rp 124,200 static)
        # because this action is a mandatory billing compliance remediation rather than a commercial subsidy save.
        if has_disc:
            dyn_tier.append("BASIC")
            playbooks.append(
                "BILLING_CORRECTION_PLAYBOOK: Audit ledger discrepancy (+Rp 410k), issue immediate refund & Rp 150k goodwill credit"
            )
            guardrails_triggered.append(
                "GUARDRAIL_LEDGER_ANOMALY: Mandatory billing audit before any commercial offer"
            )
            autonomy_tiers.append(
                "Tier 2: Agent-Prepared Refund + Human Billing Approval"
            )
            urgency_sla_hours.append(24)
            action_costs_dyn.append(COST_BASIC_IDR)
            expected_net_benefit_dyn.append(eb)
            expected_net_benefit_static_chosen.append(eb_stat)
            expected_saves_contrib.append(pi * UPLIFT_BASIC)
            continue

        # Select economic tier based on Dynamic ENEV
        if ep >= eb and ep > 0:
            chosen = "PREMIUM"
            cost = COST_PREMIUM_IDR
            net_val = ep
            net_val_stat = ep_stat
            saves_c = pi * UPLIFT_PREMIUM
        elif eb > 0:
            chosen = "BASIC"
            cost = COST_BASIC_IDR
            net_val = eb
            net_val_stat = eb_stat
            saves_c = pi * UPLIFT_BASIC
        else:
            chosen = "DO_NOT_ACT"
            cost = 0.0
            net_val = 0.0
            net_val_stat = 0.0
            saves_c = 0.0

        # Deterministic Guardrail 2: Active customer with open Pending ticket must never be ignored
        if chosen == "DO_NOT_ACT" and (has_pend_tech or has_pend_bill):
            chosen = "BASIC"
            cost = COST_BASIC_IDR
            net_val = eb
            net_val_stat = eb_stat
            saves_c = pi * UPLIFT_BASIC

        dyn_tier.append(chosen)
        action_costs_dyn.append(cost)
        expected_net_benefit_dyn.append(net_val)
        expected_net_benefit_static_chosen.append(net_val_stat)
        expected_saves_contrib.append(saves_c)

        if chosen == "DO_NOT_ACT":
            playbooks.append(
                "SILENT_MONITORING: No outbound intervention (Prevent sleeping-dog disturbance & conserve budget)"
            )
            guardrails_triggered.append("NONE")
            autonomy_tiers.append("Tier 0: Do Not Act (Silent Monitoring)")
            urgency_sla_hours.append(720)
            continue

        # Root-cause matched playbook & guardrails for BASIC / PREMIUM
        if driver == "NETWORK_TECH_FAILURE":
            if chosen == "PREMIUM":
                playbooks.append(
                    "TECH_RECOVERY_PREMIUM: Fast-track Tier-2 NOC dispatch (<12h SLA) + Free Phone Line Bundle Upgrade + Rp 450k SLA Service Credit"
                )
            else:
                playbooks.append(
                    "TECH_PROACTIVE_BASIC: Automated line diagnostics + Priority ticket check-in + Rp 150k Data/Bill Credit"
                )

            if has_pend_tech or has_repeat:
                guardrails_triggered.append(
                    "GUARDRAIL_SUPPRESS_SALES_UPSELL: Block commercial lock-in pitch until Pending technical ticket is resolved"
                )
            else:
                guardrails_triggered.append(
                    "GUARDRAIL_POST_RESOLUTION_VERIFY: Verify 7-day line stability after technical ticket closure"
                )
        elif driver == "BILLING_PRICE_FRICTION":
            if chosen == "PREMIUM":
                playbooks.append(
                    "BILLING_RETENTION_PREMIUM: Immediate Pending billing resolution + Rp 450k Loyalty Rate Lock (12-Mo Plan + Auto-Pay Migration)"
                )
            else:
                playbooks.append(
                    "BILLING_CLARITY_BASIC: Proactive bill breakdown + Rp 150k E-Check to Auto-Pay Rebate"
                )

            if has_pend_bill:
                guardrails_triggered.append(
                    "GUARDRAIL_RESOLVE_BILLING_TICKET_FIRST: Must resolve open billing inquiry inside same interaction"
                )
            else:
                guardrails_triggered.append("NONE")
        else:  # CONTRACT_ONBOARDING_RISK
            if chosen == "PREMIUM":
                playbooks.append(
                    "STRUCTURAL_LOCKIN_PREMIUM: Month-to-Month to 1-Year Bundle Migration + Free Voice Line + Rp 450k Welcome Rebate"
                )
            else:
                playbooks.append(
                    "ONBOARDING_HABIT_BASIC: Early-tenure onboarding check-in + Rp 150k Auto-Pay Enrollment Bonus"
                )
            guardrails_triggered.append("NONE")

        # Assign Autonomy Tier & SLA Urgency (72-Hour Churn Clock)
        if has_repeat or (has_pend_tech and pi >= 0.85):
            autonomy_tiers.append(
                "Tier 3: Immediate Senior Retention Specialist + NOC Escalation (<12h SLA)"
            )
            urgency_sla_hours.append(12)
        elif chosen == "PREMIUM" or has_pend_bill or has_pend_tech or conf < 0.80:
            autonomy_tiers.append(
                "Tier 2: Agent-Drafted Offer & Ticket Context + Human Approval (<24h SLA)"
            )
            urgency_sla_hours.append(24)
        else:
            autonomy_tiers.append(
                "Tier 1: Fully Automated Digital Execution (Pre-Approved Basic Offer, <48h SLA)"
            )
            urgency_sla_hours.append(48)

    df["action_tier_dynamic"] = dyn_tier
    df["recommended_playbook"] = playbooks
    df["guardrail_triggered"] = guardrails_triggered
    df["autonomy_tier"] = autonomy_tiers
    df["urgency_sla_hours"] = urgency_sla_hours
    df["action_cost_dynamic_idr"] = action_costs_dyn
    df["optimal_enev_dynamic_idr"] = np.round(expected_net_benefit_dyn, 0)
    df["enev_static_chosen_idr"] = np.round(expected_net_benefit_static_chosen, 0)
    df["expected_saves_contribution"] = np.round(expected_saves_contrib, 4)

    # Operational Queue & Unified ENEV Rank on Actioned Active Customers (N=22)
    is_active_actioned = (df["is_churn"] == 0) & (
        df["action_tier_dynamic"] != "DO_NOT_ACT"
    )
    has_ticket_or_disc = (
        (df["n_pending"] > 0)
        | (df["has_billing_discrepancy"] == 1)
        | (df["n_tech"] > 0)
    )
    df["operational_queue"] = np.where(
        is_active_actioned & has_ticket_or_disc,
        "1_IMMEDIATE_SPECIALIST_SLA_24H",
        np.where(
            is_active_actioned,
            "2_PROACTIVE_ONBOARDING_LOCKIN",
            "0_DO_NOT_ACT_OR_HISTORICAL",
        ),
    )

    # Separate Commercial Retention Subsidies from Mandatory Billing Compliance (IOH-553)
    # and flag the 34 Pre-Second-Bill Month-1 MtM Onboarding subscribers (Stage 0 Nudge).
    is_mtm_m1_active = (
        (df["is_churn"] == 0)
        & (df["is_mtm"] == 1)
        & (df["tenureMonths"] == 1)
        & (df["action_tier_dynamic"] == "DO_NOT_ACT")
    )
    df["action_category"] = np.where(
        (df["is_churn"] == 0) & (df["has_billing_discrepancy"] == 1),
        "COMPLIANCE_BILLING_REMEDIATION",
        np.where(
            is_active_actioned,
            "COMMERCIAL_RETENTION_SUBSIDY",
            np.where(
                is_mtm_m1_active,
                "STAGE0_PRE_SECOND_BILL_NUDGE",
                "NONE",
            ),
        ),
    )

    # Operational execution & contact handling cost line (Rp 35k specialist / Rp 15k digital)
    df["ops_execution_cost_idr"] = np.where(
        df["operational_queue"] == "1_IMMEDIATE_SPECIALIST_SLA_24H",
        OPS_CONTACT_COST_SPECIALIST_IDR,
        np.where(
            df["operational_queue"] == "2_PROACTIVE_ONBOARDING_LOCKIN",
            OPS_CONTACT_COST_DIGITAL_IDR,
            0.0,
        ),
    )
    df["enev_dynamic_incl_ops_idr"] = np.where(
        df["action_tier_dynamic"] != "DO_NOT_ACT",
        df["optimal_enev_dynamic_idr"] - df["ops_execution_cost_idr"],
        0.0,
    )

    # Unified ranking across all outputs: rank 1..22 strictly by optimal_enev_dynamic_idr descending
    df["enev_rank_active"] = 0
    active_act_indices = (
        df.loc[is_active_actioned]
        .sort_values(
            ["optimal_enev_dynamic_idr", "p_composite", "monthlyCharges"],
            ascending=[False, False, False],
        )
        .index
    )
    df.loc[active_act_indices, "enev_rank_active"] = np.arange(
        1, len(active_act_indices) + 1
    )

    # Single unified priority_score = optimal_enev_dynamic_idr (no competing ad-hoc multiplier)
    df["priority_score"] = df["optimal_enev_dynamic_idr"]

    return df


def simulate_policy_comparison(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate and compare targeting policies across:
    A. **Live Actionable Active Population (`is_churn == 0`, `N = 574`)**:
       Prospective expected saves and expected net economic value under both Static LTV (`Rp 6.0M`)
       and Dynamic ARPU-scaled LTV (`V_i`), using `p_composite` and brief uplift (`20%` / `50%`).
    B. **Retrospective Full-Cohort Backtest (`N = 719`, including `145` historical churners)**:
       Realized-churn upper-bound backtest under Static LTV (`Lens 1`), Dynamic LTV (`Lens 2`),
       and Hypothetical Scenario Lens 3 (`50%` uplift attenuation on unrouted technical outages),
       complete with customer-level non-parametric bootstrap `95% Confidence Intervals` (`B = 1,000`).

    Also computes the **Live Active Budget Frontier (`is_churn == 0`, `N = 574`, ranked by Dynamic ENEV)**
    alongside the retrospective full-cohort reference curve.
    """
    y = df["is_churn"].values
    active_mask = (df["is_churn"] == 0).values
    p_comp = df["p_composite"].values
    v_dyn = df["ltv_dynamic_idr"].values
    has_unrouted_tech_outage = (df["n_pending_tech"].values > 0) | (
        df["has_billing_discrepancy"].values > 0
    )

    # Pre-generate bootstrap index matrix (B=1000 resamples of N=719 customers) for fast, exact CIs
    rng = np.random.default_rng(42)
    n_total = len(df)
    boot_idx = rng.integers(0, n_total, size=(1000, n_total))

    def _eval_tier_vector(
        policy_name: str,
        tiers: np.ndarray,
        has_root_cause_routing: bool = False,
    ) -> PolicySimulationResult:
        is_basic = tiers == "BASIC"
        is_prem = tiers == "PREMIUM"
        n_basic = int(is_basic.sum())
        n_prem = int(is_prem.sum())
        n_targ = n_basic + n_prem

        cost_vec = np.where(
            is_prem, COST_PREMIUM_IDR, np.where(is_basic, COST_BASIC_IDR, 0.0)
        )
        indiv_uplift = np.where(
            is_prem, UPLIFT_PREMIUM, np.where(is_basic, UPLIFT_BASIC, 0.0)
        )

        # --- A. Live Active Campaign Metrics (is_churn == 0, N = 574) ---
        act_basic = int((is_basic & active_mask).sum())
        act_prem = int((is_prem & active_mask).sum())
        act_targ = act_basic + act_prem
        live_cost = float(cost_vec[active_mask].sum())
        live_saves_vec = p_comp[active_mask] * indiv_uplift[active_mask]
        live_saves = float(live_saves_vec.sum())
        live_net_stat = float(live_saves * STATIC_LTV_IDR - live_cost)
        live_roi_stat = (live_net_stat / live_cost * 100.0) if live_cost > 0 else 0.0
        live_net_dyn = float(
            (live_saves_vec * v_dyn[active_mask]).sum() - live_cost
        )
        live_roi_dyn = (live_net_dyn / live_cost * 100.0) if live_cost > 0 else 0.0

        # --- B. Retrospective Full-Cohort Backtest Metrics (N = 719) ---
        cost = float(cost_vec.sum())
        saves_vec = y * indiv_uplift
        saves = float(saves_vec.sum())
        gross_vec_stat = saves_vec * STATIC_LTV_IDR
        net_vec_stat = gross_vec_stat - cost_vec
        gross_val = float(gross_vec_stat.sum())
        net_val = float(net_vec_stat.sum())
        roi = (net_val / cost * 100.0) if cost > 0 else 0.0
        cps = (cost / saves) if saves > 0 else 0.0

        # Lens 2: Dynamic ARPU-Scaled LTV (V_i) + Constant Assumed Uplift
        gross_vec_dyn = saves_vec * v_dyn
        net_vec_dyn = gross_vec_dyn - cost_vec
        net_dyn = float(net_vec_dyn.sum())
        roi_dyn = (net_dyn / cost * 100.0) if cost > 0 else 0.0

        # Bootstrap 95% CIs (B = 1,000 customer-level resamples)
        if n_targ > 0:
            boot_net_stat = net_vec_stat[boot_idx].sum(axis=1)
            boot_cost = cost_vec[boot_idx].sum(axis=1)
            boot_roi_stat = np.where(
                boot_cost > 0, boot_net_stat / boot_cost * 100.0, 0.0
            )
            boot_net_dyn = net_vec_dyn[boot_idx].sum(axis=1)
            ci_stat_low, ci_stat_high = np.percentile(boot_net_stat, [2.5, 97.5])
            ci_roi_low, ci_roi_high = np.percentile(boot_roi_stat, [2.5, 97.5])
            ci_dyn_low, ci_dyn_high = np.percentile(boot_net_dyn, [2.5, 97.5])
        else:
            ci_stat_low = ci_stat_high = 0.0
            ci_roi_low = ci_roi_high = 0.0
            ci_dyn_low = ci_dyn_high = 0.0

        # Lens 3 (Hypothetical Scenario): Root-Cause Mismatch-Adjusted Uplift (50% attenuation on unrouted technical outages)
        mismatch_mult = (
            np.ones(len(df), dtype=float)
            if has_root_cause_routing
            else np.where(has_unrouted_tech_outage, 0.50, 1.00)
        )
        gross_mismatch = float((y * indiv_uplift * mismatch_mult * STATIC_LTV_IDR).sum())
        net_mismatch = gross_mismatch - cost
        roi_mismatch = (net_mismatch / cost * 100.0) if cost > 0 else 0.0

        return PolicySimulationResult(
            policy_name=policy_name,
            n_active_targeted=act_targ,
            n_active_basic=act_basic,
            n_active_premium=act_prem,
            live_active_cost_idr=round(live_cost, 0),
            live_expected_saves=round(live_saves, 2),
            live_net_value_static_idr=round(live_net_stat, 0),
            live_roi_static_pct=round(live_roi_stat, 1),
            live_net_value_dynamic_idr=round(live_net_dyn, 0),
            live_roi_dynamic_pct=round(live_roi_dyn, 1),
            n_targeted=n_targ,
            n_basic=n_basic,
            n_premium=n_prem,
            total_cost_idr=float(cost),
            expected_saves=round(saves, 2),
            gross_value_saved_idr=float(gross_val),
            net_economic_value_idr=float(net_val),
            net_value_ci95_low_idr=round(float(ci_stat_low), 0),
            net_value_ci95_high_idr=round(float(ci_stat_high), 0),
            roi_pct=round(roi, 2),
            roi_ci95_low_pct=round(float(ci_roi_low), 1),
            roi_ci95_high_pct=round(float(ci_roi_high), 1),
            cost_per_save_idr=round(cps, 0),
            net_value_dynamic_ltv_idr=float(net_dyn),
            net_value_dynamic_ci95_low_idr=round(float(ci_dyn_low), 0),
            net_value_dynamic_ci95_high_idr=round(float(ci_dyn_high), 0),
            roi_dynamic_pct=round(roi_dyn, 2),
            net_value_mismatch_adjusted_idr=float(net_mismatch),
            roi_mismatch_adjusted_pct=round(roi_mismatch, 2),
        )

    n = len(df)
    mtm_mask = (df["contractType"] == "Month-to-month").values

    # Policy 1: Do Nothing
    p1 = _eval_tier_vector("1. Do Nothing (Baseline)", np.full(n, "DO_NOT_ACT"))

    # Policy 2: Blanket Basic to all Month-to-Month (N=306 total; 161 active)
    t_mtm_basic = np.where(mtm_mask, "BASIC", "DO_NOT_ACT")
    p2 = _eval_tier_vector("2. Blanket Basic (All Month-to-Month)", t_mtm_basic)

    # Policy 3: Blanket Premium to all Month-to-Month (N=306 total; 161 active)
    t_mtm_prem = np.where(mtm_mask, "PREMIUM", "DO_NOT_ACT")
    p3 = _eval_tier_vector("3. Blanket Premium (All Month-to-Month)", t_mtm_prem)

    # Policy 4a: 3-Rule Heuristic Baseline (MtM & E-Check & Tenure <= 12 -> Premium, N=266 total; 122 active)
    rule3_mask = (
        (df["is_mtm"] == 1)
        & (df["is_echeck"] == 1)
        & (df["is_early_tenure_le12"] == 1)
    ).values
    t_rule3 = np.where(rule3_mask, "PREMIUM", "DO_NOT_ACT")
    p4a = _eval_tier_vector(
        "4a. 3-Rule Heuristic (MtM+ECheck+Tenure<=12) -> Premium", t_rule3
    )

    # Policy 4b: Naive Risk Cutoff (Top 25% risk -> Premium, no Basic tier, N=180 total; 37 active)
    cutoff_25 = np.quantile(df["p_composite"], 0.75)
    t_top25 = np.where(df["p_composite"] >= cutoff_25, "PREMIUM", "DO_NOT_ACT")
    p4b = _eval_tier_vector("4b. Naive Top-Quartile Risk -> Premium", t_top25)

    # Policy 5: Model A Only (Structural Risk EV-Optimal, N=168 total; 30 active)
    p_s = df["p_structural"].values
    t_struct = np.where(
        p_s >= BREAKEVEN_PREMIUM_VS_BASIC,
        "PREMIUM",
        np.where(p_s >= BREAKEVEN_BASIC_VS_NONE, "BASIC", "DO_NOT_ACT"),
    )
    p5 = _eval_tier_vector(
        "5. Model A Only (Structural EV-Optimal)", t_struct
    )

    # Policy 6: Proposed Dual-Horizon Static-LTV EV-Optimal Policy (N=163 total; 20 active)
    p6 = _eval_tier_vector(
        "6. Proposed Dual-Horizon EV-Optimal (Static LTV)",
        df["action_tier_static"].values,
        has_root_cause_routing=False,
    )

    # Policy 7: Proposed Dual-Horizon Dynamic-LTV + Root-Cause Guardrails Policy (N=165 total; 22 active)
    p7 = _eval_tier_vector(
        "7. Proposed Dynamic-LTV + Guardrails Policy",
        df["action_tier_dynamic"].values,
        has_root_cause_routing=True,
    )

    policy_df = pd.DataFrame(
        [vars(r) for r in [p1, p2, p3, p4a, p4b, p5, p6, p7]]
    )

    # =========================================================================
    # Budget-Constrained Pareto Frontier Simulation:
    # Primary Curve: LIVE ACTIVE COHORT (is_churn == 0, N = 574) ranked by Dynamic ENEV
    # Paired Reference: RETROSPECTIVE FULL COHORT (N = 719)
    # =========================================================================
    active_df = (
        df.loc[df["is_churn"] == 0]
        .sort_values(
            ["optimal_enev_dynamic_idr", "enev_dynamic_premium", "p_composite"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    retro_df = df.sort_values("p_composite", ascending=False).reset_index(
        drop=True
    )

    live_budgets_idr = [
        1_800_000,
        3_600_000,
        5_400_000,
        7_200_000,
        9_150_000,
        9_300_000,
        15_000_000,
        25_000_000,
        54_900_000,
    ]
    retro_budgets_idr = [
        10_000_000,
        20_000_000,
        30_000_000,
        45_000_000,
        60_000_000,
        75_000_000,
        90_000_000,
        110_000_000,
        137_700_000,
    ]

    frontier_rows: List[Dict[str, float]] = []
    for b_live, b_retro in zip(live_budgets_idr, retro_budgets_idr):
        # 1. Live Active ENEV-Gated Frontier (stops at the 22 actioned active accounts = Rp 9.30M)
        cum_live_cost = 0.0
        live_tiers = np.full(len(active_df), "DO_NOT_ACT", dtype=object)
        for idx in range(len(active_df)):
            cand_tier = active_df.loc[idx, "action_tier_dynamic"]
            if cand_tier == "DO_NOT_ACT":
                continue
            c = COST_PREMIUM_IDR if cand_tier == "PREMIUM" else COST_BASIC_IDR
            if cum_live_cost + c <= b_live:
                live_tiers[idx] = cand_tier
                cum_live_cost += c

        is_lb = live_tiers == "BASIC"
        is_lp = live_tiers == "PREMIUM"
        p_act = active_df["p_composite"].values
        v_act = active_df["ltv_dynamic_idr"].values
        u_act = np.where(is_lp, UPLIFT_PREMIUM, np.where(is_lb, UPLIFT_BASIC, 0.0))
        live_saves = float((p_act * u_act).sum())
        live_gross_dyn = float((p_act * u_act * v_act).sum())
        live_net_dyn = live_gross_dyn - cum_live_cost
        live_roi_dyn = (
            (live_net_dyn / cum_live_cost * 100.0) if cum_live_cost > 0 else 0.0
        )
        live_gross_stat = live_saves * STATIC_LTV_IDR
        live_net_stat = live_gross_stat - cum_live_cost
        live_roi_stat = (
            (live_net_stat / cum_live_cost * 100.0) if cum_live_cost > 0 else 0.0
        )

        # Also compute Forced Un-Gated Active Spend (if budget b_live were forced onto next highest-ENEV active accounts)
        cum_forced_cost = 0.0
        forced_tiers = np.full(len(active_df), "DO_NOT_ACT", dtype=object)
        for idx in range(len(active_df)):
            cand_tier = active_df.loc[idx, "action_tier_dynamic"]
            if cand_tier == "DO_NOT_ACT":
                # If forced to spend up to b_live on top active accounts by p_composite
                cand_tier = "PREMIUM"
            c = COST_PREMIUM_IDR if cand_tier == "PREMIUM" else COST_BASIC_IDR
            if cum_forced_cost + c <= b_live:
                forced_tiers[idx] = cand_tier
                cum_forced_cost += c
        u_forced = np.where(
            forced_tiers == "PREMIUM",
            UPLIFT_PREMIUM,
            np.where(forced_tiers == "BASIC", UPLIFT_BASIC, 0.0),
        )
        forced_net_dyn = float((p_act * u_forced * v_act).sum() - cum_forced_cost)
        forced_roi_dyn = (
            (forced_net_dyn / cum_forced_cost * 100.0)
            if cum_forced_cost > 0
            else 0.0
        )

        # 2. Retrospective Full-Cohort Reference Curve (N=719)
        cum_retro_cost = 0.0
        tiers_b = np.full(len(retro_df), "DO_NOT_ACT", dtype=object)
        for idx in range(len(retro_df)):
            cand_tier = retro_df.loc[idx, "action_tier_static"]
            if cand_tier == "DO_NOT_ACT":
                continue
            c = COST_PREMIUM_IDR if cand_tier == "PREMIUM" else COST_BASIC_IDR
            if cum_retro_cost + c <= b_retro:
                tiers_b[idx] = cand_tier
                cum_retro_cost += c
            elif (
                cand_tier == "PREMIUM"
                and (cum_retro_cost + COST_BASIC_IDR <= b_retro)
                and retro_df.loc[idx, "p_composite"] >= BREAKEVEN_BASIC_VS_NONE
            ):
                tiers_b[idx] = "BASIC"
                cum_retro_cost += COST_BASIC_IDR

        y_r = retro_df["is_churn"].values
        is_b = tiers_b == "BASIC"
        is_p = tiers_b == "PREMIUM"
        retro_saves = float(
            (y_r[is_b].sum() * UPLIFT_BASIC)
            + (y_r[is_p].sum() * UPLIFT_PREMIUM)
        )
        retro_gross = retro_saves * STATIC_LTV_IDR
        retro_net = retro_gross - cum_retro_cost
        retro_roi = (
            (retro_net / cum_retro_cost * 100.0) if cum_retro_cost > 0 else 0.0
        )

        frontier_rows.append(
            {
                "live_budget_cap_idr": float(b_live),
                "live_spend_idr": float(cum_live_cost),
                "live_n_targeted": int((is_lb | is_lp).sum()),
                "live_n_premium": int(is_lp.sum()),
                "live_n_basic": int(is_lb.sum()),
                "live_expected_saves": round(live_saves, 2),
                "live_net_value_static_idr": round(live_net_stat, 0),
                "live_roi_static_pct": round(live_roi_stat, 1),
                "live_net_value_dynamic_idr": round(live_net_dyn, 0),
                "live_roi_dynamic_pct": round(live_roi_dyn, 1),
                "forced_ungated_spend_idr": float(cum_forced_cost),
                "forced_ungated_net_value_dynamic_idr": round(forced_net_dyn, 0),
                "forced_ungated_roi_dynamic_pct": round(forced_roi_dyn, 1),
                # Retrospective Full-Cohort Reference Columns (kept for backtest compatibility)
                "budget_cap_idr": float(b_retro),
                "spend_idr": float(cum_retro_cost),
                "n_targeted": int((is_b | is_p).sum()),
                "n_premium": int(is_p.sum()),
                "n_basic": int(is_b.sum()),
                "expected_saves": round(retro_saves, 2),
                "gross_value_saved_idr": float(retro_gross),
                "net_economic_value_idr": float(retro_net),
                "roi_pct": round(retro_roi, 2),
            }
        )

    frontier_df = pd.DataFrame(frontier_rows)
    return policy_df, frontier_df


def simulate_uplift_ltv_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sensitivity analysis of Policy 7 (`action_tier_dynamic`) across alternative
    Treatment Uplift multipliers and Customer LTV horizons for BOTH:
      1. **Live Actionable Active Campaign (`N = 22` targeted active accounts, `Rp 9.30M` subsidy cost + `Rp 0.57M` ops handling)**
      2. **Retrospective Full-Cohort Backtest (`N = 165` targeted accounts, `Rp 73.65M` cost)**

    Because the 20% (Basic) and 50% (Premium) uplift rates in the brief are assumed
    rather than causally estimated from an RCT, this table quantifies the exact downside
    and upside economic bounds of the proposed policy:
      - Uplift Multipliers:
          * `0.25x Severe Decay` (Basic=5.0%, Premium=12.5%)
          * `0.50x Conservative` (Basic=10.0%, Premium=25.0%)
          * `0.75x Moderate`     (Basic=15.0%, Premium=37.5%)
          * `1.00x Base Brief`   (Basic=20.0%, Premium=50.0%)
      - LTV Scenarios:
          * `Conservative 12m Horizon (Rp 3.60M)`
          * `Dynamic ARPU-Scaled (20m * 72% margin, Rp 2.16M - Rp 14.40M)`
          * `Contract-Migration 24m ARPU-Scaled (24m * 72% margin, Rp 2.59M - Rp 17.28M)`
          * `Base Brief Static (Rp 6.00M)`
          * `Extended 24m Fiber Horizon (Rp 7.20M)`
    """
    y = df["is_churn"].values
    active_mask = (df["is_churn"] == 0).values
    p_comp = df["p_composite"].values
    tiers = df["action_tier_dynamic"].values
    is_basic = tiers == "BASIC"
    is_prem = tiers == "PREMIUM"

    cost_retro = float(
        is_basic.sum() * COST_BASIC_IDR + is_prem.sum() * COST_PREMIUM_IDR
    )
    cost_live = float(
        (is_basic & active_mask).sum() * COST_BASIC_IDR
        + (is_prem & active_mask).sum() * COST_PREMIUM_IDR
    )
    ops_cost_live = float(df.loc[active_mask, "ops_execution_cost_idr"].sum())

    uplift_scenarios = [
        ("0.25x Severe Decay (5% / 12.5%)", 0.25),
        ("0.50x Conservative (10% / 25%)", 0.50),
        ("0.75x Moderate (15% / 37.5%)", 0.75),
        ("1.00x Base Brief (20% / 50%)", 1.00),
    ]
    ltv_scenarios = [
        ("Conservative 12m (Rp 3.6M)", np.full(len(df), 3_600_000.0)),
        ("Dynamic ARPU-Scaled (14.4x ARPU)", df["ltv_dynamic_idr"].values),
        (
            "Contract-Migration 24m (17.28x ARPU)",
            df["ltv_contract_migration_24m_idr"].values,
        ),
        ("Base Static Brief (Rp 6.0M)", np.full(len(df), 6_000_000.0)),
        ("Extended 24m (Rp 7.2M)", np.full(len(df), 7_200_000.0)),
    ]

    rows: List[Dict[str, object]] = []
    for u_label, u_mult in uplift_scenarios:
        u_b = UPLIFT_BASIC * u_mult
        u_p = UPLIFT_PREMIUM * u_mult
        indiv_u = np.where(is_prem, u_p, np.where(is_basic, u_b, 0.0))
        saves_retro = float((y * indiv_u).sum())
        saves_live = float((p_comp[active_mask] * indiv_u[active_mask]).sum())

        for ltv_label, ltv_vec in ltv_scenarios:
            # Live Active Campaign (N=22 targeted, Subsidy Cost=Rp 9.30M, Ops Handling=Rp 0.57M)
            gross_live = float(
                (p_comp[active_mask] * indiv_u[active_mask] * ltv_vec[active_mask]).sum()
            )
            net_live = gross_live - cost_live
            roi_live = (net_live / cost_live * 100.0) if cost_live > 0 else 0.0
            net_live_incl_ops = gross_live - (cost_live + ops_cost_live)
            roi_live_incl_ops = (
                (net_live_incl_ops / (cost_live + ops_cost_live) * 100.0)
                if (cost_live + ops_cost_live) > 0
                else 0.0
            )

            # Retrospective Full-Cohort Backtest (N=165 targeted, Cost=Rp 73.65M)
            gross_retro = float((y * indiv_u * ltv_vec).sum())
            net_retro = gross_retro - cost_retro
            roi_retro = (net_retro / cost_retro * 100.0) if cost_retro > 0 else 0.0

            rows.append(
                {
                    "uplift_scenario": u_label,
                    "ltv_scenario": ltv_label,
                    "uplift_multiplier": u_mult,
                    "live_expected_saves": round(saves_live, 2),
                    "live_program_cost_idr": cost_live,
                    "live_ops_handling_cost_idr": ops_cost_live,
                    "live_gross_value_saved_idr": round(gross_live, 0),
                    "live_net_economic_value_idr": round(net_live, 0),
                    "live_roi_pct": round(roi_live, 1),
                    "live_net_value_incl_ops_idr": round(net_live_incl_ops, 0),
                    "live_roi_incl_ops_pct": round(roi_live_incl_ops, 1),
                    "live_is_profitable": bool(net_live > 0),
                    "expected_saves": round(saves_retro, 2),
                    "program_cost_idr": cost_retro,
                    "gross_value_saved_idr": round(gross_retro, 0),
                    "net_economic_value_idr": round(net_retro, 0),
                    "roi_pct": round(roi_retro, 1),
                    "is_profitable": bool(net_retro > 0),
                }
            )

    return pd.DataFrame(rows)


def summarize_actionable_active_vs_retrospective(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicitly separates the Live Deployable Campaign on the Actionable Active Population
    (`N = 574`, `is_churn == 0`) from the Retrospective Full-Cohort Backtest (`N = 719`,
    which includes `145` historical churners who have already exited).

    Reconciles all value figures across Static LTV (`Rp 6.0M`) and Dynamic ARPU-Scaled LTV (`V_i`):
      - **Live Actionable Active Campaign (`N = 574`, `22` Actioned: `20 PREMIUM` + `2 BASIC`)**:
        * Spend: `Rp 9,300,000` (`Rp 9,150,000` commercial ENEV-positive + `Rp 150,000` mandatory `IOH-553` billing remediation; plus `Rp 570,000` operational contact handling cost).
        * Expected Saves: `3.51` subscribers (`sum(p_composite * uplift)`).
        * Static LTV (`Rp 6.0M`) Expected Net Value: `Rp 11,764,920` (`126.5%` ROI; `Rp 11,889,120` / `129.9%` ROI on the 21 commercial actions).
        * Dynamic LTV (`14.4x ARPU`) Expected Net Value: `Rp 18,759,762` (`201.7%` ROI; `Rp 18,884,375` / `206.4%` ROI on the 21 commercial actions; `Rp 18,189,762` after operational handling cost).
        * Operational Queues: `12` Immediate Specialist Priority (`<24h SLA`, ticket/billing triggered) + `10` Proactive Onboarding Lock-In (`0` tickets); `0` for already-churned historical accounts.
    """
    active_mask = df["is_churn"] == 0
    churn_mask = df["is_churn"] == 1
    active_comm_mask = active_mask & (df["has_billing_discrepancy"] == 0)
    active_comp_mask = active_mask & (df["has_billing_discrepancy"] == 1)
    active_m1_mtm_mask = (
        active_mask
        & (df["is_mtm"] == 1)
        & (df["tenureMonths"] == 1)
    )

    def _summarize_subset(
        sub: pd.DataFrame,
        scope_label: str,
        deployability: str,
        is_retro_realized: bool,
    ) -> Dict[str, object]:
        tiers = sub["action_tier_dynamic"].values
        is_prem = tiers == "PREMIUM"
        is_basic = tiers == "BASIC"
        is_act = is_prem | is_basic
        n_prem = int(is_prem.sum())
        n_basic = int(is_basic.sum())
        n_none = int((tiers == "DO_NOT_ACT").sum())
        n_act = n_prem + n_basic
        n_rule3 = int(sub["rule_mtm_echeck_le12"].sum())
        budget_idr = float(n_prem * COST_PREMIUM_IDR + n_basic * COST_BASIC_IDR)
        ops_cost_idr = float(sub["ops_execution_cost_idr"].sum())

        # Operational queue counts apply ONLY to live active accounts (is_churn == 0)
        n_imm = int(
            (sub["operational_queue"] == "1_IMMEDIATE_SPECIALIST_SLA_24H").sum()
        )
        n_pro = int(
            (sub["operational_queue"] == "2_PROACTIVE_ONBOARDING_LOCKIN").sum()
        )

        # Prospective expected metrics (using p_composite * uplift, un-floored)
        expected_saves_prospective = float(
            sub.loc[is_act, "expected_saves_contribution"].sum()
        )
        expected_net_stat = float(sub.loc[is_act, "enev_static_chosen_idr"].sum())
        expected_net_dyn = float(sub.loc[is_act, "optimal_enev_dynamic_idr"].sum())

        if is_retro_realized:
            # Also compute realized-churn retrospective saves and net value for historical rows
            u_vec = np.where(
                is_prem, UPLIFT_PREMIUM, np.where(is_basic, UPLIFT_BASIC, 0.0)
            )
            y_sub = sub["is_churn"].values
            v_sub = sub["ltv_dynamic_idr"].values
            saves_val = float((y_sub * u_vec).sum())
            net_stat_val = float(saves_val * STATIC_LTV_IDR - budget_idr)
            net_dyn_val = float((y_sub * u_vec * v_sub).sum() - budget_idr)
        else:
            saves_val = expected_saves_prospective
            net_stat_val = expected_net_stat
            net_dyn_val = expected_net_dyn

        roi_stat = (net_stat_val / budget_idr * 100.0) if budget_idr > 0 else 0.0
        roi_dyn = (net_dyn_val / budget_idr * 100.0) if budget_idr > 0 else 0.0

        return {
            "population_scope": scope_label,
            "operational_status": deployability,
            "n_customers": int(len(sub)),
            "n_3rule_heuristic_flagged": n_rule3,
            "n_actioned_policy7": n_act,
            "n_immediate_specialist_priority": n_imm,
            "n_proactive_onboarding": n_pro,
            "n_premium": n_prem,
            "n_basic": n_basic,
            "n_do_not_act": n_none,
            "program_budget_idr": round(budget_idr, 0),
            "ops_handling_cost_idr": round(ops_cost_idr, 0),
            "expected_saves": round(saves_val, 2),
            "expected_net_value_static_idr": round(net_stat_val, 0),
            "roi_static_pct": round(roi_stat, 1),
            "expected_net_value_dynamic_idr": round(net_dyn_val, 0),
            "roi_dynamic_pct": round(roi_dyn, 1),
        }

    rows = [
        _summarize_subset(
            df.loc[active_mask],
            "1. Actionable Active Population (is_churn == 0)",
            "LIVE DEPLOYABLE CAMPAIGN (22 Actioned: 12 Immediate Specialist <24h SLA + 10 Proactive Onboarding)",
            is_retro_realized=False,
        ),
        _summarize_subset(
            df.loc[active_comm_mask],
            "1a. Live Commercial Retention Cohort (Positive-ENEV Only, Excl. IOH-553)",
            "COMMERCIAL RETENTION CAMPAIGN (21 Positive-ENEV Actions: 20 Premium + 1 Basic)",
            is_retro_realized=False,
        ),
        _summarize_subset(
            df.loc[active_comp_mask],
            "1b. Live Mandatory Billing Compliance Remediation (IOH-553 Anomaly)",
            "BILLING COMPLIANCE REMEDIATION (Rp 410k Overcharge Refund + Rp 150k Goodwill Credit)",
            is_retro_realized=False,
        ),
        _summarize_subset(
            df.loc[active_m1_mtm_mask],
            "1c. Pre-Second-Bill Month-1 MtM Onboarding Cohort (Stage 0 Nudge)",
            "ZERO-SUBSIDY PRE-SECOND-BILL AUTO-PAY & USAGE CHECK-IN (34 Active Month-1 MtM Accounts)",
            is_retro_realized=False,
        ),
        _summarize_subset(
            df.loc[churn_mask],
            "2. Historical Churned Population (is_churn == 1)",
            "RETROSPECTIVE OUTCOME VERIFICATION ONLY (Already Exited - 0 Live Specialist Actions)",
            is_retro_realized=True,
        ),
        _summarize_subset(
            df,
            "3. Full Dataset Extract (Active + Historical Churn)",
            "RETROSPECTIVE FULL-COHORT BACKTEST ONLY (Assumed-Uplift Upper Bound - Not a Live Campaign)",
            is_retro_realized=True,
        ),
    ]
    return pd.DataFrame(rows)
