"""
Forensic Data Hygiene & Leakage-Safe Point-in-Time Feature Engineering Pipeline.

This module implements:
1. Schema & primary-key validation (detecting and quarantining corrupted orphan ID 'IOH-1pictures').
2. Canonical string normalization for `region` (10 variants -> 5 cities) and `supportTopic` (5 variants -> 3 topics).
3. Mixed-format date parsing (`YYYY-MM-DD` and `DD/MM/YYYY`) for `support_interactions.interactionDate`.
4. Exact imputation for `tenureMonths == 1` blank `totalCharges` and billing anomaly detection (`IOH-553`).
5. Strict Point-in-Time (PIT) feature engineering and structural diagnostics that document:
   - Trap 1: Multi-row warehouse extraction leakage (`n_usage_rows == 2` only for churners: 134/145 vs 0/574 active).
   - Trap 2: Source-system region casing leakage (`bandung` 97.6% churn [40/41] vs `bdg` 7.7% churn [5/65]).
   - Trap 3: Post-churn support ticket leakage (`interactionDate > churnDate` for IOH-003, IOH-351, IOH-551).
   - Caveat 4: Temporal Snapshot / Right-Censoring Asymmetry (574 active customers are observed at `2025-09` [n=6],
     `2025-10` [n=167], and `2025-11` [n=401], whereas 145 churners are frozen at their exit month `2025-04` to
     `2025-11`, with `churnDate` spanning `2025-04-18` to `2025-11-28`).
   - Caveat 5: Near Rule-Generated Structure (`Month-to-month AND Electronic check AND tenureMonths <= 12`
     captures 144/145 churners; within that 266-customer cohort, Fiber optic churns at 78.2% [136/174] vs DSL at
     8.7% [8/92]) and Outcome-Adjacent `Pending` Ticket Status (135/142 pre-churn churners with tickets vs 5/574 active).
   - Caveat 6: Modulo-200 Synthetic Template Duplication (`template_group = (cust_num - 1) % 200`): 63.2% of
     `(id, id+200)` pairs share identical contract, payment method, internet, phone, and city (vs 4.8% random pairs)
     and 90.4% share the same churn outcome, requiring `GroupKFold` by `template_group` to prevent cross-fold leakage.
   - Caveat 7: Month-2 Churn Cliff (`126/145` [86.9%] of all churns occur at `tenureMonths == 2`, where discrete-time
     Month-to-month hazard spikes from 0.7% in Month 1 to 46.7% in Month 2 before dropping to 5.1% in Month 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REGION_CANONICAL_MAP: Dict[str, str] = {
    "jakarta": "Jakarta",
    "jkt": "Jakarta",
    "bandung": "Bandung",
    "bdg": "Bandung",
    "surabaya": "Surabaya",
    "sby": "Surabaya",
    "medan": "Medan",
    "semarang": "Semarang",
}

TOPIC_CANONICAL_MAP: Dict[str, str] = {
    "technical_issue": "technical_issue",
    "technical issue": "technical_issue",
    "billing_inquiry": "billing_inquiry",
    "billing inquiry": "billing_inquiry",
    "contract_question": "contract_question",
    "contract question": "contract_question",
}


@dataclass
class DataAuditSummary:
    """Structured summary of forensic data quality findings and leakage traps."""

    raw_customer_rows: int
    valid_customer_rows: int
    orphan_customer_ids: List[str]
    raw_region_unique_count: int
    clean_region_unique_count: int
    raw_topic_unique_count: int
    clean_topic_unique_count: int
    missing_gender_count: int
    missing_partner_count: int
    blank_total_charges_count: int
    billing_anomaly_ids: List[str]
    post_churn_ticket_ids: List[str]
    multi_row_churn_count: int
    multi_row_active_count: int
    standalone_fiber_count: int
    standalone_fiber_churn_rate: float
    bundled_fiber_count: int
    bundled_fiber_churn_rate: float
    voice_with_mobile_data_count: int


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """
    Parse dates that appear in either ISO `YYYY-MM-DD` or `DD/MM/YYYY` format.
    """
    s = series.fillna("").astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    slash_mask = s.str.contains("/", regex=False)
    dash_mask = (~slash_mask) & (s != "")

    if slash_mask.any():
        parsed.loc[slash_mask] = pd.to_datetime(
            s.loc[slash_mask], format="%d/%m/%Y", errors="coerce"
        )
    if dash_mask.any():
        parsed.loc[dash_mask] = pd.to_datetime(
            s.loc[dash_mask], format="%Y-%m-%d", errors="coerce"
        )
    return parsed


def load_and_clean_tables(
    data_dir: str | Path = ".",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, DataAuditSummary]:
    """
    Load raw CSV tables, perform forensic cleaning, quarantine corrupted records,
    and return cleaned tables plus a structured `DataAuditSummary`.
    """
    data_path = Path(data_dir)
    cb_raw = pd.read_csv(data_path / "customer_base.csv")
    su_raw = pd.read_csv(data_path / "service_usage.csv")
    si_raw = pd.read_csv(data_path / "support_interactions.csv")

    # Standardize customerId / customerID alias so both work seamlessly
    for tbl in (cb_raw, su_raw, si_raw):
        if "customerId" in tbl.columns and "customerID" not in tbl.columns:
            tbl["customerID"] = tbl["customerId"]

    # 1. Identify valid vs. corrupted/orphan customer IDs
    cb = cb_raw.copy()
    cb["cust_num"] = pd.to_numeric(
        cb["customerID"].astype(str).str.extract(r"^IOH-(\d+)$")[0], errors="coerce"
    )
    in_usage = cb["customerID"].isin(set(su_raw["customerID"]))
    in_support = cb["customerID"].isin(set(si_raw["customerID"]))
    valid_mask = cb["cust_num"].notna() & (in_usage | in_support)

    quarantine_df = cb.loc[~valid_mask].copy()
    quarantine_df["quarantine_reason"] = (
        "Corrupted customerId format ('IOH-1pictures') & zero records in service_usage/support_interactions"
    )
    orphan_ids = quarantine_df["customerID"].tolist()

    cb_clean = cb.loc[valid_mask].copy().reset_index(drop=True)
    cb_clean["cust_num"] = cb_clean["cust_num"].astype(int)

    # 2. Canonicalize customer_base fields
    raw_region_count = cb_clean["region"].nunique(dropna=True)
    cb_clean["raw_region"] = cb_clean["region"].astype(str)
    cb_clean["region"] = (
        cb_clean["region"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(REGION_CANONICAL_MAP)
        .fillna("Unknown")
    )
    clean_region_count = cb_clean["region"].nunique(dropna=True)
    # Contiguous customer ID block (5 blocks of ~144 accounts) and Modulo-200 synthetic template group (200 templates)
    cb_clean["id_block_group"] = (cb_clean["cust_num"] - 1) // 144
    cb_clean["template_group"] = (cb_clean["cust_num"] - 1) % 200

    missing_gender_count = int(
        cb_clean["gender"].isna().sum()
        + (cb_clean["gender"].fillna("").astype(str).str.strip() == "").sum()
    )
    missing_partner_count = int(
        cb_clean["partner"].isna().sum()
        + (cb_clean["partner"].fillna("").astype(str).str.strip() == "").sum()
    )

    cb_clean["gender"] = (
        cb_clean["gender"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown"})
    )
    cb_clean["partner"] = (
        cb_clean["partner"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown"})
    )
    cb_clean["dependents"] = (
        cb_clean["dependents"].fillna("No").astype(str).str.strip()
    )
    cb_clean["age"] = (
        pd.to_numeric(cb_clean["age"], errors="coerce").fillna(40).astype(int)
    )
    cb_clean["seniorCitizen"] = (cb_clean["age"] >= 60).astype(int)
    cb_clean["churnDate_dt"] = parse_mixed_dates(cb_clean["churnDate"])
    cb_clean["is_churn"] = cb_clean["churnDate_dt"].notna().astype(int)

    # 3. Clean service_usage
    su_clean = su_raw.copy()
    su_clean["monthlyCharges"] = pd.to_numeric(
        su_clean["monthlyCharges"], errors="coerce"
    )
    su_clean["tenureMonths"] = pd.to_numeric(
        su_clean["tenureMonths"], errors="coerce"
    ).astype(int)
    su_clean["dataUsageGB"] = pd.to_numeric(su_clean["dataUsageGB"], errors="coerce")

    tc_parsed = pd.to_numeric(
        su_clean["totalCharges"].astype(str).str.strip(), errors="coerce"
    )
    blank_tc_mask = tc_parsed.isna()
    blank_tc_count = int(blank_tc_mask.sum())

    su_clean["totalCharges_is_imputed"] = blank_tc_mask.astype(int)
    su_clean["expected_totalCharges"] = (
        su_clean["monthlyCharges"] * su_clean["tenureMonths"]
    )
    su_clean["totalCharges"] = tc_parsed.fillna(su_clean["expected_totalCharges"])
    su_clean["billing_discrepancy_idr"] = (
        su_clean["totalCharges"] - su_clean["expected_totalCharges"]
    )
    su_clean["has_billing_discrepancy"] = (
        su_clean["billing_discrepancy_idr"].abs() > 1.0
    ).astype(int)

    billing_anomaly_ids = (
        su_clean.loc[su_clean["has_billing_discrepancy"] == 1, "customerID"]
        .unique()
        .tolist()
    )

    # 4. Clean support_interactions
    si_clean = si_raw.copy()
    raw_topic_count = si_clean["supportTopic"].nunique(dropna=True)
    si_clean["raw_supportTopic"] = si_clean["supportTopic"].astype(str)
    si_clean["supportTopic"] = (
        si_clean["supportTopic"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(TOPIC_CANONICAL_MAP)
        .fillna("other")
    )
    clean_topic_count = si_clean["supportTopic"].nunique(dropna=True)
    si_clean["resolutionStatus"] = (
        si_clean["resolutionStatus"].astype(str).str.strip().str.title()
    )
    si_clean["interactionType"] = (
        si_clean["interactionType"].astype(str).str.strip().str.lower()
    )
    si_clean["interactionDate_dt"] = parse_mixed_dates(si_clean["interactionDate"])

    si_with_churn = si_clean.merge(
        cb_clean[["customerID", "churnDate_dt", "is_churn"]],
        on="customerID",
        how="left",
    )
    si_with_churn["days_before_churn"] = (
        si_with_churn["churnDate_dt"] - si_with_churn["interactionDate_dt"]
    ).dt.days
    post_churn_mask = (si_with_churn["is_churn"] == 1) & (
        si_with_churn["days_before_churn"] < 0
    )
    post_churn_ticket_ids = (
        si_with_churn.loc[post_churn_mask, "customerID"].unique().tolist()
    )
    si_clean["is_post_churn_leakage"] = post_churn_mask.astype(int)
    si_clean["days_before_churn"] = si_with_churn["days_before_churn"]

    # Compute audit statistics using latest usage snapshot per customer
    usage_counts = su_clean.groupby("customerID").size()
    cb_usage_counts = cb_clean["customerID"].map(usage_counts).fillna(0).astype(int)
    multi_row_churn = int(((cb_clean["is_churn"] == 1) & (cb_usage_counts > 1)).sum())
    multi_row_active = int(((cb_clean["is_churn"] == 0) & (cb_usage_counts > 1)).sum())

    su_latest_tmp = (
        su_clean.sort_values(["customerID", "yearMonth"])
        .groupby("customerID", as_index=False)
        .last()
    )
    cb_with_prod = cb_clean.merge(
        su_latest_tmp[["customerID", "hasInternetService", "hasPhoneService"]],
        on="customerID",
        how="left",
    )
    standalone_fiber_mask = (
        cb_with_prod["hasInternetService"] == "Fiber optic"
    ) & (cb_with_prod["hasPhoneService"] == "No")
    bundled_fiber_mask = (
        cb_with_prod["hasInternetService"] == "Fiber optic"
    ) & (cb_with_prod["hasPhoneService"] == "Yes")
    voice_no_internet_mask = (cb_with_prod["hasInternetService"] == "No") & (
        cb_with_prod["hasPhoneService"] == "Yes"
    )

    audit_summary = DataAuditSummary(
        raw_customer_rows=len(cb_raw),
        valid_customer_rows=len(cb_clean),
        orphan_customer_ids=orphan_ids,
        raw_region_unique_count=raw_region_count,
        clean_region_unique_count=clean_region_count,
        raw_topic_unique_count=raw_topic_count,
        clean_topic_unique_count=clean_topic_count,
        missing_gender_count=missing_gender_count,
        missing_partner_count=missing_partner_count,
        blank_total_charges_count=blank_tc_count,
        billing_anomaly_ids=billing_anomaly_ids,
        post_churn_ticket_ids=post_churn_ticket_ids,
        multi_row_churn_count=multi_row_churn,
        multi_row_active_count=multi_row_active,
        standalone_fiber_count=int(standalone_fiber_mask.sum()),
        standalone_fiber_churn_rate=float(
            cb_with_prod.loc[standalone_fiber_mask, "is_churn"].mean()
        ),
        bundled_fiber_count=int(bundled_fiber_mask.sum()),
        bundled_fiber_churn_rate=float(
            cb_with_prod.loc[bundled_fiber_mask, "is_churn"].mean()
        ),
        voice_with_mobile_data_count=int(voice_no_internet_mask.sum()),
    )

    return cb_clean, su_clean, si_clean, quarantine_df, audit_summary


def build_point_in_time_feature_matrix(
    cb_clean: pd.DataFrame,
    su_clean: pd.DataFrame,
    si_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the leakage-free, point-in-time customer feature matrix (N=719).
    """
    su_sorted = su_clean.sort_values(["customerID", "yearMonth"]).reset_index(drop=True)
    su_latest = su_sorted.groupby("customerID", as_index=False).last()
    su_first = su_sorted.groupby("customerID", as_index=False).first()
    su_counts = (
        su_sorted.groupby("customerID")
        .size()
        .rename("diagnostic_n_usage_rows")
        .reset_index()
    )

    df = cb_clean.merge(
        su_latest[
            [
                "customerID",
                "yearMonth",
                "monthlyCharges",
                "totalCharges",
                "tenureMonths",
                "hasPhoneService",
                "hasInternetService",
                "dataUsageGB",
                "totalCharges_is_imputed",
                "billing_discrepancy_idr",
                "has_billing_discrepancy",
            ]
        ],
        on="customerID",
        how="left",
    )

    df = df.merge(su_counts, on="customerID", how="left")
    df = df.merge(
        su_first[["customerID", "dataUsageGB", "monthlyCharges"]].rename(
            columns={
                "dataUsageGB": "diagnostic_first_dataUsageGB",
                "monthlyCharges": "diagnostic_first_monthlyCharges",
            }
        ),
        on="customerID",
        how="left",
    )
    df["diagnostic_usage_delta_gb"] = np.where(
        df["diagnostic_n_usage_rows"] > 1,
        df["dataUsageGB"] - df["diagnostic_first_dataUsageGB"],
        np.nan,
    )

    # Filter support interactions to strict point-in-time (no post-churn tickets)
    si_pit = si_clean.loc[si_clean["is_post_churn_leakage"] == 0].copy()
    si_pit = si_pit.sort_values(["customerID", "interactionDate_dt"]).reset_index(
        drop=True
    )

    if len(si_pit) > 0:
        si_agg = (
            si_pit.groupby("customerID")
            .agg(
                n_tickets=("interactionDate", "count"),
                n_pending=(
                    "resolutionStatus",
                    lambda x: int((x == "Pending").sum()),
                ),
                n_resolved=(
                    "resolutionStatus",
                    lambda x: int((x == "Resolved").sum()),
                ),
                n_escalated=(
                    "resolutionStatus",
                    lambda x: int((x == "Escalated").sum()),
                ),
                n_tech=(
                    "supportTopic",
                    lambda x: int((x == "technical_issue").sum()),
                ),
                n_billing=(
                    "supportTopic",
                    lambda x: int((x == "billing_inquiry").sum()),
                ),
                n_contract=(
                    "supportTopic",
                    lambda x: int((x == "contract_question").sum()),
                ),
                n_call=(
                    "interactionType",
                    lambda x: int((x == "call").sum()),
                ),
                n_pending_tech=(
                    "supportTopic",
                    lambda x: int(
                        (
                            (x == "technical_issue")
                            & (
                                si_pit.loc[x.index, "resolutionStatus"]
                                == "Pending"
                            )
                        ).sum()
                    ),
                ),
                n_pending_billing=(
                    "supportTopic",
                    lambda x: int(
                        (
                            (x == "billing_inquiry")
                            & (
                                si_pit.loc[x.index, "resolutionStatus"]
                                == "Pending"
                            )
                        ).sum()
                    ),
                ),
                last_topic=("supportTopic", "last"),
                last_status=("resolutionStatus", "last"),
                last_interaction_date=("interactionDate_dt", "max"),
                first_interaction_date=("interactionDate_dt", "min"),
            )
            .reset_index()
        )
    else:
        si_agg = pd.DataFrame(columns=["customerID"])

    df = df.merge(si_agg, on="customerID", how="left")

    int_cols = [
        "n_tickets",
        "n_pending",
        "n_resolved",
        "n_escalated",
        "n_tech",
        "n_billing",
        "n_contract",
        "n_call",
        "n_pending_tech",
        "n_pending_billing",
    ]
    for col in int_cols:
        df[col] = df[col].fillna(0).astype(int)

    df["last_topic"] = df["last_topic"].fillna("none")
    df["last_status"] = df["last_status"].fillna("none")
    df["has_ticket"] = (df["n_tickets"] > 0).astype(int)
    df["has_pending"] = (df["n_pending"] > 0).astype(int)
    df["has_repeat_ticket"] = (df["n_tickets"] >= 2).astype(int)
    df["days_from_last_ticket_to_churn"] = (
        df["churnDate_dt"] - df["last_interaction_date"]
    ).dt.days

    # Derive exact cohort activation month from latest observation yearMonth and tenureMonths:
    # activation_month = yearMonth - (tenureMonths - 1)
    obs_dt = pd.to_datetime(df["yearMonth"].astype(str) + "-01")
    act_total_months = (
        obs_dt.dt.year * 12 + (obs_dt.dt.month - 1) - (df["tenureMonths"] - 1)
    )
    act_year = act_total_months // 12
    act_month = (act_total_months % 12) + 1
    df["activation_month"] = (
        act_year.astype(str) + "-" + act_month.astype(str).str.zfill(2)
    )

    # Domain-engineered structural & behavioral ratios
    df["charge_per_gb"] = df["monthlyCharges"] / (df["dataUsageGB"] + 1.0)
    df["is_standalone_fiber"] = (
        (df["hasInternetService"] == "Fiber optic") & (df["hasPhoneService"] == "No")
    ).astype(int)
    df["is_bundled_fiber"] = (
        (df["hasInternetService"] == "Fiber optic") & (df["hasPhoneService"] == "Yes")
    ).astype(int)
    df["is_voice_mobile_only"] = (
        (df["hasInternetService"] == "No") & (df["hasPhoneService"] == "Yes")
    ).astype(int)
    df["is_mtm"] = (df["contractType"] == "Month-to-month").astype(int)
    df["is_echeck"] = (df["paymentMethod"] == "Electronic check").astype(int)
    df["is_early_tenure_le12"] = (df["tenureMonths"] <= 12).astype(int)
    df["is_onboarding_le3"] = (df["tenureMonths"] <= 3).astype(int)
    df["is_tenure_month_2"] = (df["tenureMonths"] == 2).astype(int)
    df["is_tenure_month_1_mtm"] = (
        (df["is_mtm"] == 1) & (df["tenureMonths"] == 1) & (df["is_churn"] == 0)
    ).astype(int)
    # 3-Rule Heuristic Baseline indicator (captures 144/145 churners in this near-rule-generated dataset)
    df["rule_mtm_echeck_le12"] = (
        (df["is_mtm"] == 1)
        & (df["is_echeck"] == 1)
        & (df["is_early_tenure_le12"] == 1)
    ).astype(int)

    # Cohort peer benchmarks (relative charge & usage vs. same InternetService tier)
    tier_stats = df.groupby("hasInternetService").agg(
        tier_median_charge=("monthlyCharges", "median"),
        tier_median_usage=("dataUsageGB", "median"),
    )
    df = df.merge(tier_stats, on="hasInternetService", how="left")
    df["charge_vs_tier_ratio"] = df["monthlyCharges"] / df["tier_median_charge"]
    df["usage_vs_tier_ratio"] = (df["dataUsageGB"] + 1.0) / (
        df["tier_median_usage"] + 1.0
    )

    return df


def compute_tenure_hazard_and_survival_table(
    pit_df: pd.DataFrame, max_tenure: int = 12
) -> pd.DataFrame:
    """
    Compute discrete-time hazard h(t) = d_t / n_t and Kaplan-Meier survival S(t)
    accounting for right-censoring of active customers across tenure months t = 1..max_tenure.

    Highlights the Month-2 Churn Cliff:
    - 126 of 145 total churns (86.9%) happen at tenureMonths == 2.
    - Month-to-month discrete-time hazard spikes from 0.7% (2/306) at Month 1
      to 46.7% (126/270) at Month 2, then drops to 5.1% (7/137) at Month 3.
    - 34 active Month-to-month customers are currently right-censored at tenureMonths == 1
      and are about to enter the Month-2 second-bill cliff next billing cycle.
    """
    cohorts = {
        "All Customers (N=719)": pit_df,
        "Month-to-month (All, n=306)": pit_df[pit_df["contractType"] == "Month-to-month"],
        "MtM + Fiber optic (n=178)": pit_df[
            (pit_df["contractType"] == "Month-to-month")
            & (pit_df["hasInternetService"] == "Fiber optic")
        ],
        "MtM + DSL (n=122)": pit_df[
            (pit_df["contractType"] == "Month-to-month")
            & (pit_df["hasInternetService"] == "DSL")
        ],
        "Rule Cohort: MtM + E-Check + <=12m (n=266)": pit_df[
            pit_df["rule_mtm_echeck_le12"] == 1
        ],
        "One/Two-Year Contracts (n=413)": pit_df[
            pit_df["contractType"] != "Month-to-month"
        ],
    }

    rows = []
    for cohort_name, sub in cohorts.items():
        s_t = 1.0
        for t in range(1, max_tenure + 1):
            n_at_risk = int((sub["tenureMonths"] >= t).sum())
            d_t = int(((sub["tenureMonths"] == t) & (sub["is_churn"] == 1)).sum())
            c_t = int(((sub["tenureMonths"] == t) & (sub["is_churn"] == 0)).sum())
            h_t = float(d_t / n_at_risk) if n_at_risk > 0 else 0.0
            s_t = s_t * (1.0 - h_t)
            rows.append(
                {
                    "cohort": cohort_name,
                    "tenure_month": t,
                    "at_risk_n": n_at_risk,
                    "churn_events_d": d_t,
                    "right_censored_active_c": c_t,
                    "discrete_hazard_rate": round(h_t, 4),
                    "kaplan_meier_survival": round(s_t, 4),
                    "cumulative_incidence": round(1.0 - s_t, 4),
                }
            )
    return pd.DataFrame(rows)


def compute_modulo200_duplication_audit(pit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantify near-duplicate synthetic template structure across customer IDs separated
    by 200 (cust_num vs cust_num + 200, template_group = (cust_num - 1) % 200)
    compared against random customer pairs.
    """
    struct_cols = [
        "contractType",
        "paymentMethod",
        "hasInternetService",
        "hasPhoneService",
        "region",
    ]
    full_cols = struct_cols + ["tenureMonths", "monthlyCharges", "is_churn"]

    # Create composite tuple keys for O(1) comparison
    struct_key = pit_df[struct_cols].astype(str).agg("|".join, axis=1)
    full_key = pit_df[full_cols].astype(str).agg("|".join, axis=1)
    key_df = pd.DataFrame(
        {
            "cust_num": pit_df["cust_num"].values,
            "s_key": struct_key.values,
            "f_key": full_key.values,
            "churn": pit_df["is_churn"].values,
        }
    )
    paired = key_df.merge(
        key_df.assign(cust_num=key_df["cust_num"] - 200),
        on="cust_num",
        suffixes=("_1", "_2"),
    )
    n_pairs = len(paired)
    mod200_struct_matches = int((paired["s_key_1"] == paired["s_key_2"]).sum())
    mod200_churn_matches = int((paired["churn_1"] == paired["churn_2"]).sum())
    mod200_exact_matches = int((paired["f_key_1"] == paired["f_key_2"]).sum())

    rng = np.random.default_rng(42)
    n_rows = len(key_df)
    n_rand = 2000
    idx1 = rng.integers(0, n_rows, size=n_rand)
    idx2 = (idx1 + rng.integers(1, n_rows, size=n_rand)) % n_rows
    s_arr = key_df["s_key"].to_numpy()
    f_arr = key_df["f_key"].to_numpy()
    c_arr = key_df["churn"].to_numpy()

    rand_struct = int((s_arr[idx1] == s_arr[idx2]).sum())
    rand_churn = int((c_arr[idx1] == c_arr[idx2]).sum())
    rand_exact = int((f_arr[idx1] == f_arr[idx2]).sum())

    return pd.DataFrame(
        [
            {
                "pair_type": "Modulo-200 Template Pairs (id vs id+200)",
                "n_pairs_evaluated": n_pairs,
                "structural_5col_match_rate": round(mod200_struct_matches / n_pairs, 4),
                "churn_label_match_rate": round(mod200_churn_matches / n_pairs, 4),
                "exact_8col_clone_rate": round(mod200_exact_matches / n_pairs, 4),
            },
            {
                "pair_type": "Random Customer Pairs (Baseline)",
                "n_pairs_evaluated": n_rand,
                "structural_5col_match_rate": round(rand_struct / n_rand, 4),
                "churn_label_match_rate": round(rand_churn / n_rand, 4),
                "exact_8col_clone_rate": round(rand_exact / n_rand, 4),
            },
        ]
    )
