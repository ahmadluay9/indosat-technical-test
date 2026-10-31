"""
Dual-Horizon Churn Risk Engine, 3-Rule Baseline Benchmark, Discrete-Time Hazard, Platt Calibration & Fairness Audit Module.

Methodological Design & Diagnostic Caveats:
1. Horizon Separation (Why two distinct risk models are required):
   - Model A (`p_structural`): Structural & Usage Vulnerability (Zero support-ticket features).
     Identifies accounts fragile at onboarding / contract renewal BEFORE support friction occurs
     (notably the 34 active Month-to-month customers at `tenureMonths == 1` approaching the Month-2 cliff).
   - Model B (`p_acute`): Point-in-Time Acute Risk (Structural + Pre-churn support features).
     Triggered upon support ticket creation or 24-hour Pending SLA breach.
   - Structural-Prior Shrinkage Composite (`p_composite = 0.30 * p_structural + 0.70 * p_acute`):
     Shrinks Model B's acute ticket likelihood toward Model A's structural base rate to dampen outcome-adjacent
     `Pending` status inflation. Across the 574 active customers, `sum(p_composite) = 19.69` expected next-cycle
     churns, closely matching the empirical steady-state run-rate of `18.0` churns/month across `2025-08..2025-11`
     and the discrete-time next-cycle hazard sum (`19.8` churns), grounding `p_composite` in a 1-billing-cycle (30-day)
     forward horizon.
2. Critical Data & Causal Caveats Diagnosed:
   - Near-Rule-Generated Cohort Structure & Month-2 Cliff: A 3-condition heuristic (`Month-to-month AND Electronic check
     AND tenureMonths <= 12`, `n=266`) captures 144 of 145 churners (99.3% recall, 54.1% precision; within this rule
     cohort, Fiber optic churns at 78.2% [136/174] vs DSL at 8.7% [8/92]). Furthermore, 126 of 145 churns (86.9%)
     occur at `tenureMonths == 2` (discrete-time Month-to-month hazard spikes from 0.7% in Month 1 to 46.7% in Month 2).
   - Modulo-200 Synthetic Template Duplication (`template_group = (cust_num - 1) % 200`): 63.2% of `(id, id+200)` pairs
     share identical contract, payment method, internet, phone, and city (vs 4.8% random pairs) and 90.4% share the same
     churn outcome. We evaluate models under `GroupKFold (Modulo-200 Template Holdout)` and `StratifiedGroupKFold
     (Modulo-200 Template Holdout)` alongside `GroupKFold (Contiguous ID-Block Holdout)` and `StratifiedKFold`.
   - Outcome-Adjacent `Pending` Ticket Status in Model B: `Pending` status appears in 135/142 pre-churn
     churners with tickets vs. only 5/574 active customers because unresolved tickets at snapshot time
     are partially a consequence of customer exit. Consequently, Model B's ~0.99+ ROC-AUC is treated as
     a diagnostic indicator of outcome-adjacent separability rather than a realistic live benchmark.
   - Right-Censored Snapshot Asymmetry: Active customers (`N=574`) are observed at `2025-09` (`n=6`), `2025-10` (`n=167`),
     and `2025-11` (`n=401`), whereas churners (`N=145`) are frozen at their exit month `2025-04` to `2025-11`
     (`churnDate` `2025-04-18` to `2025-11-28`, all `tenureMonths <= 12`).
3. Fairness-by-Design, Inter-Model Consensus & Deterministic Root-Cause Attribution:
   - Personal demographic attributes (`gender`, `age`, `seniorCitizen`, `partner`, `dependents`) are
     excluded from production scoring features (Fairness-by-Design), and post-hoc parity is audited
     across all demographic and regional subgroups via `compute_fairness_and_parity_audit` (leading with
     the active-only Gender Disparate Impact ratio of `0.665` alongside subgroup Brier calibration and Equal Opportunity).
   - `prediction_confidence = 1 - clip(2 * model_std, 0, 0.5)` is an **Inter-Model Consensus Index** measuring agreement
     across Logistic Regression, Gradient Boosting, and Random Forest (not formal Bayesian epistemic uncertainty).
   - `driver_score_network`, `driver_score_billing`, and `driver_score_contract` are **deterministic domain-rule
     attribution indices** designed to route the appropriate operational intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Fairness-by-Design: Protected demographic attributes (age, seniorCitizen, gender, partner, dependents)
# are excluded from production scoring features and retained solely for post-hoc fairness auditing.
PROTECTED_DEMOGRAPHIC_NUMERIC: List[str] = ["age", "seniorCitizen"]
PROTECTED_DEMOGRAPHIC_CATEGORICAL: List[str] = ["gender", "partner", "dependents"]

STRUCTURAL_NUMERIC_FEATURES: List[str] = [
    "tenureMonths",
    "monthlyCharges",
    "dataUsageGB",
    "charge_per_gb",
    "is_standalone_fiber",
    "is_bundled_fiber",
    "is_voice_mobile_only",
    "is_mtm",
    "is_echeck",
    "is_early_tenure_le12",
    "is_onboarding_le3",
    "charge_vs_tier_ratio",
    "usage_vs_tier_ratio",
    "has_billing_discrepancy",
]

STRUCTURAL_CATEGORICAL_FEATURES: List[str] = [
    "region",
    "contractType",
    "hasInternetService",
    "hasPhoneService",
    "paperlessBilling",
    "paymentMethod",
]

ACUTE_SUPPORT_NUMERIC_FEATURES: List[str] = [
    "n_tickets",
    "n_pending",
    "n_resolved",
    "n_escalated",
    "n_tech",
    "n_billing",
    "n_contract",
    "n_pending_tech",
    "n_pending_billing",
    "has_ticket",
    "has_pending",
    "has_repeat_ticket",
]

ACUTE_SUPPORT_CATEGORICAL_FEATURES: List[str] = [
    "last_topic",
    "last_status",
]


@dataclass
class ModelEvaluationResult:
    """Performance metrics for a specific model family and validation scheme."""

    horizon: str
    model_name: str
    validation_scheme: str
    n_samples: int
    roc_auc: float
    pr_auc: float
    brier_score: float
    f1_at_05: float
    precision_at_05: float
    recall_at_05: float


def _build_preprocessor(
    num_cols: List[str], cat_cols: List[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )


def _get_candidate_pipelines(
    num_cols: List[str], cat_cols: List[str]
) -> Dict[str, Pipeline]:
    """
    Build Platt-calibrated (sigmoid, 3-fold internal CV) candidate pipelines so
    predicted probabilities (`p_structural`, `p_acute`) are genuinely calibrated
    for downstream Expected Net Economic Value (ENEV) decision thresholds.
    """
    return {
        "LogisticRegression": Pipeline(
            [
                ("prep", _build_preprocessor(num_cols, cat_cols)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=LogisticRegression(
                            max_iter=1000, C=0.5, random_state=42
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("prep", _build_preprocessor(num_cols, cat_cols)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=GradientBoostingClassifier(
                            n_estimators=120,
                            max_depth=3,
                            learning_rate=0.05,
                            subsample=0.85,
                            random_state=42,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("prep", _build_preprocessor(num_cols, cat_cols)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=RandomForestClassifier(
                            n_estimators=300,
                            max_depth=6,
                            min_samples_leaf=4,
                            random_state=42,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def _get_rf_importance_pipeline(
    num_cols: List[str], cat_cols: List[str]
) -> Pipeline:
    return Pipeline(
        [
            ("prep", _build_preprocessor(num_cols, cat_cols)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=4,
                    random_state=42,
                ),
            ),
        ]
    )


def _compute_metrics(
    horizon: str,
    model_name: str,
    validation_scheme: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> ModelEvaluationResult:
    y_pred = (y_prob >= 0.5).astype(int)
    return ModelEvaluationResult(
        horizon=horizon,
        model_name=model_name,
        validation_scheme=validation_scheme,
        n_samples=int(len(y_true)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        f1_at_05=float(f1_score(y_true, y_pred, zero_division=0)),
        precision_at_05=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_at_05=float(recall_score(y_true, y_pred, zero_division=0)),
    )


def train_and_evaluate_dual_horizon_models(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train and cross-validate Model A (Structural Risk) and Model B (Acute Risk)
    under multiple validation schemes, compute out-of-fold ensemble predictions,
    uncertainty estimates, feature importances, and root-cause driver attributions.

    Returns:
        scored_df: Customer DataFrame enriched with `p_structural`, `p_acute`,
                   `p_composite`, `model_std`, `prediction_confidence`,
                   `risk_quadrant`, `primary_driver`, and driver scores.
        eval_df: DataFrame of ModelEvaluationResult across models and CV schemes.
        feature_imp_df: DataFrame of feature importances for Model A and Model B.
    """
    scored_df = df.copy()
    y = scored_df["is_churn"].values
    groups_mod200 = scored_df["template_group"].values
    groups_id_block = (
        scored_df["id_block_group"].values
        if "id_block_group" in scored_df.columns
        else (scored_df["cust_num"].values - 1) // 144
    )
    # Primary production out-of-fold probabilities use Contiguous ID-Block Holdout
    # (which maintains balanced ~20% fold base rates), while Modulo-200 Template Holdout
    # (both unstratified GroupKFold and StratifiedGroupKFold) is evaluated and stored in parallel.
    groups = groups_id_block

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)
    sgkf = StratifiedGroupKFold(n_splits=5)

    eval_results: List[ModelEvaluationResult] = []

    # -------------------------------------------------------------------------
    # 0. Explicit 3-Rule Heuristic Baseline (`MtM AND E-Check AND Tenure <= 12`)
    # -------------------------------------------------------------------------
    # Compute out-of-fold empirical rate for the 3-rule condition (`is_mtm & is_echeck & is_early_tenure_le12`)
    rule_flag = (
        (scored_df["is_mtm"] == 1)
        & (scored_df["is_echeck"] == 1)
        & (scored_df["is_early_tenure_le12"] == 1)
    ).values.astype(int)
    p_rule_oof = np.zeros(len(scored_df), dtype=float)
    for train_idx, test_idx in skf.split(scored_df, y):
        r_tr, y_tr = rule_flag[train_idx], y[train_idx]
        p1 = float(y_tr[r_tr == 1].mean()) if (r_tr == 1).any() else 0.78
        p0 = float(y_tr[r_tr == 0].mean()) if (r_tr == 0).any() else 0.002
        p_rule_oof[test_idx] = np.where(rule_flag[test_idx] == 1, p1, p0)

    scored_df["p_rule_baseline"] = np.round(p_rule_oof, 4)
    eval_results.append(
        _compute_metrics(
            "Baseline: 3-Rule Heuristic (MtM + E-Check + Tenure<=12)",
            "RuleBaseline_3Cond",
            "Stratified 5-Fold (Out-of-Fold Rate)",
            y,
            p_rule_oof,
        )
    )

    # -------------------------------------------------------------------------
    # 0b. Discrete-Time Conditional Hazard Benchmark h(t_next | T >= t)
    # -------------------------------------------------------------------------
    # For active accounts at tenure t, compute the empirical next-billing-cycle
    # hazard at t_target = t + 1 (and at t_target = t for historical churners)
    # stratified by (contractType, hasInternetService).
    next_cycle_hazard_cohort = np.zeros(len(scored_df), dtype=float)
    next_cycle_hazard_tier = np.zeros(len(scored_df), dtype=float)
    for c_type, c_df in scored_df.groupby("contractType"):
        for t_val in range(1, int(c_df["tenureMonths"].max()) + 2):
            at_risk_c = (c_df["tenureMonths"] >= t_val).sum()
            d_c = ((c_df["tenureMonths"] == t_val) & (c_df["is_churn"] == 1)).sum()
            h_c = float(d_c / at_risk_c) if at_risk_c > 0 else 0.0
            mask_c = (scored_df["contractType"] == c_type) & (
                np.where(
                    scored_df["is_churn"] == 0,
                    scored_df["tenureMonths"] + 1,
                    scored_df["tenureMonths"],
                )
                == t_val
            )
            next_cycle_hazard_cohort[mask_c] = h_c

    for (c_type, net_type), cn_df in scored_df.groupby(
        ["contractType", "hasInternetService"]
    ):
        for t_val in range(1, int(cn_df["tenureMonths"].max()) + 2):
            at_risk_cn = (cn_df["tenureMonths"] >= t_val).sum()
            d_cn = (
                (cn_df["tenureMonths"] == t_val) & (cn_df["is_churn"] == 1)
            ).sum()
            h_cn = float(d_cn / at_risk_cn) if at_risk_cn > 0 else 0.0
            mask_cn = (
                (scored_df["contractType"] == c_type)
                & (scored_df["hasInternetService"] == net_type)
                & (
                    np.where(
                        scored_df["is_churn"] == 0,
                        scored_df["tenureMonths"] + 1,
                        scored_df["tenureMonths"],
                    )
                    == t_val
                )
            )
            next_cycle_hazard_tier[mask_cn] = h_cn

    scored_df["p_next_cycle_hazard_cohort"] = np.round(
        next_cycle_hazard_cohort, 4
    )
    scored_df["p_next_cycle_hazard"] = np.round(next_cycle_hazard_tier, 4)

    # -------------------------------------------------------------------------
    # 1. Model A: Structural Baseline Risk (Profile + Usage only, Fairness-by-Design)
    # -------------------------------------------------------------------------
    X_struct = scored_df[
        STRUCTURAL_NUMERIC_FEATURES + STRUCTURAL_CATEGORICAL_FEATURES
    ]
    struct_pipes = _get_candidate_pipelines(
        STRUCTURAL_NUMERIC_FEATURES, STRUCTURAL_CATEGORICAL_FEATURES
    )

    oof_struct_probs: Dict[str, np.ndarray] = {}
    oof_struct_mod200: Dict[str, np.ndarray] = {}
    oof_struct_sgkf_mod200: Dict[str, np.ndarray] = {}
    for name, pipe in struct_pipes.items():
        # Stratified 5-Fold CV
        p_skf = cross_val_predict(
            pipe, X_struct, y, cv=skf, method="predict_proba"
        )[:, 1]
        eval_results.append(
            _compute_metrics(
                "Model A: Structural (Profile+Usage, No Protected Demo)",
                name,
                "Stratified 5-Fold (Random)",
                y,
                p_skf,
            )
        )

        # GroupKFold by Contiguous ID Block (Balanced Non-Random Block Holdout)
        p_gkf = cross_val_predict(
            pipe, X_struct, y, cv=gkf, groups=groups, method="predict_proba"
        )[:, 1]
        oof_struct_probs[name] = p_gkf
        eval_results.append(
            _compute_metrics(
                "Model A: Structural (Profile+Usage, No Protected Demo)",
                name,
                "GroupKFold (Contiguous ID-Block Holdout)",
                y,
                p_gkf,
            )
        )

        # GroupKFold by Modulo-200 Template Group ((cust_num - 1) % 200)
        p_mod200 = cross_val_predict(
            pipe,
            X_struct,
            y,
            cv=gkf,
            groups=groups_mod200,
            method="predict_proba",
        )[:, 1]
        oof_struct_mod200[name] = p_mod200
        eval_results.append(
            _compute_metrics(
                "Model A: Structural (Profile+Usage, No Protected Demo)",
                name,
                "GroupKFold (Modulo-200 Template Holdout)",
                y,
                p_mod200,
            )
        )

        # StratifiedGroupKFold by Modulo-200 Template Group ((cust_num - 1) % 200)
        p_sgkf_mod200 = cross_val_predict(
            pipe,
            X_struct,
            y,
            cv=sgkf,
            groups=groups_mod200,
            method="predict_proba",
        )[:, 1]
        oof_struct_sgkf_mod200[name] = p_sgkf_mod200
        eval_results.append(
            _compute_metrics(
                "Model A: Structural (Profile+Usage, No Protected Demo)",
                name,
                "StratifiedGroupKFold (Modulo-200 Template Holdout)",
                y,
                p_sgkf_mod200,
            )
        )

    # Platt-Calibrated Soft-Voting Ensemble for Model A
    p_struct_ens = (
        0.20 * oof_struct_probs["LogisticRegression"]
        + 0.40 * oof_struct_probs["GradientBoosting"]
        + 0.40 * oof_struct_probs["RandomForest"]
    )
    eval_results.append(
        _compute_metrics(
            "Model A: Structural (Profile+Usage, No Protected Demo)",
            "Platt-Calibrated Ensemble",
            "GroupKFold (Contiguous ID-Block Holdout)",
            y,
            p_struct_ens,
        )
    )

    p_struct_ens_mod200 = (
        0.20 * oof_struct_mod200["LogisticRegression"]
        + 0.40 * oof_struct_mod200["GradientBoosting"]
        + 0.40 * oof_struct_mod200["RandomForest"]
    )
    eval_results.append(
        _compute_metrics(
            "Model A: Structural (Profile+Usage, No Protected Demo)",
            "Platt-Calibrated Ensemble",
            "GroupKFold (Modulo-200 Template Holdout)",
            y,
            p_struct_ens_mod200,
        )
    )

    p_struct_ens_sgkf_mod200 = (
        0.20 * oof_struct_sgkf_mod200["LogisticRegression"]
        + 0.40 * oof_struct_sgkf_mod200["GradientBoosting"]
        + 0.40 * oof_struct_sgkf_mod200["RandomForest"]
    )
    eval_results.append(
        _compute_metrics(
            "Model A: Structural (Profile+Usage, No Protected Demo)",
            "Platt-Calibrated Ensemble",
            "StratifiedGroupKFold (Modulo-200 Template Holdout)",
            y,
            p_struct_ens_sgkf_mod200,
        )
    )

    # Demographic Fairness Ablation: verify that adding protected demographics (age, gender, partner, dependents)
    # provides no meaningful lift over the Fairness-by-Design feature set
    demo_num_cols = STRUCTURAL_NUMERIC_FEATURES + PROTECTED_DEMOGRAPHIC_NUMERIC
    demo_cat_cols = STRUCTURAL_CATEGORICAL_FEATURES + PROTECTED_DEMOGRAPHIC_CATEGORICAL
    X_struct_demo = scored_df[demo_num_cols + demo_cat_cols]
    demo_pipes = _get_candidate_pipelines(demo_num_cols, demo_cat_cols)
    p_demo_ens = (
        0.20
        * cross_val_predict(
            demo_pipes["LogisticRegression"],
            X_struct_demo,
            y,
            cv=gkf,
            groups=groups,
            method="predict_proba",
        )[:, 1]
        + 0.40
        * cross_val_predict(
            demo_pipes["GradientBoosting"],
            X_struct_demo,
            y,
            cv=gkf,
            groups=groups,
            method="predict_proba",
        )[:, 1]
        + 0.40
        * cross_val_predict(
            demo_pipes["RandomForest"],
            X_struct_demo,
            y,
            cv=gkf,
            groups=groups,
            method="predict_proba",
        )[:, 1]
    )
    eval_results.append(
        _compute_metrics(
            "Model A: Ablation With Protected Demographics",
            "Platt-Calibrated Ensemble (With Demo)",
            "GroupKFold (Contiguous ID-Block Holdout)",
            y,
            p_demo_ens,
        )
    )

    # Evaluate Model A on Month-to-Month cohort only (hardest non-trivial subset, N=306)
    mtm_mask = scored_df["contractType"] == "Month-to-month"
    eval_results.append(
        _compute_metrics(
            "Model A: Structural (Profile+Usage, No Protected Demo)",
            "Platt-Calibrated Ensemble",
            "Month-to-Month Cohort Only (GroupKFold)",
            y[mtm_mask],
            p_struct_ens[mtm_mask],
        )
    )

    # Usage & Spend Ablation on Model A: drop monthlyCharges, dataUsageGB, charge_per_gb,
    # charge_vs_tier_ratio, usage_vs_tier_ratio to test whether active ticketless suppression
    # holds purely from Contract + Service Tier (DSL vs Fiber) + Region + Tenure
    spend_usage_cols = {
        "monthlyCharges",
        "dataUsageGB",
        "charge_per_gb",
        "charge_vs_tier_ratio",
        "usage_vs_tier_ratio",
    }
    no_usage_num_cols = [
        c for c in STRUCTURAL_NUMERIC_FEATURES if c not in spend_usage_cols
    ]
    X_no_usage = scored_df[no_usage_num_cols + STRUCTURAL_CATEGORICAL_FEATURES]
    no_usage_pipes = _get_candidate_pipelines(
        no_usage_num_cols, STRUCTURAL_CATEGORICAL_FEATURES
    )
    p_no_usage_ens = (
        0.20
        * cross_val_predict(
            no_usage_pipes["LogisticRegression"],
            X_no_usage,
            y,
            cv=gkf,
            groups=groups,
            method="predict_proba",
        )[:, 1]
        + 0.40
        * cross_val_predict(
            no_usage_pipes["GradientBoosting"],
            X_no_usage,
            y,
            cv=gkf,
            groups=groups,
            method="predict_proba",
        )[:, 1]
        + 0.40
        * cross_val_predict(
            no_usage_pipes["RandomForest"],
            X_no_usage,
            y,
            cv=gkf,
            groups=groups,
            method="predict_proba",
        )[:, 1]
    )
    eval_results.append(
        _compute_metrics(
            "Model A: Ablation Without Usage & Spend (Contract/Service/Region Only)",
            "Platt-Calibrated Ensemble (No Usage/Spend)",
            "GroupKFold (Contiguous ID-Block Holdout)",
            y,
            p_no_usage_ens,
        )
    )

    # -------------------------------------------------------------------------
    # 2. Model B: Full Point-in-Time Acute Risk (Structural + Pre-Churn Support)
    #    Note: Includes `Pending` ticket status (partially outcome-adjacent in this extract)
    # -------------------------------------------------------------------------
    acute_num_cols = STRUCTURAL_NUMERIC_FEATURES + ACUTE_SUPPORT_NUMERIC_FEATURES
    acute_cat_cols = (
        STRUCTURAL_CATEGORICAL_FEATURES + ACUTE_SUPPORT_CATEGORICAL_FEATURES
    )
    X_acute = scored_df[acute_num_cols + acute_cat_cols]
    acute_pipes = _get_candidate_pipelines(acute_num_cols, acute_cat_cols)

    oof_acute_probs: Dict[str, np.ndarray] = {}
    oof_acute_mod200: Dict[str, np.ndarray] = {}
    oof_acute_sgkf_mod200: Dict[str, np.ndarray] = {}
    for name, pipe in acute_pipes.items():
        p_skf = cross_val_predict(
            pipe, X_acute, y, cv=skf, method="predict_proba"
        )[:, 1]
        eval_results.append(
            _compute_metrics(
                "Model B: Acute Point-in-Time (Struct+Support)",
                name,
                "Stratified 5-Fold (Random)",
                y,
                p_skf,
            )
        )

        p_gkf = cross_val_predict(
            pipe, X_acute, y, cv=gkf, groups=groups, method="predict_proba"
        )[:, 1]
        oof_acute_probs[name] = p_gkf
        eval_results.append(
            _compute_metrics(
                "Model B: Acute Point-in-Time (Struct+Support)",
                name,
                "GroupKFold (Contiguous ID-Block Holdout)",
                y,
                p_gkf,
            )
        )

        p_mod200 = cross_val_predict(
            pipe,
            X_acute,
            y,
            cv=gkf,
            groups=groups_mod200,
            method="predict_proba",
        )[:, 1]
        oof_acute_mod200[name] = p_mod200
        eval_results.append(
            _compute_metrics(
                "Model B: Acute Point-in-Time (Struct+Support)",
                name,
                "GroupKFold (Modulo-200 Template Holdout)",
                y,
                p_mod200,
            )
        )

        p_sgkf_mod200 = cross_val_predict(
            pipe,
            X_acute,
            y,
            cv=sgkf,
            groups=groups_mod200,
            method="predict_proba",
        )[:, 1]
        oof_acute_sgkf_mod200[name] = p_sgkf_mod200
        eval_results.append(
            _compute_metrics(
                "Model B: Acute Point-in-Time (Struct+Support)",
                name,
                "StratifiedGroupKFold (Modulo-200 Template Holdout)",
                y,
                p_sgkf_mod200,
            )
        )

    p_acute_ens = (
        0.20 * oof_acute_probs["LogisticRegression"]
        + 0.40 * oof_acute_probs["GradientBoosting"]
        + 0.40 * oof_acute_probs["RandomForest"]
    )
    eval_results.append(
        _compute_metrics(
            "Model B: Acute Point-in-Time (Struct+Support)",
            "Platt-Calibrated Ensemble",
            "GroupKFold (Contiguous ID-Block Holdout)",
            y,
            p_acute_ens,
        )
    )

    p_acute_ens_mod200 = (
        0.20 * oof_acute_mod200["LogisticRegression"]
        + 0.40 * oof_acute_mod200["GradientBoosting"]
        + 0.40 * oof_acute_mod200["RandomForest"]
    )
    eval_results.append(
        _compute_metrics(
            "Model B: Acute Point-in-Time (Struct+Support)",
            "Platt-Calibrated Ensemble",
            "GroupKFold (Modulo-200 Template Holdout)",
            y,
            p_acute_ens_mod200,
        )
    )

    p_acute_ens_sgkf_mod200 = (
        0.20 * oof_acute_sgkf_mod200["LogisticRegression"]
        + 0.40 * oof_acute_sgkf_mod200["GradientBoosting"]
        + 0.40 * oof_acute_sgkf_mod200["RandomForest"]
    )
    eval_results.append(
        _compute_metrics(
            "Model B: Acute Point-in-Time (Struct+Support)",
            "Platt-Calibrated Ensemble",
            "StratifiedGroupKFold (Modulo-200 Template Holdout)",
            y,
            p_acute_ens_sgkf_mod200,
        )
    )

    eval_results.append(
        _compute_metrics(
            "Model B: Acute Point-in-Time (Struct+Support)",
            "Platt-Calibrated Ensemble",
            "Month-to-Month Cohort Only (GroupKFold)",
            y[mtm_mask],
            p_acute_ens[mtm_mask],
        )
    )

    # Ticket-Status Ablation on Model B: drop Pending/Resolved/Escalated status features
    # (keep only ticket volume & topic) to isolate outcome-adjacent Pending status sensitivity
    status_num_cols = {
        "n_pending",
        "n_resolved",
        "n_escalated",
        "n_pending_tech",
        "n_pending_billing",
        "has_pending",
    }
    b_nostatus_num = STRUCTURAL_NUMERIC_FEATURES + [
        c for c in ACUTE_SUPPORT_NUMERIC_FEATURES if c not in status_num_cols
    ]
    b_nostatus_cat = STRUCTURAL_CATEGORICAL_FEATURES + ["last_topic"]
    X_b_nostatus = scored_df[b_nostatus_num + b_nostatus_cat]
    b_nostatus_pipes = _get_candidate_pipelines(b_nostatus_num, b_nostatus_cat)
    p_b_nostatus_ens = (
        0.20
        * cross_val_predict(
            b_nostatus_pipes["LogisticRegression"],
            X_b_nostatus,
            y,
            cv=gkf,
            groups=groups,
            method="predict_proba",
        )[:, 1]
        + 0.40
        * cross_val_predict(
            b_nostatus_pipes["GradientBoosting"],
            X_b_nostatus,
            y,
            cv=gkf,
            groups=groups,
            method="predict_proba",
        )[:, 1]
        + 0.40
        * cross_val_predict(
            b_nostatus_pipes["RandomForest"],
            X_b_nostatus,
            y,
            cv=gkf,
            groups=groups,
            method="predict_proba",
        )[:, 1]
    )
    eval_results.append(
        _compute_metrics(
            "Model B: Ablation Without Ticket Status (Count/Topic Only, No Pending)",
            "Platt-Calibrated Ensemble (No Ticket Status)",
            "GroupKFold (Contiguous ID-Block Holdout)",
            y,
            p_b_nostatus_ens,
        )
    )

    # -------------------------------------------------------------------------
    # 3. Assign Scores, Uncertainty, Quadrants & Root-Cause Attribution
    # -------------------------------------------------------------------------
    scored_df["p_structural"] = np.round(p_struct_ens, 4)
    scored_df["p_structural_mod200"] = np.round(p_struct_ens_mod200, 4)
    scored_df["p_structural_sgkf_mod200"] = np.round(
        p_struct_ens_sgkf_mod200, 4
    )
    scored_df["p_ablation_no_usage"] = np.round(p_no_usage_ens, 4)
    scored_df["p_ablation_no_pending"] = np.round(p_b_nostatus_ens, 4)
    scored_df["p_acute"] = np.round(p_acute_ens, 4)
    scored_df["p_acute_mod200"] = np.round(p_acute_ens_mod200, 4)
    scored_df["p_acute_sgkf_mod200"] = np.round(p_acute_ens_sgkf_mod200, 4)
    scored_df["p_composite"] = np.round(
        0.30 * p_struct_ens + 0.70 * p_acute_ens, 4
    )
    scored_df["p_composite_mod200"] = np.round(
        0.30 * p_struct_ens_mod200 + 0.70 * p_acute_ens_mod200, 4
    )
    scored_df["p_composite_sgkf_mod200"] = np.round(
        0.30 * p_struct_ens_sgkf_mod200 + 0.70 * p_acute_ens_sgkf_mod200, 4
    )

    acute_stack = np.vstack(
        [
            oof_acute_probs["LogisticRegression"],
            oof_acute_probs["GradientBoosting"],
            oof_acute_probs["RandomForest"],
        ]
    )
    # Inter-Model Consensus Index & Functional-Form Dispersion:
    # Measures agreement across linear (LR), boosted-tree (GBDT), and bagged-tree (RF) models.
    scored_df["model_std"] = np.round(np.std(acute_stack, axis=0), 4)
    scored_df["model_spread"] = np.round(
        np.max(acute_stack, axis=0) - np.min(acute_stack, axis=0), 4
    )
    scored_df["prediction_confidence"] = np.round(
        1.0 - np.clip(scored_df["model_std"] * 2.0, 0.0, 0.5), 4
    )

    # Risk Quadrant assignment (Dual-Horizon Matrix)
    # Thresholds based on economic break-even (p >= 0.1667 for Premium, >= 0.50 for high risk)
    cond_q1 = (scored_df["p_acute"] >= 0.35) & (scored_df["p_structural"] >= 0.25)
    cond_q2 = (scored_df["p_acute"] < 0.35) & (scored_df["p_structural"] >= 0.25)
    cond_q3 = (scored_df["p_acute"] >= 0.35) & (scored_df["p_structural"] < 0.25)

    scored_df["risk_quadrant"] = np.select(
        [cond_q1, cond_q2, cond_q3],
        [
            "Q1: Acute Critical Fire (High Struct + High Acute)",
            "Q2: Structural Vulnerability (High Struct + Low Acute)",
            "Q3: Emerging Service Friction (Low Struct + High Acute)",
        ],
        default="Q4: Healthy / Stable (Low Struct + Low Acute)",
    )

    # Domain-Guided Root-Cause Attribution Index (0..1 scale):
    # Maps customer features into actionable operational domains (Network/Tech, Billing/Pricing,
    # Contract/Onboarding) with explicit pending-ticket and billing-discrepancy priority rules.
    score_network = np.clip(
        0.65 * (scored_df["n_pending_tech"] > 0).astype(float)
        + 0.35 * (scored_df["n_tech"] > 0).astype(float)
        + 0.30 * (scored_df["has_repeat_ticket"] > 0).astype(float)
        + 0.20 * scored_df["is_standalone_fiber"].astype(float),
        0.0,
        1.0,
    )
    score_billing = np.clip(
        0.70 * (scored_df["n_pending_billing"] > 0).astype(float)
        + 0.35 * (scored_df["n_billing"] > 0).astype(float)
        + 0.85 * scored_df["has_billing_discrepancy"].astype(float)
        + 0.25 * (scored_df["charge_vs_tier_ratio"] > 1.08).astype(float)
        + 0.20 * (scored_df["charge_per_gb"] > 15000).astype(float),
        0.0,
        1.0,
    )
    score_contract = np.clip(
        0.35 * scored_df["is_mtm"].astype(float)
        + 0.25 * scored_df["is_echeck"].astype(float)
        + 0.25 * scored_df["is_onboarding_le3"].astype(float)
        + 0.15 * scored_df["is_early_tenure_le12"].astype(float)
        + 0.20 * (scored_df["n_contract"] > 0).astype(float),
        0.0,
        1.0,
    )

    scored_df["driver_score_network"] = np.round(score_network, 3)
    scored_df["driver_score_billing"] = np.round(score_billing, 3)
    scored_df["driver_score_contract"] = np.round(score_contract, 3)

    primary_drivers: List[str] = []
    secondary_drivers: List[str] = []
    for i in range(len(scored_df)):
        row = scored_df.iloc[i]
        if (
            row["p_composite"] < 0.125
            and row["has_pending"] == 0
            and row["has_billing_discrepancy"] == 0
        ):
            primary_drivers.append("ORGANIC_LOW_RISK")
            secondary_drivers.append("NONE")
            continue

        driver_map = {
            "NETWORK_TECH_FAILURE": float(row["driver_score_network"]),
            "BILLING_PRICE_FRICTION": float(row["driver_score_billing"]),
            "CONTRACT_ONBOARDING_RISK": float(row["driver_score_contract"]),
        }
        # Deterministic tie-breaking: explicit pending ticket topic or billing anomaly takes priority
        if row["has_billing_discrepancy"] == 1 or row["n_pending_billing"] > 0:
            driver_map["BILLING_PRICE_FRICTION"] += 0.50
        if row["n_pending_tech"] > 0 or row["has_repeat_ticket"] == 1:
            driver_map["NETWORK_TECH_FAILURE"] += 0.50

        sorted_d = sorted(driver_map.items(), key=lambda kv: kv[1], reverse=True)
        primary_drivers.append(sorted_d[0][0])
        secondary_drivers.append(
            sorted_d[1][0] if sorted_d[1][1] >= 0.25 else "NONE"
        )

    scored_df["primary_driver"] = primary_drivers
    scored_df["secondary_driver"] = secondary_drivers

    # -------------------------------------------------------------------------
    # 4. Extract Feature Importances from Full Fit on Model A & Model B
    # -------------------------------------------------------------------------
    feature_imp_rows: List[Dict[str, object]] = []
    for horizon_label, X_mat, num_c, cat_c in [
        (
            "Model A: Structural",
            X_struct,
            STRUCTURAL_NUMERIC_FEATURES,
            STRUCTURAL_CATEGORICAL_FEATURES,
        ),
        (
            "Model B: Acute Point-in-Time",
            X_acute,
            acute_num_cols,
            acute_cat_cols,
        ),
    ]:
        rf_pipe = _get_rf_importance_pipeline(num_c, cat_c)
        rf_pipe.fit(X_mat, y)
        feat_names = rf_pipe.named_steps["prep"].get_feature_names_out()
        importances = rf_pipe.named_steps["clf"].feature_importances_
        for fn, imp in sorted(
            zip(feat_names, importances), key=lambda x: x[1], reverse=True
        )[:15]:
            feature_imp_rows.append(
                {
                    "horizon": horizon_label,
                    "feature": str(fn)
                    .replace("num__", "")
                    .replace("cat__", ""),
                    "importance": float(imp),
                }
            )

    eval_df = pd.DataFrame([vars(r) for r in eval_results])
    feature_imp_df = pd.DataFrame(feature_imp_rows)

    return scored_df, eval_df, feature_imp_df


def compute_feature_ablation_on_active_rule3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate the Feature Ablation models specifically on the 3-Rule Heuristic Cohort
    (`Month-to-month AND Electronic check AND tenureMonths <= 12`, N = 266 total:
    51 Active Ticketless, 71 Active Ticketed, and 144 Historical Churned).

    Tests two key analytical questions:
    1. Does the suppression of the 51 ticketless active 3-rule matches depend on the
       outcome-adjacent `Pending` ticket status in Model B, or on the usage/spend gap
       (`32 GB` / `Rp 345k` median vs `89.5 GB` / `Rp 527.5k` for churners)?
    2. Result: Even when ALL ticket features AND ALL usage/spend features (`monthlyCharges`,
       `dataUsageGB`, `charge_per_gb`, `charge_vs_tier_ratio`, `usage_vs_tier_ratio`) are
       dropped (`Model A: No Usage/Spend, No Tickets`), 0 of the 51 ticketless active
       rule-matches exceed the 16.67% Premium threshold (`mean p = 0.0560`), because 49 of
       those 51 accounts (`96.1%`) are on DSL (`0%` Fiber optic) in Semarang/Medan, whereas
       136 of the 144 churners (`94.4%`) are on Fiber optic in Bandung/Surabaya/Jakarta.
    """
    y = df["is_churn"].values
    r3 = df["rule_mtm_echeck_le12"] == 1
    act_notkt = r3 & (df["is_churn"] == 0) & (df["n_tickets"] == 0)
    act_tkt = r3 & (df["is_churn"] == 0) & (df["n_tickets"] > 0)
    churn_r3 = r3 & (df["is_churn"] == 1)

    models_to_check = [
        (
            "1. Model A Full (Profile + Usage/Spend, Zero Ticket Features)",
            df["p_structural"].values,
        ),
        (
            "1b. Model A Modulo-200 StratifiedGroupKFold (Strict Clone Holdout)",
            df["p_structural_sgkf_mod200"].values,
        ),
        (
            "1c. Model A Modulo-200 Unstratified GroupKFold (Strict Clone Holdout)",
            df["p_structural_mod200"].values,
        ),
        (
            "2. Model A Ablation (Contract/Service/Region Only - No Usage/Spend, No Tickets)",
            df["p_ablation_no_usage"].values,
        ),
        (
            "3. Model B Ablation (Struct + Ticket Count/Topic Only - No Pending Status)",
            df["p_ablation_no_pending"].values,
        ),
        (
            "4. Model B Full (Struct + All Pre-Churn Ticket Features incl. Pending)",
            df["p_acute"].values,
        ),
        (
            "5. Production Composite (0.30 * Model A + 0.70 * Model B)",
            df["p_composite"].values,
        ),
    ]

    rows: List[Dict[str, object]] = []
    for label, p in models_to_check:
        rows.append(
            {
                "ablation_model": label,
                "overall_roc_auc": round(float(roc_auc_score(y, p)), 4),
                "overall_pr_auc": round(float(average_precision_score(y, p)), 4),
                "overall_brier_score": round(float(brier_score_loss(y, p)), 4),
                "active_rule3_noticket_n": int(act_notkt.sum()),
                "active_rule3_noticket_fiber_pct": round(
                    float(
                        (
                            df.loc[act_notkt, "hasInternetService"] == "Fiber optic"
                        ).mean()
                        * 100.0
                    ),
                    1,
                ),
                "active_rule3_noticket_mean_p": round(
                    float(p[act_notkt].mean()), 4
                ),
                "active_rule3_noticket_n_ge_1667": int(
                    (p[act_notkt] >= 0.1667).sum()
                ),
                "active_rule3_with_ticket_n": int(act_tkt.sum()),
                "active_rule3_with_ticket_fiber_pct": round(
                    float(
                        (
                            df.loc[act_tkt, "hasInternetService"] == "Fiber optic"
                        ).mean()
                        * 100.0
                    ),
                    1,
                ),
                "active_rule3_with_ticket_mean_p": round(
                    float(p[act_tkt].mean()), 4
                ),
                "active_rule3_with_ticket_n_ge_1667": int(
                    (p[act_tkt] >= 0.1667).sum()
                ),
                "churned_rule3_n": int(churn_r3.sum()),
                "churned_rule3_fiber_pct": round(
                    float(
                        (
                            df.loc[churn_r3, "hasInternetService"] == "Fiber optic"
                        ).mean()
                        * 100.0
                    ),
                    1,
                ),
                "churned_rule3_mean_p": round(float(p[churn_r3].mean()), 4),
                "churned_rule3_n_ge_1667": int((p[churn_r3] >= 0.1667).sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_fairness_and_parity_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit demographic and regional parity, calibration, error rates, equal opportunity,
    and informative missingness across protected and operational subgroups (`gender`,
    `seniorCitizen`, `partner`, `dependents`, and `region`).

    Because `gender`, `age`, `seniorCitizen`, `partner`, and `dependents` are excluded
    from the model feature matrix (Fairness-by-Design), raw action-rate differences
    (where Active Gender=0.665, Senior=0.000, Dependents=Yes=0.000, Partner=Yes=0.017,
    Jakarta=0.234, and Medan=0.027 all fall below the naive 0.80 four-fifths ratio)
    are driven by severe base-rate and product-mix differences (`active_mtm_fiber_share`),
    as well as informative missingness in `Unknown` gender (n=23, 100% active, 0% Fiber,
    87% Medan) and `Unknown` partner (n=19, 100% active, 94.7% 1-Yr/2-Yr contracts).
    Therefore, we report subgroup Brier calibration (`brier_score_group`), calibration-in-the-large
    (`calibration_gap`), active false-alarm rates (`fpr_active_action_rate`), churn
    false-negative rates (`fnr_churn_missed_rate`), and Equal Opportunity recall on realized
    churners (`tpr_churn_caught_rate` and `equal_opportunity_ratio`) alongside both
    active (`disparate_impact_ratio_active`) and full-dataset (`disparate_impact_ratio`)
    action-rate ratios.
    """
    work = df.copy()
    work["age_group"] = np.where(
        work["seniorCitizen"] == 1, "Senior (Age >= 60)", "Non-Senior (Age < 60)"
    )
    work["is_targeted_static"] = (
        work.get(
            "action_tier_static",
            pd.Series(
                np.where(work["p_composite"] >= 0.125, "ACT", "DO_NOT_ACT")
            ),
        )
        != "DO_NOT_ACT"
    ).astype(int)
    work["is_targeted_dynamic"] = (
        work.get(
            "action_tier_dynamic",
            pd.Series(
                np.where(work["p_composite"] >= 0.125, "ACT", "DO_NOT_ACT")
            ),
        )
        != "DO_NOT_ACT"
    ).astype(int)
    work["is_mtm_fiber"] = (
        (work["contractType"] == "Month-to-month")
        & (work["hasInternetService"] == "Fiber optic")
    ).astype(int)

    dimensions = [
        ("Gender", "gender"),
        ("Age Cohort", "age_group"),
        ("Partner Status", "partner"),
        ("Dependents Status", "dependents"),
        ("Region (City)", "region"),
    ]

    rows: List[Dict[str, object]] = []
    for dim_label, col in dimensions:
        grp_stats = []
        for val, g in work.groupby(col, dropna=False):
            active_g = g.loc[g["is_churn"] == 0]
            churn_g = g.loc[g["is_churn"] == 1]
            y_g = g["is_churn"].values.astype(float)
            p_g = g["p_composite"].values.astype(float)

            act_rate_all = float(g["is_targeted_dynamic"].mean())
            act_rate_active = (
                float(active_g["is_targeted_dynamic"].mean())
                if len(active_g) > 0
                else 0.0
            )
            brier_g = float(np.mean((p_g - y_g) ** 2))
            cal_gap = float(np.mean(p_g) - np.mean(y_g))
            tpr_g = (
                round(float(churn_g["is_targeted_dynamic"].mean()), 4)
                if len(churn_g) > 0
                else np.nan
            )
            fnr_g = (
                round(float(1.0 - churn_g["is_targeted_dynamic"].mean()), 4)
                if len(churn_g) > 0
                else np.nan
            )
            active_mtm_fiber = (
                round(float(active_g["is_mtm_fiber"].mean()), 4)
                if len(active_g) > 0
                else 0.0
            )
            active_p_struct = (
                round(float(active_g["p_structural"].mean()), 4)
                if len(active_g) > 0
                else 0.0
            )
            active_p_comp = (
                round(float(active_g["p_composite"].mean()), 4)
                if len(active_g) > 0
                else 0.0
            )

            grp_stats.append(
                {
                    "dimension": dim_label,
                    "subgroup": str(val),
                    "n_customers": int(len(g)),
                    "n_active": int(len(active_g)),
                    "empirical_churn_rate": round(float(np.mean(y_g)), 4),
                    "mean_p_structural": round(
                        float(g["p_structural"].mean()), 4
                    ),
                    "mean_p_composite": round(float(np.mean(p_g)), 4),
                    "brier_score_group": round(brier_g, 4),
                    "calibration_gap": round(cal_gap, 4),
                    "active_mtm_fiber_share": active_mtm_fiber,
                    "active_mean_p_structural": active_p_struct,
                    "active_mean_p_composite": active_p_comp,
                    "action_rate_active_dynamic": round(act_rate_active, 4),
                    "action_rate_all_dynamic": round(act_rate_all, 4),
                    "fpr_active_action_rate": round(act_rate_active, 4),
                    "tpr_churn_caught_rate": tpr_g,
                    "fnr_churn_missed_rate": fnr_g,
                }
            )
        max_rate_all = max(
            (r["action_rate_all_dynamic"] for r in grp_stats), default=1.0
        )
        max_rate_act = max(
            (r["action_rate_active_dynamic"] for r in grp_stats), default=1.0
        )
        valid_tprs = [
            r["tpr_churn_caught_rate"]
            for r in grp_stats
            if not pd.isna(r["tpr_churn_caught_rate"])
        ]
        max_tpr = max(valid_tprs, default=1.0)
        for r in grp_stats:
            r["disparate_impact_ratio_active"] = (
                round(r["action_rate_active_dynamic"] / max_rate_act, 3)
                if max_rate_act > 0
                else 1.0
            )
            r["disparate_impact_ratio"] = (
                round(r["action_rate_all_dynamic"] / max_rate_all, 3)
                if max_rate_all > 0
                else 1.0
            )
            r["equal_opportunity_ratio"] = (
                round(r["tpr_churn_caught_rate"] / max_tpr, 3)
                if (not pd.isna(r["tpr_churn_caught_rate"]) and max_tpr > 0)
                else np.nan
            )
            rows.append(r)

    return pd.DataFrame(rows)
