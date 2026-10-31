# Evaluation: IOH Churn & Agentic Retention Submission

**Reviewed:** [IOH_Customer_Retention_Intelligence.ipynb](IOH_Customer_Retention_Intelligence.ipynb) (the notebook inside the submission zip), `src/*.py`, `outputs/*.csv`, `figures/*.png`, and the 12-slide deck. Verified against raw CSV files (`customer_base.csv`, `service_usage.csv`, `support_interactions.csv`).

## Scorecard

| Criterion | Score | One-line verdict |
| :--- | :---: | :--- |
| Business understanding | **3.5 / 5** | The decision framework is right (ENEV, break-evens, budget frontier, do-not-act). The headline ROI comes from a backtest that can't support it, and the strongest business insight in the data isn't mentioned. |
| Technical soundness | **3 / 5** | The data forensics are excellent and scores are out-of-fold. But the problem is framed as classification when it's really survival, near-duplicate rows leak across CV folds, the weights are arbitrary, and several numbers are wrong. |
| Analytical judgment | **3.5 / 5** | You're unusually candid about leakage, the 3-rule baseline and the prototype's limits. But the fairness number is the favourable one, and the advantage you claim for Policy 7 is built into the assumptions. |
| Clarity of communication | **2.5 / 5** | It's complete and well sectioned. It's also very dense and repetitive, contradicts itself in places, has one broken figure and one with outdated numbers, and still contains wording from earlier drafts. |

**Overall:** This is an ambitious, above-average submission. The data hygiene and the governance design would stand out. What's most likely to cost you points in the interview is credibility: a reviewer who checks two or three numbers will find mismatches, and that undermines the rest. Most of the fixes are cheap.

---

## 1. Business understanding — 3.5 / 5

### Strengths
- **Prediction is kept separate from the action decision.** The closed-form break-evens are correct (Basic ≥ 12.5%, Premium vs Basic ≥ 16.67%), and the code assigns tiers in the right order (`src/decision_engine.py:100-104`).
- **LTV varies by customer (scaled by ARPU),** so low-value accounts aren't over-subsidised. Good commercial instinct.
- **"Do not act" is a real output** (552 of 574), and there's a budget frontier and a 16-scenario sensitivity matrix. These are the trade-offs a retention manager actually faces.
- **Separating live customers (N=574) from the retrospective cohort (N=719)** shows you understand you can't run a campaign on people who have already left.
- **Root-cause routing** (billing goes to ledger correction, technical issues go to NOC, upsells are blocked while an outage is open) is realistic.

### Weaknesses
1. **The headline ROI isn't decision-grade.** "Rp 355M net, 482% ROI" counts "saves" as `is_churn × uplift` over people who already churned, scored with features that include their pre-exit `Pending` tickets. That measures how well you can recognise churners after the fact, not campaign value. The figure that matters for a decision is the live campaign: **Rp 9.3M spend and about Rp 19M expected net**. It's buried. The sensitivity matrix has the same issue: it's computed on realised churners, not on live `p`.
2. **On a like-for-like basis, the model adds very little economic value.** Under Lens 1 (`economic_policy_comparison.csv`), Policy 6 beats naive top-quartile targeting by **Rp 5.55M (1.6%)**. The "+Rp 164.85M" Lens 3 advantage exists only because the code applies a 0.5× penalty to every other policy and exempts Policy 7 (`src/decision_engine.py:342-347`). Saying so openly would make a stronger interview answer.
3. **Made-up expected values.** Forced actions get a hard-coded ENEV floor: Rp 50k for pending tickets and Rp 250k for IOH-553 (`src/decision_engine.py:169, 190`). Refunding an overcharge is a compliance obligation, not a retention return. Book it as a cost, not as value.
4. **The most actionable insight in the data is missing: the month-2 cliff.** Among Month-to-month customers, **126 of 145 churns happen at tenure month 2** (hazard 47%, compared with 0.7% at month 1 and 5% at month 3). That points to second-bill shock, and the business action is to intervene during month 1, before the second invoice, for example by moving customers to auto-pay. There are **34 active Month-to-month customers at tenure 1 right now.** The notebook only says "median 2.0 months".
5. **The probability has no time horizon.** ENEV multiplies `p` by a multi-month LTV, but `p` is "looks like a historical churner", not P(churn within X days). Summed over active customers, `p` predicts about 20 churners, while history shows roughly 18 churns per month. You need to reconcile those, or at least explain the gap.
6. **Missing cost lines:** contact and agent time, call-centre capacity, and the LTV change when the Premium offer moves a customer onto a 1-year contract.

---

## 2. Technical soundness — 3 / 5

### Strengths
- **The forensics are top-tier:** the orphan ID, region-casing leakage, multi-row extraction leakage (134 of 145 churners have 2 rows vs 0 of 574 actives), the three post-churn tickets, mixed date formats, blank `totalCharges`, and the IOH-553 overcharge. Each one has a stated fix.
- **Active customers are scored out-of-fold** (`cross_val_predict`), so they aren't scored by a model that saw their labels.
- **Useful ablations:** with and without demographics, without ticket status, without usage, and Month-to-month only. Plus Platt calibration and the 3-rule baseline as a proper benchmark.

### Weaknesses
1. **The data is templated, and the CV split lets near-duplicates leak.** **63.2% of customer pairs 200 IDs apart share the same contract, payment method, internet, phone and region, versus 4.8% for random pairs, and 90% share the same churn label.** IOH-488 and IOH-688 are identical except for gender, and so are 464/664 and 457/657. The notebook explicitly rejects this ("do not share duplicated profiles"), based only on a `monthlyCharges` match rate. Contiguous 144-ID blocks (`src/data_pipeline.py:145-146`) put siblings in different folds, so the reported AUCs are probably inflated. Group the folds by `(id−1) % 200` and report the AUC again.
2. **The "two-clock" problem is right-censored survival data.** Churners give tenure at the event and actives give tenure at censoring. A Kaplan–Meier curve or a discrete-time hazard model is the textbook fix, and it directly gives the horizon probability ENEV needs. Also, the claim that "without activationDate no out-of-time split is possible" is wrong: activation month = `yearMonth − tenureMonths + 1` can be derived for everyone, which makes as-of-date backtests possible.
3. **Arbitrary weights, presented as rigour:**
   - The ensemble is 0.2/0.4/0.4 and the composite is 0.3·A + 0.7·B (`src/models.py:587-589`). Neither is tuned or justified.
   - B already contains A's features, so A is counted twice, and the composite isn't itself calibrated.
   - Binned check: mean `p` 0.074 vs observed 0.018 in the 5–12.5% bin.
4. **`prediction_confidence = 1 − 2·std` across three models** (`src/models.py:598-601`) isn't a measure of uncertainty, yet it drives the autonomy tier.
5. **The "root-cause attribution" is hand-weighted rules** (`src/models.py:620-645`). That's fine as deterministic logic, but describe it as rules, not as a model output.
6. **Reproducibility gaps:**
   - There's no `requirements.txt` (langgraph, langchain and sklearn versions).
   - **No code generates the 8 figures** (no `savefig` anywhere).
   - The zip's `src/models.py` differs from your working copy (+296 lines locally).
7. **Factual errors (checked against the raw CSVs):**

| Claim in notebook/code | Actual |
| :--- | :--- |
| Churners frozen at exit "2024-07 to 2025-10" | churnDate is 2025-04-18 to 2025-11-28 |
| Actives "observed at 2025-11" | Actives span 2025-09, 2025-10 and 2025-11 (your own IOH-198 payload says 2025-10) |
| ARPU "Rp 150k–850k", so LTV max "Rp 12.24M" | Max is **Rp 1,000,000** (97 rows above 850k), so LTV max is Rp 14.4M. A code comment says 515k. |
| "Every churner's last ticket was 2–5 days before churn" | 119/142 (84%). The range is 1–19 days. |
| Lens 3 uses "0.35× tech / 0.40× billing" | The code uses **0.50×** for both (the outputs confirm 0.5×) |
| Live campaign "Rp 14,507,968 expected net" | **The notebook's own Table 4.0 prints Rp 19,134,375** |
| Docstring: "bandung 66.7% vs bdg 0.0%" | 97.6% vs 7.7% (the notebook text is correct) |

---

## 3. Analytical judgment — 3.5 / 5

### Strengths
- **You don't oversell the model.** You say openly that the 3-rule heuristic catches 144/145, and you reframe the model's value as decomposition, prioritisation and sizing. Interviewers will like this.
- **You call out outcome-adjacent `Pending` tickets** and the possibility that the 72-hour pattern is an artifact of how the data was built.
- **You're honest about the LLM stub and the n=2 holdout,** and you include a "what we chose not to do" section, as the brief asked.
- **The governance design is good:** deterministic guardrails first, the LLM limited to synthesis, HITL by tier, mandatory remediations exempt from holdout, and propensity logging for later AIPW estimation.

### Weaknesses
1. **The fairness headline is the favourable number.** The notebook reports gender disparate-impact ratio = **0.828 (passes the 4/5ths rule)**. That figure uses action rates that include people who already churned. On the live active population it's **3.20% / 4.81% = 0.665, which fails.** The code's docstring even admits this. A better position: DI is the wrong test when a *benefit* is allocated by base risk. Show within-group calibration (which you already have) and equal opportunity, and state the 0.665 openly.
2. **Policy 7's advantage is assumed into existence (Lens 3), then reported as a finding.** This undercuts the credibility you build elsewhere.
3. **The experiment design needs a power calculation.**
   - "N ≥ 2,000 at 5% holdout" gives about 100 controls.
   - That's enough to detect the claimed Premium effect (50% → 25% churn), but not the Basic effect (50% → 40%), which needs about 200 controls: roughly 4,000 interventions at 5%, or a 10% holdout.
   - Also, nothing below the threshold is ever randomised, so you can never find out whether the threshold, or the "sleeping dog" assumption, is wrong.
4. **The deck says more than the notebook supports.** Slide 5 says "monthly batch churn scoring is operationally obsolete" because of the 72-hour clock, while the notebook calls that pattern a likely dataset artifact. The deck's AUCs (0.995 / 0.981) don't match the notebook's (0.9970 / 0.9911).
5. **Prioritisation:** a lot of space goes to the LangGraph mechanics, SHA-256 hashing and the AIPW formula. The month-2 cliff and the Fiber-vs-DSL split (78% vs 9% churn within the rule cohort) would carry more weight with a business panel.

---

## 4. Clarity of communication — 2.5 / 5

### Strengths
- **Logical order:** hygiene, then cohorts, models, economics, the live campaign, the agent, fairness, limitations, and trade-offs.
- **The summary table, the explicit live vs retrospective split, and the JSON output contract** make the work easy for a downstream system to consume.

### Weaknesses
1. **Too dense:**
   - The executive summary runs to about 1,300 words.
   - Almost every number is in backticks.
   - "574 → 22 → 12" appears about 10 times and "144/145" about 6 times.
   - Section titles are long strings of jargon ("Dual-Horizon Platt-Calibrated Risk Engine…").
2. **Wording left over from revisions.** Phrases like "Replacing Unverified Modulo-200 Claims", "Caveating the '2–5 Day Pattern'", "To ensure complete clarity…", "candid principles" and "We chose NOT to overclaim" read as replies to an earlier reviewer and confuse a first-time reader.
3. **Figure quality:**
   - **Fig 3 panel B is broken:** every bar runs off the top of the axis, and the title claims a −2.27 GB drop that the chart doesn't show.
   - **Fig 5 has outdated numbers:** ROI 413% / 420%, while the tables say 492% / 482%. It even shows naive top-25% *beating* the proposed policy (437%), which contradicts your own claim.
   - Fig 7 has arrows running through the text.
4. **No clear "what IOH should do on Monday" statement.** For example: "Call these 12 within 24h (Rp 5.4M). Run a pre-second-bill auto-pay nudge on all tenure-1 Month-to-month customers. Hold out 10% to measure uplift."

---

## Priority fixes (highest return first)

| # | Fix | Effort | Why |
| :---: | :--- | :---: | :--- |
| 1 | Correct the 7 factual errors above; regenerate Fig 3B and Fig 5 from code | Low | Removes the easiest credibility hits |
| 2 | Lead with the live campaign (Rp 9.3M, about Rp 19M); relabel the Rp 355M as an illustrative backtest; present Lens 3 openly as an assumption | Low | Puts the decision-relevant figure first |
| 3 | Report the active-only DI of 0.665 and argue for calibration / equal-opportunity metrics | Low | Turns a potential gotcha into a strength |
| 4 | Add the month-2 cliff with a KM / hazard curve by contract × internet; define `p` over a horizon | Medium | The strongest business insight, and it fixes the horizon problem |
| 5 | Group CV folds by `(id−1) % 200`; retract the "no duplicated profiles" claim; report the AUC again | Low | Fixes a real leakage issue |
| 6 | Cut the notebook text by about 50%: a one-page summary with 3 numbers and 1 decision, details in an appendix; remove the leftover revision wording | Medium | Clarity score |
| 7 | Add `requirements.txt` and the figure code; rebuild the zip from the working tree | Low | Reproducibility |
| 8 | Add a power calculation; raise the holdout to 10%; add a small randomised probe below the threshold | Low | Makes the learning loop credible |

---

## Likely interview questions to prepare for
- "Your model beats top-quartile ranking by 1.6%. Why not just ship the rule plus a ranking?"
- "What horizon is `p_composite` a probability over, and how does that square with an LTV of Rp 6M?"
- "Why is churn concentrated at month 2, and what would you do about it?"
- "How do you know your AUC isn't inflated by duplicated customers?"
- "Your live action rate for men is two-thirds of the rate for women. Is that a problem?"
- "Which actions would you let the agent execute without a human, and what's the blast radius if it's wrong?"

