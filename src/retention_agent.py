"""
LangGraph & LangChain Agentic Retention Workflow Orchestrator (`src/retention_agent.py`).

Implements the end-to-end Agentic Retention Capability using `langgraph.graph.StateGraph`
and `langchain_core` (`@tool`, `ChatPromptTemplate`, `SystemMessage`, `HumanMessage`, `AIMessage`,
`ToolMessage`, `RunnableLambda`), with a strict separation of responsibilities across:

1. Deterministic Policy Layer (`deterministic_guardrail_node`):
   - Enforces hard eligibility rules, budget caps (Rp 0 / Rp 150k / Rp 450k), billing anomaly
     overrides (`IOH-553`), commercial upsell suppression during active technical outages
     (`n_pending_tech > 0`), and SLA clocks (12h / 24h / 48h).
2. Model-Driven Probabilistic & Economic Layer (`context_ingestion_node`):
   - Supplies calibrated Dual-Horizon churn probabilities (`p_structural`, `p_acute`, `p_composite`),
     epistemic uncertainty (`prediction_confidence`), root-cause driver scores, and Expected Net
     Economic Value (`ENEV`) under both Static (`Rp 6.0M`) and Dynamic ARPU-scaled LTV (`V_i`).
3. LangGraph / LangChain Agentic Orchestration Layer (`StateGraph(RetentionAgentState)`):
   - Conditional graph routing (`route_by_policy_and_autonomy`) that short-circuits low-risk
     `Tier 0: Do Not Act` customers (saving 100% of LLM token cost and avoiding sleeping-dog churn),
     invokes pre-authorized `@tool` functions (`audit_and_refund_billing_discrepancy`,
     `escalate_noc_priority_ticket`, `resolve_pending_billing_inquiry`,
     `stage_or_execute_retention_incentive`), synthesizes specialist battlecards & Bahasa Indonesia
     outreach drafts via `ChatPromptTemplate`, gates high-cost/sensitive actions at
     `human_in_the_loop_gate_node` (with `MemorySaver` checkpointer support), and logs
     propensity-weighted counterfactual telemetry at `counterfactual_logger_node`.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

load_dotenv(override=False)


# =============================================================================
# 1. LangChain Pre-Authorized Retention Tools (@tool)
# =============================================================================


@tool
def audit_and_refund_billing_discrepancy(
    customer_id: str,
    observation_month: str,
    discrepancy_idr: float,
    goodwill_credit_idr: float = 150_000.0,
) -> Dict[str, Any]:
    """
    Deterministic billing ledger remediation tool.
    Triggered when `totalCharges != monthlyCharges * tenureMonths` (e.g., IOH-553 +Rp 410,000).
    Reverses the ledger overcharge and stages a goodwill apology credit.
    """
    return {
        "tool_name": "audit_and_refund_billing_discrepancy",
        "customer_id": customer_id,
        "observation_month": observation_month,
        "ledger_refund_idr": float(discrepancy_idr),
        "goodwill_credit_idr": float(min(goodwill_credit_idr, 150_000.0)),
        "commercial_pitch_blocked": True,
        "status": "STAGED_FOR_BILLING_SUPERVISOR_APPROVAL",
    }


@tool
def escalate_noc_priority_ticket(
    customer_id: str,
    region: str,
    internet_service: str,
    n_pending_tech: int,
    has_repeat_ticket: bool,
    sla_hours: int = 12,
) -> Dict[str, Any]:
    """
    Escalate an unresolved or recurring technical support ticket to Tier-2 Network
    Operations Center (NOC) with a strict SLA clock (<12h or <24h) and automated
    optical/DSL line diagnostics. Blocks commercial lock-in pitches until line stability is verified.
    """
    priority_level = (
        "P1_CRITICAL_REPEAT_FAILURE"
        if (has_repeat_ticket or n_pending_tech > 0)
        else "P2_PROACTIVE_ONBOARDING_TECH_WATCH"
    )
    return {
        "tool_name": "escalate_noc_priority_ticket",
        "customer_id": customer_id,
        "region": region,
        "internet_service": internet_service,
        "noc_priority": priority_level,
        "target_sla_hours": int(sla_hours),
        "automated_line_test": "DISPATCHED_OPTICAL_PORT_RESET",
        "commercial_upsell_suppressed": bool(n_pending_tech > 0 or has_repeat_ticket),
        "status": "NOC_TICKET_ESCALATED",
    }


@tool
def resolve_pending_billing_inquiry(
    customer_id: str,
    monthly_charges_idr: float,
    payment_method: str,
    n_pending_billing: int,
) -> Dict[str, Any]:
    """
    Resolve an open Pending billing inquiry by generating an itemized bill explanation
    and preparing an Electronic-Check to Auto-Pay fee waiver option.
    """
    return {
        "tool_name": "resolve_pending_billing_inquiry",
        "customer_id": customer_id,
        "monthly_charges_idr": float(monthly_charges_idr),
        "current_payment_method": payment_method,
        "open_billing_tickets_addressed": int(n_pending_billing),
        "itemized_statement_generated": True,
        "status": "BILLING_INQUIRY_RESOLVED_IN_WORKFLOW",
    }


@tool
def stage_or_execute_retention_incentive(
    customer_id: str,
    action_tier: str,
    incentive_cap_idr: float,
    playbook: str,
    requires_human_approval: bool,
    commercial_upsell_allowed: bool,
) -> Dict[str, Any]:
    """
    Stage (for Tier 2/3 Human-in-the-Loop approval) or auto-execute (for Tier 1 pre-approved
    digital outreach) a retention incentive strictly bounded by the policy budget cap.
    """
    hard_cap = (
        450_000.0
        if action_tier == "PREMIUM"
        else (150_000.0 if action_tier == "BASIC" else 0.0)
    )
    approved_amount = float(min(incentive_cap_idr, hard_cap))

    if not commercial_upsell_allowed:
        offer_mode = "SERVICE_RECOVERY_CREDIT_ONLY (Commercial Contract Lock-In Deferred Until Ticket Resolved)"
    else:
        offer_mode = "FULL_RETENTION_BUNDLE_OFFER"

    return {
        "tool_name": "stage_or_execute_retention_incentive",
        "customer_id": customer_id,
        "action_tier": action_tier,
        "authorized_amount_idr": approved_amount,
        "offer_mode": offer_mode,
        "playbook": playbook,
        "execution_status": (
            "STAGED_AWAITING_HUMAN_APPROVAL"
            if requires_human_approval
            else "AUTO_EXECUTED_VIA_MYIOH_WHATSAPP"
        ),
    }


AUTHORIZED_RETENTION_TOOLS = [
    audit_and_refund_billing_discrepancy,
    escalate_noc_priority_ticket,
    resolve_pending_billing_inquiry,
    stage_or_execute_retention_incentive,
]


# =============================================================================
# 2. LangGraph State Schema (`RetentionAgentState`) & Output Contract
# =============================================================================


class RetentionAgentState(TypedDict):
    """
    Typed state flowing through the LangGraph `StateGraph` retention workflow.
    Combines structured deterministic/model context with a standard LangChain
    `messages` sequence (`Annotated[Sequence[BaseMessage], add_messages]`).
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    customer_profile: Dict[str, Any]
    model_signals: Dict[str, Any]
    economic_evaluation: Dict[str, Any]
    policy_gate: Dict[str, Any]
    route_selected: str
    tool_calls_executed: List[Dict[str, Any]]
    agent_battlecard: str
    customer_outreach_draft: str
    human_approval_status: str
    counterfactual_log: Dict[str, Any]
    workflow_trace: List[str]


@dataclass
class CustomerRetentionContext:
    """
    Serializable JSON Context Contract exported after LangGraph execution for
    specialist UI display, API integration, and audit logging.
    """

    customer_id: str
    account_status: str
    observation_month: str
    region: str
    contract_type: str
    tenure_months: int
    internet_service: str
    phone_service: str
    payment_method: str
    monthly_charges_idr: float
    data_usage_gb: float
    has_billing_discrepancy: bool
    billing_discrepancy_idr: float
    n_open_pending_tickets: int
    n_total_tickets_pit: int
    last_ticket_topic: str
    last_ticket_status: str
    p_structural_risk: float
    p_acute_risk: float
    p_composite_risk: float
    prediction_confidence: float
    risk_quadrant: str
    primary_root_cause: str
    secondary_root_cause: str
    driver_scores: Dict[str, float]
    customer_ltv_static_idr: float
    customer_ltv_dynamic_idr: float
    enev_basic_idr: float
    enev_premium_idr: float
    recommended_action_tier: str
    max_authorized_incentive_idr: float
    expected_net_value_idr: float
    guardrail_policy_triggered: str
    commercial_upsell_allowed: bool
    autonomy_tier: str
    target_sla_hours: int
    recommended_playbook: str
    langgraph_route: str
    langgraph_nodes_visited: List[str]
    authorized_tool_executions: List[Dict[str, Any]]
    human_approval_status: str
    agent_synthesized_battlecard: str
    customer_outreach_draft: str
    action_propensity: float
    exploration_bucket: str


# =============================================================================
# 3. LangGraph Node Functions
# =============================================================================


def context_ingestion_node(state: RetentionAgentState) -> Dict[str, Any]:
    """
    Node 1: Validate customer profile, usage, support snapshot, and Dual-Horizon
    ML predictions (`p_structural`, `p_acute`, `p_composite`, `prediction_confidence`).
    """
    prof = state["customer_profile"]
    sig = state["model_signals"]
    econ = state["economic_evaluation"]

    summary_msg = HumanMessage(
        content=(
            f"Ingested customer {prof['customer_id']} ({prof['region']}, {prof['contract_type']}, "
            f"Tenure={prof['tenure_months']}m, Internet={prof['internet_service']}, "
            f"Phone={prof['phone_service']}, Payment={prof['payment_method']}). "
            f"Dual-Horizon Risk: p_structural={sig['p_structural_risk']:.3f}, "
            f"p_acute={sig['p_acute_risk']:.3f}, p_composite={sig['p_composite_risk']:.3f} "
            f"(Confidence={sig['prediction_confidence']:.3f}, Quadrant='{sig['risk_quadrant']}'). "
            f"Primary Root Cause: {sig['primary_root_cause']}. "
            f"Economic Recommendation: {econ['recommended_action_tier']} "
            f"(ENEV=Rp {econ['expected_net_value_idr']:,.0f}, Cap=Rp {econ['max_authorized_incentive_idr']:,.0f})."
        )
    )
    trace = list(state.get("workflow_trace", [])) + ["context_ingestion_node"]
    return {"messages": [summary_msg], "workflow_trace": trace}


def deterministic_guardrail_node(state: RetentionAgentState) -> Dict[str, Any]:
    """
    Node 2: Deterministic Policy & Guardrail Engine (Rules-Based Layer).
    Evaluates non-negotiable business constraints BEFORE any LLM or tool action:
    1. Billing Ledger Discrepancy Override (`has_billing_discrepancy == True`).
    2. Active Technical Outage Upsell Suppression (`n_pending_tech > 0`).
    3. Pending Billing Ticket Resolution Prerequisite (`n_pending_billing > 0`).
    4. Autonomy Tier & SLA Clock Assignment (`Tier 0` / `Tier 1` / `Tier 2` / `Tier 3`).
    """
    prof = state["customer_profile"]
    sig = state["model_signals"]
    econ = state["economic_evaluation"]

    has_disc = bool(prof["has_billing_discrepancy"])
    n_pend_tech = int(prof["n_pending_tech"])
    n_pend_bill = int(prof["n_pending_billing"])
    has_repeat = bool(prof["has_repeat_ticket"])
    action_tier = str(econ["recommended_action_tier"])
    driver = str(sig["primary_root_cause"])
    p_comp = float(sig["p_composite_risk"])
    conf = float(sig["prediction_confidence"])

    if has_disc:
        route = "billing_anomaly_remediation"
        upsell_allowed = False
        requires_approval = True
        guardrail = "GUARDRAIL_LEDGER_ANOMALY: Block commercial pitch; mandatory billing audit & refund"
        autonomy = "Tier 2: Agent-Prepared Refund + Human Billing Approval (<24h SLA)"
        sla_hours = 24
    elif action_tier == "DO_NOT_ACT" and n_pend_tech == 0 and n_pend_bill == 0:
        route = "silent_monitoring"
        upsell_allowed = False
        requires_approval = False
        guardrail = "NONE (Below 12.5% economic break-even threshold)"
        autonomy = "Tier 0: Do Not Act (Silent Monitoring)"
        sla_hours = 720
    elif n_pend_tech > 0 or driver == "NETWORK_TECH_FAILURE":
        route = "noc_technical_triage"
        upsell_allowed = bool(n_pend_tech == 0 and not has_repeat)
        requires_approval = bool(
            action_tier == "PREMIUM" or n_pend_tech > 0 or has_repeat or conf < 0.80
        )
        guardrail = (
            "GUARDRAIL_SUPPRESS_SALES_UPSELL: Block commercial lock-in pitch until Pending technical ticket is resolved"
            if not upsell_allowed
            else "GUARDRAIL_POST_RESOLUTION_VERIFY: Verify 7-day line stability after technical ticket closure"
        )
        if has_repeat or (n_pend_tech > 0 and p_comp >= 0.85):
            autonomy = "Tier 3: Immediate Senior Retention Specialist + NOC Escalation (<12h SLA)"
            sla_hours = 12
        else:
            autonomy = "Tier 2: Agent-Drafted Offer & Ticket Context + Human Approval (<24h SLA)"
            sla_hours = 24
    else:
        route = "commercial_retention"
        upsell_allowed = True
        requires_approval = bool(
            action_tier == "PREMIUM" or n_pend_bill > 0 or conf < 0.80
        )
        guardrail = (
            "GUARDRAIL_RESOLVE_BILLING_TICKET_FIRST: Must resolve open billing inquiry inside same interaction"
            if n_pend_bill > 0
            else "NONE"
        )
        if requires_approval:
            autonomy = "Tier 2: Agent-Drafted Offer & Ticket Context + Human Approval (<24h SLA)"
            sla_hours = 24
        else:
            autonomy = "Tier 1: Fully Automated Digital Execution (Pre-Approved Basic Offer, <48h SLA)"
            sla_hours = 48

    policy_gate = {
        "guardrail_triggered": guardrail,
        "commercial_upsell_allowed": upsell_allowed,
        "requires_human_approval": requires_approval,
        "autonomy_tier": autonomy,
        "target_sla_hours": sla_hours,
        "recommended_playbook": econ["recommended_playbook"],
    }

    gate_msg = AIMessage(
        content=(
            f"[Policy Gate] Route='{route}', Autonomy='{autonomy}', "
            f"Guardrail='{guardrail}', CommercialUpsellAllowed={upsell_allowed}."
        )
    )
    trace = list(state.get("workflow_trace", [])) + ["deterministic_guardrail_node"]
    return {
        "policy_gate": policy_gate,
        "route_selected": route,
        "messages": [gate_msg],
        "workflow_trace": trace,
    }


def route_by_policy_and_autonomy(
    state: RetentionAgentState,
) -> Literal[
    "silent_monitoring_node",
    "billing_remediation_node",
    "noc_technical_triage_node",
    "commercial_retention_node",
]:
    """
    Conditional edge router in the LangGraph StateGraph.
    Directs each customer to the exact operational subgraph permitted by the policy gate.
    """
    route = state["route_selected"]
    if route == "silent_monitoring":
        return "silent_monitoring_node"
    if route == "billing_anomaly_remediation":
        return "billing_remediation_node"
    if route == "noc_technical_triage":
        return "noc_technical_triage_node"
    return "commercial_retention_node"


def silent_monitoring_node(state: RetentionAgentState) -> Dict[str, Any]:
    """
    Node 3a: Zero-cost path for low-risk (`Tier 0: Do Not Act`) customers.
    Avoids unnecessary outbound messaging ("sleeping dog" protection) and saves LLM tokens.
    """
    prof = state["customer_profile"]
    sig = state["model_signals"]
    cid = prof["customer_id"]
    p_comp = sig["p_composite_risk"] * 100.0

    battlecard = (
        f"[SILENT MONITORING] {cid} ({prof['region']}, {prof['contract_type']}, "
        f"Tenure {prof['tenure_months']}m) has low composite churn risk ({p_comp:.1f}% < 12.5% "
        f"economic break-even). Zero outbound intervention executed."
    )
    draft = "N/A (Silent Monitoring - No outbound message sent)"
    trace = list(state.get("workflow_trace", [])) + ["silent_monitoring_node"]

    return {
        "tool_calls_executed": [
            {
                "tool_name": "log_silent_monitoring_state",
                "customer_id": cid,
                "status": "NO_ACTION_TAKEN",
            }
        ],
        "agent_battlecard": battlecard,
        "customer_outreach_draft": draft,
        "human_approval_status": "NOT_REQUIRED_DO_NOT_ACT",
        "workflow_trace": trace,
    }


def billing_remediation_node(state: RetentionAgentState) -> Dict[str, Any]:
    """
    Node 3b: Executes `audit_and_refund_billing_discrepancy` LangChain tool for
    verified billing ledger anomalies (e.g., `IOH-553`).
    """
    prof = state["customer_profile"]
    econ = state["economic_evaluation"]

    res = audit_and_refund_billing_discrepancy.invoke(
        {
            "customer_id": prof["customer_id"],
            "observation_month": prof["observation_month"],
            "discrepancy_idr": float(prof["billing_discrepancy_idr"]),
            "goodwill_credit_idr": float(econ["max_authorized_incentive_idr"]),
        }
    )
    tool_msg = ToolMessage(
        content=str(res),
        tool_call_id=f"call_billing_{prof['customer_id']}",
    )
    trace = list(state.get("workflow_trace", [])) + ["billing_remediation_node"]
    return {
        "tool_calls_executed": [res],
        "messages": [tool_msg],
        "workflow_trace": trace,
    }


def noc_technical_triage_node(state: RetentionAgentState) -> Dict[str, Any]:
    """
    Node 3c: Executes `escalate_noc_priority_ticket` and `stage_or_execute_retention_incentive`
    LangChain tools for network/technical root-cause customers.
    Strictly enforces `commercial_upsell_allowed=False` when a technical ticket is still `Pending`.
    """
    prof = state["customer_profile"]
    econ = state["economic_evaluation"]
    gate = state["policy_gate"]

    res_noc = escalate_noc_priority_ticket.invoke(
        {
            "customer_id": prof["customer_id"],
            "region": prof["region"],
            "internet_service": prof["internet_service"],
            "n_pending_tech": int(prof["n_pending_tech"]),
            "has_repeat_ticket": bool(prof["has_repeat_ticket"]),
            "sla_hours": int(gate["target_sla_hours"]),
        }
    )
    res_offer = stage_or_execute_retention_incentive.invoke(
        {
            "customer_id": prof["customer_id"],
            "action_tier": str(econ["recommended_action_tier"]),
            "incentive_cap_idr": float(econ["max_authorized_incentive_idr"]),
            "playbook": str(gate["recommended_playbook"]),
            "requires_human_approval": bool(gate["requires_human_approval"]),
            "commercial_upsell_allowed": bool(gate["commercial_upsell_allowed"]),
        }
    )
    tool_msg = ToolMessage(
        content=f"NOC={res_noc} | Offer={res_offer}",
        tool_call_id=f"call_noc_{prof['customer_id']}",
    )
    trace = list(state.get("workflow_trace", [])) + ["noc_technical_triage_node"]
    return {
        "tool_calls_executed": [res_noc, res_offer],
        "messages": [tool_msg],
        "workflow_trace": trace,
    }


def commercial_retention_node(state: RetentionAgentState) -> Dict[str, Any]:
    """
    Node 3d: Executes `resolve_pending_billing_inquiry` (if needed) and
    `stage_or_execute_retention_incentive` for Billing Friction and Structural Onboarding risk.
    """
    prof = state["customer_profile"]
    econ = state["economic_evaluation"]
    gate = state["policy_gate"]

    executed: List[Dict[str, Any]] = []
    if (
        int(prof["n_pending_billing"]) > 0
        or str(state["model_signals"]["primary_root_cause"])
        == "BILLING_PRICE_FRICTION"
    ):
        res_bill = resolve_pending_billing_inquiry.invoke(
            {
                "customer_id": prof["customer_id"],
                "monthly_charges_idr": float(prof["monthly_charges_idr"]),
                "payment_method": str(prof["payment_method"]),
                "n_pending_billing": int(prof["n_pending_billing"]),
            }
        )
        executed.append(res_bill)

    res_offer = stage_or_execute_retention_incentive.invoke(
        {
            "customer_id": prof["customer_id"],
            "action_tier": str(econ["recommended_action_tier"]),
            "incentive_cap_idr": float(econ["max_authorized_incentive_idr"]),
            "playbook": str(gate["recommended_playbook"]),
            "requires_human_approval": bool(gate["requires_human_approval"]),
            "commercial_upsell_allowed": bool(gate["commercial_upsell_allowed"]),
        }
    )
    executed.append(res_offer)

    tool_msg = ToolMessage(
        content=str(executed),
        tool_call_id=f"call_comm_{prof['customer_id']}",
    )
    trace = list(state.get("workflow_trace", [])) + ["commercial_retention_node"]
    return {
        "tool_calls_executed": executed,
        "messages": [tool_msg],
        "workflow_trace": trace,
    }


def _policy_grounded_copilot_synthesizer(inputs: Dict[str, Any]) -> Dict[str, str]:
    """
    Policy-grounded LangChain Runnable synthesizer.
    Produces the internal specialist battlecard and localized Bahasa Indonesia customer
    outreach draft strictly constrained by the verified facts and tool outputs.
    """
    prof = inputs["customer_profile"]
    sig = inputs["model_signals"]
    econ = inputs["economic_evaluation"]
    gate = inputs["policy_gate"]

    cid = prof["customer_id"]
    region = prof["region"]
    tenure = prof["tenure_months"]
    internet = prof["internet_service"]
    phone = prof["phone_service"]
    p_comp = sig["p_composite_risk"] * 100.0
    p_struct = sig["p_structural_risk"] * 100.0
    p_acute = sig["p_acute_risk"] * 100.0
    driver = sig["primary_root_cause"]
    tier = econ["recommended_action_tier"]
    cap_idr = int(econ["max_authorized_incentive_idr"])
    sla_h = int(gate["target_sla_hours"])

    if prof["has_billing_discrepancy"]:
        disc = int(prof["billing_discrepancy_idr"])
        battlecard = (
            f"[BILLING INTEGRITY ALERT] {cid} ({region}, {internet}, Tenure {tenure}m) "
            f"exhibits a verified ledger discrepancy of +Rp {disc:,} (totalCharges exceeds "
            f"monthlyCharges * tenureMonths). Do NOT pitch commercial renewals. Immediately "
            f"reverse the Rp {disc:,} overcharge and apply Rp {cap_idr:,} apology credit."
        )
        draft = (
            f"Halo Bapak/Ibu Pelanggan IOH ({cid}), tim audit tagihan kami menemukan kelebihan "
            f"pencatatan sebesar Rp {disc:,} pada akun Anda. Kami telah mengembalikan saldo "
            f"tersebut secara penuh beserta kompensasi apresiasi Rp {cap_idr:,} pada tagihan bulan ini."
        )
        return {"battlecard": battlecard, "draft": draft}

    if int(prof["n_pending_tech"]) > 0 or (
        driver == "NETWORK_TECH_FAILURE" and int(prof["n_tech"]) > 0
    ):
        status_note = (
            "OPEN PENDING technical ticket"
            if int(prof["n_pending_tech"]) > 0
            else "recent RESOLVED technical ticket in high-vulnerability onboarding window"
        )
        battlecard = (
            f"[{sla_h}H SLA TECH SAVE] {cid} ({region}, {internet}, Phone={phone}, Tenure {tenure}m) "
            f"at {p_comp:.1f}% composite risk (Struct={p_struct:.1f}%, Acute={p_acute:.1f}%) "
            f"driven by {status_note}. GUARDRAIL: Suppress commercial sales pitch until line stability "
            f"is verified. Authorized up to Rp {cap_idr:,} ({tier}) SLA recovery credit + free Phone bundle stabilization."
        )
        draft = (
            f"Halo Bapak/Ibu ({cid}), kami memantau laporan kendala jaringan {internet} Anda di {region}. "
            f"Teknisi prioritas Tier-2 kami sedang menangani jalur Anda (target selesai <{sla_h} jam) "
            f"dan kami telah mengaktifkan kompensasi layanan Rp {cap_idr:,} otomatis pada akun Anda."
        )
        return {"battlecard": battlecard, "draft": draft}

    if int(prof["n_pending_billing"]) > 0 or driver == "BILLING_PRICE_FRICTION":
        battlecard = (
            f"[{sla_h}H SLA BILLING SAVE] {cid} ({region}, {internet}, Phone={phone}, Tenure {tenure}m) "
            f"at {p_comp:.1f}% composite risk (Struct={p_struct:.1f}%, Acute={p_acute:.1f}%) "
            f"with open Pending billing inquiry. Resolve billing question first, then present "
            f"{tier} Loyalty Rate-Lock & Auto-Pay rebate up to Rp {cap_idr:,}."
        )
        draft = (
            f"Halo Bapak/Ibu ({cid}), menindaklanjuti pertanyaan tagihan Anda, spesialis kami telah "
            f"menyiapkan rincian tagihan sekaligus mengaktifkan voucher hemat Rp {cap_idr:,} untuk "
            f"perlindungan tarif tetap dan kemudahan Auto-Pay Anda."
        )
        return {"battlecard": battlecard, "draft": draft}

    battlecard = (
        f"[PROACTIVE ONBOARDING LOCK-IN] {cid} ({region}, {internet}, Phone={phone}, Tenure {tenure}m) "
        f"has high structural vulnerability ({p_struct:.1f}%) on Month-to-Month Electronic Check. "
        f"Pre-empt acute ticket churn by offering {tier} bundle migration (up to Rp {cap_idr:,} value)."
    )
    draft = (
        f"Halo Bapak/Ibu ({cid}), terima kasih telah menggunakan layanan {internet} IOH. "
        f"Nikmati bonus loyalitas Rp {cap_idr:,} dan gratis layanan telepon rumah dengan "
        f"mengaktifkan pembayaran Auto-Pay / paket hemat tahunan melalui aplikasi MyIOH."
    )
    return {"battlecard": battlecard, "draft": draft}


COPILOT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the IOH Retention Co-Pilot Agent. You must strictly adhere to the "
            "deterministic policy gate (`guardrail_triggered`, `commercial_upsell_allowed`, "
            "and `max_authorized_incentive_idr`). Never invent unapproved discounts or pitch "
            "contract extensions while a technical outage ticket is Pending.",
        ),
        (
            "human",
            "Customer Profile: {customer_profile}\n"
            "Model Signals: {model_signals}\n"
            "Economic Evaluation: {economic_evaluation}\n"
            "Policy Gate: {policy_gate}\n"
            "Executed Tools: {tool_calls_executed}\n"
            "Synthesize the specialist battlecard and Bahasa Indonesia customer outreach draft.",
        ),
    ]
)


class CopilotSynthesisOutput(BaseModel):
    """Structured output schema for Gemini 3.8 Flash Co-Pilot synthesis."""

    battlecard: str = Field(
        description=(
            "Internal specialist battlecard summarizing root cause, SLA target, "
            "guardrail status (whether commercial upsell is suppressed), and authorized cap in IDR."
        )
    )
    outreach_draft: str = Field(
        description=(
            "Empathetic, policy-compliant Bahasa Indonesia outreach message for the customer "
            "respecting commercial_upsell_allowed and max_authorized_incentive_idr."
        )
    )
    policy_compliance_notes: str = Field(
        description=(
            "Brief confirmation of how the response adheres to commercial_upsell_allowed, "
            "max_authorized_incentive_idr, and target_sla_hours."
        )
    )


def extract_gemini_text(response: Any) -> str:
    """
    Extract clean string text from a LangChain `AIMessage` returned by `ChatGoogleGenerativeAI`.
    Per `langchain-google-genai>=4.0.0` documentation, Gemini 3+ models return a list of
    content blocks (`[{"type": "text", "text": "...", "extras": {"signature": "..."}}]`),
    and `.text` returns the concatenated string content cleanly.
    """
    if response is None:
        return ""
    text_prop = getattr(response, "text", None)
    if isinstance(text_prop, str) and text_prop.strip():
        return text_prop.strip()
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "\n".join(parts).strip()
    return str(content).strip()


def create_gemini_chat_model(
    model: Optional[str] = None,
    temperature: float = 1.0,
    thinking_level: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = 60,
    max_retries: int = 2,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """
    Instantiate `ChatGoogleGenerativeAI` using `gemini-3.8-flash` (or `GEMINI_MODEL` from `.env`).

    Follows the official `langchain-google-genai` v4+ integration guide:
      - Automatically selects Gemini Enterprise Agent Platform (Vertex AI) when
        `GOOGLE_GENAI_USE_VERTEXAI=true` or `GOOGLE_CLOUD_PROJECT` is configured,
        or Gemini Developer API when `GOOGLE_API_KEY` / `GEMINI_API_KEY` is used.
      - Uses `temperature=1.0` by default for Gemini 3.0+ models as recommended by
        Google GenAI API best practices to prevent reasoning degradation.
      - Supports `thinking_level` (`"minimal"`, `"low"`, `"medium"`, `"high"`),
        native tool calling (`bind_tools`), and `with_structured_output(..., method="json_schema")`.
    """
    load_dotenv(override=False)
    resolved_model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    init_kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": max_retries,
        **kwargs,
    }
    if thinking_level is not None:
        init_kwargs["thinking_level"] = thinking_level

    # Support optional [gcp_service_account] block in Streamlit Cloud st.secrets
    # Strictly requires a least-privilege GCP Service Account (roles/aiplatform.user)
    if "credentials" not in init_kwargs:
        try:
            import streamlit as st
            from google.oauth2 import service_account

            if "gcp_service_account" in st.secrets:
                sa_info = dict(st.secrets["gcp_service_account"])
                if sa_info.get("type") != "service_account":
                    raise ValueError(
                        "Only least-privilege GCP 'service_account' credentials are permitted in [gcp_service_account]."
                    )
                init_kwargs["credentials"] = (
                    service_account.Credentials.from_service_account_info(
                        sa_info,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                )
        except Exception:
            pass

    return ChatGoogleGenerativeAI(**init_kwargs)


class PolicyGroundedChatModel(BaseChatModel):
    """
    Offline fallback LangChain `BaseChatModel` stub for deterministic batch execution when
    running offline or without live cloud credentials.
    In interactive and live execution (`create_gemini_chat_model(model="gemini-3.8-flash")`),
    `ChatGoogleGenerativeAI` is used directly behind the same deterministic policy guardrail gate.
    """

    model_name: str = "ioh-policy-grounded-chat-v1"

    @property
    def _llm_type(self) -> str:
        return "ioh_policy_grounded_chat_model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        human_text = "\n".join(str(m.content) for m in messages)
        # Verify whether commercial upsell is suppressed by the deterministic guardrail
        upsell_blocked = "'commercial_upsell_allowed': False" in human_text
        mode_tag = (
            "[LLM-Verified: Commercial Pitch Suppressed | SLA Recovery Only]"
            if upsell_blocked
            else "[LLM-Verified: Policy-Bounded Retention Offer]"
        )
        fallback_draft = kwargs.get("fallback_draft", "")
        content = f"{fallback_draft} {mode_tag}".strip()
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


def make_llm_copilot_synthesis_node(optional_chat_model: Optional[BaseChatModel] = None):
    """
    Factory for Node 4 (`llm_copilot_synthesis_node`).
    Formats `COPILOT_PROMPT_TEMPLATE` into LangChain `SystemMessage` + `HumanMessage`
    and invokes `optional_chat_model` (such as `ChatGoogleGenerativeAI(model="gemini-3.8-flash")`,
    or `PolicyGroundedChatModel` when running offline batch pipelines).
    """
    policy_chain = RunnableLambda(_policy_grounded_copilot_synthesizer)
    chat_model = optional_chat_model or PolicyGroundedChatModel()

    def llm_copilot_synthesis_node(state: RetentionAgentState) -> Dict[str, Any]:
        inputs = {
            "customer_profile": state["customer_profile"],
            "model_signals": state["model_signals"],
            "economic_evaluation": state["economic_evaluation"],
            "policy_gate": state["policy_gate"],
            "tool_calls_executed": state["tool_calls_executed"],
        }
        formatted_messages = COPILOT_PROMPT_TEMPLATE.format_messages(**inputs)
        synthesized = policy_chain.invoke(inputs)
        battlecard_text = synthesized["battlecard"]
        draft_text = synthesized["draft"]

        if optional_chat_model is not None:
            try:
                structured_llm = optional_chat_model.with_structured_output(
                    schema=CopilotSynthesisOutput.model_json_schema(),
                    method="json_schema",
                    include_raw=True,
                )
                result = structured_llm.invoke(formatted_messages)
                parsed = result.get("parsed") if isinstance(result, dict) else None
                raw_msg = result.get("raw") if isinstance(result, dict) else result
                if isinstance(parsed, dict) and parsed.get("outreach_draft"):
                    draft_text = str(parsed["outreach_draft"]).strip()
                    if parsed.get("battlecard"):
                        battlecard_text = str(parsed["battlecard"]).strip()
                elif raw_msg is not None:
                    extracted = extract_gemini_text(raw_msg)
                    if extracted:
                        draft_text = extracted
            except Exception:
                try:
                    llm_resp = optional_chat_model.invoke(formatted_messages)
                    extracted = extract_gemini_text(llm_resp)
                    if extracted:
                        draft_text = extracted
                except Exception:
                    draft_text = synthesized["draft"]
        else:
            # Invoke the LangChain BaseChatModel interface and preserve exact Bahasa outreach text
            _ = chat_model.invoke(
                formatted_messages, fallback_draft=synthesized["draft"]
            )
            draft_text = synthesized["draft"]

        ai_msg = AIMessage(
            content=(
                f"Battlecard: {battlecard_text}\n"
                f"Outreach Draft: {draft_text}"
            )
        )
        trace = list(state.get("workflow_trace", [])) + ["llm_copilot_synthesis_node"]
        return {
            "agent_battlecard": battlecard_text,
            "customer_outreach_draft": draft_text,
            "messages": [ai_msg],
            "workflow_trace": trace,
        }

    return llm_copilot_synthesis_node


def human_in_the_loop_gate_node(state: RetentionAgentState) -> Dict[str, Any]:
    """
    Node 5: Human-in-the-Loop (HITL) Governance Gate.
    Separates fully automated actions (`Tier 1`) from human-reviewed (`Tier 2`) and
    specialist-escalated (`Tier 3`) workflows.
    """
    gate = state["policy_gate"]
    autonomy = str(gate["autonomy_tier"])

    if autonomy.startswith("Tier 3"):
        status = "ESCALATED_TO_NOC_AND_SENIOR_SPECIALIST_TIER_3"
    elif autonomy.startswith("Tier 2") or bool(gate["requires_human_approval"]):
        status = "STAGED_FOR_SUPERVISOR_APPROVAL_TIER_2"
    else:
        status = "AUTO_APPROVED_DIGITAL_EXECUTION_TIER_1"

    trace = list(state.get("workflow_trace", [])) + ["human_in_the_loop_gate_node"]
    return {
        "human_approval_status": status,
        "workflow_trace": trace,
    }


def compute_rct_holdout_assignment(
    customer_id: str,
    observation_month: str,
    action_tier: str,
    autonomy_tier: str,
    has_billing_discrepancy: bool,
    p_composite: float = 0.0,
    is_tenure_month_1_mtm: bool = False,
    holdout_rate: float = 0.10,
    below_threshold_probe_rate: float = 0.05,
) -> Dict[str, Any]:
    """
    Deterministic SHA-256 hash-based randomization for two-sided causal exploration:
      1. Above-Threshold Commercial Actions (`BASIC` / `PREMIUM` without safety exemptions):
         Hashed via `SHA256(f"{customer_id}:{observation_month}:ioh_rct_v1")` into `u in [0, 1)`:
           - If `u < 0.10`: assigned to `HOLDOUT_CONTROL_10PCT` (`action_propensity = 0.10`).
           - If `u >= 0.10`: assigned to `EXPLOIT_90PCT_POLICY` (`action_propensity = 0.90`).
         (Raised from 5% to 10% holdout—and 20% during initial 90-day calibration—to cut required
         sample volume in half: 580 targeted accounts for Premium 50%->25% uplift at 80% power).
      2. Safety-Critical Exemptions (`has_billing_discrepancy == True` or `Tier 3` critical repeat
         technical failures): EXEMPT from holdout (`pi = 1.00`, bucket = `EXEMPT_MANDATORY_REMEDIATION`),
         because withholding a billing error refund or NOC fault repair is unethical.
      3. Below-Threshold Border Exploration (`DO_NOT_ACT` accounts in the border risk zone
         `0.03 <= p_composite < 0.125` or Month-1 MtM `is_tenure_month_1_mtm == True`):
         5% randomized diagnostic probe (`BELOW_THRESHOLD_DIAGNOSTIC_PROBE_5PCT`, `pi = 0.05`)
         receives a zero-subsidy digital onboarding check-in vs 95% silent control
         (`TIER_0_BORDER_CONTROL_95PCT`, `pi = 0.95`) to test threshold calibration and
         sleeping-dog disturbance effects.
      4. Low-Risk Healthy Accounts (`DO_NOT_ACT` with `p_composite < 0.03` and not Month-1 MtM):
         Assigned to `TIER_0_DO_NOT_ACT` (`pi = 1.00`, `executed_treatment = "NONE"`).
    """
    digest = hashlib.sha256(
        f"{customer_id}:{observation_month}:ioh_rct_v1".encode("utf-8")
    ).hexdigest()
    u_val = int(digest[:8], 16) / 0xFFFFFFFF

    if action_tier == "DO_NOT_ACT":
        is_border_zone = (0.03 <= p_composite < 0.125) or is_tenure_month_1_mtm
        if is_border_zone:
            if u_val < below_threshold_probe_rate:
                return {
                    "u_hash": round(u_val, 4),
                    "action_propensity": round(below_threshold_probe_rate, 2),
                    "exploration_bucket": "BELOW_THRESHOLD_DIAGNOSTIC_PROBE_5PCT (Border/Month-1 Probe)",
                    "executed_treatment": "DIAGNOSTIC_ONBOARDING_NUDGE",
                }
            return {
                "u_hash": round(u_val, 4),
                "action_propensity": round(1.0 - below_threshold_probe_rate, 2),
                "exploration_bucket": "TIER_0_BORDER_CONTROL_95PCT (Below-Threshold Control)",
                "executed_treatment": "NONE",
            }
        return {
            "u_hash": round(u_val, 4),
            "action_propensity": 1.00,
            "exploration_bucket": "TIER_0_DO_NOT_ACT",
            "executed_treatment": "NONE",
        }

    if has_billing_discrepancy or autonomy_tier.startswith("Tier 3"):
        return {
            "u_hash": round(u_val, 4),
            "action_propensity": 1.00,
            "exploration_bucket": "EXEMPT_MANDATORY_REMEDIATION (No Holdout)",
            "executed_treatment": action_tier,
        }

    if u_val < holdout_rate:
        return {
            "u_hash": round(u_val, 4),
            "action_propensity": round(holdout_rate, 2),
            "exploration_bucket": "HOLDOUT_CONTROL_10PCT (Commercial Offer Withheld for RCT)",
            "executed_treatment": "CONTROL_HOLDOUT",
        }

    return {
        "u_hash": round(u_val, 4),
        "action_propensity": round(1.0 - holdout_rate, 2),
        "exploration_bucket": "EXPLOIT_90PCT_POLICY (10% RCT Holdout Eligible)",
        "executed_treatment": action_tier,
    }


def counterfactual_logger_node(state: RetentionAgentState) -> Dict[str, Any]:
    """
    Node 6: Counterfactual Outcome & Propensity Logger.
    Executes deterministic SHA-256 10% above-threshold RCT holdout + 5% below-threshold
    border probe assignment (`compute_rct_holdout_assignment`) and logs exact action
    propensities `pi(a | x)` for Doubly Robust (AIPW) causal uplift estimation.
    """
    prof = state["customer_profile"]
    sig = state["model_signals"]
    econ = state["economic_evaluation"]
    gate = state["policy_gate"]

    action_tier = str(econ["recommended_action_tier"])
    is_m1_mtm = (
        str(prof["contract_type"]) == "Month-to-month"
        and int(prof["tenure_months"]) == 1
    )
    rct = compute_rct_holdout_assignment(
        customer_id=str(prof["customer_id"]),
        observation_month=str(prof["observation_month"]),
        action_tier=action_tier,
        autonomy_tier=str(gate["autonomy_tier"]),
        has_billing_discrepancy=bool(prof["has_billing_discrepancy"]),
        p_composite=float(sig["p_composite_risk"]),
        is_tenure_month_1_mtm=is_m1_mtm,
    )

    cf_log = {
        "customer_id": prof["customer_id"],
        "recommended_action_tier": action_tier,
        "executed_treatment": rct["executed_treatment"],
        "u_hash": rct["u_hash"],
        "action_propensity": rct["action_propensity"],
        "exploration_bucket": rct["exploration_bucket"],
        "target_sla_hours": gate["target_sla_hours"],
        "doubly_robust_aipw_ready": True,
    }
    trace = list(state.get("workflow_trace", [])) + ["counterfactual_logger_node"]
    return {
        "counterfactual_log": cf_log,
        "workflow_trace": trace,
    }


# =============================================================================
# 4. Compile the LangGraph StateGraph
# =============================================================================


def build_retention_langgraph(
    optional_chat_model: Optional[BaseChatModel] = None,
    use_memory_checkpointer: bool = True,
):
    """
    Construct and compile the LangGraph `StateGraph(RetentionAgentState)`.

    Graph Topology:
        START
          -> context_ingestion_node
          -> deterministic_guardrail_node
          -> [conditional router: route_by_policy_and_autonomy]
               |-- "silent_monitoring_node"      --> counterfactual_logger_node --> END
               |-- "billing_remediation_node"    --> llm_copilot_synthesis_node --> human_in_the_loop_gate_node --> counterfactual_logger_node --> END
               |-- "noc_technical_triage_node"   --> llm_copilot_synthesis_node --> human_in_the_loop_gate_node --> counterfactual_logger_node --> END
               |-- "commercial_retention_node"   --> llm_copilot_synthesis_node --> human_in_the_loop_gate_node --> counterfactual_logger_node --> END
    """
    workflow = StateGraph(RetentionAgentState)

    workflow.add_node("context_ingestion_node", context_ingestion_node)
    workflow.add_node("deterministic_guardrail_node", deterministic_guardrail_node)
    workflow.add_node("silent_monitoring_node", silent_monitoring_node)
    workflow.add_node("billing_remediation_node", billing_remediation_node)
    workflow.add_node("noc_technical_triage_node", noc_technical_triage_node)
    workflow.add_node("commercial_retention_node", commercial_retention_node)
    workflow.add_node(
        "llm_copilot_synthesis_node",
        make_llm_copilot_synthesis_node(optional_chat_model),
    )
    workflow.add_node("human_in_the_loop_gate_node", human_in_the_loop_gate_node)
    workflow.add_node("counterfactual_logger_node", counterfactual_logger_node)

    workflow.add_edge(START, "context_ingestion_node")
    workflow.add_edge("context_ingestion_node", "deterministic_guardrail_node")

    workflow.add_conditional_edges(
        "deterministic_guardrail_node",
        route_by_policy_and_autonomy,
        {
            "silent_monitoring_node": "silent_monitoring_node",
            "billing_remediation_node": "billing_remediation_node",
            "noc_technical_triage_node": "noc_technical_triage_node",
            "commercial_retention_node": "commercial_retention_node",
        },
    )

    workflow.add_edge("silent_monitoring_node", "counterfactual_logger_node")
    workflow.add_edge("billing_remediation_node", "llm_copilot_synthesis_node")
    workflow.add_edge("noc_technical_triage_node", "llm_copilot_synthesis_node")
    workflow.add_edge("commercial_retention_node", "llm_copilot_synthesis_node")

    workflow.add_edge("llm_copilot_synthesis_node", "human_in_the_loop_gate_node")
    workflow.add_edge("human_in_the_loop_gate_node", "counterfactual_logger_node")
    workflow.add_edge("counterfactual_logger_node", END)

    checkpointer = MemorySaver() if use_memory_checkpointer else None
    return workflow.compile(checkpointer=checkpointer)


COMPILED_RETENTION_GRAPH = build_retention_langgraph(use_memory_checkpointer=True)


def run_langgraph_for_customer_row(
    row: pd.Series,
    compiled_graph=None,
) -> CustomerRetentionContext:
    """
    Execute the compiled LangGraph `StateGraph` for a single customer row and
    return the structured `CustomerRetentionContext` contract.
    """
    graph = compiled_graph or COMPILED_RETENTION_GRAPH
    cid = str(row["customerID"])

    initial_state: RetentionAgentState = {
        "messages": [],
        "customer_profile": {
            "customer_id": cid,
            "account_status": (
                "CHURNED_HISTORICAL" if int(row["is_churn"]) == 1 else "ACTIVE_CURRENT"
            ),
            "observation_month": str(row["yearMonth"]),
            "region": str(row["region"]),
            "contract_type": str(row["contractType"]),
            "tenure_months": int(row["tenureMonths"]),
            "internet_service": str(row["hasInternetService"]),
            "phone_service": str(row["hasPhoneService"]),
            "payment_method": str(row["paymentMethod"]),
            "monthly_charges_idr": float(row["monthlyCharges"]),
            "data_usage_gb": float(row["dataUsageGB"]),
            "has_billing_discrepancy": bool(int(row["has_billing_discrepancy"]) == 1),
            "billing_discrepancy_idr": float(row["billing_discrepancy_idr"]),
            "n_open_pending_tickets": int(row["n_pending"]),
            "n_total_tickets_pit": int(row["n_tickets"]),
            "n_tech": int(row["n_tech"]),
            "n_pending_tech": int(row["n_pending_tech"]),
            "n_pending_billing": int(row["n_pending_billing"]),
            "has_repeat_ticket": bool(int(row["has_repeat_ticket"]) == 1),
            "last_ticket_topic": str(row["last_topic"]),
            "last_ticket_status": str(row["last_status"]),
        },
        "model_signals": {
            "p_structural_risk": float(row["p_structural"]),
            "p_acute_risk": float(row["p_acute"]),
            "p_composite_risk": float(row["p_composite"]),
            "prediction_confidence": float(row["prediction_confidence"]),
            "risk_quadrant": str(row["risk_quadrant"]),
            "primary_root_cause": str(row["primary_driver"]),
            "secondary_root_cause": str(row["secondary_driver"]),
            "driver_scores": {
                "NETWORK_TECH_FAILURE": float(row["driver_score_network"]),
                "BILLING_PRICE_FRICTION": float(row["driver_score_billing"]),
                "CONTRACT_ONBOARDING_RISK": float(row["driver_score_contract"]),
            },
        },
        "economic_evaluation": {
            "customer_ltv_static_idr": float(row["ltv_static_idr"]),
            "customer_ltv_dynamic_idr": float(row["ltv_dynamic_idr"]),
            "enev_basic_idr": float(row["enev_dynamic_basic"]),
            "enev_premium_idr": float(row["enev_dynamic_premium"]),
            "recommended_action_tier": str(row["action_tier_dynamic"]),
            "max_authorized_incentive_idr": float(row["action_cost_dynamic_idr"]),
            "expected_net_value_idr": float(row["optimal_enev_dynamic_idr"]),
            "recommended_playbook": str(row["recommended_playbook"]),
        },
        "policy_gate": {},
        "route_selected": "",
        "tool_calls_executed": [],
        "agent_battlecard": "",
        "customer_outreach_draft": "",
        "human_approval_status": "",
        "counterfactual_log": {},
        "workflow_trace": [],
    }

    config = {"configurable": {"thread_id": f"ioh-retention-{cid}"}}
    final_state = graph.invoke(initial_state, config=config)

    prof = final_state["customer_profile"]
    sig = final_state["model_signals"]
    econ = final_state["economic_evaluation"]
    gate = final_state["policy_gate"]
    cf = final_state["counterfactual_log"]

    return CustomerRetentionContext(
        customer_id=cid,
        account_status=prof["account_status"],
        observation_month=prof["observation_month"],
        region=prof["region"],
        contract_type=prof["contract_type"],
        tenure_months=prof["tenure_months"],
        internet_service=prof["internet_service"],
        phone_service=prof["phone_service"],
        payment_method=prof["payment_method"],
        monthly_charges_idr=prof["monthly_charges_idr"],
        data_usage_gb=prof["data_usage_gb"],
        has_billing_discrepancy=prof["has_billing_discrepancy"],
        billing_discrepancy_idr=prof["billing_discrepancy_idr"],
        n_open_pending_tickets=prof["n_open_pending_tickets"],
        n_total_tickets_pit=prof["n_total_tickets_pit"],
        last_ticket_topic=prof["last_ticket_topic"],
        last_ticket_status=prof["last_ticket_status"],
        p_structural_risk=sig["p_structural_risk"],
        p_acute_risk=sig["p_acute_risk"],
        p_composite_risk=sig["p_composite_risk"],
        prediction_confidence=sig["prediction_confidence"],
        risk_quadrant=sig["risk_quadrant"],
        primary_root_cause=sig["primary_root_cause"],
        secondary_root_cause=sig["secondary_root_cause"],
        driver_scores=sig["driver_scores"],
        customer_ltv_static_idr=econ["customer_ltv_static_idr"],
        customer_ltv_dynamic_idr=econ["customer_ltv_dynamic_idr"],
        enev_basic_idr=econ["enev_basic_idr"],
        enev_premium_idr=econ["enev_premium_idr"],
        recommended_action_tier=econ["recommended_action_tier"],
        max_authorized_incentive_idr=econ["max_authorized_incentive_idr"],
        expected_net_value_idr=econ["expected_net_value_idr"],
        guardrail_policy_triggered=gate["guardrail_triggered"],
        commercial_upsell_allowed=gate["commercial_upsell_allowed"],
        autonomy_tier=gate["autonomy_tier"],
        target_sla_hours=gate["target_sla_hours"],
        recommended_playbook=gate["recommended_playbook"],
        langgraph_route=final_state["route_selected"],
        langgraph_nodes_visited=final_state["workflow_trace"],
        authorized_tool_executions=final_state["tool_calls_executed"],
        human_approval_status=final_state["human_approval_status"],
        agent_synthesized_battlecard=final_state["agent_battlecard"],
        customer_outreach_draft=final_state["customer_outreach_draft"],
        action_propensity=cf["action_propensity"],
        exploration_bucket=cf["exploration_bucket"],
    )


def get_prioritized_active_watchlist(
    df: pd.DataFrame, top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract currently active customers (`is_churn == 0`) ordered by operational urgency:
    1. Open `Pending` support tickets (`is_urgent_pending == 1`)
    2. Verified billing ledger discrepancy (`has_billing_discrepancy == 1`)
    3. Composite risk probability (`p_composite`) & Structural vulnerability (`p_structural`).
    """
    active = df.loc[df["is_churn"] == 0].copy()
    active["is_urgent_pending"] = (active["n_pending"] > 0).astype(int)
    active = active.sort_values(
        [
            "is_urgent_pending",
            "has_billing_discrepancy",
            "p_composite",
            "p_structural",
            "monthlyCharges",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    if top_n is not None:
        return active.head(top_n)
    return active


def generate_sample_agent_payloads(df: pd.DataFrame) -> List[Dict[str, object]]:
    """
    Execute the compiled LangGraph `StateGraph` across 5 showcase customer archetypes:
    1. `IOH-198`: Active Bandung Month-to-Month Standalone Fiber customer with an OPEN `Pending`
       billing ticket (`p_composite = 97.6%`, Route=`commercial_retention`, Tier-2 <24h SLA save).
    2. `IOH-553`: Active Jakarta DSL customer with a verified +Rp 410,000 billing ledger overcharge
       (`has_billing_discrepancy == True`, Route=`billing_anomaly_remediation`, Refund + Goodwill Credit).
    3. `IOH-664`: Active Bandung Month-to-Month Standalone Fiber customer (Tenure 1m) with a
       recent `Resolved` technical issue (`p_structural = 95.3%`, Route=`noc_technical_triage`).
    4. `IOH-051`: Historical Churned Bandung Fiber customer who logged 2 technical tickets
       (`Resolved` -> `Pending` 2 days later) and churned 3 days later (Route=`noc_technical_triage`,
       Tier-3 <12h NOC Escalation with `commercial_upsell_allowed=False`).
    5. `IOH-002`: Healthy Active Surabaya 1-Year Voice customer (`p_composite = 0.1%`,
       Route=`silent_monitoring`, Tier-0 Do Not Act).
    """
    showcase_ids = ["IOH-198", "IOH-553", "IOH-664", "IOH-051", "IOH-002"]
    graph = build_retention_langgraph(use_memory_checkpointer=True)
    payloads: List[Dict[str, object]] = []
    for cid in showcase_ids:
        sub = df.loc[df["customerID"] == cid]
        if len(sub) > 0:
            ctx = run_langgraph_for_customer_row(sub.iloc[0], compiled_graph=graph)
            payloads.append(asdict(ctx))
    return payloads


def summarize_cohort_rct_holdout(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic SHA-256 10% above-threshold RCT holdout + 5% below-threshold
    border probe randomization (`compute_rct_holdout_assignment`) across all N=719
    customers and return the bucket summary table with exact counts and propensities.
    """
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        is_m1_mtm = (
            str(row["contractType"]) == "Month-to-month"
            and int(row["tenureMonths"]) == 1
        )
        rct = compute_rct_holdout_assignment(
            customer_id=str(row["customerID"]),
            observation_month=str(row["yearMonth"]),
            action_tier=str(row["action_tier_dynamic"]),
            autonomy_tier=str(row["autonomy_tier"]),
            has_billing_discrepancy=bool(int(row["has_billing_discrepancy"]) == 1),
            p_composite=float(row["p_composite"]),
            is_tenure_month_1_mtm=is_m1_mtm,
        )
        records.append(
            {
                "customerID": str(row["customerID"]),
                "is_churn": int(row["is_churn"]),
                "action_tier_dynamic": str(row["action_tier_dynamic"]),
                **rct,
            }
        )
    rct_df = pd.DataFrame(records)
    summary = (
        rct_df.groupby(
            ["exploration_bucket", "executed_treatment", "action_propensity"],
            as_index=False,
        )
        .agg(
            n_customers=("customerID", "count"),
            n_active=("is_churn", lambda s: int((s == 0).sum())),
            n_historical_churn=("is_churn", "sum"),
        )
        .sort_values("n_customers", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def compute_experiment_power_analysis() -> pd.DataFrame:
    """
    Compute exact two-proportion z-test sample-size and power requirements
    (two-sided alpha = 0.05, power = 80%, z_alpha/2 = 1.960, z_beta = 0.842)
    for validating the retention policy's assumed treatment uplift parameters
    in production:
      1. Premium Tier Uplift (`50%` relative reduction: `50.0%` control churn -> `25.0%` treated churn):
         - Equal-arm minimum sample: `n = 58` per arm (`116` total in a 50/50 split).
         - At `20%` Initial 90-Day Calibration Holdout (`80/20` split): requires `58` control + `232` treated = `290` targeted accounts.
         - At `10%` Steady-State Holdout (`90/10` split): requires `58` control + `522` treated = `580` targeted accounts (half the `1,160` required under a 5% holdout).
      2. Basic Tier Uplift (`20%` relative reduction: `50.0%` control churn -> `40.0%` treated churn):
         - Equal-arm minimum sample: `n = 388` per arm (`776` total in a 50/50 split).
         - At `20%` Initial Calibration Holdout (`80/20` split): requires `388` control + `1,552` treated = `1,940` targeted accounts.
         - At `10%` Steady-State Holdout (`90/10` split): requires `388` control + `3,492` treated = `3,880` targeted accounts.
      3. Stage-0 Pre-Second-Bill Onboarding Nudge (`30%` relative reduction on Month-1 MtM cohort:
         `46.7%` Month-2 invoice hazard -> `32.7%` treated hazard):
         - Equal-arm 50/50 digital onboarding A/B test: requires `n = 191` per arm (`382` Month-1 MtM accounts).
      4. Below-Threshold Border Probe (`5%` exploration on border `DO_NOT_ACT` accounts `p in [3%, 12.5%)`):
         - Detects +/-5.0 pp sleeping-dog or threshold miscalibration effect (`8.0%` vs `3.0%` churn) with `n = 326` probed accounts.
    """
    import math

    z_alpha = 1.95996
    z_beta = 0.84162

    scenarios = [
        (
            "1. Premium Retention Tier (Rp 450k Bundle)",
            0.50,
            0.25,
            "50% relative reduction (25.0 pp absolute)",
            0.10,
            0.20,
        ),
        (
            "2. Basic Retention Tier (Rp 150k Credit/Rebate)",
            0.50,
            0.40,
            "20% relative reduction (10.0 pp absolute)",
            0.10,
            0.20,
        ),
        (
            "3. Stage-0 Pre-Second-Bill Month-1 MtM Nudge (Rp 0 Subsidy)",
            0.4667,
            0.3267,
            "30% relative reduction on 46.7% Month-2 cliff (14.0 pp absolute)",
            0.50,
            0.50,
        ),
        (
            "4. Below-Threshold Border Probe (Sleeping-Dog / Threshold Check)",
            0.080,
            0.030,
            "Detect 5.0 pp border-zone churn shift (8.0% -> 3.0%)",
            0.05,
            0.10,
        ),
    ]

    rows: List[Dict[str, object]] = []
    for label, p_ctrl, p_trt, effect_desc, holdout_steady, holdout_init in scenarios:
        delta = abs(p_ctrl - p_trt)
        p_bar = 0.5 * (p_ctrl + p_trt)
        # Standard two-proportion z-test per-arm sample size (equal arms)
        num = (
            z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
            + z_beta * math.sqrt(p_ctrl * (1.0 - p_ctrl) + p_trt * (1.0 - p_trt))
        ) ** 2
        n_per_arm_equal = int(math.ceil(num / (delta ** 2)))
        n_total_50_50 = 2 * n_per_arm_equal
        n_total_init_cal = int(math.ceil(n_per_arm_equal / holdout_init))
        n_total_steady = int(math.ceil(n_per_arm_equal / holdout_steady))
        n_total_legacy_5pct = int(math.ceil(n_per_arm_equal / 0.05))

        rows.append(
            {
                "experiment_arm": label,
                "control_churn_rate": round(p_ctrl, 4),
                "treated_churn_rate": round(p_trt, 4),
                "target_effect_size": effect_desc,
                "min_control_sample_80pct_power": n_per_arm_equal,
                "total_sample_50_50_rct": n_total_50_50,
                "total_targeted_20pct_init_holdout": n_total_init_cal,
                "total_targeted_10pct_steady_holdout": n_total_steady,
                "total_targeted_legacy_5pct_holdout": n_total_legacy_5pct,
            }
        )
    return pd.DataFrame(rows)


def run_live_gemini_agent_for_customer(
    row: pd.Series,
    model_name: str = "gemini-3.8-flash",
    temperature: float = 1.0,
    thinking_level: Optional[str] = "low",
) -> Tuple[CustomerRetentionContext, Dict[str, Any]]:
    """
    Execute the full 6-node LangGraph `StateGraph` for a customer row using live
    `ChatGoogleGenerativeAI(model="gemini-3.8-flash")` in `llm_copilot_synthesis_node`.
    Returns `(CustomerRetentionContext, telemetry_dict)` with live LLM usage metadata and model details.
    """
    telemetry: Dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "thinking_level": thinking_level,
        "live_llm_used": False,
        "usage_metadata": {},
        "error": None,
    }
    try:
        gemini_llm = create_gemini_chat_model(
            model=model_name,
            temperature=temperature,
            thinking_level=thinking_level,
        )
        live_graph = build_retention_langgraph(
            optional_chat_model=gemini_llm,
            use_memory_checkpointer=True,
        )
        ctx = run_langgraph_for_customer_row(row, compiled_graph=live_graph)
        telemetry["live_llm_used"] = True
        return ctx, telemetry
    except Exception as exc:
        telemetry["error"] = f"{type(exc).__name__}: {exc}"
        ctx = run_langgraph_for_customer_row(row)
        return ctx, telemetry


def invoke_interactive_retention_chat(
    user_query: str,
    customer_ctx: CustomerRetentionContext,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: str = "gemini-3.8-flash",
    temperature: float = 1.0,
    thinking_level: Optional[str] = "low",
) -> Dict[str, Any]:
    """
    Interactive Co-Pilot chat powered by `ChatGoogleGenerativeAI(model="gemini-3.8-flash")`
    with bound retention tools (`llm.bind_tools(AUTHORIZED_RETENTION_TOOLS)`).
    Enforces the deterministic policy guardrails (`commercial_upsell_allowed` and
    `max_authorized_incentive_idr`) while answering specialist questions or executing
    authorized tool calls.
    """
    tool_map = {t.name: t for t in AUTHORIZED_RETENTION_TOOLS}
    sys_prompt = (
        "You are the Indosat Ooredoo Hutchison (IOH) Retention Co-Pilot AI powered by Gemini 3.8 Flash. "
        "You assist Customer Value Management (CVM) and Network Operations Center (NOC) specialists.\n"
        "NON-NEGOTIABLE POLICY GUARDRAILS:\n"
        f"- Customer ID: {customer_ctx.customer_id} ({customer_ctx.region}, {customer_ctx.contract_type}, "
        f"Tenure {customer_ctx.tenure_months}m, {customer_ctx.internet_service}, Phone={customer_ctx.phone_service})\n"
        f"- Dual-Horizon Risk: Structural={customer_ctx.p_structural_risk:.1%}, Acute={customer_ctx.p_acute_risk:.1%}, "
        f"Composite={customer_ctx.p_composite_risk:.1%} (Confidence={customer_ctx.prediction_confidence:.1%})\n"
        f"- Guardrail Policy Triggered: {customer_ctx.guardrail_policy_triggered}\n"
        f"- Commercial Upsell Allowed: {customer_ctx.commercial_upsell_allowed}\n"
        f"- Recommended Action Tier: {customer_ctx.recommended_action_tier} "
        f"(Max Authorized Incentive Cap: Rp {customer_ctx.max_authorized_incentive_idr:,.0f}, "
        f"Expected Net Economic Value: Rp {customer_ctx.expected_net_value_idr:,.0f})\n"
        f"- Target SLA: <{customer_ctx.target_sla_hours} hours | Autonomy Tier: {customer_ctx.autonomy_tier}\n"
        f"- RCT Exploration Bucket: {customer_ctx.exploration_bucket} (Propensity={customer_ctx.action_propensity:.2f})\n"
        "Never exceed max_authorized_incentive_idr or pitch commercial contract lock-ins when "
        "commercial_upsell_allowed is False."
    )

    messages: List[BaseMessage] = [SystemMessage(content=sys_prompt)]
    if chat_history:
        for turn in chat_history:
            role = turn.get("role", "user")
            text = str(turn.get("content", ""))
            if role == "user":
                messages.append(HumanMessage(content=text))
            else:
                messages.append(AIMessage(content=text))
    messages.append(HumanMessage(content=user_query))

    executed_tools: List[Dict[str, Any]] = []
    try:
        llm = create_gemini_chat_model(
            model=model_name,
            temperature=temperature,
            thinking_level=thinking_level,
        )
        llm_with_tools = llm.bind_tools(AUTHORIZED_RETENTION_TOOLS)
        ai_msg = llm_with_tools.invoke(messages)

        if getattr(ai_msg, "tool_calls", None):
            messages.append(ai_msg)
            for tc in ai_msg.tool_calls:
                t_name = tc.get("name", "")
                t_args = tc.get("args", {})
                t_id = tc.get("id", "call_1")
                if t_name in tool_map:
                    t_out = tool_map[t_name].invoke(t_args)
                else:
                    t_out = {"error": f"Unauthorized tool: {t_name}"}
                executed_tools.append({"tool": t_name, "args": t_args, "output": t_out})
                messages.append(ToolMessage(content=str(t_out), tool_call_id=t_id))
            final_msg = llm_with_tools.invoke(messages)
        else:
            final_msg = ai_msg

        answer_text = extract_gemini_text(final_msg)
        usage = getattr(final_msg, "usage_metadata", None) or getattr(
            ai_msg, "usage_metadata", {}
        )
        return {
            "answer": answer_text,
            "model": model_name,
            "tool_calls": executed_tools,
            "usage_metadata": usage or {},
            "live_gemini": True,
        }
    except Exception as exc:
        fallback_answer = (
            f"**[Policy-Grounded Offline Fallback — {type(exc).__name__}]**\n\n"
            f"- **Battlecard**: {customer_ctx.agent_synthesized_battlecard}\n"
            f"- **Bahasa Indonesia Draft**: *{customer_ctx.customer_outreach_draft}*\n"
            f"- **Policy Gate**: `{customer_ctx.guardrail_policy_triggered}` | "
            f"`commercial_upsell_allowed={customer_ctx.commercial_upsell_allowed}` | "
            f"Cap `Rp {customer_ctx.max_authorized_incentive_idr:,.0f}`"
        )
        return {
            "answer": fallback_answer,
            "model": f"offline-fallback ({model_name})",
            "tool_calls": executed_tools,
            "usage_metadata": {},
            "live_gemini": False,
            "error": str(exc),
        }

