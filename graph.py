"""
LangGraph Orchestration Module
================================
Defines the three agent nodes (Intake, Recommender, Auditor),
wires them into a linear LangGraph pipeline, and compiles the graph.

Pipeline: START → Intake → Recommender → Auditor → END

Usage:
    from graph import build_graph, run_pipeline

    graph = build_graph()
    result = run_pipeline(graph, user_input="...", protocol="json")
"""

import json
import time
import traceback
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from llm_config import get_llm
from prompts import get_intake_prompt, get_recommender_prompt, get_auditor_prompt

# ── Retry config for free-tier rate limits ───────────────────────────────
MAX_RETRIES = 5
BASE_DELAY = 10  # seconds


def _invoke_with_retry(llm, messages, retries=MAX_RETRIES):
    """Invoke the LLM with automatic retry on rate-limit errors."""
    for attempt in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "rate" in err_str:
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  ⏳ Rate limited (attempt {attempt+1}/{retries}). Waiting {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Max retries ({retries}) exceeded due to rate limiting.")

# ── Lazily initialised LLM (created once on first use) ──────────────────
_llm = None


def _get_llm():
    """Get or create the shared LLM instance."""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


# ═══════════════════════════════════════════════════════════════════════
#  NODE 1 — INTAKE AGENT
# ═══════════════════════════════════════════════════════════════════════

def node_intake(state: AgentState) -> dict:
    """
    Parses the raw user input into either a free-text summary (Protocol A)
    or a structured JSON string (Protocol B).

    Updates: intake_payload, error
    """
    try:
        llm = _get_llm()
        protocol = state["protocol"]
        user_input = state["user_input"]

        system_prompt = get_intake_prompt(protocol)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        response = _invoke_with_retry(llm, messages)
        payload = response.content.strip()

        # For JSON protocol, try to validate the JSON
        if protocol == "json":
            # Strip markdown code fences if the LLM added them
            if payload.startswith("```"):
                lines = payload.split("\n")
                # Remove first and last lines (fences)
                payload = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                ).strip()

            # Validate it's parseable JSON
            json.loads(payload)  # Will raise if invalid

        return {"intake_payload": payload, "error": None}

    except json.JSONDecodeError as e:
        return {
            "intake_payload": payload if 'payload' in dir() else "",
            "error": f"Intake JSON parse error: {str(e)}",
        }
    except Exception as e:
        return {
            "intake_payload": "",
            "error": f"Intake error: {str(e)}\n{traceback.format_exc()}",
        }


# ═══════════════════════════════════════════════════════════════════════
#  NODE 2 — RECOMMENDER AGENT
# ═══════════════════════════════════════════════════════════════════════

def node_recommend(state: AgentState) -> dict:
    """
    Reads the Intake Agent's payload and selects a policy from the database.

    Updates: recommended_policy_id, recommendation_reasoning, error
    """
    # If a previous node already errored, propagate but still try
    try:
        llm = _get_llm()
        protocol = state["protocol"]
        intake_payload = state["intake_payload"]

        system_prompt = get_recommender_prompt(protocol)

        # Build the human message with the intake payload
        if protocol == "text":
            human_msg = f"Here is the user profile from the Intake Agent:\n\n{intake_payload}"
        else:
            human_msg = f"Here is the structured user profile from the Intake Agent:\n\n{intake_payload}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg),
        ]

        response = _invoke_with_retry(llm, messages)
        raw_output = response.content.strip()

        # Strip markdown code fences if present
        if raw_output.startswith("```"):
            lines = raw_output.split("\n")
            raw_output = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        # Parse the JSON response
        result = json.loads(raw_output)

        return {
            "recommended_policy_id": result.get("recommended_policy_id", "UNKNOWN"),
            "recommendation_reasoning": result.get("reasoning", "No reasoning provided."),
            "error": state.get("error"),  # preserve prior errors
        }

    except json.JSONDecodeError as e:
        return {
            "recommended_policy_id": "PARSE_ERROR",
            "recommendation_reasoning": f"Raw output: {raw_output if 'raw_output' in dir() else 'N/A'}",
            "error": f"{state.get('error', '') or ''} | Recommender JSON parse error: {str(e)}".strip(" |"),
        }
    except Exception as e:
        return {
            "recommended_policy_id": "ERROR",
            "recommendation_reasoning": "",
            "error": f"{state.get('error', '') or ''} | Recommender error: {str(e)}".strip(" |"),
        }


# ═══════════════════════════════════════════════════════════════════════
#  NODE 3 — COMPLIANCE AUDITOR AGENT
# ═══════════════════════════════════════════════════════════════════════

def node_audit(state: AgentState) -> dict:
    """
    Validates the Recommender's choice against the original user input.

    Updates: audit_result, audit_reasoning, error
    """
    try:
        llm = _get_llm()

        system_prompt = get_auditor_prompt()

        # Give the auditor ALL context: original query + intake payload + recommendation
        human_msg = f"""Please audit the following recommendation:

═══ ORIGINAL USER QUERY ═══
{state["user_input"]}

═══ INTAKE AGENT OUTPUT ({state["protocol"].upper()} protocol) ═══
{state["intake_payload"]}

═══ RECOMMENDER AGENT DECISION ═══
Recommended Policy ID: {state["recommended_policy_id"]}
Reasoning: {state["recommendation_reasoning"]}

Please perform your compliance audit now."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg),
        ]

        response = _invoke_with_retry(llm, messages)
        raw_output = response.content.strip()

        # Strip markdown code fences if present
        if raw_output.startswith("```"):
            lines = raw_output.split("\n")
            raw_output = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        result = json.loads(raw_output)

        return {
            "audit_result": result.get("audit_result", "Rejected"),
            "audit_reasoning": result.get("audit_reasoning", "No reasoning provided."),
            "error": state.get("error"),  # preserve prior errors
        }

    except json.JSONDecodeError as e:
        return {
            "audit_result": "Rejected",
            "audit_reasoning": f"Audit parse error. Raw: {raw_output if 'raw_output' in dir() else 'N/A'}",
            "error": f"{state.get('error', '') or ''} | Auditor JSON parse error: {str(e)}".strip(" |"),
        }
    except Exception as e:
        return {
            "audit_result": "Rejected",
            "audit_reasoning": "",
            "error": f"{state.get('error', '') or ''} | Auditor error: {str(e)}".strip(" |"),
        }


# ═══════════════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Builds and compiles the LangGraph:
        START → node_intake → node_recommend → node_audit → END

    Returns a compiled StateGraph ready for .invoke()
    """
    from langgraph.graph import StateGraph, START, END

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("intake", node_intake)
    workflow.add_node("recommend", node_recommend)
    workflow.add_node("audit", node_audit)

    # Wire edges: linear pipeline
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "recommend")
    workflow.add_edge("recommend", "audit")
    workflow.add_edge("audit", END)

    # Compile
    graph = workflow.compile()
    return graph


# ═══════════════════════════════════════════════════════════════════════
#  PIPELINE RUNNER (convenience wrapper)
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(graph, user_input: str, protocol: str = "json") -> AgentState:
    """
    Runs the full pipeline on a single user query.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph instance.
    user_input : str
        Raw natural language query from the user.
    protocol : str
        "text" for Protocol A, "json" for Protocol B.

    Returns
    -------
    AgentState
        The final state after all three agents have run.
    """
    initial_state: AgentState = {
        "user_input": user_input,
        "protocol": protocol,
        "intake_payload": "",
        "recommended_policy_id": None,
        "recommendation_reasoning": None,
        "audit_result": None,
        "audit_reasoning": None,
        "error": None,
    }

    result = graph.invoke(initial_state)
    return result
