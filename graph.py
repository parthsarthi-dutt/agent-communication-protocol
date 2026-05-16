"""
LangGraph Orchestration Module
================================
Defines the three agent nodes (Intake, Recommender, Auditor),
wires them into a linear LangGraph pipeline, and compiles the graph.

Pipeline: START → Intake → Recommender → Auditor → END

Supports three protocols: text, json, markdown_json
Supports per-agent LLM routing (different providers per node).

Usage:
    from graph import build_graph, run_pipeline

    graph = build_graph()
    result = run_pipeline(graph, user_input="...", protocol="json")

    # Or with per-agent LLM routing:
    result = run_pipeline_multi_llm(
        graph, user_input="...", protocol="json",
        intake_config=("ollama", "qwen2.5:7b-instruct-q4_K_M"),
        recommend_config=("gemini", "gemma-4-31b-it"),
        auditor_config=("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),
    )
"""

import json
import re
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


def _extract_text_content(response) -> str:
    """
    Extract the text content from an LLM response.

    Thinking models (e.g. gemma-4-31b-it) return content as a list of dicts:
        [{'type': 'thinking', 'thinking': '...'}, {'type': 'text', 'text': '...'}]
    Normal models return content as a plain string.

    This helper handles both formats.
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        # Extract all 'text' type blocks and join them
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts).strip()
    else:
        return str(content).strip()


def _get_llm(provider: str | None = None, model: str | None = None, **kwargs):
    """Get a fresh LLM instance for the given provider/model (caching removed for API Key rotation)."""
    return get_llm(provider=provider, model=model, **kwargs)


def _extract_json_from_markdown(text: str) -> str:
    """
    Extract the contents of the first ```json ... ``` code block from text.
    Falls back to the raw text if no fenced block is found.
    """
    match = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try generic code fence
    match = re.search(r"```\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences if the LLM wrapped its output in them."""
    if text.startswith("```"):
        lines = text.split("\n")
        return "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════
#  NODE 1 — INTAKE AGENT
# ═══════════════════════════════════════════════════════════════════════

def node_intake(state: AgentState) -> dict:
    """
    Parses the raw user input into either a free-text summary (Protocol A),
    a structured JSON string (Protocol B), or a hybrid markdown-JSON
    response (Protocol C).

    Updates: intake_payload, error
    """
    try:
        # Use per-agent LLM config if provided in state, else default
        llm = _get_llm(
            state.get("_intake_provider"),
            state.get("_intake_model"),
        )
        protocol = state["protocol"]
        user_input = state["user_input"]

        system_prompt = get_intake_prompt(protocol)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]

        response = _invoke_with_retry(llm, messages)
        payload = _extract_text_content(response)

        # For JSON protocol, validate the JSON
        if protocol == "json":
            payload = _strip_code_fences(payload)
            json.loads(payload)  # Will raise if invalid

        # For markdown_json protocol, validate that a JSON block exists
        elif protocol == "markdown_json":
            extracted = _extract_json_from_markdown(payload)
            json.loads(extracted)  # Validate the JSON block is parseable

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
        llm = _get_llm(
            state.get("_recommend_provider"),
            state.get("_recommend_model"),
        )
        protocol = state["protocol"]
        intake_payload = state["intake_payload"]

        system_prompt = get_recommender_prompt(protocol)

        # Build the human message with the intake payload
        if protocol == "text":
            human_msg = f"Here is the user profile from the Intake Agent:\n\n{intake_payload}"
        elif protocol == "json":
            human_msg = f"Here is the structured user profile from the Intake Agent:\n\n{intake_payload}"
        else:  # markdown_json
            human_msg = f"Here is the hybrid payload from the Intake Agent:\n\n{intake_payload}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg),
        ]

        response = _invoke_with_retry(llm, messages)
        raw_output = _extract_text_content(response)

        # Strip markdown code fences if present
        raw_output = _strip_code_fences(raw_output)

        # Parse the JSON response
        result = json.loads(raw_output, strict=False)

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
        llm = _get_llm(
            state.get("_auditor_provider"),
            state.get("_auditor_model"),
        )

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
        raw_output = _extract_text_content(response)

        # Strip markdown code fences if present
        raw_output = _strip_code_fences(raw_output)

        result = json.loads(raw_output, strict=False)

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
#  PIPELINE RUNNERS
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(graph, user_input: str, protocol: str = "json") -> AgentState:
    """
    Runs the full pipeline on a single user query using the default LLM.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph instance.
    user_input : str
        Raw natural language query from the user.
    protocol : str
        "text" for Protocol A, "json" for Protocol B, "markdown_json" for Protocol C.

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


def run_pipeline_multi_llm(
    graph,
    user_input: str,
    protocol: str = "json",
    intake_config: tuple[str | None, str | None] = (None, None),
    recommend_config: tuple[str | None, str | None] = (None, None),
    auditor_config: tuple[str | None, str | None] = (None, None),
) -> AgentState:
    """
    Runs the full pipeline with per-agent LLM routing.

    Each config is a tuple of (provider, model).
    If None/None, falls back to the default from .env.

    Parameters
    ----------
    graph : CompiledStateGraph
    user_input : str
    protocol : str
    intake_config : tuple
        (provider, model) for the Intake Agent.
    recommend_config : tuple
        (provider, model) for the Recommender Agent.
    auditor_config : tuple
        (provider, model) for the Auditor Agent.

    Returns
    -------
    AgentState
    """
    initial_state = {
        "user_input": user_input,
        "protocol": protocol,
        "intake_payload": "",
        "recommended_policy_id": None,
        "recommendation_reasoning": None,
        "audit_result": None,
        "audit_reasoning": None,
        "error": None,
        # Per-agent LLM routing (consumed by node functions via state.get())
        "_intake_provider": intake_config[0],
        "_intake_model": intake_config[1],
        "_recommend_provider": recommend_config[0],
        "_recommend_model": recommend_config[1],
        "_auditor_provider": auditor_config[0],
        "_auditor_model": auditor_config[1],
    }

    result = graph.invoke(initial_state)
    return result
