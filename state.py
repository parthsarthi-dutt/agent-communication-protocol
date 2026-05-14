"""
State Definition for the Insurance Policy Advisory Multi-Agent System.

This module defines the shared TypedDict state that flows through the LangGraph
pipeline across all three agents (Intake, Recommender, Auditor).

Two communication protocols are evaluated:
  - Protocol A ("text"): Free-text unstructured summaries between agents.
  - Protocol B ("json"): Structured JSON schemas between agents.
"""

from typing import TypedDict, Optional, Literal


class AgentState(TypedDict):
    """
    Shared state dictionary passed between LangGraph nodes.

    Attributes
    ----------
    user_input : str
        The original natural-language query from the user describing their
        insurance needs (age, conditions, smoking status, budget, etc.).

    protocol : Literal["text", "json"]
        Which communication protocol is active for this run.
        - "text"  → Protocol A (free-text summaries between agents)
        - "json"  → Protocol B (structured JSON schemas between agents)

    intake_payload : str
        The output of Agent 1 (Intake Agent).
        - Under Protocol A this is a free-form paragraph summarising the user.
        - Under Protocol B this is a JSON-formatted string with explicit keys
          such as age, is_smoker, pre_existing_conditions, budget, etc.

    recommended_policy_id : Optional[str]
        The policy_id chosen by Agent 2 (Recommender Agent) from the mock
        database. None until the Recommender node has run.

    recommendation_reasoning : Optional[str]
        Agent 2's reasoning / explanation for its policy choice.

    audit_result : Optional[Literal["Approved", "Rejected"]]
        Agent 3's (Compliance Auditor) final verdict on the recommendation.
        "Approved" means the recommended policy satisfies all user constraints.
        "Rejected" means a rule violation was detected.

    audit_reasoning : Optional[str]
        Agent 3's detailed compliance notes explaining why the recommendation
        was approved or rejected (e.g., "Policy excludes smokers but user is
        a smoker").

    error : Optional[str]
        Captures any parsing or runtime errors that occur during the pipeline
        so that the evaluation engine can track parsing-error rates.
    """

    # ── User Input ──────────────────────────────────────────────────────
    user_input: str

    # ── Protocol Selector ───────────────────────────────────────────────
    protocol: Literal["text", "json"]

    # ── Agent 1 → Agent 2 Payload ───────────────────────────────────────
    intake_payload: str

    # ── Agent 2 Output ──────────────────────────────────────────────────
    recommended_policy_id: Optional[str]
    recommendation_reasoning: Optional[str]

    # ── Agent 3 Output ──────────────────────────────────────────────────
    audit_result: Optional[Literal["Approved", "Rejected"]]
    audit_reasoning: Optional[str]

    # ── Error Tracking ──────────────────────────────────────────────────
    error: Optional[str]
