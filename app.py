"""
Phase 5 — Interactive UI (Multi-Provider)
==========================================
Streamlit app demonstrating hybrid Edge/Cloud orchestration.
Each agent can be independently routed to a different LLM provider
(Ollama local, Google GenAI, or Groq).

Usage:
    streamlit run app.py
"""

import streamlit as st
import json
import os
import time
from graph import build_graph, run_pipeline_multi_llm
from state import AgentState

# ── Load policy database for detail lookups ────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
with open(os.path.join(_DATA_DIR, "mock_policies.json"), "r", encoding="utf-8") as f:
    _POLICIES = {p["policy_id"]: p for p in json.load(f)}

st.set_page_config(
    page_title="Multi-Agent Insurance Advisor",
    page_icon="🛡️",
    layout="wide",
)

# ── Cached graph ────────────────────────────────────────────────────────
@st.cache_resource
def get_graph():
    return build_graph()

graph = get_graph()

# ── Provider / Model Defaults ──────────────────────────────────────────
PROVIDER_OPTIONS = ["Google GenAI", "Groq", "Ollama (Local)"]
PROVIDER_MAP = {
    "Google GenAI": "gemini",
    "Groq": "groq",
    "Ollama (Local)": "ollama",
}

DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:7b-instruct-q4_K_M",
}

PRESET_MODELS = {
    "gemini": [
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant",
    ],
    "ollama": [
        "qwen2.5:7b-instruct-q4_K_M",
    ]
}

PROTOCOL_OPTIONS = {
    "Free-Text (Protocol A)": "text",
    "Strict JSON (Protocol B)": "json",
    "Markdown JSON (Protocol C)": "markdown_json",
}


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR — Model Routing & Info
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Agent Model Routing")
    st.caption("Route each agent to a different LLM provider for hybrid Edge/Cloud orchestration.")

    st.markdown("---")

    # ── Intake Agent ──
    st.subheader("🤖 Agent 1: Intake")
    intake_provider_label = st.selectbox(
        "Provider", PROVIDER_OPTIONS, index=0, key="intake_prov",
        help="Ollama runs locally (edge), GenAI/Groq run in the cloud."
    )
    intake_provider = PROVIDER_MAP[intake_provider_label]
    intake_model = st.selectbox(
        "Model", options=PRESET_MODELS[intake_provider], key="intake_model"
    )

    st.markdown("---")

    # ── Recommender Agent ──
    st.subheader("🧠 Agent 2: Recommender")
    rec_provider_label = st.selectbox(
        "Provider", PROVIDER_OPTIONS, index=0, key="rec_prov",
        help="The recommender benefits from a larger model for better reasoning."
    )
    rec_provider = PROVIDER_MAP[rec_provider_label]
    rec_model = st.selectbox(
        "Model", options=PRESET_MODELS[rec_provider], key="rec_model"
    )

    st.markdown("---")

    # ── Auditor Agent ──
    st.subheader("⚖️ Agent 3: Auditor")
    aud_provider_label = st.selectbox(
        "Provider", PROVIDER_OPTIONS, index=0, key="aud_prov",
        help="The auditor cross-checks the recommendation for compliance."
    )
    aud_provider = PROVIDER_MAP[aud_provider_label]
    aud_model = st.selectbox(
        "Model", options=PRESET_MODELS[aud_provider], key="aud_model"
    )

    st.markdown("---")

    # ── About ──
    st.header("📖 About")
    st.markdown("""
    This app simulates a **3-agent LangGraph pipeline**:
    1. **Intake Agent**: Parses natural language into the selected protocol format.
    2. **Recommender Agent**: Queries the mock DB to recommend a policy.
    3. **Compliance Auditor**: Cross-references the recommendation for rule violations.

    Each agent can run on a **different LLM provider**, demonstrating
    hybrid Edge/Cloud orchestration.
    """)
    st.markdown("---")
    st.subheader("💡 Try this test query:")
    st.code(
        "I'm 42 and I have Type 2 diabetes. I don't smoke. "
        "I need something that covers my diabetes treatment "
        "and costs under $400 a month.",
        language="text",
    )


# ═══════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════

st.title("🛡️ MAS Insurance Policy Advisory")
st.markdown(
    "Evaluating Communication Protocol Design between Agents: "
    "**Free-Text (A)** vs **Structured JSON (B)** vs **Markdown JSON (C)**"
)
st.divider()

# ── Two-column layout ──
col_front, col_back = st.columns([1, 1.5], gap="large")

with col_front:
    st.header("👤 Frontend UI")
    st.markdown("Simulate the end-user facing experience.")

    # ── Protocol Selection ──
    protocol_label = st.radio(
        "Communication Protocol:",
        options=list(PROTOCOL_OPTIONS.keys()),
        index=1,
        horizontal=True,
    )
    protocol = PROTOCOL_OPTIONS[protocol_label]

    # ── Model routing summary ──
    with st.expander("🔌 Active Model Routing", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Intake", f"{intake_provider_label}", intake_model)
        col_b.metric("Recommender", f"{rec_provider_label}", rec_model)
        col_c.metric("Auditor", f"{aud_provider_label}", aud_model)

    # ── User Input ──
    user_query = st.text_area(
        "Describe your insurance needs:",
        placeholder="E.g., I am 28 years old, non-smoker, no health issues...",
    )

    submit_btn = st.button("Get Recommendation", type="primary", use_container_width=True)

    # ── Result Display Placeholder ──
    result_container = st.container()

with col_back:
    st.header("⚙️ Backend MAS Inspector")
    st.markdown("Observe real-time state transitions between the agents.")

    # Placeholders for backend components
    node1_placeholder = st.empty()
    node2_placeholder = st.empty()
    node3_placeholder = st.empty()


# ═══════════════════════════════════════════════════════════════════════
#  PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════

if submit_btn and user_query.strip():
    with st.spinner("Pipeline running..."):
        # Show progress in backend inspector
        with node1_placeholder.container():
            st.info("🤖 Agent 1 (Intake) is parsing the query...")
        time.sleep(0.3)

        # Run graph with per-agent model routing
        start_time = time.time()
        final_state = run_pipeline_multi_llm(
            graph,
            user_input=user_query,
            protocol=protocol,
            intake_config=(intake_provider, intake_model),
            recommend_config=(rec_provider, rec_model),
            auditor_config=(aud_provider, aud_model),
        )
        elapsed = time.time() - start_time

        # ── Compute payload size ──
        payload_chars = len(final_state.get("intake_payload", ""))

        # ────────────────────────────────────────────────────────────
        #  UPDATE FRONTEND
        # ────────────────────────────────────────────────────────────
        with result_container:
            st.subheader("Outcome")
            if final_state["error"]:
                st.error("Pipeline encountered errors during processing.")
                st.code(final_state["error"])
            else:
                if final_state["audit_result"] == "Approved":
                    st.success(f"**Recommended Policy:** {final_state['recommended_policy_id']}")
                else:
                    st.error(
                        f"**Recommender suggested:** {final_state['recommended_policy_id']} "
                        f"(but it was REJECTED by Audit)"
                    )

                st.markdown(f"**Reasoning:** {final_state['recommendation_reasoning']}")

                # ── Policy Details Card ──
                policy_id = final_state.get("recommended_policy_id", "")
                policy = _POLICIES.get(policy_id)
                if policy:
                    with st.expander(f"📋 Policy Details — {policy['policy_name']}", expanded=True):
                        dc1, dc2, dc3 = st.columns(3)
                        dc1.metric("Monthly Premium", f"${policy['monthly_premium']}")
                        dc2.metric("Annual Deductible", f"${policy['annual_deductible']:,}")
                        dc3.metric("Max Coverage", f"${policy['max_coverage']:,}")

                        dc4, dc5, dc6 = st.columns(3)
                        dc4.metric("Age Range", f"{policy['min_age']} – {policy['max_age']}")
                        dc5.metric("Smokers", "✅ Allowed" if policy['allows_smokers'] else "❌ Not Allowed")
                        dc6.metric("Pre-existing", "✅ Covered" if policy['covers_pre_existing'] else "❌ Excluded")

                        st.markdown(f"**Covers:** {', '.join(policy['covered_conditions'])}")
                        st.caption(f"Network: {policy['network_type']} | Waiting Period: {policy['waiting_period_months']} months")

            # Metrics row
            met_col1, met_col2, met_col3 = st.columns(3)
            met_col1.metric("Latency", f"{elapsed:.2f}s")
            met_col2.metric("Payload Size", f"{payload_chars} chars")
            met_col3.metric("Protocol", protocol.upper())

        # ────────────────────────────────────────────────────────────
        #  UPDATE BACKEND INSPECTOR
        # ────────────────────────────────────────────────────────────
        node1_placeholder.empty()
        with node1_placeholder.container():
            with st.expander("🤖 Agent 1: Intake Payload", expanded=True):
                st.caption(f"Protocol: {protocol.upper()} | Provider: {intake_provider_label} ({intake_model}) | Payload: {payload_chars} chars")
                if protocol == "text":
                    st.markdown(f"> {final_state['intake_payload']}")
                elif protocol == "json":
                    try:
                        st.json(json.loads(final_state["intake_payload"]))
                    except Exception:
                        st.code(final_state["intake_payload"])
                else:  # markdown_json
                    st.markdown(final_state["intake_payload"])

        with node2_placeholder.container():
            with st.expander("🧠 Agent 2: Recommender Output", expanded=True):
                st.caption(f"Provider: {rec_provider_label} ({rec_model})")
                st.markdown(f"**Selected Policy:** `{final_state['recommended_policy_id']}`")
                st.markdown(f"**Thought Process:** {final_state['recommendation_reasoning']}")

        with node3_placeholder.container():
            with st.expander("⚖️ Agent 3: Auditor Logs", expanded=True):
                st.caption(f"Provider: {aud_provider_label} ({aud_model})")
                if final_state["audit_result"] == "Approved":
                    st.success("**Status: APPROVED**")
                else:
                    st.error(f"**Status: {final_state['audit_result']}**")

                st.markdown(f"**Compliance Notes:** {final_state['audit_reasoning']}")
