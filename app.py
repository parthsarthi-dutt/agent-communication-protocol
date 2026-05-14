"""
Phase 5 — Interactive UI
========================
Streamlit app to visualize the frontend user experience and backend MAS orchestration.
It demonstrates why Protocol B (JSON) is typically more reliable than Protocol A.

Usage:
    streamlit run app.py
"""

import streamlit as st
import json
import time
from graph import build_graph, run_pipeline
from state import AgentState

st.set_page_config(page_title="Multi-Agent Insurance Advisor", layout="wide")

# Cached graph loading
@st.cache_resource
def get_graph():
    return build_graph()

graph = get_graph()

st.title("🛡️ MAS Insurance Policy Advisory")
st.markdown("Evaluating Communication Protocol Design between Agents: Free-Text (A) vs Structured JSON (B)")
st.divider()

# Sidebar for Instructions
with st.sidebar:
    st.header("About the Simulation")
    st.markdown("""
    This app simulates a 3-agent LangGraph pipeline:
    1. **Intake Agent**: Parses natural language into Text or JSON.
    2. **Recommender Agent**: Queries the mock DB to recommend a policy.
    3. **Compliance Auditor**: Cross-references the recommendation with the original query to ensure no rules are violated.
    """)
    st.markdown("---")
    st.subheader("💡 Try this test query:")
    st.code("I'm 42 and I have Type 2 diabetes. I don't smoke. I need something that covers my diabetes treatment and costs under $400 a month.", language="text")

# CREATE TWO COLUMNS
col_front, col_back = st.columns([1, 1.5], gap="large")

with col_front:
    st.header("👤 Frontend UI")
    st.markdown("Simulate the end-user facing experience.")
    
    # ── Protocol Selection ──
    protocol_selector = st.radio(
        "Communication Protocol:", 
        options=["Free-Text (Protocol A)", "Strict JSON (Protocol B)"],
        index=1,
        horizontal=True
    )
    protocol = "text" if "A" in protocol_selector else "json"
    
    # ── User Input ──
    user_query = st.text_area(
        "Describe your insurance needs:", 
        placeholder="E.g., I am 28 years old, non-smoker, no health issues..."
    )
    
    submit_btn = st.button("Get Recommendation", type="primary", use_container_width=True)
    
    # ── Result Display Placeholders ──
    result_container = st.container()

with col_back:
    st.header("⚙️ Backend MAS Inspector")
    st.markdown("Observe real-time state transitions between the agents.")
    
    # Placeholders for backend components
    node1_placeholder = st.empty()
    node2_placeholder = st.empty()
    node3_placeholder = st.empty()


if submit_btn and user_query.strip():
    with st.spinner("Pipeline running..."):
        # Reset backend inspector views
        with node1_placeholder.container():
            st.info("Agent 1 (Intake) is parsing the query...")
        time.sleep(0.5)
        
        # Run graph
        start_time = time.time()
        final_state = run_pipeline(graph, user_query, protocol)
        elapsed = time.time() - start_time
        
        # --- UPDATE FRONTEND ---
        with result_container:
            st.subheader("Outcome")
            if final_state["error"]:
                st.error("Pipeline crashed due to formatting/parsing errors.")
                st.code(final_state["error"])
            else:
                if final_state["audit_result"] == "Approved":
                    st.success(f"**Recommended Policy:** {final_state['recommended_policy_id']}")
                else:
                    st.error(f"**Recommender suggested:** {final_state['recommended_policy_id']} (but it was REJECTED by Audit)")
                
                st.markdown(f"**Reasoning:** {final_state['recommendation_reasoning']}")
                st.caption(f"Latency: {elapsed:.2f}s")
                
        # --- UPDATE BACKEND ---
        node1_placeholder.empty()
        with node1_placeholder.container():
            with st.expander("🤖 Agent 1: Intake Payload", expanded=True):
                st.caption(f"Protocol: {protocol.upper()}")
                if protocol == "text":
                    st.markdown(f"> {final_state['intake_payload']}")
                else:
                    try:
                        st.json(json.loads(final_state['intake_payload']))
                    except:
                        st.code(final_state['intake_payload']) # Fallback if invalid JSON
                
        with node2_placeholder.container():
            with st.expander("🧠 Agent 2: Recommender Output", expanded=True):
                st.markdown(f"**Selected Policy:** `{final_state['recommended_policy_id']}`")
                st.markdown(f"**Thought Process:** {final_state['recommendation_reasoning']}")
                
        with node3_placeholder.container():
            with st.expander("⚖️ Agent 3: Auditor Logs", expanded=True):
                if final_state["audit_result"] == "Approved":
                    st.success("**Status: APPROVED**")
                else:
                    st.error(f"**Status: {final_state['audit_result']}**")
                    
                st.markdown(f"**Compliance Notes:** {final_state['audit_reasoning']}")
