"""
Phase 1 Validation Script
==========================
Run this to verify that mock_policies.json, mock_users.json, and the
AgentState TypedDict are all correctly defined and loadable.

Usage:
    python validate_phase1.py
"""

import json
import os
import sys

# ── Colour helpers for console output ─────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{RESET}\n")


def ok(msg: str):
    print(f"  {GREEN}✔ {msg}{RESET}")


def fail(msg: str):
    print(f"  {RED}✘ {msg}{RESET}")
    sys.exit(1)


# ── 1. Validate mock_policies.json ────────────────────────────────────
section("1. Validating mock_policies.json")

policies_path = os.path.join("data", "mock_policies.json")
if not os.path.exists(policies_path):
    fail(f"File not found: {policies_path}")

with open(policies_path, "r", encoding="utf-8") as f:
    policies = json.load(f)

ok(f"Loaded {len(policies)} policies")

required_policy_fields = [
    "policy_id", "policy_name", "monthly_premium", "min_age", "max_age",
    "allows_smokers", "covers_pre_existing", "covered_conditions",
    "excluded_conditions", "fine_print"
]

for p in policies:
    for field in required_policy_fields:
        if field not in p:
            fail(f"Policy {p.get('policy_id', '???')} missing field: {field}")
    ok(f"{p['policy_id']} — {p['policy_name']} (${p['monthly_premium']}/mo, "
       f"age {p['min_age']}-{p['max_age']}, "
       f"smokers={'✔' if p['allows_smokers'] else '✘'}, "
       f"pre-existing={'✔' if p['covers_pre_existing'] else '✘'})")


# ── 2. Validate mock_users.json ───────────────────────────────────────
section("2. Validating mock_users.json")

users_path = os.path.join("data", "mock_users.json")
if not os.path.exists(users_path):
    fail(f"File not found: {users_path}")

with open(users_path, "r", encoding="utf-8") as f:
    users = json.load(f)

ok(f"Loaded {len(users)} user queries")

policy_ids = {p["policy_id"] for p in policies}

for u in users:
    for field in ["user_id", "query", "expected_policy_id", "rationale"]:
        if field not in u:
            fail(f"User {u.get('user_id', '???')} missing field: {field}")
    if u["expected_policy_id"] not in policy_ids:
        fail(f"User {u['user_id']} references unknown policy: {u['expected_policy_id']}")
    ok(f"{u['user_id']} → expects {u['expected_policy_id']}: "
       f"\"{u['query'][:60]}...\"")


# ── 3. Validate AgentState TypedDict ──────────────────────────────────
section("3. Validating AgentState TypedDict")

try:
    from state import AgentState
    ok("AgentState imported successfully")
except ImportError as e:
    fail(f"Could not import AgentState: {e}")

expected_keys = [
    "user_input", "protocol", "intake_payload",
    "recommended_policy_id", "recommendation_reasoning",
    "audit_result", "audit_reasoning", "error"
]

annotations = AgentState.__annotations__
for key in expected_keys:
    if key not in annotations:
        fail(f"AgentState missing key: {key}")
    ok(f"AgentState.{key}: {annotations[key]}")


# ── 4. Quick integration sanity check ─────────────────────────────────
section("4. Integration Sanity Check")

sample_state: AgentState = {
    "user_input": "I am 28, non-smoker, no conditions, budget $200.",
    "protocol": "json",
    "intake_payload": "",
    "recommended_policy_id": None,
    "recommendation_reasoning": None,
    "audit_result": None,
    "audit_reasoning": None,
    "error": None,
}
ok(f"Sample AgentState created: protocol={sample_state['protocol']}")
ok(f"User input: \"{sample_state['user_input']}\"")


# ── Summary ───────────────────────────────────────────────────────────
section("PHASE 1 VALIDATION COMPLETE")
print(f"  {GREEN}{BOLD}All checks passed!{RESET}")
print(f"  • {len(policies)} policies loaded")
print(f"  • {len(users)} user test cases loaded")
print(f"  • AgentState has {len(expected_keys)} fields\n")
