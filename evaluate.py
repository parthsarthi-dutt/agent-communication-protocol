"""
Phase 4 — Evaluation Engine
============================
Loops through all mock users and runs the pipeline for both Protocol A (Text) 
and Protocol B (JSON). Computes and prints metrics at the end:
- Task Success Rate (Accuracy)
- Auditor Rejection Rate
- Parsing Error Rate

To respect free-tier API rate limits, a short delay is introduced between calls.

Usage:
    python evaluate.py
"""

import json
import time
import os
from tqdm import tqdm
from graph import build_graph, run_pipeline

# Configure delay to avoid free-tier rate limits (e.g., 15 RPM for Gemini free)
API_DELAY_SECONDS = 4.0

def evaluate():
    from prompts import _DATA_DIR
    
    users_path = os.path.join(_DATA_DIR, "mock_users.json")
    with open(users_path, "r", encoding="utf-8") as f:
        users = json.load(f)

    print(f"Starting evaluation on {len(users)} users...\n")
    graph = build_graph()

    metrics = {
        "text": {
            "total_runs": 0,
            "successes": 0,
            "auditor_rejections": 0,
            "parsing_errors": 0,
            "latency": 0.0
        },
        "json": {
            "total_runs": 0,
            "successes": 0,
            "auditor_rejections": 0,
            "parsing_errors": 0,
            "latency": 0.0
        }
    }

    # For testing, we might want to evaluate a subset if rate limits are an issue, 
    # but the prompt requires looping through all entries.
    for i, user in enumerate(users):
        user_input = user["query"]
        expected_policy_id = user["expected_policy_id"]
        
        print(f"\n[{i+1}/{len(users)}] User ID: {user['user_id']} | Expected: {expected_policy_id}")

        for protocol in ["text", "json"]:
            start_time = time.time()
            try:
                result = run_pipeline(graph, user_input=user_input, protocol=protocol)
                elapsed = time.time() - start_time
                
                # Check metrics
                is_success = (result["recommended_policy_id"] == expected_policy_id and result["audit_result"] == "Approved")
                is_rejected = (result["audit_result"] == "Rejected")
                has_error = (result["error"] is not None)
                
                metrics[protocol]["total_runs"] += 1
                metrics[protocol]["latency"] += elapsed
                
                if is_success and not has_error:
                    metrics[protocol]["successes"] += 1
                if is_rejected and not has_error:
                    metrics[protocol]["auditor_rejections"] += 1
                if has_error or result["recommended_policy_id"] in ["UNKNOWN", "ERROR", "PARSE_ERROR"]:
                    metrics[protocol]["parsing_errors"] += 1
                
                match_str = "SUCCESS" if is_success else "FAIL"
                err_str = f" [Errors: {result['error']}]" if has_error else ""
                print(f"  [{protocol.upper()}] -> {result['recommended_policy_id']} | {result['audit_result']} | {match_str}{err_str}")

            except Exception as e:
                print(f"  [{protocol.upper()}] -> EXCEPTION: {e}")
                metrics[protocol]["total_runs"] += 1
                metrics[protocol]["parsing_errors"] += 1
            
            # Delay to avoid rate limiting
            time.sleep(API_DELAY_SECONDS)

    # ── Summary Report ──────────────────────────────────────────────────
    print("\n" + "="*50)
    print(" EVALUATION RESULTS ".center(50, "="))
    print("="*50)
    
    for protocol in ["text", "json"]:
        m = metrics[protocol]
        total = max(m["total_runs"], 1)
        success_rate = (m["successes"] / total) * 100
        rejection_rate = (m["auditor_rejections"] / total) * 100
        error_rate = (m["parsing_errors"] / total) * 100
        avg_latency = m["latency"] / total
        
        print(f"\n--- Protocol: {protocol.upper()} ---")
        print(f"Total Runs           : {m['total_runs']}")
        print(f"Task Success Rate    : {success_rate:.1f}% ({m['successes']}/{total})")
        print(f"Auditor Rejection Rate: {rejection_rate:.1f}% ({m['auditor_rejections']}/{total})")
        print(f"Parsing Error Rate   : {error_rate:.1f}% ({m['parsing_errors']}/{total})")
        print(f"Average Latency      : {avg_latency:.2f}s")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    evaluate()
