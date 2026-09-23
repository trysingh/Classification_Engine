import os
import json
import time

# Bypass network checks for instant local cache loading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from laya import Router

print("Loading Laya decision engine into local memory...")
start_load = time.perf_counter()
router = Router(preload=True)
end_load = time.perf_counter()
print(f"Engine loaded successfully in {((end_load - start_load) * 1000):.2f} ms.\n")

incoming_payloads = [
    {
        "id": "TICKET-101",
        "state": {
            "body": "The user reported an unexpected HTTP 500 error while trying to hit /v1/checkout. Database connection timed out after 30 seconds."
        },
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this infrastructure error?",
                "criteria": ["frontend_bug", "database_timeout", "billing_gateway", "user_error"]
            },
            "urgency": {
                "type": "choice",
                "instructions": "Rate the operational impact and triage level.",
                "criteria": ["low", "medium", "critical"]
            }
        }
    },
    {
        "id": "TICKET-102",
        "state": {
            "body": "I want to cancel my subscription immediately. Your app charges too much and lacks basic team features."
        },
        "questions": {
            "churn_risk": {
                "type": "choice",
                "instructions": "Evaluate subscription retention risk.",
                "criteria": ["low", "moderate", "high"]
            },
            "sentiment": {
                "type": "choice",
                "instructions": "Identify user emotional state.",
                "criteria": ["positive", "neutral", "frustrated"]
            }
        }
    }
]

print("--- Running Dynamic Inferences ---")
print("\n| Ticket ID | Query | Response | Probabilities | Time Taken (ms) |")
print("|---|---|---|---|---|")

for item in incoming_payloads:
    start_inference = time.perf_counter()
    
    # Generate the JSON dictionary
    result = router.predict(state=item["state"], questions=item["questions"])
    
    end_inference = time.perf_counter()
    latency_ms = (end_inference - start_inference) * 1000
    
    # Extract the nested "answers" dictionary
    answers_dict = result.get("answers", {})
    
    for query_name in item["questions"].keys():
        # Look up the specific query inside the "answers" block
        query_data = answers_dict.get(query_name, {})
        
        if isinstance(query_data, dict):
            answer = query_data.get('choice', 'N/A')
            
            probs_dict = query_data.get('probabilities', {})
            if probs_dict:
                probs = ", ".join([f"{k}: {float(v):.2f}" for k, v in probs_dict.items()])
            else:
                probs = "N/A"
        else:
            answer = "N/A"
            probs = "N/A"

        print(f"| {item['id']} | {query_name} | {answer} | {probs} | {latency_ms:.2f} |")