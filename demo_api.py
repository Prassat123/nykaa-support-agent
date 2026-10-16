"""
demo_api.py - Tasks 11 & 12 transcript: calls the real FastAPI app over HTTP
(FastAPI TestClient) and then prints the JSON-Lines log written for each request.
"""
import json

from fastapi.testclient import TestClient

import config
from api import app

if config.LOG_FILE.exists():
    config.LOG_FILE.unlink()          # start a clean log for the demo

with TestClient(app) as client:
    print("GET /health ->", client.get("/health").json(), "\n")

    calls = [
        ("/ask", {"query": "Can I exchange my shoes for a bigger size?", "session_id": "api-demo"}),
        ("/ask", {"query": "My number is 9876543210, card ending 4321. Status of NYK-1029?",
                  "session_id": "api-demo"}),
        ("/add-document", {"doc_id": "13_gift_card_policy",
                           "text": "Nykaa gift cards are valid for 12 months from the date of purchase. "
                                   "Gift cards cannot be exchanged for cash."}),
        ("/ask", {"query": "How long is a Nykaa gift card valid?", "session_id": "api-demo"}),
    ]
    for path, body in calls:
        r = client.post(path, json=body)
        print(f"POST {path}  body={json.dumps(body)}\n  status={r.status_code}\n  response={json.dumps(r.json(), indent=2)}\n")

    r = client.post("/ask", json={"query": ""})
    print(f"POST /ask with empty query -> status {r.status_code} (Pydantic validation)\n")

from rag import remove_document
remove_document("13_gift_card_policy")   # clean up so the official knowledge base stays at 12 docs
print("(demo document 13_gift_card_policy removed again so the graded KB stays at 12 documents)\n")

print("=== logs/requests.jsonl (one JSON line per request) ===")
raw = config.LOG_FILE.read_text()
print(raw)
assert "9876543210" not in raw and "4321" not in raw, "PII leaked to log!"
print("CHECK PASSED: raw phone number / card digits do NOT appear anywhere in the log file.")
