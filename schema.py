"""
schema.py - Task 9: the JSON Schema every agent response must follow.
validate_response() is called on EVERY response inside the agent (finalize node).
"""
from jsonschema import validate

RESPONSE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "NykaaAgentResponse",
    "type": "object",
    "required": ["session_id", "query", "route", "answer", "sources", "order",
                 "guardrails", "nodes_executed"],
    "additionalProperties": False,
    "properties": {
        "session_id": {"type": "string", "minLength": 1},
        "query": {"type": "string"},                      # PII-masked query
        "route": {"enum": ["rag", "order_lookup", "blocked", "clarify"]},
        "answer": {"type": "string", "minLength": 1},
        "sources": {"type": "array", "items": {"type": "string"}},
        "order": {
            "type": ["object", "null"],
            "properties": {
                "record_id": {"type": "string"},
                "status": {"type": "string"},
                "order_value_inr": {"type": "integer"},
                "escalation_score": {"type": "number", "minimum": 0, "maximum": 1},
                "escalate": {"type": "boolean"},
            },
        },
        "guardrails": {
            "type": "object",
            "required": ["pii_masked", "injection_detected", "grounded"],
            "properties": {
                "pii_masked": {"type": "array", "items": {"type": "string"}},
                "injection_detected": {"type": "boolean"},
                "grounded": {"type": "boolean"},
                "note": {"type": "string"},
            },
        },
        "nodes_executed": {"type": "array", "items": {"type": "string"}},
    },
}


def validate_response(response: dict) -> dict:
    """Raises jsonschema.ValidationError if the response breaks the schema."""
    validate(instance=response, schema=RESPONSE_SCHEMA)
    return response
