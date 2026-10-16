"""
memory.py - Task 8: conversation memory persisted to a JSON file (memory/sessions.json).

Each session_id keeps its own turn history and the last order ID mentioned, so a
follow-up like "is it delayed?" knows which order "it" is. A new session_id starts empty.
"""
import json

import config


def _load_all() -> dict:
    if config.MEMORY_FILE.exists():
        return json.loads(config.MEMORY_FILE.read_text(encoding="utf-8"))
    return {}


def _save_all(data: dict) -> None:
    config.MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.MEMORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_session(session_id: str) -> dict:
    return _load_all().get(session_id, {"turns": [], "last_order_id": None})


def save_turn(session_id: str, query: str, answer: str, route: str, order_id: str | None) -> None:
    data = _load_all()
    session = data.setdefault(session_id, {"turns": [], "last_order_id": None})
    session["turns"].append({"user": query, "assistant": answer, "route": route})
    if order_id:
        session["last_order_id"] = order_id
    _save_all(data)


def reset_session(session_id: str) -> None:
    data = _load_all()
    data.pop(session_id, None)
    _save_all(data)
