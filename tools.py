"""
tools.py - Task 6: check_order_status tool with a designed escalation score.

escalation_score = 0.6 * delayed_shipment + 0.4 * (days_since_created / 30)

* delayed_shipment (0 or 1) carries most of the weight (0.6): a late order is the
  main reason a customer gets upset.
* the recency signal (days_since_created / 30, already in 0..1) adds up to 0.4:
  the older the order, the more urgent it is.
* the score always stays inside [0, 1].

Threshold = 0.6 + 0.4 * (median days_since_created / 30), computed from OUR dataset.
Meaning: escalate a delayed order once it is older than the median order age.
A non-delayed order can never escalate (its max score is 0.4).
"""
import statistics

from dataset import ORDERS

ORDERS_BY_ID = {o["record_id"]: o for o in ORDERS}

DELAY_WEIGHT = 0.6
RECENCY_WEIGHT = 0.4
MAX_DAYS = 30

MEDIAN_DAYS = statistics.median(o["days_since_created"] for o in ORDERS)
ESCALATION_THRESHOLD = round(DELAY_WEIGHT + RECENCY_WEIGHT * MEDIAN_DAYS / MAX_DAYS, 4)


def escalation_score(order: dict) -> float:
    recency = min(order["days_since_created"], MAX_DAYS) / MAX_DAYS
    return round(DELAY_WEIGHT * int(order["delayed_shipment"]) + RECENCY_WEIGHT * recency, 4)


def check_order_status(record_id: str) -> dict:
    """Look up a Nykaa order and return its status, value and escalation score.

    Args:
        record_id: The order ID, for example "NYK-1010".

    Returns:
        A dict with found, record_id, category, status, order_value_inr,
        delayed_shipment, days_since_created, escalation_score,
        escalation_threshold and escalate (True if score > threshold).
    """
    record_id = record_id.strip().upper()
    order = ORDERS_BY_ID.get(record_id)
    if order is None:
        return {"found": False, "record_id": record_id,
                "message": f"No order found with ID {record_id}."}
    score = escalation_score(order)
    return {
        "found": True,
        "record_id": record_id,
        "category": order["category"],
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "delayed_shipment": order["delayed_shipment"],
        "days_since_created": order["days_since_created"],
        "escalation_score": score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "escalate": score > ESCALATION_THRESHOLD,
    }


if __name__ == "__main__":
    escalated = [o["record_id"] for o in ORDERS if escalation_score(o) > ESCALATION_THRESHOLD]
    print(f"Formula: score = {DELAY_WEIGHT} * delayed_shipment + {RECENCY_WEIGHT} * (days_since_created / {MAX_DAYS})")
    print(f"Median days_since_created in our data = {MEDIAN_DAYS}")
    print(f"Threshold = {DELAY_WEIGHT} + {RECENCY_WEIGHT} * {MEDIAN_DAYS}/{MAX_DAYS} = {ESCALATION_THRESHOLD}")
    print(f"Orders above threshold: {len(escalated)}/{len(ORDERS)} -> {escalated}\n")
    for rid in ["NYK-1010", "NYK-1004", "NYK-1026", "NYK-9999"]:
        print(rid, "->", check_order_status(rid))
