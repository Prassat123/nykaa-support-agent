"""
dataset.py - Nykaa order dataset generator (Capstone Task 1)
"""
import random
from collections import Counter

# ---------- Design choices (README mein bhi likhenge) ----------
SEED = 42            # same seed = har baar same data
NUM_ORDERS = 50

# Kaunsi category kitni baar aaye (Beauty Nykaa ka core hai, isliye thoda zyada)
CATEGORY_WEIGHTS = {
    "Apparel": 0.22,
    "Beauty": 0.26,
    "Footwear": 0.18,
    "Home": 0.17,
    "Electronics": 0.17,
}

# Kaunsa status kitni baar aaye (zyada orders delivered hote hain)
STATUS_WEIGHTS = {
    "Placed": 0.15,
    "Shipped": 0.25,
    "Delivered": 0.35,
    "Returned": 0.12,
    "Refunded": 0.13,
}

# Har category ki realistic price range (₹)
# Price reasoning: ranges follow typical Nykaa prices - beauty items from a INR 199 lipstick to a
# INR 3999 premium serum, while electronics (hair dryers, trimmers, straighteners) go up to INR 24999.
PRICE_RANGE_INR = {
    "Beauty": (199, 3999),
    "Apparel": (499, 4999),
    "Footwear": (799, 5999),
    "Home": (299, 7999),
    "Electronics": (999, 24999),
}

DELAY_PROBABILITY = 0.20   # ~20% orders late hone chahiye


def generate_orders(seed=SEED, n=NUM_ORDERS):
    rng = random.Random(seed)          # seeded random generator
    categories = list(CATEGORY_WEIGHTS.keys())
    statuses = list(STATUS_WEIGHTS.keys())

    orders = []
    for i in range(n):
        category = rng.choices(categories, weights=list(CATEGORY_WEIGHTS.values()))[0]
        status = rng.choices(statuses, weights=list(STATUS_WEIGHTS.values()))[0]
        low, high = PRICE_RANGE_INR[category]
        orders.append({
            "record_id": f"NYK-{1001 + i}",
            "category": category,
            "status": status,
            "order_value_inr": rng.randint(low, high),
            "days_since_created": rng.randint(0, 30),
            "delayed_shipment": rng.random() < DELAY_PROBABILITY,
        })
    return orders


ORDERS = generate_orders()


def validate(orders):
    cat_counts = Counter(o["category"] for o in orders)
    status_counts = Counter(o["status"] for o in orders)
    delayed_pct = 100 * sum(o["delayed_shipment"] for o in orders) / len(orders)

    assert len(orders) >= 40, "Need at least 40 orders"
    for c in CATEGORY_WEIGHTS:
        assert cat_counts[c] >= 3, f"Category {c} has fewer than 3 orders"
    for s in STATUS_WEIGHTS:
        assert status_counts[s] >= 1, f"Status {s} is missing"
    assert 10 <= delayed_pct <= 30, f"Delayed % is {delayed_pct:.1f}, must be 10-30"
    return cat_counts, status_counts, delayed_pct


if __name__ == "__main__":
    cat_counts, status_counts, delayed_pct = validate(ORDERS)

    print(f"Total orders: {len(ORDERS)}  (seed = {SEED})\n")
    print("Count per category:")
    for c in CATEGORY_WEIGHTS:
        print(f"  {c:<12} {cat_counts[c]}")
    print("\nCount per status:")
    for s in STATUS_WEIGHTS:
        print(f"  {s:<12} {status_counts[s]}")
    print(f"\nDelayed shipments: {delayed_pct:.1f}%")
    print("\nAll checks passed!")
    print("\nSample order:", ORDERS[0])
