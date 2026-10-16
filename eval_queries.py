"""eval_queries.py - the fixed query sets used by Tasks 4, 5 and 13."""

# Task 4 + 5: in-scope queries with the document(s) that truly answer them
IN_SCOPE = [
    {"query": "How many days do I have to return footwear?",
     "relevant": ["01_return_window", "09_size_exchange"]},
    {"query": "How long does a COD refund take to reach my bank account?",
     "relevant": ["02_cod_refund_timeline"]},
    {"query": "How long does standard delivery take in metro cities?",
     "relevant": ["03_delivery_sla"]},
    {"query": "Can I cancel my order after it has been shipped?",
     "relevant": ["06_order_cancellation"]},
    {"query": "What is the warranty on electronics like hair dryers?",
     "relevant": ["05_warranty_terms"]},
    {"query": "How do I raise a claim for a damaged item?",
     "relevant": ["10_damaged_item_claim"]},
]

# Task 4: deliberately out-of-scope queries (must trigger "I don't know")
OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Can you give me a recipe for butter chicken?",
    "Who won the cricket world cup in 2011?",
]

# Task 13: 15 queries = 12 (one per KB topic) + 3 out-of-scope / edge cases
TRIAD_QUERIES = [
    ("return window by category", "Within how many days can I return a sealed lipstick or Electronics item?"),
    ("COD refund timelines", "I paid cash on delivery, how will I get my refund and how long does it take?"),
    ("delivery SLAs", "How many business days does delivery take to a non-metro pin code?"),
    ("reverse-pickup eligibility", "What happens if reverse pickup is not available at my pin code?"),
    ("warranty terms by category", "Do apparel or beauty products come with a warranty?"),
    ("order-cancellation policy", "Can I cancel an order while its status is Placed?"),
    ("loyalty-points redemption", "How much is one loyalty point worth and when do points expire?"),
    ("payment failure / retry", "My payment failed but money was debited, when will I get it back?"),
    ("size-exchange policy", "Can I exchange my shoes for a different size?"),
    ("damaged-item claim process", "My item arrived broken, how many hours do I have to raise a claim?"),
    ("international shipping restrictions", "Can I ship perfume internationally to the UK?"),
    ("support escalation matrix", "When is my complaint escalated to the grievance officer?"),
    ("OUT-OF-SCOPE", "What is the best laptop for gaming in 2026?"),
    ("OUT-OF-SCOPE", "Tell me a joke about cats."),
    ("EDGE CASE (vague)", "help"),
]
