"""
mcp_server.py - Task 14: exposes the check_order_status lookup tool over MCP.

Run:  python mcp_server.py
Serves FastMCP over HTTP at  http://127.0.0.1:8000/mcp   (note the /mcp path)
"""
from fastmcp import FastMCP

from tools import check_order_status as _check_order_status

mcp = FastMCP("nykaa-order-tools")


@mcp.tool
def check_order_status(record_id: str) -> dict:
    """Look up a Nykaa order by its ID and return its current status.

    Use this tool when a customer asks where their order is, whether it is delayed,
    or whether it needs escalation.

    Args:
        record_id: The Nykaa order ID, for example "NYK-1010".

    Returns:
        found, record_id, category, status, order_value_inr, delayed_shipment,
        days_since_created, escalation_score (0..1), escalation_threshold and
        escalate (True when the score is above the threshold).
    """
    return _check_order_status(record_id)


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)
