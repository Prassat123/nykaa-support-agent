"""
mcp_client.py - Task 14: a SEPARATE process that connects to the MCP server and
calls check_order_status for two different order IDs.

Run (with mcp_server.py running in another terminal):  python mcp_client.py
"""
import asyncio
import json

from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"


async def main():
    async with Client(SERVER_URL) as client:
        tools = await client.list_tools()
        print(f"Connected to {SERVER_URL}")
        print("Tools exposed by server:", [t.name for t in tools])
        print("Tool description:", tools[0].description.splitlines()[0], "\n")
        for record_id in ["NYK-1010", "NYK-1004"]:
            result = await client.call_tool("check_order_status", {"record_id": record_id})
            print(f"call_tool('check_order_status', {{'record_id': '{record_id}'}})")
            print("  standardized MCP response (content[0].text):")
            print("  " + result.content[0].text)
            print("  structured_content:", json.dumps(result.structured_content))
            print("  is_error:", result.is_error, "\n")


if __name__ == "__main__":
    asyncio.run(main())
