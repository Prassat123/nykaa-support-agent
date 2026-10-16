"""
demo_resilience.py - Task 16: retry with exponential backoff, per-node timeout,
and global (whole-graph) timeout.

Policies (config.py / agent.py):
  * RetryPolicy on node `lookup_order`: max_attempts=4, initial_interval=0.2s,
    backoff_factor=2.0, max_interval=2.0s, jitter=True, retry_on=TransientLookupError
  * Per-node timeout on `retrieve_policy`: 2.0s  (LangGraph TimeoutPolicy)
  * Global timeout on the whole graph run: 10.0s by default (asyncio.wait_for)
"""
import asyncio
import time

from langgraph.errors import NodeTimeoutError

import agent
import config
from rag import ensure_index


async def demo_retry():
    print("(a) RETRY POLICY - lookup_order fails 2 times (simulated DB outage), then succeeds")
    print(f"    policy: max_attempts={config.RETRY_MAX_ATTEMPTS}, initial_interval={config.RETRY_INITIAL_INTERVAL}s, "
          f"backoff_factor={config.RETRY_BACKOFF_FACTOR}, max_interval={config.RETRY_MAX_INTERVAL}s, "
          f"jitter={config.RETRY_JITTER}")
    t = time.perf_counter()
    resp = await agent.aask("What is the status of order NYK-1029?", "resilience-demo",
                            simulate={"transient_failures": 2})
    print(f"    RECOVERED after {agent._TRANSIENT_CALLS['count'] + 1} attempts "
          f"(within the {config.RETRY_MAX_ATTEMPTS} allowed) in {time.perf_counter() - t:.2f}s")
    print(f"    answer: {resp['answer']}\n")


async def demo_node_timeout():
    print(f"(b) PER-NODE TIMEOUT - retrieve_policy is made to take 5s; its timeout is "
          f"{config.NODE_TIMEOUT_SECONDS}s")
    t = time.perf_counter()
    try:
        await agent.aask("How long does delivery take?", "resilience-demo",
                         simulate={"slow_retrieve_seconds": 5})
        print("    ERROR: no timeout fired")
    except NodeTimeoutError as e:
        print(f"    CLEAN ERROR after {time.perf_counter() - t:.2f}s (no hang): "
              f"{type(e).__name__}: {str(e).splitlines()[0]}\n")


async def demo_global_timeout():
    budget = 3.0
    print(f"(c) GLOBAL TIMEOUT - every node takes 1s (each under its own limit) but the whole run "
          f"needs ~6s; global budget = {budget}s")
    t = time.perf_counter()
    try:
        await agent.aask("How long does delivery take?", "resilience-demo",
                         simulate={"slow_every_node_seconds": 1.0}, global_timeout=budget)
        print("    ERROR: no timeout fired")
    except agent.GraphTimeoutError as e:
        print(f"    WHOLE RUN CANCELLED after {time.perf_counter() - t:.2f}s: "
              f"{type(e).__name__}: {e}")
        print(f"    nodes completed before cancellation: {agent.EXECUTION_LOG}")


async def main():
    ensure_index()
    agent.VERBOSE["on"] = True
    await demo_retry()
    agent.EXECUTION_LOG.clear()
    await demo_node_timeout()
    agent.EXECUTION_LOG.clear()
    await demo_global_timeout()


if __name__ == "__main__":
    asyncio.run(main())
