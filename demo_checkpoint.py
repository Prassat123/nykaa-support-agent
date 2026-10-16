"""
demo_checkpoint.py - Task 15: SQLite checkpointing + resume without re-running nodes.

Run in TWO separate processes (that is the point - state survives a crash/stop):
    python demo_checkpoint.py phase1   # runs 3 nodes, then STOPS before generate_answer
    python demo_checkpoint.py phase2   # new process, same thread_id -> resumes from checkpoints.sqlite
"""
import asyncio
import sys

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import agent
import config
from rag import ensure_index

THREAD_ID = "checkpoint-demo-thread-1"
RUN_CONFIG = {"configurable": {"thread_id": THREAD_ID}}
QUERY = "How long does a COD refund take to reach my bank account?"
FIRST_NODES = ["input_guard", "classify_intent", "retrieve_policy"]


async def phase1():
    config.CHECKPOINT_DB.unlink(missing_ok=True)          # fresh demo
    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as saver:
        graph = agent.build_graph(checkpointer=saver, interrupt_before=["generate_answer"])
        print(f"PHASE 1 (process A) - thread_id = {THREAD_ID}")
        print(f"Query: {QUERY}")
        print("Graph is configured to STOP before 'generate_answer' (simulated interruption).\n")
        await graph.ainvoke({"session_id": "checkpoint-demo", "raw_query": QUERY,
                             "simulate": {}, "nodes_executed": []}, RUN_CONFIG)
        snap = await graph.aget_state(RUN_CONFIG)
        print(f"\nRun stopped. Nodes executed in this process: {agent.EXECUTION_LOG}")
        print(f"Checkpoint saved to {config.CHECKPOINT_DB.name}")
        print(f"  saved nodes_executed = {snap.values['nodes_executed']}")
        print(f"  saved route          = {snap.values['route']}")
        print(f"  saved chunks         = {[c['doc_id'] for c in snap.values['chunks']]}")
        print(f"  next node to run     = {snap.next}")
        print("Process A now exits without finishing the run.")


async def phase2():
    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as saver:
        graph = agent.build_graph(checkpointer=saver)
        snap = await graph.aget_state(RUN_CONFIG)
        print(f"PHASE 2 (NEW process B) - resuming the SAME thread_id = {THREAD_ID}")
        print("State LOADED from checkpoints.sqlite (nothing recomputed):")
        print(f"  nodes already completed = {snap.values['nodes_executed']}")
        print(f"  route (from checkpoint) = {snap.values['route']}")
        print(f"  chunks (from checkpoint)= {[c['doc_id'] for c in snap.values['chunks']]}")
        print(f"  pending next node       = {snap.next}\n")
        print("Resuming with graph.ainvoke(None, same config):")
        result = await graph.ainvoke(None, RUN_CONFIG)
        print(f"\nNodes executed in process B : {agent.EXECUTION_LOG}")
        re_run = [n for n in FIRST_NODES if n in agent.EXECUTION_LOG]
        print(f"Already-completed nodes re-executed in process B: {re_run or 'NONE'}")
        assert not re_run, "checkpoint resume re-ran completed nodes!"
        print(f"Full node history of the run: {result['response']['nodes_executed']}")
        print(f"\nFinal answer: {result['response']['answer']}")
        print("CHECK PASSED: run resumed from the checkpoint and finished without repeating nodes.")


if __name__ == "__main__":
    ensure_index()
    agent.VERBOSE["on"] = True
    asyncio.run({"phase1": phase1, "phase2": phase2}[sys.argv[1]]())
