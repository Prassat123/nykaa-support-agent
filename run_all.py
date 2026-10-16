"""
run_all.py - runs EVERY task in order and saves the output to transcripts/.

    python run_all.py

Everything runs under MOCK_LLM (zero API keys). The first run downloads the free
all-MiniLM-L6-v2 embedding model (~90 MB) from HuggingFace once.
"""
import os
import socket
import subprocess
import sys
import time

import config

PY = sys.executable
ENV = {**os.environ, "PYTHONWARNINGS": "ignore", "TOKENIZERS_PARALLELISM": "false",
       "PYTHONUNBUFFERED": "1"}

STEPS = [
    ("task01_dataset.txt", [["dataset.py"]]),
    ("task02_03_knowledge_base_and_chunking.txt", [["rag.py"]]),
    ("task05_chunking_evaluation.txt", [["eval_chunking.py"]]),
    ("task04_threshold_and_grounded_generation.txt", [["calibrate.py"]]),
    ("task06_order_status_tool.txt", [["tools.py"]]),
    ("task07_langgraph_routing.txt", [["demo_agent.py", "task7"]]),
    ("task08_memory_multiturn.txt", [["demo_agent.py", "task8a"]]),
    ("task08_memory_fresh_session.txt", [["demo_agent.py", "task8b"]]),
    ("task09_structured_output.txt", [["demo_agent.py", "task9"]]),
    ("task10_guardrails.txt", [["demo_agent.py", "task10"]]),
    ("task11_12_fastapi_and_logging.txt", [["demo_api.py"]]),
    ("task13_rag_triad.txt", [["eval_triad.py"]]),
    ("task15_checkpointing.txt", [["demo_checkpoint.py", "phase1"], ["demo_checkpoint.py", "phase2"]]),
    ("task16_retries_and_timeouts.txt", [["demo_resilience.py"]]),
]


def run(args: list[str]) -> str:
    print(f"   $ python {' '.join(args)}")
    p = subprocess.run([PY, *args], capture_output=True, text=True, env=ENV, cwd=config.ROOT)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-4000:])
        raise SystemExit(f"FAILED: python {' '.join(args)}")
    return p.stdout


def port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_mcp() -> str:
    print("   $ python mcp_server.py   (background)  +  python mcp_client.py")
    server = subprocess.Popen([PY, "mcp_server.py"], env=ENV, cwd=config.ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            if port_open(8000):
                break
            time.sleep(0.5)
        else:
            raise SystemExit("MCP server did not start on port 8000")
        return ("MCP server: python mcp_server.py  ->  http://127.0.0.1:8000/mcp (separate process)\n"
                "MCP client: python mcp_client.py  (separate process)\n\n" + run(["mcp_client.py"]))
    finally:
        server.terminate()


def main():
    config.TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    config.MEMORY_FILE.unlink(missing_ok=True)           # clean memory for reproducible transcripts
    print(f"MOCK_LLM = {config.MOCK_LLM}\n")
    steps = STEPS[:12] + [("task14_mcp.txt", None)] + STEPS[12:]
    for i, (fname, cmds) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {fname}")
        if cmds is None:
            out = run_mcp()
        else:
            out = "\n\n".join(run(c) for c in cmds)
        (config.TRANSCRIPTS_DIR / fname).write_text(out, encoding="utf-8")
    print("\nALL TASKS RAN SUCCESSFULLY. Transcripts saved in transcripts/")


if __name__ == "__main__":
    main()
