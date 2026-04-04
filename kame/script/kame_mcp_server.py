#!/usr/bin/env python3
"""
MCP server that connects to KAME's embedded IPython kernel.

Provides Claude (or any MCP client) with the ability to execute Python code
in KAME's interpreter, where Root(), Snapshot(), Transaction() etc. are
pre-imported.

Requirements:
    pip install mcp jupyter_client

Usage (stdio transport, launched by Claude Code):
    python kame_mcp_server.py
"""
import json
import os
import sys
import queue
from pathlib import Path

from mcp.server.fastmcp import FastMCP
import jupyter_client

CONN_INFO_PATH = Path.home() / ".kame_kernel_connection.json"

server = FastMCP("kame", instructions="""You are connected to a running KAME measurement application
via its embedded IPython kernel. The `kame` module is pre-imported:
Root(), Snapshot(), Transaction() are available directly.

Use execute_code to interact with instruments, read data, and control experiments.
Use read_node to quickly inspect node values without writing full Python.""")

def _get_client() -> jupyter_client.BlockingKernelClient:
    """Connect to KAME's embedded IPython kernel."""
    if not CONN_INFO_PATH.exists():
        raise RuntimeError(
            "KAME is not running (no ~/.kame_kernel_connection.json). "
            "Start KAME first."
        )
    with open(CONN_INFO_PATH) as f:
        info = json.load(f)
    cf = info["connection_file"]
    if not os.path.exists(cf):
        raise RuntimeError(f"Kernel connection file not found: {cf}")
    client = jupyter_client.BlockingKernelClient()
    client.load_connection_file(cf)
    client.start_channels()
    # Verify kernel is alive
    try:
        client.wait_for_ready(timeout=5)
    except RuntimeError:
        client.stop_channels()
        raise RuntimeError("KAME kernel is not responding.")
    return client


def _execute(code: str, timeout: float = 30.0) -> str:
    """Execute code on the kernel and collect output."""
    client = _get_client()
    try:
        msg_id = client.execute(code)
        outputs = []
        while True:
            try:
                msg = client.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                outputs.append("[Timeout waiting for output]")
                break
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            msg_type = msg["msg_type"]
            content = msg["content"]
            if msg_type == "stream":
                outputs.append(content["text"])
            elif msg_type in ("execute_result", "display_data"):
                data = content.get("data", {})
                outputs.append(data.get("text/plain", ""))
            elif msg_type == "error":
                tb = "\n".join(content.get("traceback", []))
                outputs.append(f"ERROR: {content.get('ename')}: {content.get('evalue')}\n{tb}")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        return "\n".join(outputs).strip() or "(no output)"
    finally:
        client.stop_channels()


@server.tool()
def execute_code(code: str) -> str:
    """Execute Python code in KAME's interpreter.

    The kame module is pre-imported. Available globals include:
    - Root() — measurement root node
    - Snapshot(node) — immutable read view of a subtree
    - Transaction(node) — optimistic write access
    - sleep(sec) — KAME-aware sleep (use instead of time.sleep)

    Example: Read a DMM value
        shot = Snapshot(Root()["Drivers"]["DMM1"])
        print(shot[Root()["Drivers"]["DMM1"]]["Value"].to_str())

    Example: Set a parameter
        Root()["Drivers"]["TempCtrl"]["TargetTemp"].set("300")

    Returns the stdout/stderr output and execution results.
    """
    return _execute(code)


@server.tool()
def read_node(path: str) -> str:
    """Read the current value of a KAME node by path.

    Args:
        path: Slash-separated node path from Root, e.g. "Drivers/DMM1/Value"

    Returns the node's string value.
    """
    parts = path.strip("/").split("/")
    nav = 'Root()'
    for p in parts:
        nav += f'["{p}"]'
    code = f"""
_node = {nav}
_shot = Snapshot(_node)
print(str(_shot[_node]))
"""
    return _execute(code)


@server.tool()
def list_children(path: str = "") -> str:
    """List child nodes at a given path.

    Args:
        path: Slash-separated path from Root (empty string for root children).
              e.g. "Drivers" or "Drivers/DMM1"

    Returns names of all child nodes.
    """
    if path.strip("/"):
        parts = path.strip("/").split("/")
        nav = 'Root()'
        for p in parts:
            nav += f'["{p}"]'
    else:
        nav = 'Root()'
    code = f"""
_node = {nav}
_shot = Snapshot(_node)
for _child in _shot.list(_node):
    print(_child.getName())
"""
    return _execute(code)


@server.tool()
def kame_status() -> str:
    """Check if KAME is running and show basic measurement info."""
    if not CONN_INFO_PATH.exists():
        return "KAME is not running."
    try:
        result = _execute("""
import os
print(f"KAME PID: {os.getpid()}")
_shot = Snapshot(Root())
_drivers = Root()["Drivers"]
_dshot = Snapshot(_drivers)
_names = [str(_c.getName()) for _c in _dshot.list(_drivers)]
print(f"Drivers ({len(_names)}): {', '.join(_names) if _names else '(none)'}")
""", timeout=10)
        return result
    except Exception as e:
        return f"KAME kernel error: {e}"


if __name__ == "__main__":
    server.run(transport="stdio")
