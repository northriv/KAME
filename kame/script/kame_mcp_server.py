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

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: 'mcp' package not installed. Run:", file=sys.stderr)
    print(f"  {sys.executable} -m pip install mcp jupyter_client", file=sys.stderr)
    sys.exit(1)
try:
    import jupyter_client
except ImportError:
    print("Error: 'jupyter_client' package not installed. Run:", file=sys.stderr)
    print(f"  {sys.executable} -m pip install jupyter_client", file=sys.stderr)
    sys.exit(1)

CONN_INFO_PATH = Path.home() / ".kame_kernel_connection.json"
API_DOC_PATH = Path(__file__).parent / "kame_python_api.md"

server = FastMCP("kame", instructions="""You are connected to a running KAME measurement application
via its embedded IPython kernel. The `kame` module is pre-imported:
Root(), Snapshot(), Transaction() are available directly.

IMPORTANT: Call kame_api first to learn the Python API before writing code.

Key patterns:
- Read: shot = Snapshot(node); float(shot[node]) or str(shot[node])
- List children: shot.list(node) returns list[XNode], use child.getName()
- Write: node["Child"] = value (auto-transactional)
- Drivers: Root()["Drivers"]["DriverName"]
- Scalar entry value: entry["Value"] is XDoubleNode, float(shot[entry["Value"]])

NOTE: print() in KAME's kernel produces HTML, not plain text.
In execute_code, use expression results (last line as bare expression)
instead of print(). Example: use `result` not `print(result)`.
""")

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
                # Prefer text/plain; skip HTML object reprs
                text = data.get("text/plain", "")
                if "HTML object" not in text:
                    outputs.append(text)
            elif msg_type == "error":
                tb = "\n".join(content.get("traceback", []))
                outputs.append(f"ERROR: {content.get('ename')}: {content.get('evalue')}\n{tb}")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        return "\n".join(outputs).strip() or "(no output)"
    finally:
        client.stop_channels()


def _nav_code(path: str) -> str:
    """Build Python navigation expression from a slash-separated path."""
    parts = path.strip("/").split("/")
    nav = 'Root()'
    for p in parts:
        nav += f'["{p}"]'
    return nav


@server.tool()
def kame_api() -> str:
    """Return the KAME Python API quick reference.

    Call this FIRST before writing any code to learn the correct patterns
    for reading values, navigating nodes, and controlling instruments.
    """
    if API_DOC_PATH.exists():
        return API_DOC_PATH.read_text()
    return "API documentation not found at " + str(API_DOC_PATH)


@server.tool()
def execute_code(code: str) -> str:
    """Execute Python code in KAME's interpreter.

    The kame module is pre-imported. Available globals include:
    - Root() — measurement root node
    - Snapshot(node) — immutable read view of a subtree
    - Transaction(node) — optimistic write access
    - sleep(sec) — KAME-aware sleep (use instead of time.sleep)

    IMPORTANT: print() produces HTML in KAME's kernel. Use a bare expression
    as the last line to return results. Example:
        snap = Snapshot(Root()["Drivers"]["DMM1"])
        float(snap[Root()["Drivers"]["DMM1"]["Value"]])
    NOT:
        print(float(...))

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
    nav = _nav_code(path)
    code = f"""
_node = {nav}
str(_node)
"""
    return _execute(code)


@server.tool()
def list_children(path: str = "") -> str:
    """List child nodes at a given path with their types and values.

    Args:
        path: Slash-separated path from Root (empty string for root children).
              e.g. "Drivers" or "Drivers/DMM1"

    Returns structured list: name, typename, and value for each child.
    """
    nav = _nav_code(path) if path.strip("/") else 'Root()'
    code = f"""
import json as _json
_node = {nav}
_shot = Snapshot(_node)
_result = []
for _child in _shot.list(_node):
    _entry = {{"name": _child.getName(), "type": _child.getTypename()}}
    try:
        _entry["value"] = str(_child)
    except Exception:
        pass
    _result.append(_entry)
_json.dumps(_result, indent=2, ensure_ascii=False)
"""
    return _execute(code)


@server.tool()
def read_scalar(path: str) -> str:
    """Read a scalar value from a KAME node by path.

    For value nodes (XDoubleNode etc.), returns the numeric value directly.
    For XScalarEntry nodes, reads the "Value" child node.

    Args:
        path: Slash-separated path, e.g. "Drivers/TempControl/Ch.B"

    Returns JSON with path, label, and value.
    """
    nav = _nav_code(path)
    code = f"""
import json as _json
_node = {nav}
_shot = Snapshot(_node)
_result = {{"path": {repr(path)}, "label": _node.getLabel() or _node.getName()}}
try:
    _result["value"] = float(_shot[_node])
except (TypeError, ValueError):
    # Might be XScalarEntry — try Value child
    _val = _node["Value"]
    if _val is not None:
        _result["value"] = float(_shot[_val])
    else:
        _result["value"] = str(_node)
_json.dumps(_result, ensure_ascii=False)
"""
    return _execute(code)


@server.tool()
def list_scalars() -> str:
    """List all scalar entries with their current values.

    Returns every XScalarEntry registered in the measurement,
    with name, driver, and current value. Useful for orientation.
    """
    code = """
import json as _json
_entries = Root()["ScalarEntries"]
_shot = Snapshot(_entries)
_result = []
for _e in _shot.list(_entries):
    _entry = {"name": _e.getName(), "label": _e.getLabel() or _e.getName()}
    try:
        _val_node = _e["Value"]
        _entry["value"] = float(_shot[_val_node])
    except Exception:
        _entry["value"] = None
    try:
        _entry["driver"] = _e.driver().getName()
    except Exception:
        pass
    _result.append(_entry)
_json.dumps(_result, indent=2, ensure_ascii=False)
"""
    return _execute(code)


@server.tool()
def kame_status() -> str:
    """Check if KAME is running and show basic measurement info."""
    if not CONN_INFO_PATH.exists():
        return "KAME is not running."
    try:
        code = """
import os as _os, json as _json
_drivers = Root()["Drivers"]
_dshot = Snapshot(_drivers)
_dlist = []
for _d in _dshot.list(_drivers):
    _dlist.append({"name": _d.getName(), "type": _d.getTypename()})
_json.dumps({"pid": _os.getpid(), "drivers": _dlist}, indent=2, ensure_ascii=False)
"""
        return _execute(code, timeout=10)
    except Exception as e:
        return f"KAME kernel error: {e}"


if __name__ == "__main__":
    server.run(transport="stdio")
