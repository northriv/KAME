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
import base64
import json
import os
import sys
import queue
import time
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP, Image
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

_client = None

def _get_client() -> jupyter_client.BlockingKernelClient:
    """Connect to KAME's embedded IPython kernel (reuses connection)."""
    global _client
    if _client is not None and _client.is_alive():
        return _client
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
    # Enable inline matplotlib so plots produce image/png
    client.execute("%matplotlib inline")
    _client = client
    return client


def _execute(code: str, timeout: float = 30.0) -> list:
    """Execute code on the kernel and collect output (text and images)."""
    client = _get_client()
    try:
        msg_id = client.execute(code)
        outputs = []  # str for text, Image for images
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
                # Return images via MCP Image content
                if "image/png" in data:
                    outputs.append(Image(
                        data=base64.b64decode(data["image/png"]),
                        format="png",
                    ))
                else:
                    # Prefer text/plain; skip HTML object reprs
                    text = data.get("text/plain", "")
                    if "HTML object" not in text:
                        outputs.append(text)
            elif msg_type == "error":
                tb = "\n".join(content.get("traceback", []))
                outputs.append(f"ERROR: {content.get('ename')}: {content.get('evalue')}\n{tb}")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        if not outputs:
            return ["(no output)"]
        return outputs
    except Exception:
        # Reset client on error so next call reconnects
        global _client
        _client = None
        raise


def _execute_text(code: str, timeout: float = 30.0) -> str:
    """Execute code and return only text output (no images)."""
    results = _execute(code, timeout)
    return "\n".join(r for r in results if isinstance(r, str)).strip() or "(no output)"


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
def execute_code(code: str) -> list:
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

    Returns the stdout/stderr output, execution results, and matplotlib plots.
    """
    return _execute(code)


@server.tool()
def execute_code_async(code: str) -> str:
    """Execute long-running Python code asynchronously.

    Use this for experiments that take minutes or hours (sweeps, scans,
    relaxation measurements). The code runs in a background thread on the
    kernel while the kernel remains responsive for other tool calls.

    Store results in global variables so you can read them afterward
    with execute_code. Example:

        execute_code_async('''
        results = []
        for field in [0, 1, 2, 3, 4, 5]:
            dc_source["Value"] = field
            sleep(1.0)
            snap = Snapshot(root)
            results.append({"field": field, "signal": float(snap[nmr_val])})
        ''')

    Then check status with get_result(job_id), and when done, use
    execute_code to read and plot the results variable.

    CAUTION: The background thread shares globals() with the kernel.
    KAME node operations (Snapshot, Transaction, node[]=) are thread-safe,
    but avoid reading Python variables from other tools while the async
    job is still writing to them. Wait for "done" status before reading
    result variables.

    Returns a job_id string for use with get_result().
    """
    job_id = f"_mcp_{int(time.time() * 1000)}"
    wrapper = f"""
import threading as _th, traceback as _tb
_mcp_jobs = globals().setdefault("_mcp_jobs", {{}})
_mcp_jobs[{repr(job_id)}] = {{"status": "running"}}
def _mcp_run():
    try:
        exec({repr(code)}, globals())
        _mcp_jobs[{repr(job_id)}]["status"] = "done"
    except Exception:
        _mcp_jobs[{repr(job_id)}]["status"] = "error"
        _mcp_jobs[{repr(job_id)}]["error"] = _tb.format_exc()
_th.Thread(target=_mcp_run, daemon=True).start()
{repr(job_id)}
"""
    return _execute_text(wrapper)


@server.tool()
def get_result(job_id: str) -> str:
    """Check the status of an async job started with execute_code_async.

    Args:
        job_id: The job ID returned by execute_code_async.

    Returns JSON with "status" ("running", "done", or "error").
    If error, includes "error" with the traceback.
    """
    code = f"""
import json as _json
_json.dumps(_mcp_jobs.get({repr(job_id)}, {{"status": "unknown"}}))
"""
    return _execute_text(code)


@server.tool()
def tree(path: str = "", depth: int = 2) -> str:
    """List child nodes at a given path as an indented tree.

    Args:
        path: Slash-separated path from Root (empty string for root children).
              e.g. "Drivers" or "Drivers/DMM1"
        depth: Max depth to traverse (default 2, max 5). 1 = immediate children only.

    Returns compact indented tree: name (type) = value
    """
    depth = max(1, min(depth, 5))
    nav = _nav_code(path) if path.strip("/") else 'Root()'
    code = f"""
def _mcp_tree(_node, _depth, _max_depth, _indent=0):
    _shot = Snapshot(_node)
    _lines = []
    for _child in _shot.list(_node):
        _name = _child.getName()
        _typ = _child.getTypename()
        try:
            _val = str(_child)
            # Skip verbose object reprs
            if _val.startswith("<node["):
                _val = None
        except Exception:
            _val = None
        _prefix = "  " * _indent
        if _val is not None:
            _lines.append(f"{{_prefix}}{{_name}} ({{_typ}}) = {{_val}}")
        else:
            _lines.append(f"{{_prefix}}{{_name}} ({{_typ}})")
        if _depth < _max_depth:
            try:
                _lines.extend(_mcp_tree(_child, _depth + 1, _max_depth, _indent + 1))
            except Exception:
                pass
    return _lines
"\\n".join(_mcp_tree({nav}, 1, {depth}))
"""
    return _execute_text(code)


@server.tool()
def kame_status() -> str:
    """Check if KAME is running and show basic measurement info."""
    if not CONN_INFO_PATH.exists():
        return "KAME is not running."
    try:
        code = """
import os as _os
_drivers = Root()["Drivers"]
_dshot = Snapshot(_drivers)
_lines = [f"PID: {_os.getpid()}", f"Drivers ({_dshot.size(_drivers)}):"]
for _d in _dshot.list(_drivers):
    _lines.append(f"  {_d.getName()} ({_d.getTypename()})")
"\\n".join(_lines)
"""
        return _execute_text(code, timeout=10)
    except Exception as e:
        return f"KAME kernel error: {e}"


if __name__ == "__main__":
    server.run(transport="stdio")
