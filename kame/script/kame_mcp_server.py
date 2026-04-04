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

# Preamble injected before tool-generated code to bypass KAME's HTML stdout
# redirect. Uses STDOUT (the original sys.stdout saved by xpythonsupport.py).
_PRINT_PREAMBLE = "import sys as _sys, builtins as _builtins; _print = lambda *a, **kw: _builtins.print(*a, **kw, file=_sys.__stdout__)\n"

server = FastMCP("kame", instructions="""You are connected to a running KAME measurement application
via its embedded IPython kernel. The `kame` module is pre-imported:
Root(), Snapshot(), Transaction() are available directly.

IMPORTANT: Call kame_api first to learn the Python API before writing code.

Key patterns:
- Read: shot = Snapshot(node); float(shot[node]) or str(shot[node])
- List children: shot.list(node) returns list[XNode], use child.getName()
- Write: node["Child"] = value (auto-transactional)
- Drivers: Root()["Drivers"]["DriverName"]
- Dump children: for c in shot.list(node): print(c.getName(), str(shot[c]))
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
                outputs.append(data.get("text/plain", ""))
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

    NOTE: print() in KAME's kernel may produce HTML. For clean text output,
    use: print(..., file=sys.__stdout__)

    Returns the stdout/stderr output and execution results.
    """
    return _execute(code)


@server.tool()
def read_node(path: str) -> str:
    """Read the current value of a KAME node by path.

    Args:
        path: Slash-separated node path from Root, e.g. "Drivers/DMM1/Value"

    Returns the node's string value and numeric value if applicable.
    """
    nav = _nav_code(path)
    code = _PRINT_PREAMBLE + f"""
_node = {nav}
_shot = Snapshot(_node)
_val = str(_shot[_node])
_print(_val)
try:
    _print("(numeric:", float(_shot[_node]), ")")
except (TypeError, ValueError):
    pass
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
    code = _PRINT_PREAMBLE + f"""
import json as _json
_node = {nav}
_shot = Snapshot(_node)
_result = []
for _child in _shot.list(_node):
    _entry = {{"name": _child.getName(), "type": _child.getTypename()}}
    try:
        _entry["value"] = str(_shot[_child])
    except Exception:
        pass
    _result.append(_entry)
_print(_json.dumps(_result, indent=2, ensure_ascii=False))
"""
    return _execute(code)


@server.tool()
def read_scalar(path: str) -> str:
    """Read a scalar value from a KAME node by path.

    Traverses the path, reads a Snapshot, and returns the numeric value.

    Args:
        path: Slash-separated path, e.g. "Drivers/TempControl/Ch.B"

    Returns the numeric value as a string, or the raw string value.
    """
    nav = _nav_code(path)
    code = _PRINT_PREAMBLE + f"""
import json as _json
_node = {nav}
_shot = Snapshot(_node)
_result = {{"path": {repr(path)}}}
try:
    _result["value"] = float(_shot[_node])
except (TypeError, ValueError):
    _result["value"] = str(_shot[_node])
_result["label"] = _node.getLabel() or _node.getName()
_print(_json.dumps(_result, ensure_ascii=False))
"""
    return _execute(code)


@server.tool()
def list_scalars() -> str:
    """List all scalar entries with their current values.

    Returns every XScalarEntry registered in the measurement,
    with name, driver, and current value. Useful for orientation.
    """
    code = _PRINT_PREAMBLE + """
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
_print(_json.dumps(_result, indent=2, ensure_ascii=False))
"""
    return _execute(code)


@server.tool()
def kame_status() -> str:
    """Check if KAME is running and show basic measurement info."""
    if not CONN_INFO_PATH.exists():
        return "KAME is not running."
    try:
        code = _PRINT_PREAMBLE + """
import os as _os
import json as _json
_drivers = Root()["Drivers"]
_dshot = Snapshot(_drivers)
_dlist = []
for _d in _dshot.list(_drivers):
    _dlist.append({"name": _d.getName(), "type": _d.getTypename()})
_print(_json.dumps({"pid": _os.getpid(), "drivers": _dlist}, indent=2, ensure_ascii=False))
"""
        return _execute(code, timeout=10)
    except Exception as e:
        return f"KAME kernel error: {e}"


if __name__ == "__main__":
    server.run(transport="stdio")
