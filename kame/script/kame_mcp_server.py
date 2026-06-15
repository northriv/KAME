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
import re
import sys
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
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
MANUAL_DOC_PATHS = [
    Path(__file__).parent / "kame-8-en.md",  # deployed (e.g. Contents/Resources)
    Path(__file__).parent.parent.parent / "doc" / "manual" / "kame-8-en.md",  # source tree
]

server = FastMCP("kame", instructions="""You are connected to a running KAME measurement application
via its embedded IPython kernel. The `kame` module is pre-imported:
Root(), Snapshot(), Transaction() are available directly.

IMPORTANT: Call kame_api first to learn the Python API before writing code.
For end-user topics (UI operation, driver-specific settings, NMR measurement
workflow, installation), consult kame_manual.

Camera/2D images: read quantitative values through 2D math tools, whose
functors receive the raw uint32 count matrix (see kame_api, "2D Math
Tools"). to_png() is the gamma-encoded display image: fine for viewing
and for binary segmentation / mask generation (rank-based thresholds,
1:1 pixel coordinates), but never read signal values from its pixels.

Notebook cell editing: the user's measurements live in notebook cells.
Workflow: notebook_status (kernel busy? which cell is running?) →
notebook_read → notebook_edit. Edits change the .ipynb on disk only —
never a running execution — and after EVERY edit you must relay the
reload instruction to the user. While the kernel is busy executing a
cell, execute_code queues behind it; the notebook_* tools keep working.

Key patterns:
- Read: shot = Snapshot(node); float(shot[node]) or str(shot[node])
- List children: shot.list(node) returns list[XNode], use child.getName()
- Write single: node["Child"] = value (auto-transactional)
- Write atomic (multi-step):
      def tx(tr):
          tr[a] = v1
          tr[b] = float(tr[a]) * 2   # read-after-write in same tr
      node.iterate_commit(tx)
  ⚠️ The closure may be re-invoked on conflict — keep it idempotent.
     Do external side effects (file I/O, hardware commands) AFTER
     iterate_commit returns, not inside the closure.
- Conditional commit: node.iterate_commit_if(fn) — return True to commit, False to retry
- Bounded retry:      node.iterate_commit_while(fn) — return False to give up
- Drivers: Root()["Drivers"]["DriverName"]
- Scalar entry value: entry["Value"] is XDoubleNode, float(shot[entry["Value"]])

NOTE: This MCP session runs at `Priority.SCRIPTING` — your Tx will
yield to active measurement traffic for ~1 s before claiming privilege.
**SCRIPTING is a one-way trapdoor**: attempting
`kame.setCurrentPriorityMode(...)` to any other level raises
`RuntimeError`.  This is a safety guarantee — your generated code
cannot disrupt the measurement loop regardless of priority calls.
If a measurement-critical operation is needed, ask the user to run
it from their own Jupyter notebook (which is not locked).

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
    # Enable inline matplotlib so plots produce image/png. Pin the format
    # to PNG at a modest dpi with a tight bbox so figures come back crisp
    # but compact (fast over MCP); a bare `plt.plot(...)` in a cell is
    # captured automatically at cell end — no explicit display() needed.
    client.execute("%matplotlib inline")
    client.execute(
        "try:\n"
        "    import matplotlib as _mpl\n"
        "    _mpl.rcParams['figure.dpi'] = 100\n"
        "    _mpl.rcParams['savefig.bbox'] = 'tight'\n"
        "    get_ipython().run_line_magic('config', "
        "\"InlineBackend.figure_formats = {'png'}\")\n"
        "except Exception:\n"
        "    pass\n"
    )
    # MCP-driven Tx are external scripting — should yield to the
    # measurement loop for the first ~1 s of any contention before
    # claiming privilege.  Falls back silently on older KAME builds
    # without the Priority binding.
    client.execute(
        "try:\n"
        "    import kame\n"
        "    kame.setCurrentPriorityMode(kame.Priority.SCRIPTING)\n"
        "except (AttributeError, ImportError):\n"
        "    pass\n"
    )
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
def kame_manual(section: str = "") -> str:
    """Return the KAME user's manual (Markdown), whole-section at a time.

    The manual covers installation, basic UI operation, driver-specific
    settings (DMM, DSO, lock-in amplifier, magnet power supply, temperature
    controller, NMR pulser, ...), NMR measurement workflow, scripting, and
    STM internals.

    Args:
        section: Heading to retrieve, case-insensitive substring match
                 (e.g. "Lock-in", "NMR Pulser", "Script Control").
                 Empty string returns the table of contents.
    """
    path = next((p for p in MANUAL_DOC_PATHS if p.exists()), None)
    if path is None:
        return "Manual not found: " + ", ".join(str(p) for p in MANUAL_DOC_PATHS)
    lines = path.read_text().splitlines()
    headings = []  # (level, title, line_idx)
    in_fence = False
    for i, ln in enumerate(lines):
        if ln.startswith("```"):
            in_fence = not in_fence
            continue
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", ln)
        if m and not in_fence:
            headings.append((len(m.group(1)), m.group(2), i))
    if not section.strip():
        toc = ["Table of contents — call kame_manual(section=<heading>) to read one:"]
        toc += ["  " * (lvl - 1) + "- " + title for lvl, title, _ in headings]
        return "\n".join(toc)
    sec = section.strip().lower()
    idx = next((k for k, h in enumerate(headings) if h[1].lower() == sec), None)
    if idx is None:
        idx = next((k for k, h in enumerate(headings) if sec in h[1].lower()), None)
    if idx is None:
        return f"Section {section!r} not found. Call kame_manual() for the table of contents."
    lvl, _, start = headings[idx]
    end = next((h[2] for h in headings[idx + 1:] if h[0] <= lvl), len(lines))
    body = "\n".join(lines[start:end]).strip()
    # Image references are dead weight over MCP (text-only consumers)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)
    return body


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

    Inside the code, call mcp_checkpoint(progress) at every loop
    iteration. It (a) publishes a progress string readable via
    get_result, and (b) terminates the job (status "stopped") if
    stop_job was called — code that never checkpoints cannot be
    stopped. Prefer short sleeps inside a loop over one long sleep so
    checkpoints are reached promptly.

    Store results in global variables so you can read them afterward
    with execute_code. Example:

        execute_code_async('''
        results = []
        fields = [0, 1, 2, 3, 4, 5]
        for i, field in enumerate(fields):
            mcp_checkpoint(f"{i}/{len(fields)} field={field} T")
            dc_source["Value"] = field
            sleep(1.0)
            snap = Snapshot(root)
            results.append({"field": field, "signal": float(snap[nmr_val])})
        ''')

    Then check status/progress with get_result(job_id), stop early with
    stop_job(job_id), and when done, use execute_code to read and plot
    the results variable.

    CAUTION: The background thread shares globals() with the kernel.
    KAME node operations (Snapshot, Transaction, node[]=) are thread-safe,
    but report progress via mcp_checkpoint rather than reading the job's
    Python variables from other tools while it is still writing to them.
    Wait for "done" or "stopped" status before reading result variables.

    Returns a job_id string for use with get_result() / stop_job().
    """
    job_id = f"_mcp_{int(time.time() * 1000)}"
    wrapper = f"""
import threading as _th, traceback as _tb
_mcp_jobs = globals().setdefault("_mcp_jobs", {{}})
_mcp_tls = globals().setdefault("_mcp_tls", _th.local())
if "_McpStopped" not in globals():
    class _McpStopped(Exception):
        pass
if "mcp_checkpoint" not in globals():
    def mcp_checkpoint(progress=None):
        _job = getattr(_mcp_tls, "job", None)
        if _job is None:
            return
        if progress is not None:
            _job["progress"] = str(progress)
        if _job.get("stop"):
            raise _McpStopped()
_mcp_jobs[{job_id!r}] = {{"status": "running", "progress": ""}}
def _mcp_run():
    _mcp_tls.job = _mcp_jobs[{job_id!r}]
    try:
        exec({code!r}, globals())
        _mcp_jobs[{job_id!r}]["status"] = "done"
    except _McpStopped:
        _mcp_jobs[{job_id!r}]["status"] = "stopped"
    except Exception:
        _mcp_jobs[{job_id!r}]["status"] = "error"
        _mcp_jobs[{job_id!r}]["error"] = _tb.format_exc()
_th.Thread(target=_mcp_run, daemon=True).start()
{job_id!r}
"""
    return _execute_text(wrapper)


@server.tool()
def get_result(job_id: str) -> str:
    """Check the status of an async job started with execute_code_async.

    Args:
        job_id: The job ID returned by execute_code_async.

    Returns JSON with "status" ("running", "done", "stopped", or "error")
    and "progress" (the latest string the job passed to mcp_checkpoint).
    If error, includes "error" with the traceback.
    """
    code = f"""
import json as _json
_json.dumps(_mcp_jobs.get({repr(job_id)}, {{"status": "unknown"}}))
"""
    return _execute_text(code)


@server.tool()
def stop_job(job_id: str) -> str:
    """Request a cooperative stop of an async job.

    Sets the job's stop flag; the job terminates (status "stopped") at
    its next mcp_checkpoint() call. Code that never calls
    mcp_checkpoint cannot be stopped this way.

    Args:
        job_id: The job ID returned by execute_code_async.

    Returns JSON with the job's status at the time of the request.
    """
    code = f"""
import json as _json
_job = globals().get("_mcp_jobs", {{}}).get({job_id!r})
if _job is None:
    _r = {{"status": "unknown"}}
else:
    _job["stop"] = True
    _r = {{"status": _job["status"], "stop_requested": True}}
_json.dumps(_r)
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


# ---------- Jupyter notebook document access (contents API) ----------
# The kernel (ZMQ) knows nothing about the notebook *document*; cells are
# read/edited through the Jupyter server's REST API. KAME generates the
# server token itself and records it (with the workspace dir) in
# ~/.kame_kernel_connection.json, so the server can be located among the
# Jupyter runtime info files.

_nb_server_cache = None


def _conn_info() -> dict:
    if not CONN_INFO_PATH.exists():
        raise RuntimeError(
            "KAME is not running (no ~/.kame_kernel_connection.json). "
            "Start KAME and use Script → Launch Jupyter notebook.")
    with open(CONN_INFO_PATH) as f:
        return json.load(f)


def _match_server_file(runtime_dir: Path, want_token, want_dir):
    """Pick KAME's notebook server among Jupyter runtime info files.

    Match by the token KAME generated, or by the workspace dir; always
    authenticate with the token found in the runtime file itself."""
    candidates = []
    for p in list(runtime_dir.glob("jpserver-*.json")) \
            + list(runtime_dir.glob("nbserver-*.json")):
        try:
            with open(p) as f:
                s = json.load(f)
            mtime = p.stat().st_mtime
        except (OSError, ValueError):
            continue
        root = s.get("root_dir") or s.get("notebook_dir")
        if (want_token and s.get("token") == want_token) \
                or (want_dir and root == want_dir):
            candidates.append((mtime, s))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])[1]


def _notebook_server():
    """(base_url, token) of KAME's Jupyter notebook server."""
    global _nb_server_cache
    if _nb_server_cache is not None:
        return _nb_server_cache
    info = _conn_info()
    from jupyter_core.paths import jupyter_runtime_dir
    s = _match_server_file(Path(jupyter_runtime_dir()),
                           info.get("notebook_token"), info.get("notebook_dir"))
    if s is None:
        raise RuntimeError(
            "KAME's Jupyter notebook server was not found. Is the notebook "
            "running? (Older KAME builds don't record notebook_token — "
            "relaunch the notebook from a current build.)")
    url = s.get("url") or \
        f"http://127.0.0.1:{s.get('port', 8888)}{s.get('base_url', '/')}"
    _nb_server_cache = (url.rstrip("/"), s.get("token", ""))
    return _nb_server_cache


def _nb_api(path: str, method: str = "GET", body=None):
    global _nb_server_cache
    base, token = _notebook_server()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Authorization": f"token {token}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Jupyter API {method} {path}: HTTP {e.code} {e.read()[:300]!r}")
    except urllib.error.URLError as e:
        _nb_server_cache = None  # stale port — rediscover on next call
        raise RuntimeError(f"Jupyter server unreachable ({e}); retry once.")
    return json.loads(raw) if raw else {}


# Watch the kernel's iopub broadcasts on a dedicated ZMQ connection.
# A busy kernel cannot answer execute_code, but iopub still broadcasts
# execute_input/status for cells started from the notebook browser — this
# is the only way to see WHAT is running during a measurement cell.
_activity = {"state": "unknown", "execution_count": None, "code": "",
             "since": None}
_watcher_lock = threading.Lock()
_watcher_started = False


def _start_activity_watcher():
    global _watcher_started
    with _watcher_lock:
        if _watcher_started:
            return
        info = _conn_info()
        client = jupyter_client.BlockingKernelClient()
        client.load_connection_file(info["connection_file"])
        client.start_channels()

        def run():
            while True:
                try:
                    msg = client.get_iopub_msg(timeout=60)
                except queue.Empty:
                    continue
                except Exception:
                    time.sleep(2)
                    continue
                typ, content = msg["msg_type"], msg["content"]
                if typ == "status":
                    _activity["state"] = content.get("execution_state",
                                                     "unknown")
                elif typ == "execute_input":
                    _activity["execution_count"] = content.get(
                        "execution_count")
                    _activity["code"] = content.get("code", "")
                    _activity["since"] = time.time()

        threading.Thread(target=run, daemon=True).start()
        _watcher_started = True


def _cell_source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _default_notebook(path: str) -> str:
    if path.strip():
        return path.strip().lstrip("/")
    sessions = [s for s in _nb_api("/api/sessions")
                if s.get("type", "notebook") == "notebook"]
    if len(sessions) == 1:
        return sessions[0]["path"]
    names = ", ".join(s.get("path", "?") for s in sessions) or "(none)"
    raise RuntimeError(f"Specify the notebook path; open sessions: {names}")


def _kernel_busy() -> bool:
    try:
        return any(k.get("execution_state") == "busy"
                   for k in _nb_api("/api/kernels"))
    except RuntimeError:
        return False


RELOAD_NOTICE = (
    "⚠️ The edit is saved to the .ipynb on disk only. TELL THE USER NOW to "
    "reload the notebook browser tab before touching it — the open tab "
    "still holds the old version, and saving from it would silently "
    "overwrite this edit. The change does NOT affect any execution already "
    "in progress.")


@server.tool()
def notebook_status() -> str:
    """KAME notebook overview: open notebooks, kernel busy/idle, and the
    currently (or last) started cell.

    ALWAYS call this before notebook_edit. If the kernel is busy, the
    shown cell is probably still running: do not edit that cell, and any
    execute_code call would queue behind it until it finishes (KAME's
    sleep() inside the cell does not release the kernel).
    """
    watcher_note = ""
    try:
        _start_activity_watcher()
    except Exception as e:
        watcher_note = f"(cell watcher unavailable: {e})"
    sessions = _nb_api("/api/sessions")
    kernels = {k.get("id"): k for k in _nb_api("/api/kernels")}
    lines = []
    for s in sessions:
        kid = (s.get("kernel") or {}).get("id")
        k = kernels.get(kid, s.get("kernel") or {})
        lines.append(f"notebook: {s.get('path')}  "
                     f"kernel: {k.get('execution_state', '?')}")
    a = dict(_activity)
    if a["execution_count"] is not None:
        ago = f", started {int(time.time() - a['since'])}s ago" \
            if a["since"] else ""
        code = a["code"] if len(a["code"]) <= 2000 \
            else a["code"][:2000] + "\n...[truncated]"
        lines.append(f"last started cell: In[{a['execution_count']}] "
                     f"(kernel now {a['state']}{ago}):")
        lines.append(code)
    else:
        lines.append("(cell watcher sees only cells started after it "
                     "attaches — the busy/idle state above is authoritative)")
    if watcher_note:
        lines.append(watcher_note)
    return "\n".join(lines) if lines else "No notebook sessions."


@server.tool()
def notebook_read(path: str = "", with_outputs: bool = False) -> str:
    """Read a notebook's cells with indices (for notebook_edit).

    Args:
        path: Path relative to the Jupyter workspace (see notebook_status).
              Empty: the single open notebook session.
        with_outputs: include trimmed text outputs.
    """
    path = _default_notebook(path)
    model = _nb_api("/api/contents/" + urllib.parse.quote(path))
    cells = (model.get("content") or {}).get("cells", [])
    out = [f"{path} — {len(cells)} cells"]
    for i, c in enumerate(cells):
        ec = c.get("execution_count")
        out.append(f"--- cell {i} [{c.get('cell_type')}]"
                   + (f" In[{ec}]" if ec else "") + " ---")
        out.append(_cell_source(c))
        if with_outputs:
            for o in c.get("outputs", [])[:5]:
                txt = o.get("text") or (o.get("data") or {}).get(
                    "text/plain") or ""
                txt = "".join(txt) if isinstance(txt, list) else str(txt)
                if txt:
                    out.append("  out: " + (txt[:500] + "...[truncated]"
                                            if len(txt) > 500 else txt))
    return "\n".join(out)


@server.tool()
def notebook_edit(path: str, index: int, source: str = "",
                  mode: str = "replace", cell_type: str = "code") -> str:
    """Replace/insert/delete a notebook cell on disk via the Jupyter API.

    Args:
        path: notebook path ("" = the single open notebook).
        index: cell index from notebook_read. For insert, the new cell is
               placed at this index (-1 = append).
        source: new cell source (replace/insert).
        mode: "replace", "insert", or "delete".
        cell_type: for insert: "code" or "markdown".

    Check notebook_status first; never edit the cell that is currently
    executing. After a successful edit, relay the reload notice in the
    response to the user — this is mandatory, not optional.
    """
    path = _default_notebook(path)
    model = _nb_api("/api/contents/" + urllib.parse.quote(path))
    content = model.get("content") or {}
    cells = content.get("cells", [])
    busy = _kernel_busy()
    if mode == "insert":
        cell = {"cell_type": cell_type, "metadata": {}, "source": source}
        if cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        if any("id" in c for c in cells):
            cell["id"] = uuid.uuid4().hex[:8]
        pos = len(cells) if index == -1 else index
        if not 0 <= pos <= len(cells):
            return f"Bad index {index}; notebook has {len(cells)} cells."
        cells.insert(pos, cell)
        action = f"Inserted {cell_type} cell at index {pos}."
    elif mode in ("replace", "delete"):
        if not 0 <= index < len(cells):
            return f"Bad index {index}; notebook has {len(cells)} cells."
        old_src = _cell_source(cells[index])
        if busy and _activity.get("code") \
                and old_src.strip() == _activity["code"].strip():
            return (f"REFUSED: cell {index} appears to be the one currently "
                    "executing (kernel busy). Wait for it to finish or have "
                    "the user stop it first.")
        if mode == "delete":
            cells.pop(index)
            action = f"Deleted cell {index}."
        else:
            cells[index]["source"] = source
            if cells[index].get("cell_type") == "code":
                cells[index]["outputs"] = []
                cells[index]["execution_count"] = None
            action = f"Replaced source of cell {index}."
    else:
        return f"Unknown mode {mode!r} (use replace/insert/delete)."
    _nb_api("/api/contents/" + urllib.parse.quote(path), method="PUT",
            body={"type": "notebook", "content": content})
    note = ("\nNOTE: the kernel is BUSY — a cell is still executing; this "
            "edit does not affect it.") if busy else ""
    return f"{action} ({path}){note}\n\n{RELOAD_NOTICE}"


def _run_http_with_token(server, host, port, token):
    """Run streamable-http server with Bearer-token middleware.

    Used when the launching process supplies --token; clients must send
    `Authorization: Bearer <token>` on every request. Mitigates the
    "any local user can hit 127.0.0.1 and execute Python in KAME"
    failure mode.

    Falls back to plain server.run() if the underlying Starlette app
    isn't exposed (older mcp versions): the warning is loud so the
    deployer knows to upgrade.
    """
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import Response
        import uvicorn
        app = server.streamable_http_app()
        expected = f"Bearer {token}"

        class _AuthMW(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.headers.get("authorization", "") != expected:
                    return Response("Unauthorized", status_code=401)
                return await call_next(request)

        app.add_middleware(_AuthMW)
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except (AttributeError, ImportError) as e:
        print(
            f"Warning: token auth unavailable ({e}); falling back to "
            f"unauthenticated streamable-http on {host}:{port}.",
            file=sys.stderr)
        server.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="KAME MCP server")
    p.add_argument("--transport", choices=["stdio", "sse", "http"],
                   default="stdio",
                   help="MCP transport (default: stdio)")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address for sse/http (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=0,
                   help="port for sse/http (0 = OS-assigned)")
    p.add_argument("--token", default="",
                   help="bearer token for http transport (optional)")
    args = p.parse_args()

    if args.transport == "stdio":
        server.run(transport="stdio")
    elif args.transport == "sse":
        server.run(transport="sse", host=args.host, port=args.port)
    else:
        # streamable-http is the MCP 1.0+ recommended transport.
        if args.token:
            _run_http_with_token(server, args.host, args.port, args.token)
        else:
            server.run(transport="streamable-http",
                       host=args.host, port=args.port)
