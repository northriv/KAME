# KAME Python API Quick Reference

## MCP tool selection (read first)

If you reach this doc through MCP, the relevant tools are:

- **`execute_code`** — synchronous; the kernel blocks the MCP response
  until the code returns. **Hard timeout ≈ 30 s.** Use only for *quick*
  reads, single transactions, plotting a snapshot.
- **`execute_code_async` + `get_result` + `stop_job`** — runs the code
  in a daemon thread on the kernel; returns a job id immediately.
  **Use this for anything that loops, sleeps, samples over time, or
  polls hardware.** Call `mcp_checkpoint("i/N ...")` at every loop
  iteration: it publishes progress (returned by `get_result`) and is
  the only point where a `stop_job` request can terminate the job.
  KAME STM is thread-safe; read the job's result variables only after
  status is "done" or "stopped".

Anti-pattern (will MCP-timeout):
```python
# Running this with execute_code blocks until 30 s and dies.
for _ in range(60):
    sleep(1.0)
    print(float(Snapshot(node)[node]))
```
Switch to `execute_code_async` + `get_result(job_id)` for the same code.

## Globals (pre-imported from `kame`)

```python
Root()                    # Root measurement node (XMeasure)
Snapshot(node)            # Immutable read view of a subtree
Transaction(node)         # Write context (optimistic, auto-retry)
sleep(sec)                # KAME-aware sleep (use instead of time.sleep)
```

## Node Navigation

```python
root = Root()
drivers = root["Drivers"]        # Child by name (returns XNode or None)
driver = drivers["DMM1"]         # Nested access
child = node[0]                  # Child by index (takes fresh Snapshot)
node.getName()                   # Internal name (str)
node.getLabel()                  # UI label (str)
node.getTypename()               # Type name without 'X' prefix
node.dynamic_cast()              # Downcast to most specific Python type
```

**Important:** `node["name"]` returns `None` if the child doesn't exist.
Always check for `None` before using the result.

## Reading Values (Snapshot)

A Snapshot must be taken from the node or an ancestor that contains it.

```python
shot = Snapshot(node)             # Snapshot of node's subtree

# Access payload — shot[x] only works if x is within shot's subtree
payload = shot[node]              # Node's payload object

# Convert payload to Python types (depends on node type)
float(shot[double_node])          # XDoubleNode -> float
int(shot[int_node])               # XIntNode/XUIntNode -> int
bool(shot[bool_node])             # XBoolNode -> bool
str(node)                         # Any XValueNodeBase -> str (takes own snapshot)

# List children — node must be in the snapshot's subtree
children = shot.list(node)        # list[XNode]
for child in shot.list(node):
    print(child.getName())

shot.size(node)                   # Number of children
len(shot)                         # Children of snapshot root
```

## Writing Values

```python
# Simple assignment (auto-transactional, by child name)
node["ChildName"] = 3.14         # float, int, bool, or str

# Set a value node directly
value_node.set("300.0")           # XValueNodeBase.set(str)

# Explicit transaction
for tr in Transaction(node):
    tr[node] = 42                 # Set value
    break                         # Commit

# Transaction with commit control
tr = Transaction(node)
tr[node["SetPoint"]] = 300.0
tr.commit()                       # Returns final Snapshot

# Touchable node (trigger action)
touchable_node.touch()
```

## Transactional Patterns (atomic multi-step writes)

For any operation that needs **multiple reads/writes to commit
together atomically** (e.g. read A, compute, write B based on A's
value), use the closure-style transaction API:

```python
# Closure-style — recommended for atomic multi-step writes
def step(tr):
    tr[driver_a] = v_a
    tr[driver_b] = v_b
    # Read-after-write within the same tr sees the new value:
    cur = float(tr[driver_a])
    tr[driver_c] = cur * 2.0

shot = node.iterate_commit(step)   # returns the committed Snapshot
```

### Conditional commit (`iterate_commit_if`)

Closure returns `True` to commit, `False` to retry (e.g. when an
intermediate `insert`/`release` failed because the tree shape
changed under us):

```python
def try_insert(tr):
    if not parent.insert(tr, child, True):
        return False     # tree shape changed → retry
    tr[child] = 0
    return True          # commit

parent.iterate_commit_if(try_insert)
```

### Bounded retry (`iterate_commit_while`)

Closure returns `True` to keep retrying, `False` to give up
(no commit happens on give-up):

```python
attempts = [0]
def try_once(tr):
    attempts[0] += 1
    if attempts[0] > 10:
        return False     # give up
    tr[node] += 1
    return True          # continue (but commit happens here regardless on success)

node.iterate_commit_while(try_once)
```

### Retry semantics — IMPORTANT

If another thread commits a conflicting change between snapshot and
commit, the **entire closure body is re-invoked from the start**.
Write closures so they are *idempotent under retry*:

  - ✅ Pure data updates: `tr[x] = value`, `tr[y] = tr[x] * 2`
  - ✅ Read-then-write within the same `tr`
  - ⚠️  Mutating Python-side variables (counters, lists): expect
         multiple increments / appends on retry; use carefully
  - ⚠️  `print()` / logging — fires once per retry; be aware
  - ❌ External side effects inside the closure: file I/O,
         network calls, hardware commands (e.g. driver "send"
         actions).  Do these **after** `iterate_commit` returns
         and you have the committed Snapshot in hand.

### Priority (light vs. measurement-critical Tx)

`kame.setCurrentPriorityMode(kame.Priority.<level>)` sets the current
thread's priority for the **privilege ("oldest-Tx escape")** mechanism.
After waiting longer than the level's threshold, a Tx claims privilege
to force forward progress; a longer threshold = more deferential.

| Level | Threshold | Use case |
|---|---|---|
| `HIGHEST` / `NORMAL` | ~300 µs | Measurement, driver activity |
| `UI_DEFERRABLE` | 50 ms | Interactive UI updates |
| `LOWEST` | 30 ms | Bulk / analysis |
| `SCRIPTING` | **1 s** | **External scripting (MCP / AI / ZMQ)** — yields to *everything* for the first second of contention before escalating; ensures the script command eventually completes without disrupting a live measurement |

The MCP server sets `SCRIPTING` on connection.

**Important: SCRIPTING is a one-way trapdoor.** Once a thread has been
set to `SCRIPTING`, any subsequent `setCurrentPriorityMode(...)` call
raises `RuntimeError`.  This is a safety guarantee: an MCP / AI session
cannot elevate its own priority to disrupt a live measurement loop,
no matter what code the AI generates.

```python
# Allowed: initial entry into SCRIPTING (set by MCP server on connect)
kame.setCurrentPriorityMode(kame.Priority.SCRIPTING)

# Allowed: re-asserting SCRIPTING (silent no-op)
kame.setCurrentPriorityMode(kame.Priority.SCRIPTING)

# Rejected: attempting to escape SCRIPTING → RuntimeError
try:
    kame.setCurrentPriorityMode(kame.Priority.NORMAL)
except RuntimeError as e:
    print(e)  # "Priority::SCRIPTING is sticky and cannot be changed..."
```

Non-MCP Python sessions (a user-launched Jupyter notebook, a Ruby
script via `XScriptingThread`) inherit the kernel's default
(`UI_DEFERRABLE`) and may switch freely among the non-SCRIPTING
levels.  The trapdoor only triggers once SCRIPTING has been set.

If you legitimately need NORMAL priority for measurement-critical
work, **do not** call `setCurrentPriorityMode(SCRIPTING)` first;
work from `UI_DEFERRABLE` (the default) and switch as needed.

### SIGINT / KeyboardInterrupt — IS interruptible

The C++ STM retry loop runs with the **GIL released**, and we
check `PyErr_CheckSignals()` after every closure invocation.
A `KeyboardInterrupt` (e.g. from a Jupyter notebook's "interrupt
kernel" button or Ctrl+C in a script) **propagates out of
`iterate_commit` cleanly**, even if the closure has been retrying
in a livelock:

```python
def conflict_prone(tr):
    tr[node] += 1
try:
    node.iterate_commit(conflict_prone)
except KeyboardInterrupt:
    print("interrupted before commit")
```

Note: GIL release during retry means that **other Python threads
can run between retries**.  Closures that touch Python-side global
state (without proper locking) may observe inconsistent values
across retries.  Stick to data flowing through the `tr` Transaction
for full STM consistency guarantees.

### When to use what

| Situation | API |
|---|---|
| Set a single value | `node["X"] = v` (auto-transactional) |
| Multi-step atomic write | `iterate_commit(closure)` |
| Insert/release that may need retry on tree-shape change | `iterate_commit_if(closure)` |
| Want a retry-cap (give up after N tries) | `iterate_commit_while(closure)` |
| External side effect alongside write | `iterate_commit` for the write, then act on the returned Snapshot |

## Driver Patterns

```python
# List all drivers
drivers = Root()["Drivers"]
shot = Snapshot(drivers)
for d in shot.list(drivers):
    print(d.getName(), d.getTypename())

# Read driver record timestamps
shot = Snapshot(driver)
payload = shot[driver]
payload.time()                    # When phenomenon occurred
payload.timeAwared()              # Acquisition start time (when measurement began)

# Read driver values via its scalar entries (the `Drivers/<name>/<entry>`
# children for DMMs, thermometers, etc.)
#   driver["EntryName"]["Value"] is the XDoubleNode holding the latest
#   reading; take a snapshot covering it and convert to float.
val_node = driver["Voltage"]["Value"]
voltage = float(Snapshot(val_node)[val_node])
# Or iterate through scalar entries owned by this driver:
for e in Snapshot(Root()["ScalarEntries"]).list(Root()["ScalarEntries"]):
    if e.driver() == driver:
        v_node = e["Value"]
        print(e.getName(), float(Snapshot(v_node)[v_node]))

# Access driver child nodes (settings, readings)
shot = Snapshot(driver)
for child in shot.list(driver):
    name = child.getName()
    try:
        val = str(child)          # Works for XValueNodeBase children
        print(f"{name} = {val}")
    except Exception:
        print(name)
```

## Scalar Entries

Each `XScalarEntry` has child nodes: `Value` (XDoubleNode), `StoredValue`,
`Delta`, `Store`. Read the numeric value via `entry["Value"]`.

```python
entries = Root()["ScalarEntries"]
shot = Snapshot(entries)
for e in shot.list(entries):
    val_node = e["Value"]
    print(e.getName(), "=", float(shot[val_node]))
    # e.driver() returns the owning XDriver
```

## Root Children

The root node (`Root()`) has these children:

| Name | Type | Runtime |
|---|---|---|
| `Drivers` | XDriverList | no |
| `Interfaces` | XInterfaceList | yes |
| `ScalarEntries` | XScalarEntryList | yes |
| `Thermometers` | XThermometerList | no |
| `GraphList` | XGraphList | no |
| `ChartList` | XChartList | yes |
| `CalibratedEntries` | XCalibratedEntryList | no |

## Creating and releasing drivers

```python
drivers = Root()["Drivers"]
dc = drivers.dynamic_cast()         # downcast to XDriverList

# Discover registered driver types — don't guess. Returns model-
# specific keys (the strings shown in KAME's "Add driver" dialog),
# which depend on which modules the running KAME has loaded.
dc.typenames()
# e.g. ['TestDriver', 'DMMKE2700', 'DCSrcKE2400', 'OxfordITC503', ...]
dc.typelabels()                     # parallel list of human-readable labels

# Create — pass an exact key from typenames(); returns None for an
# unregistered key, so verify first.
if "TestDriver" not in dc.typenames():
    raise RuntimeError("TestDriver module not loaded")
driver = dc.createByTypename("TestDriver", "Test1")

# After creation, the driver's child structure is type-specific:
# inspect children with Snapshot(driver).list(driver). For interface-
# backed drivers (most C++ ones), an "Interface" subtree exists with
# at least a Boolean "Control" child to open/close the connection.

# Release: also removes the driver's ChartList / ScalarEntries
dc.release(driver)
```

## Common Recipes

### List all drivers and their types
```python
drivers = Root()["Drivers"]
shot = Snapshot(drivers)
for d in shot.list(drivers):
    print(d.getName(), d.getTypename())
```

### Read a value node by path
```python
# Path elements after the driver are driver-specific child names —
# inspect with Snapshot(driver).list(driver) to discover them.
# Example: a Lakeshore-style temperature controller exposes per-channel
# readings as XDoubleNode children named "Ch.A", "Ch.B", ...
node = Root()["Drivers"]["TempControl"]["Ch.B"]
shot = Snapshot(node)
print(float(shot[node]))
```

### Dump all children of a driver
```python
driver = Root()["Drivers"]["DMM1"]
shot = Snapshot(driver)
for child in shot.list(driver):
    try:
        print(child.getName(), "=", str(child))
    except Exception:
        print(child.getName())
```

### Set a parameter and wait
```python
source = Root()["Drivers"]["DCSource"]
source["Value"] = "1.0"
sleep(2)
```

### Sweep and measure
```python
results = []
source = Root()["Drivers"]["DCSource"]
dmm = Root()["Drivers"]["DMM1"]
for v in [0.1, 0.2, 0.5, 1.0]:
    source["Value"] = str(v)
    sleep(1)
    shot = Snapshot(dmm)
    reading = float(shot[dmm].value(0))
    results.append((v, reading))
    print(f"V={v}, Reading={reading}")
```

### Read all scalar entry values
```python
entries = Root()["ScalarEntries"]
shot = Snapshot(entries)
for e in shot.list(entries):
    val = float(shot[e["Value"]])
    print(f"{e.getName()} ({e.driver().getName()}): {val}")
```

## Graph Data Access

### XWaveNGraph (1D plots)

```python
shot = Snapshot(wave_graph)
p = shot[wave_graph]
p.colCount()                      # Number of columns
p.rowCount()                      # Number of rows
p.labels()                        # Column labels (list[str])
p.getColumn(0)                    # Column data (list[float])

# Plot with matplotlib
import matplotlib.pyplot as plt
x = p.getColumn(0)
y = p.getColumn(1)
labels = p.labels()
plt.plot(x, y)
plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.gcf()  # ← last expression returns the Figure; MCP `execute_code`
           #    captures it as inline image content. Don't `plt.show()`.
```

### X2DImagePlot (2D images) — display only

`to_png()` returns the *rendered display image*: gamma-encoded, possibly
colormapped/scaled. Use it **only to show a picture to the human**.

**Anti-pattern: never decode `to_png()` bytes (plt.imread etc.) for
analysis.** PNG pixel values are not raw counts; thresholds, averages and
masks derived from them are quantitatively wrong. For ROI statistics,
masks and raw frames use **2D Math Tools** (below) — their functors
receive the true uint32 count matrix before any display rendering.

To reach `to_png()`, navigate: X2DImage → Graph child → Plots → ImagePlot.

```python
# Full path example for a camera LiveImage:
# Root()["Drivers"]["JAI"]["LiveImage"]        → X2DImage (XGraphNToolBox)
#   ["LiveImage"]                               → XGraph
#     ["Plots"]["ImagePlot"]                    → X2DImagePlot (has to_png())

imgplot = Root()["Drivers"]["JAI"]["LiveImage"]["LiveImage"]["Plots"]["ImagePlot"]
shot = Snapshot(imgplot)
png_bytes = shot[imgplot].to_png()  # PNG-encoded image bytes
                                    # (or None if no image is available yet)

# Image dimensions (pixels)
w = shot[imgplot].imageWidth()
h = shot[imgplot].imageHeight()

# Display via IPython (returned as image through MCP)
from IPython.display import display, Image
display(Image(data=png_bytes, format='png'))
```

## 2D Math Tools

X2DImage nodes have `Graph2DMathToolList` children (one per channel, e.g. `CH0`, `CH1`).
Use `createByTypename()` to add math tools; `release()` to remove them.

### Available types

| Type name | Description |
|---|---|
| `Graph2DMathToolAverage` | Average pixel value in ROI |
| `Graph2DMathToolSum` | Sum of pixel values in ROI |

### Creating and configuring

ROI coordinates use inclusive pixel endpoints: `FirstX/FirstY` (top-left) and
`LastX/LastY` (bottom-right, inclusive). The ROI width in pixels is `LastX - FirstX + 1`.

```python
ch0 = Root()["Drivers"]["JAI"]["LiveImage"]["CH0"]
dc = ch0.dynamic_cast()  # needed for createByTypename

# Create — returns the new tool node
tool = dc.createByTypename("Graph2DMathToolAverage", "MyAvg")

# Set ROI in pixel coordinates (inclusive endpoints)
tool["FirstX"] = "100"
tool["FirstY"] = "100"
tool["LastX"] = "500"
tool["LastY"] = "500"

# Read the scalar entry value
val_node = tool["CH0-MyAvg"]["Value"]  # name = "{channel}-{toolname}"
shot = Snapshot(val_node)
float(shot[val_node])
```

### Mask types

Each tool has a `MaskType` node: Rectangle (default), Ellipse, Arbitrary.

For Arbitrary masks, use `setArbitraryMask()` which atomically sets MaskType
and the mask bitmap in one transaction:

```python
# Mask: uint8 list, 1=included, 0=excluded.
# Dimensions must be (LastX - FirstX + 1) * (LastY - FirstY + 1).
tool.dynamic_cast().setArbitraryMask(mask_bytes)

# Read back mask
shot = Snapshot(tool)
mask = shot[tool].mask()  # list[uint8]
```

**Important:** Clamp ROI to actual image dimensions to avoid display offset.
Use `shot[imgplot].imageWidth()` and `shot[imgplot].imageHeight()` to get bounds.

### Python functor tools — direct raw-frame access

Math-tool functors are the **only quantitative pixel access**: on every
frame the tool receives the raw uint32 count matrix of its ROI
(zero-copy, before any display rendering or gamma). Subclass
`XPythonGraph2DMathTool`, register with `exportClass`, then create it via
`createByTypename` exactly like the C++ tools.

The functor takes 7 arguments:
`f(matrix, width, stride, numlines, coefficient, offset, mask)`
- `matrix` — `(numlines, width)` uint32 array, raw ROI counts
- `coefficient`, `offset` — physical calibration: value = coefficient*count + offset
- `mask` — `(numlines, width)` uint8, 1 = included (all-ones for Rectangle)

Return one value per entry as `np.array` (entry names are the
semicolon-separated fields of the exportClass label).

**CAUTION:** `matrix` and `mask` are views of transient C++ buffers,
valid only during the call — `np.array(..., copy=True)` anything you keep.

```python
import numpy as np
latest = {}

class FrameGrabber(XPythonGraph2DMathTool):
    def __init__(self, name, runtime, tr, entries, driver, plot, parent, entryname):
        XPythonGraph2DMathTool.__init__(self, name, runtime, tr, entries,
            driver, plot, parent, entryname)  # super().__init__ cannot be used
        self.setFunctor(self.func)
    def func(self, matrix, width, stride, numlines, coefficient, offset, mask):
        frame = coefficient * np.array(matrix, dtype=np.float64, copy=True) + offset
        latest["frame"] = frame
        latest["mask"] = np.array(mask, copy=True)
        return np.array([float(frame.mean())])

XPythonGraph2DMathTool.exportClass("FrameGrabber", FrameGrabber, "Mean")

ch0 = Root()["Drivers"]["JAI"]["LiveImage"]["CH0"]
tool = ch0.dynamic_cast().createByTypename("FrameGrabber", "Grab")
# Set the ROI to the full frame, wait for the next frame, then read
# latest["frame"] (true counts) from a later execute_code call.
```

For simple ROI statistics (e.g. ODMR signal/reference regions) you do not
need a functor at all: create two `Graph2DMathToolAverage` tools and read
their scalar entries at each sweep point. No PNG is involved anywhere.

### Creating a mask from image analysis

Helper that builds a clean bright-area mask from a **raw frame** captured
by a functor tool (see above), with morphological cleanup and connected
component filtering. Requires `scipy`.

```python
import numpy as np
from scipy import ndimage

def make_bright_mask(frame,
                     method="median",    # "median", "otsu", or "percentile"
                     threshold_frac=0.5, # for median: fraction between median and p99
                     percentile=90,      # for percentile method
                     min_area=500,       # discard components smaller than this
                     morph_close=5,      # morphological closing kernel size (0=skip)
                     morph_open=3):      # morphological opening kernel size (0=skip)
    """frame: 2D array of raw counts (full image), e.g. latest["frame"]
    captured by the FrameGrabber recipe with the ROI covering the frame.
    Returns (mask_bytes, fx, fy, lx, ly, img_w, img_h); mask_bytes is a
    flat list of uint8 with dimensions (lx-fx+1) * (ly-fy+1)."""
    plot_area = np.asarray(frame, dtype=float)
    img_h, img_w = plot_area.shape

    # Threshold
    if method == "otsu":
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(plot_area)
    elif method == "percentile":
        thresh = np.percentile(plot_area, percentile)
    else:  # median
        med = np.median(plot_area)
        p99 = np.percentile(plot_area, 99)
        thresh = med + threshold_frac * (p99 - med)

    binary = (plot_area > thresh).astype(np.uint8)

    # Morphological cleanup
    if morph_close > 0:
        binary = ndimage.binary_closing(binary, structure=np.ones((morph_close, morph_close))).astype(np.uint8)
    if morph_open > 0:
        binary = ndimage.binary_opening(binary, structure=np.ones((morph_open, morph_open))).astype(np.uint8)

    # Keep only the largest connected component
    labeled, n = ndimage.label(binary)
    if n > 1:
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        largest = np.argmax(sizes) + 1
        binary = (labeled == largest).astype(np.uint8)
    # Remove small components
    if min_area > 0:
        labeled, n = ndimage.label(binary)
        for i in range(1, n + 1):
            if ndimage.sum(binary, labeled, i) < min_area:
                binary[labeled == i] = 0

    rows, cols = np.where(binary)
    if len(rows) == 0:
        return [], 0, 0, 0, 0, img_w, img_h
    fx, lx = int(cols.min()), int(cols.max())
    fy, ly = int(rows.min()), int(rows.max())
    mask = binary[fy:ly + 1, fx:lx + 1]   # already in image pixel coordinates
    return mask.flatten().tolist(), fx, fy, lx, ly, img_w, img_h
```

Usage:
```python
mask_bytes, fx, fy, lx, ly, img_w, img_h = make_bright_mask(latest["frame"])

ch0 = Root()["Drivers"]["JAI"]["LiveImage"]["CH0"]
dc = ch0.dynamic_cast()
tool = dc.createByTypename("Graph2DMathToolAverage", "BrightAvg")
tool["FirstX"] = str(fx)
tool["FirstY"] = str(fy)
tool["LastX"] = str(lx)
tool["LastY"] = str(ly)
tool.dynamic_cast().setArbitraryMask(mask_bytes)
```

### Removing a tool

```python
ch0 = Root()["Drivers"]["JAI"]["LiveImage"]["CH0"]
dc = ch0.dynamic_cast()
shot = Snapshot(ch0)
children = shot.list(ch0)
dc.release(children[0])  # release by reference from list
```
