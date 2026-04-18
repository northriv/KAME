# KAME: AI-Assisted Automation Program for Physical Property Measurements

[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2%2B-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)
[![GitHub](https://img.shields.io/badge/GitHub-northriv%2FKAME-181717?logo=github)](https://github.com/northriv/KAME)
[![Version](https://img.shields.io/badge/version-8.1-green)]()

KAME is an open-source, multi-threaded program for automated physical property measurements,
developed at [Kitagawa Laboratory, ISSP, University of Tokyo](https://kitag.issp.u-tokyo.ac.jp/).
It is particularly suited to NMR and ODMR experiments, and supports AI-assisted measurement
orchestration across compatible instruments.

**License:** GPL v2 or later (prior to 8.0: LGPL v2 or later)
**Authors:** Kentaro Kitagawa, Shota Suetsugu
**Platforms:** macOS, Windows (64-bit); Linux support discontinued
**Manual:** [日本語](https://kitag.issp.u-tokyo.ac.jp/%e8%87%aa%e5%8b%95%e5%8c%96%e5%af%be%e5%bf%9c%e6%b8%ac%e5%ae%9a%e3%83%97%e3%83%ad%e3%82%b0%e3%83%a9%e3%83%a0kame/) · [English](https://kitag.issp.u-tokyo.ac.jp/web/kame/kame-7-en.pdf)

![KAME screenshot](https://kitag.issp.u-tokyo.ac.jp/wordpress/wp-content/uploads/2025/01/dd21dff192ba7bde3beb0830a80d886c-930x620.png)

---

## Features

- Transactional, lock-free node/data model (Software Transactional Memory)
- Python (+Jupyter notebook) and Ruby scripting — nearly full control from scripts
- **AI-assisted experiment automation via [MCP](https://modelcontextprotocol.io/)** — Claude and other AI assistants can read instruments, control parameters, and run measurement sequences through natural language
- OpenGL-based 2-D / 1-D graph display; arbitrary scalar combinations (T, V, …)
- Real-time NMR relaxation fitting (T1, T2, Tst.e.), Inverse Laplace Transform
- Fourier step-sum spectrum measurement with field / frequency sweeping
- Complete data logging with post-measurement re-analysis
- Save / restore full measurement config to `.kam` files
- Modular driver plug-in architecture; Python drivers redefinable at runtime
- Calibration curves (cspline, Chebyshev, polynomial) for resistance thermometers and generic sensors; calibrated entries feed into graphs, charts, and data recording like any native scalar

### Released versions/Binaries
Source: [kame-8.1.zip](https://kitag.issp.u-tokyo.ac.jp/web/kame/src/kame-8.1.zip) (2MB, Apr. 14, 2026).
[All other source archives](https://kitag.issp.u-tokyo.ac.jp/web/kame/src).
Windows 64-bit binaries: [8.1](https://kitag.issp.u-tokyo.ac.jp/web/kame/src/kame-win32-llvm64-8.1.zip). At least Qt is additionally needed, follow instructions below to install.

### Supported instruments

| Category | Models |
|---|---|
| **Oscilloscopes (DSO)** | Tektronix TDS, Lecroy/Teledyne/Iwatsu, Thamway PROT3 streaming DSO, Thamway DV14U25 A/D board, NI-DAQmx as DSO, Digilent WaveForms AIN |
| **Signal generators** | Kenwood SG7130/7200, HP/Agilent 8643/8644/8648/8664/8665, Keysight/Agilent E44xB SCPI, Rohde-Schwarz SML01/02/03/SMV03, DSTech DPL-3.2XGF, LibreVNA SG SCPI |
| **Function / pulse generators** | NF WAVE-FACTORY, LXI 3390 arbitrary function generator |
| **Network analysers** | HP/Agilent 8711/8712/8713/8714, Agilent E5061/E5062, Copper Mountain TR1300/1504/4530, DG8SAQ VNWA3E, LibreVNA SCPI, Thamway T300-1049A impedance analyser |
| **Lock-in amplifiers / bridges** | Stanford SR830, NF LI5640, Signal Recovery 7265, LakeShore M81-SSM, Agilent/HP 4284A LCR meter, Andeen-Hagerling 2500A capacitance bridge |
| **DC sources** | Yokogawa 7651, Advantest TR6142/R6142/R6144, MICROTASK/Leiden triple current source, Optotune ICC4C-2000 |
| **Multimeters / picoammeters** | Keithley 2000/2001, 2182 nanovolt meter, 2700+7700, 6482 picoammeter; Agilent 34420A, 3458A, 3478A; Sanwa PC500/5000 |
| **Temperature controllers** | Cryocon M32/M62, LakeShore 218/340/350/370/372 (1ch, 8ch, 16ch scanner), Picowatt AVS-47, Oxford ITC-503, Neocera LTC-21, Scientific Instruments 9302/9304/9308, LinearResearch LR-700, OMRON E5\*C Modbus |
| **Magnet power supplies** | Oxford PS-120, Oxford IPS-120, Cryogenic SMS10/30/120C |
| **NMR pulsers** | Thamway N210-1026 PG32U40 (USB), PG027QAM (USB), N210-1026S/T (GPIB/TCP); NI-DAQ analog+digital output, digital output only, M+S Series; handmade H8, handmade SH2 |
| **NMR / RF measurement** | Thamway PROT NMR (USB/TCP), NMR FID/echo analyser, T1/T2 relaxation, field-swept spectrum, frequency-swept spectrum, NMR built-in network analyser, NMR LC autotuner |
| **Cameras / imaging** | IEEE 1394 IIDC, Euresys eGrabber (CoaXPress), Euresys Grablink (CameraLink), Hamamatsu via Grablink, JAI via Grablink, OceanOptics/Insight USB/HR2000+/4000 spectrometer |
| **Laser modules** | Coherent Stingray, Newport/ILX LDX-3200, Newport/ILX LDC-3700(C) |
| **ODMR** | Frequency-swept spectrum, FM peak tracker, 2-D image analysis, filter wheel (STM-driven) |
| **Motors / positioners** | OrientalMotor FLEX CRK, CVD2B, CVD5B, FLEX AR/DG2, EMP401; SigmaOptics PAMC-104 piezo-assisted; Micro CAM z/x/φ; Two-axis rotator |
| **Flow controllers** | Fujikin FCST1000 series |
| **Level meters** | Oxford ILM helium level meter, Cryomagnetics LM-500 |
| **Vacuum gauges** | Pfeiffer TPG361/362 |
| **Pump controllers** | Pfeiffer TC110 turbopump controller |
| **Counters** | Mutoh Digital Counter NPS |
| **Quantum Design PPMS** | PPMS low-level interface |
| **NI DAQmx** | Pulser (AO+DO, DO-only, M+S Series), DSO |
| **Resistance measurement** | Four-terminal with polarity switching; Python-based 4-terminal (simple and multi-current variants) |
| **Monte Carlo simulation** | Monte Carlo driver |

---

## What's New in 8.0

- **MCP server for AI-assisted experiment automation** — built-in [Model Context Protocol](https://modelcontextprotocol.io/) server lets AI assistants (Claude Code, Claude Desktop, etc.) execute Python code in the running KAME process, read instrument values, and control measurements through natural language. Matplotlib plots are returned inline. Long-running experiments (sweeps, scans) run asynchronously. To our knowledge, this is the first measurement software to integrate an MCP server.
- **Calibrated scalar entries** — `XCalibratedEntry` applies a calibration curve to any scalar entry; the result appears in graphs, charts, and data recording like a native scalar.
- **Usermode NI USB-GPIB on Apple Silicon** — the embedded userspace linux-gpib port now works reliably on macOS ARM64 without any kernel module.
- **Window cascade placement** — instrument windows are automatically arranged on show.
- **Comprehensive bug audit** — 20 bug fixes across 12 source files (GIL safety, buffer bounds, null-pointer guards, logic errors).
- **Arbitrary mask support for 2D math tools** — ROI math tools (Average, Sum) now support arbitrary binary masks in addition to Rectangle and Ellipse shapes. Masks can be set programmatically from Python via `setArbitraryMask()`. Highlighted masks are rendered as GPU textures.
- **Math tool API cleanup** — ROI endpoint naming changed from `Begin/End` to `First/Last` (inclusive endpoints, avoids STL naming confusion). Added `imageWidth()`/`imageHeight()` to `X2DImagePlot` for Python access. Old `.kam` files with `Begin/End` names load transparently via compatibility aliases.

---

## Architecture

### Driver / Plug-in Architecture

Instrument drivers are **shared libraries** under `modules/` loaded at runtime via `ltdl`.
Each driver subclasses `XDriver` (`kame/driver/driver.h`), which carries a timestamped
`Payload` (`time()` = phenomenon time, `timeAwared()` = acquisition start time) and emits
`onRecord` / `onVisualization` signals.

Hardware communication is abstracted in `modules/charinterface/` (serial, TCP, GPIB, USB).
Drivers can also be subclassed in Python via `XPythonDriver` (`kame/driver/pythondriver.h`).

Scalar values extracted from driver records are represented as `XScalarEntry` objects
(`kame/analyzer/`). A derived `XCalibratedEntry` applies any registered calibration curve
to an existing entry, and the result appears in graphs, charts, and data recording
exactly like a native scalar. Calibration curves (`kame/thermometer/`) include cubic
spline (`XApproxThermometer`, `XGenericCalibration`), Chebyshev polynomial (`XLakeShore`),
and polynomial (`XScientificInstruments`) types. `XGenericCalibration` supports
user-configured labels and units, making it applicable to any sensor, not just thermometers.

#### Usermode NI USB-GPIB

`modules/charinterface/usermode-linux-gpib/` contains a userspace port of the NI USB-GPIB
kernel driver from linux-gpib 4.3.6. The upstream `ni_usb_gpib.c` is minimally patched
(Linux-only headers guarded with `#ifdef __KERNEL__`); a compatibility header
(`osx_compat.h` / `win_compat.h`) replaces every Linux kernel API — `kmalloc`, spinlocks,
wait queues, USB URBs — with POSIX/libusb or Win32 equivalents.

The result is a standalone executable that speaks to NI USB-B, USB-HS, USB-HS+, KUSB-488A,
and MC USB-488 adapters on macOS, Linux, and Windows without installing a kernel module or
any proprietary driver. On macOS this is the only viable path for USB-GPIB on Apple Silicon.

### Python Integration

*This section was written by Claude (Anthropic) based on analysis of the source code.*

Python access is provided via [pybind11](https://pybind11.readthedocs.io/). The embedded
interpreter runs in its own OS thread; the Qt main thread and the Python thread communicate
through the Talker/Listener signal mechanism.

**Accessing the node tree from Python:**

```python
root = Root()                      # root of the instrument node tree

# Read a value (Snapshot)
shot = Snapshot(root)
print(shot[root])                  # payload of the root node

# Navigate children
tempcontrol = root["tempcontrol"]  # by name
print(float(tempcontrol["temp"]))  # XDoubleNode coerces to float

# Write a value (Transaction)
for tr in Transaction(tempcontrol["setpoint"]):
    tr[tempcontrol["setpoint"]] = 4.2   # retry loop, just like C++
```

**Writing instrument drivers in Python:**

Any C++ driver base class can be subclassed in Python via `XPythonDriver<T>`.
The subclass is registered at runtime with `exportClass()` and instantiated by the
framework exactly like a compiled driver. This enables rapid prototyping of new
instrument interfaces without recompiling KAME.

```python
class MyDriver(kame.XPythonCharDeviceDriverWithThread):
    def analyzeRaw(self, reader, payload):
        payload.local()["value"] = float(reader.pop_string())
    def visualize(self, shot):
        ...
MyDriver.exportClass("MyDriver", MyDriver, "My Instrument")
```

The driver's `Payload.local()` dict is deep-copied per transaction, giving Python
state the same snapshot-isolation semantics as C++ Payload fields.

**Jupyter notebook support:**

KAME optionally embeds an IPython kernel. When IPython is available, a Jupyter client
can connect to the running process for interactive exploration and live plotting
alongside the native KAME UI. The kernel integrates with the asyncio event loop via
a custom ipykernel integration (`loop_kamepysupport`).

**AI-assisted experiment automation (MCP):**

KAME includes an [MCP](https://modelcontextprotocol.io/) (Model Context Protocol) server
that lets AI assistants such as Claude execute Python code directly in the running KAME
interpreter. The MCP server connects to the embedded IPython kernel, giving the AI full
access to `Root()`, `Snapshot()`, `Transaction()`, and all loaded drivers — the same
environment available in Jupyter notebooks.

This enables scenarios like:
- Conversational experiment control ("sweep temperature from 100 K to 300 K and record resistance")
- Automated data collection with adaptive logic
- Real-time monitoring and analysis

See [MCP setup](#mcp-setup-ai-assisted-experiment-automation) below for configuration.

**Threading notes:**

- Long-running C++ calls release the GIL (`gil_scoped_release`) so the Python thread
  does not block Qt.
- Any Qt UI operation (loading `.ui` files, showing forms) must be dispatched to the
  main thread via `kame.kame_mainthread(closure)`.
- Payload garbage collection uses a deferred deque + mutex to avoid holding the
  GIL during snapshot cleanup (GIL-enabled builds only); Python 3.13 free-threading
  (`Py_GIL_DISABLED`) is also supported.

### Serialization (`.kam` files)

A `.kam` file is a Ruby script generated by `XRubyWriter` and re-executed on load.
Nodes marked `runtime=true` are written as comments and not restored.
`XListNode` children are recreated via `createByTypename()`; the typename must match
the key registered in `XTypeHolder`.

### Software Transactional Memory (STM)

KAME's core data model is a lock-free, snapshot-based STM (`kame/transaction.h`).
All instrument data lives in a tree of `Node<XN>` objects; reads and writes are
expressed as **snapshots** and **transactions** rather than locks.

```
Node<XN>
 └─ Linkage  ──atomic_shared_ptr──▶  PacketWrapper
                                          └─ Packet
                                              ├─ Payload   (user data)
                                              └─ PacketList (child packets)
```

**Reading — O(1) snapshot:**

```cpp
Snapshot<NodeA> shot(node);         // atomic load, no lock
double x = shot[node].m_x;
```

**Writing — optimistic transaction with automatic retry:**

```cpp
node.iterate_commit([](Transaction<NodeA> &tr) {
    tr[node].m_x += 1;             // copy-on-write on first access
});                                 // retried automatically on conflict
```

**How commits work:**

1. `Transaction` saves `m_oldpacket` at construction.
2. `operator[]` clones the payload (copy-on-write) on first write, stamping it with a unique serial.
3. `commit()` does a single CAS on `Linkage`; if `packet != m_oldpacket` a conflict is detected and the transaction retries.
4. Listeners receive deferred events only after a successful commit — no intermediate states are visible.

#### Lock-free atomic shared pointer

The O(1) snapshot reads and CAS-based commits above require a shared pointer that is itself lock-free. `atomic_shared_ptr` (in `kame/atomic_smart_ptr.h`, introduced in January 2006 as part of the 2.0-beta3 rewrite) provides this. It is a custom implementation of what C++20 calls `std::atomic<shared_ptr>`.

The core technique embeds a small **local reference counter** in the low bits of the pointer to the reference-control block — bits guaranteed zero by allocator alignment. `acquire_tag_ref_()` atomically increments this local counter via CAS to "pin" the pointer for reading; `release_tag_ref_()` decrements it. Between these two calls, even if another thread swaps the pointer, the object cannot be freed because the local count is non-zero. A separate **global reference counter** in the control block tracks long-lived ownership (copies held across scopes). Setters transfer any outstanding local count to the global counter before swapping, so `release_tag_ref_()` can fall back to decrementing the global counter if the pointer changed.

For types that inherit `atomic_countable` (notably `Payload`), the global reference counter is stored inside the object itself (**intrusive counting**), eliminating a separate heap allocation per shared-pointer instance. Non-intrusive types get an external control block (`atomic_shared_ptr_gref_`).

**Comparison with standard-library implementations (as of late 2024):**

| Implementation | Technique | Lock-free? |
|---|---|---|
| libstdc++ (GCC) | Spinlock on internal table | No — vulnerable to priority inversion |
| MSVC | Lock bit + `WaitOnAddress` | No — blocking under contention |
| libc++ (Clang) | Not yet implemented | N/A |
| KAME (2006–) | Tagged-pointer CAS | Yes — lock-free reads and writes |

On modern compilers (GCC 5.1+, Clang, MSVC), the CAS primitives delegate to `std::atomic` (`atomic_prv_std.h`). Hand-written assembly fallbacks for x86, PowerPC, and ARM remain in the tree for older toolchains.

**Multi-node consistency** is achieved through a *bundling* protocol: a parent packet absorbs child packets via multi-phase CAS protocol, making the entire subtree consistent under a single atomic pointer. A `m_missing` flag marks packets with stale children, driving re-bundling on demand.

**Collision backoff:** `Linkage::negotiate()` uses a `m_transaction_started_time` timestamp to impose a proportional wait on detected collisions, preventing live-lock under high write contention.

`iterate_commit_while(lambda)` lets the caller abort the retry loop (return `false` from the lambda to stop), enabling conditional transactions.

> **Caution:** Taking a nested `Snapshot` inside a transaction can trigger bundling, which may cause the transaction's CAS to always fail. This is not a data corruption issue but a liveness issue — the transaction retries indefinitely. This occurs when the `Snapshot` target is an ancestor of the transaction target, or when hard links exist (a child with two parents) and a `Snapshot` on one parent's tree interferes with the other. Use `tr[*node]` instead of a nested `Snapshot` in these situations.

#### Comparison with other STM designs

*The following comparison was written by Claude (Anthropic) based on analysis of the source code.*

Most widely-used STMs (GHC/Haskell `TVar`, Clojure `Ref`/`dosync`, ScalaSTM) are **flat**: the unit of transaction is a set of independent transactional variables. KAME's STM is instead **tree-structured** — the entire instrument node tree is the shared state, and snapshots are always subtree-consistent. This difference drives several design choices:

| Aspect | Flat STMs (Haskell, Clojure, ScalaSTM) | KAME STM |
|---|---|---|
| Conflict granularity | Per-variable | Per-packet (subtree root) |
| Read model | `readTVar` / `deref` inside transaction | `Snapshot` (outside) or `tr[*node]` (inside) |
| Consistency scope | Variables listed explicitly | Entire subtree, guaranteed by bundling |
| Commit log | Redo log or write set | Copy-on-write + CAS on single `Linkage` |
| Retry primitive | `retry` / `orElse` (Haskell) | `iterate_commit` / `iterate_commit_while` |
| Blocking | `retry` suspends on read-set change | No blocking; backoff via timestamp |
| Memory management | GC | Lock-free `atomic_shared_ptr` (ref-counted) |
| Hard real-time suitability | Limited (GC pauses) | Good (no GC, bounded CAS retries) |

**Compared to Hardware Transactional Memory (Intel TSX/RTM):** HTM aborts on cache-line conflicts regardless of logical independence, and has strict capacity limits. KAME's STM aborts only on semantic conflicts (packet identity change), tolerates large read sets, and degrades gracefully to software backoff rather than falling back to a global lock.

**Compared to TinySTM / NOrec (C libraries):** These use a global version clock and per-object version stamps with a full read/write log per transaction. KAME avoids the read log entirely — a `Snapshot` is just an immutable pointer, so reads outside a transaction are truly zero-overhead. The trade-off is that KAME's write path must clone the payload upfront (copy-on-write), whereas log-based STMs defer that cost to commit time.

**What makes KAME's design distinctive** is the *bundling* protocol: rather than tracking which variables a transaction touched, it tracks whether the packet at the subtree root has been replaced since the transaction started. This is efficient for KAME's access pattern (many readers of a stable tree, infrequent writes from acquisition threads) but would be coarser than necessary for workloads with many independent fine-grained variables.

**Why STM?** Laboratory software must acquire data on tight hardware timings while
simultaneously updating a UI and running user scripts — all from different threads.
Traditional mutex-based designs either serialize too aggressively (dropping samples)
or require intricate lock ordering that is error-prone to extend. The STM approach
offers three concrete benefits for this domain:

- **Deadlock-free by design.** No locks are held across hardware I/O or UI redraws.
  A slow UI thread can never stall a fast acquisition thread.
- **Consistent multi-instrument views.** A `Snapshot` of any subtree is always
  internally consistent — the UI always sees a coherent set of readings even when
  multiple drivers update simultaneously.
- **Safe scripting from Python/Ruby.** Scripts read and write the node tree through
  the same transaction API as C++ code, so user scripts cannot corrupt instrument
  state regardless of when they run.

#### Formal verification (TLA+)

The STM protocol is formally specified and model-checked with TLA+ / TLC:

- **`atomic_shared_ptr`:** tagged-pointer CAS protocol with local/global reference counting ([spec](tests/tlaplus/atomic_shared_ptr.tla))
- **`BundleUnbundle`:** subtree bundling/unbundling with modular serial arithmetic ([spec](tests/tlaplus/BundleUnbundle.tla))

Slide decks: [Layer 1 — atomic_shared_ptr](https://northriv.github.io/KAME/tests/tlaplus/doc/slides_layer1_en.html) ([JA](https://northriv.github.io/KAME/tests/tlaplus/doc_ja/slides_layer1.html)), [Layer 2 — Bundle/Unbundle + Commit](https://northriv.github.io/KAME/tests/tlaplus/doc/slides_layer2_en.html) ([JA](https://northriv.github.io/KAME/tests/tlaplus/doc_ja/slides_layer2.html))

C11 translations of each layer are verified with [GenMC](https://github.com/MPI-SWS/genmc) under the RC11 memory model: TLA+-derived tests (`tests/tlaplus/test_*.c`) and C++-derived protocol tests (`tests/cds_atomic_shared_ptr/`).

---

## Dependencies

| Library | Notes |
|---|---|
| **Qt** ≥ 5.7 or Qt 6 | Qt 5 compatibility module required for Qt 6 |
| **Ruby** | scripting |
| **pybind11** | Python scripting |
| **GSL** | |
| **FFTW 3** | |
| **Eigen 3** | |
| LAPACK / ATLAS / BLAS *(optional)* | |
| **libtool-ltdl** | runtime plug-in loading |
| **zlib** | |
| **libusb** | USB instrument interfaces |
| linux-gpib or NI 488.2 *(optional)* | GPIB interfaces |
| NI DAQmx *(optional)* | NI data-acquisition hardware |

A C++11-capable compiler is required (the build uses `CONFIG += c++11` via qmake).

Optional: IPython / Jupyter notebook, linux-gpib or NI 488.2, NI DAQmx, libdc1394 (macOS cameras).

---

## Building

### macOS

> Open `kame.pro` in **Qt Creator** (use the genuine open-source Qt, **not** the MacPorts Qt).

Install dependencies via MacPorts:

```sh
sudo port install gsl fftw-3 libtool-ltdl libusb eigen3 pybind11
```

Optionally, for a universal (arm64 + x86_64) binary, build fftw-3 with:

```sh
sudo port install fftw-3 +universal +clang13 -gfortran
```

Additional notes:

- Add `/opt/local/bin` to PATH in the Qt Creator build-environment pane if needed.
- In Qt Creator's **executable environment** pane, **deactivate** "Add build library search path to DYLD_LIBRARY_PATH …", otherwise KAME crashes on launch.
- If `ruby.h` is not found, reinstall Xcode command-line tools: `xcode-select --install`.
- Qt 6: the **Qt5 compatibility module** must be selected during Qt installation.
- NI 488.2 is not supported on Apple Silicon; use the built-in usermode NI USB-GPIB driver instead (no kernel module required).

---

### Windows (x86-64, MSYS2 / MinGW)

> Requires **Qt ≥ 6.10** with the llvm-mingw64 toolchain.
> Open `kame.pro` in **Qt Creator**.

Install dependencies via MSYS2:

```sh
pacman -S make \
    mingw-w64-x86_64-zlib \
    mingw-w64-x86_64-fftw \
    mingw-w64-x86_64-gsl \
    mingw-w64-x86_64-eigen3 \
    mingw-w64-x86_64-pybind11 \
    mingw-w64-x86_64-libusb \
    mingw-w64-x86_64-python-numpy \
    mingw-w64-x86_64-ruby
```

NI 488.2 or DAQmx drivers are optional.

**Before running KAME**, copy the following DLLs from `C:\msys64\mingw64\bin` alongside the KAME executable:

```
libfftw3-3.dll  libgsl.dll  libgslcblas-0.dll
zlib1.dll  libgmp-10.dll  libusb-1.0.dll
x64-msvcrt-ruby3**.dll
```

Also copy `kame/script/rubylineshell.rb` and `kame/script/pythonlineshell.py` to `./Resources`.

**Launch scripts:**

| Script | Purpose |
|---|---|
| `kame.bat` | Standard launch (system Python) |
| `kame-msyspython.bat` | Launch with MSYS2 Python (numpy, etc.) |

To launch from Qt Creator, add to **Projects → Environment**:

```
PATH=C:\msys64\usr\bin;C:\msys64\mingw64\bin;C:\msys64\mingw64\lib
PYTHONHOME=C:\msys64\mingw64
```

---

## Scripting

KAME exposes its entire node tree to **Ruby** and **Python**. Scripts can be run
from the **Script** tab in the UI, loaded from `.kam` files, or executed
interactively in a Jupyter notebook connected to KAME's embedded IPython kernel.

A `.kam` file is a Ruby script that recreates the full measurement state when
executed. When Python is available, `.kam` files are loaded via a fast Python-based
translator instead of the Ruby interpreter.

---

## AI-Assisted Experiment Automation (MCP)

KAME 8.0 ships a built-in [MCP](https://modelcontextprotocol.io/) (Model Context
Protocol) server that lets AI assistants execute Python code directly in the running
KAME interpreter. The MCP server connects to the embedded IPython kernel via
`jupyter_client`, giving the AI full access to `Root()`, `Snapshot()`,
`Transaction()`, and all loaded drivers — the same environment available in Jupyter
notebooks.

This enables conversational experiment control:

```
"Read the current temperature from LakeShore1"
"Sweep the magnetic field from 0 to 5 T in 0.1 T steps, recording NMR signal at each point"
"Plot the last 100 DMM readings"
```

### Available MCP tools

| Tool | Description |
|---|---|
| `kame_api` | Return the Python API quick reference (call first) |
| `execute_code` | Run Python in KAME's interpreter (returns text + matplotlib plots) |
| `execute_code_async` | Run long experiments asynchronously (sweeps, scans) |
| `get_result` | Check status of an async job |
| `tree` | Browse the node tree with configurable depth (compact indented output) |
| `kame_status` | Check if KAME is running and list active drivers (JSON) |

### Quick start

1. Install prerequisites:
   ```sh
   pip install mcp jupyter_client
   ```
2. Start KAME and launch a Jupyter notebook (Script → Launch Jupyter Notebook).
   KAME writes `.mcp.json` to the notebook workspace directory automatically.
3. Open Claude Code in the same directory — the MCP server is discovered and
   connected automatically.
4. Ask Claude to interact with your instruments. The `.mcp.json` file is removed
   when KAME exits.

**Manual setup** (without Jupyter):

```sh
claude mcp add kame /path/to/python /path/to/KAME/Resources/kame_mcp_server.py
```

### How it works

1. When KAME launches a Jupyter notebook, it writes the kernel connection path to
   `~/.kame_kernel_connection.json`.
2. The MCP server reads that file and connects to the kernel via ZMQ (`jupyter_client`).
3. The AI client launches the MCP server as a subprocess (stdio transport).
4. The server ships `kame_python_api.md` — an API reference that Claude reads
   automatically before writing code, reducing trial-and-error.

---

## Contributing

Bug reports and pull requests are welcome on [GitHub](https://github.com/northriv/KAME).

---

*This README was written with the assistance of [Claude](https://claude.ai) (Anthropic).*
