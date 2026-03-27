# KAME — K's Adaptive Measurement Engine

[![License: GPL v2+](https://img.shields.io/badge/License-GPL%20v2%2B-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)
[![GitHub](https://img.shields.io/badge/GitHub-northriv%2FKAME-181717?logo=github)](https://github.com/northriv/KAME)
[![Version](https://img.shields.io/badge/version-7.8.1-green)]()

KAME is an open-source, multi-threaded framework for automated physical property measurements,
developed at [Kitagawa Laboratory, ISSP, University of Tokyo](https://kitag.issp.u-tokyo.ac.jp/).
It is particularly suited to NMR and ODMR experiments, and supports flexible measurement
orchestration across compatible instruments without custom programming.

**License:** GPL v2 or later
**Authors:** Kentaro Kitagawa, Shota Suetsugu
**Platforms:** macOS, Windows (64-bit); Linux support discontinued

---

## Features

- Transactional, lock-free node/data model (Software Transactional Memory)
- Python (+Jupyter notebook) and Ruby scripting — nearly full control from scripts
- OpenGL-based 2-D / 1-D graph display; arbitrary scalar combinations (T, V, …)
- Real-time NMR relaxation fitting (T1, T2, Tst.e.), Inverse Laplace Transform
- Fourier step-sum spectrum measurement with field / frequency sweeping
- Complete data logging with post-measurement re-analysis
- Save / restore full measurement state to `.kam` files
- Modular driver plug-in architecture; Python drivers redefinable at runtime

### Supported instruments

| Category | Models |
|---|---|
| DC sources | Yokogawa 7651, Advantest models, Optotune ICC4C2000 |
| Multimeters / picoammeters | Keithley 2000/2001/6482, HP/Agilent 34420A/3458A/3478A |
| Signal generators | NF WAVE-FACTORY, Rohde & Schwarz, HP/Agilent, DSTech, Thamway NMR PROT |
| Temperature controllers | Cryocon M32/M62, LakeShore 340/350/370, Oxford ITC-503, Picowatt AVS-47 |
| Network analysers | Agilent E5061/E5062, Copper Mountain TR1300/1504/4530, LibreVNA, Thamway T300-1049A |
| Lock-in amplifiers | Stanford SR830, NF LI5640, Signal Recovery 7265, Andeen-Hagerling 2500A |
| Magnet power supplies | (various) |
| Oscilloscopes | (various) |
| Cameras | IEEE 1394, eGrabber (ODMR imaging) |
| Flow controllers / motors | (various) |
| Quantum Design PPMS | `qd` |
| NI DAQmx / counter / GPIB | `nidaq`, `counter`, `charinterface` |
| Monte Carlo simulation | `montecarlo` |

---

## Architecture

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

**Multi-node consistency** is achieved through a *bundling* protocol: a parent packet absorbs child packets via two sequential CAS operations, making the entire subtree consistent under a single atomic pointer. A `m_missing` flag marks packets with stale children, driving re-bundling on demand.

**Collision backoff:** `Linkage::negotiate()` uses a `m_transaction_started_time` timestamp to impose a proportional wait on detected collisions, preventing live-lock under high write contention.

`iterate_commit_if(lambda)` lets the caller abort the retry loop (return `true` from the lambda to stop), enabling conditional transactions.

> **Caution:** Taking a nested `Snapshot` inside a transaction on a tree that contains a hard link (a child with two parents) can break consistency. Use `tr[*node]` instead of a nested `Snapshot` in that situation.

#### Comparison with other STM designs

*The following comparison was written by Claude (Anthropic) based on analysis of the source code.*

Most widely-used STMs (GHC/Haskell `TVar`, Clojure `Ref`/`dosync`, ScalaSTM) are **flat**: the unit of transaction is a set of independent transactional variables. KAME's STM is instead **tree-structured** — the entire instrument node tree is the shared state, and snapshots are always subtree-consistent. This difference drives several design choices:

| Aspect | Flat STMs (Haskell, Clojure, ScalaSTM) | KAME STM |
|---|---|---|
| Conflict granularity | Per-variable | Per-packet (subtree root) |
| Read model | `readTVar` / `deref` inside transaction | `Snapshot` (outside) or `tr[*node]` (inside) |
| Consistency scope | Variables listed explicitly | Entire subtree, guaranteed by bundling |
| Commit log | Redo log or write set | Copy-on-write + CAS on single `Linkage` |
| Retry primitive | `retry` / `orElse` (Haskell) | `iterate_commit` / `iterate_commit_if` |
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

### Driver / Plug-in Architecture

Instrument drivers are **shared libraries** under `modules/` loaded at runtime via `ltdl`.
Each driver subclasses `XDriver` (`kame/driver/driver.h`), which carries a timestamped
`Payload` (`time()` = phenomenon time, `timeAwared()` = operator-visible time) and emits
`onRecord` / `onVisualization` signals.

Hardware communication is abstracted in `modules/charinterface/` (serial, TCP, GPIB, USB).
Drivers can also be subclassed in Python via `XPythonDriver` (`kame/driver/pythondriver.h`).

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

**Threading notes:**

- Long-running C++ calls release the GIL (`gil_scoped_release`) so the Python thread
  does not block Qt.
- Any Qt UI operation (loading `.ui` files, showing forms) must be dispatched to the
  main thread via `kame.kame_mainthread(closure)`.
- Python 3.13 free-threading (`Py_GIL_DISABLED`) is supported: Payload garbage
  collection uses a deferred deque + mutex rather than holding the GIL during
  snapshot cleanup.

### Serialization (`.kam` files)

A `.kam` file is a Ruby script generated by `XRubyWriter` and re-executed on load.
Nodes marked `runtime=true` are written as comments and not restored.
`XListNode` children are recreated via `createByTypename()`; the typename must match
the key registered in `XTypeHolder`.

---

## Dependencies

| Library | Notes |
|---|---|
| **Qt** ≥ 5.7 or Qt 6 | Qt 5 compatibility module required for Qt 6 |
| **boost** | |
| **Ruby** | scripting |
| **pybind11** | Python scripting |
| **GSL** | |
| **FFTW 3** | |
| **Eigen 3** | |
| LAPACK / ATLAS / BLAS *(optional)* | |
| **libusb** | USB instrument interfaces |
| linux-gpib or NI 488.2 *(optional)* | GPIB interfaces |
| NI DAQmx *(optional)* | NI data-acquisition hardware |

A C++ compiler supporting C++17 or later is required (GCC ≥ 10, Clang ≥ 2.1 with appropriate dialect flags).

Optional: IPython / Jupyter notebook, linux-gpib or NI 488.2, NI DAQmx, libdc1394 (macOS cameras).

---

## Building

### Generic (Linux / Unix)

```sh
qmake [options] /path/to/kame/source
make
sudo make install
```

Or with CMake (KDE4 build):

```sh
mkdir build && cd build
cmake /path/to/kame/source
make
make install DESTDIR=/path/to/install
```

---

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
- NI 488.2 is not supported on Apple Silicon.

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

KAME exposes its node tree to Ruby and Python. Scripts can be run from the **Script** tab in the UI or loaded from `.kam` files. A `.kam` file is a Ruby script that recreates the full measurement state when executed.

---

## Contributing

Bug reports and pull requests are welcome on [GitHub](https://github.com/northriv/KAME).

A manual (Japanese) is available on the [project page](https://kitag.issp.u-tokyo.ac.jp/%e8%87%aa%e5%8b%95%e5%8c%96%e5%af%be%e5%bf%9c%e6%b8%ac%e5%ae%9a%e3%83%97%e3%83%ad%e3%82%b0%e3%83%a9%e3%83%a0kame/).
An English manual is in progress.

---

*This README was written with the assistance of [Claude](https://claude.ai) (Anthropic).*
