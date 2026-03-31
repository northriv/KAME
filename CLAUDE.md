# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KAME is a scientific instrument control and measurement software framework written in C++11/Qt. It provides a plugin-based architecture for controlling laboratory instruments (oscilloscopes, lock-in amplifiers, temperature controllers, magnet power supplies, etc.) with Python and Ruby scripting support. Version 8.0.

**Platforms:** macOS, Windows (64-bit). Linux support is discontinued; the `CMakeLists.txt` / KDE4 build is legacy and not maintained.

## Build System

The primary build method on macOS is via **Qt Creator** using `kame.pro` (qmake). The `CMakeLists.txt` targets KDE4 (Linux, legacy/discontinued).

**macOS dependencies** (via MacPorts under `/opt/local`):
- `gsl`, `fftw3`, `libtool-ltdl`, `zlib`, `libusb`, `eigen3`, `pybind11` (no boost)
- Use genuine Qt (not from MacPorts); Qt5 compatibility module required for Qt 6
- Do NOT enable "Add build library search path..." in Qt Creator's executable environment pane — this causes crashes

**Running tests** (CMake build only):
```sh
cd build && ctest                    # run all tests
cd build && ./tests/transaction_test # run a specific test
```

Tests live in `tests/` and cover the core STM framework: `allocator_test`, `atomic_shared_ptr_test`, `atomic_queue_test`, `mutex_test`, `transaction_test`, `transaction_negotiation_test`, `transaction_dynamic_node_test`.

## Architecture

### Software Transactional Memory (STM) Framework

The core abstraction is a lock-free, snapshot-based STM in `kame/transaction.h`. All data nodes inherit from `Transactional::Node<XNode>` (via `XNode` in `kame/xnode.h`).

- **`Snapshot<XN>`** — immutable, consistent read view of a subtree, taken in O(1). A `Snapshot` taken *inside* a transaction reads the last **committed** state, not the in-progress `tr` state — use `tr[*node]` to see uncommitted writes within the same transaction. **Caution:** taking a `Snapshot` of a subtree during a transaction can be harmful if the tree contains a hard link (a child node with two or more parents): snapshotting one parent's subtree may destroy the snapshot consistency for the other parent. Avoid `Snapshot` inside transactions where hard links exist; use `tr[*node]` instead.
- **`Snapshot::isOlderThan(other)`** — temporal ordering via a Lamport clock embedded in `m_serial`. Serial layout: 48-bit monotonic counter in the upper bits, 16-bit thread ID in the lower 16 bits; `SerialGenerator::gen(last_serial)` advances the thread-local counter past `last_serial` before incrementing (Lamport step). `snapshot()` reads the node's PacketWrapper before calling `gen()` and passes its `m_bundle_serial` so the snapshot serial is always greater than the committed state it observes. `bundle()` additionally advances against each child's bundle serial (propagated recursively), and `unbundle()` carries the super-node's serial into the new PacketWrapper. All comparisons use unsigned subtraction reinterpreted as signed, correct as long as the true counter difference is well below 2^47. To compare snapshots at different levels of the hierarchy, project the super-node snapshot down to the sub-node first: `Snapshot<XN>(nodeB, shotA).isOlderThan(shotB)`.
- **`Transaction<XN>`** — optimistic write access with copy-on-write on `operator[]`; commits can fail and are retried:
  - `iterate_commit(lambda)` — retries unconditionally on conflict
  - `iterate_commit_if(lambda)` — retries on conflict **only if** the lambda returns `false`; returning `true` means "give up"
- **Online insertion** — `insert(tr, child, true)` commits the child to the live tree immediately and makes it accessible via `tr[*child]` within the same transaction. Without the `true` flag, the child is only visible after the transaction commits.
- **`XNode::Payload`** — the data a node holds; subclass this in each driver/node to add fields
- `atomic_shared_ptr` (in `kame/atomic_smart_ptr.h`) is the lock-free primitive underpinning snapshots and commits. It uses a **tagged-pointer** scheme: the lower bits of the heap pointer (guaranteed zero by allocator alignment) store a small local reference counter. `ATOMIC_SHARED_REF_ALIGNMENT` is the alignment value — it defines the pointer mask, the refcnt mask, and the max refcnt. **Do not use `alignas(N)` or `alignof(Ref)` for this constant.** Use `sizeof(intptr_t)` (non-intrusive) or `sizeof(double)` (intrusive) instead: both equal 8 on 64-bit and reflect the minimum alignment any conforming allocator guarantees. `alignas(N > max_align_t)` on the struct does *not* force `operator new` to honour that alignment pre-C++17, so the lower bits may not be zero — causing silent pointer corruption and rare crashes.

```cpp
// Read snapshot
Snapshot<NodeA> shot(node);
double x = shot[node].m_x;

// Transactional write
node.iterate_commit([](Transaction<NodeA> &tr) {
    tr[node].m_x += 1;
});

// Online insertion — child accessible via tr immediately
parent.iterate_commit_if([&](Transaction<NodeA> &tr) -> bool {
    if( !parent.insert(tr, child, true)) return false;
    tr[ *child].m_x = 42; // accessible right away
    return true;
});
```

### Driver / Module Architecture

- **`XDriver`** (`kame/driver/driver.h`) — base for all instrument drivers; holds a timestamped `Payload` with `time()` (when the phenomenon occurred) and `timeAwared()` (when visible to the operator); emits `onRecord` and `onVisualization` signals
- **`onVisualization` signal** — `Talker<bool, XDriver*>`; callbacks receive `(const Snapshot&, bool afterRecorded, XDriver*)`. `afterRecorded = !skipped` (i.e. `record()` was called). Do NOT use `time().isSet()` as a proxy — it stays `true` from the previous record across `XSkippedRecordError` cycles. `XRecordError` resets time to zero but still sets `afterRecorded = true`. Connect without `FLAG_AVOID_DUP` or `FLAG_MAIN_THREAD_CALL` unless the callback does Qt UI work.
- Instrument drivers are built as **shared libraries** under `modules/` and loaded at runtime (via ltdl)
- Each module subdirectory contains one or more drivers and registers them with the framework
- Communication with hardware is abstracted in `modules/charinterface/` (serial, TCP, GPIB, USB)
- Drivers are registered with `REGISTER_TYPE(XDriverList, ClassName, "Human-readable label")` — this is what populates the driver selection UI
- `modules/charinterface/usermode-linux-gpib/` — userspace port of the NI USB-GPIB kernel driver (linux-gpib 4.3.6); compiles `ni_usb_gpib.c` unchanged, replacing all kernel APIs via `osx_compat.h` / `win_compat.h` shims (libusb + pthreads / Win32). Supports NI USB-B, USB-HS, USB-HS+, KUSB-488A, MC USB-488 without a kernel module. On macOS this is the only USB-GPIB path on Apple Silicon.

### Key Subsystems

| Directory | Purpose |
|---|---|
| `kame/` | Core framework: XNode, STM, thread/scheduler, scripting glue |
| `kame/driver/` | `XDriver` base, `XPrimaryDriver`, `XSecondaryDriver`, Python driver bridge |
| `kame/analyzer/` | `XAnalyzer`, `XScalarEntry`, `XCalibratedEntry` — extract and calibrate scalar values from driver records |
| `kame/math/` | FFT, AR, spectral analysis helpers |
| `kame/script/` | Python (pybind11) and Ruby scripting integration |
| `kame/graph/` | Plotting/graphing framework |
| `modules/charinterface/` | Hardware communication (serial/TCP/GPIB) |
| `modules/<instrument>/` | Per-instrument driver plugins |
| `tests/` | STM/concurrency unit tests |

### Scalar Entry and Calibration

- **`XScalarEntry`** — scalar value extracted from a driver record; holds `value()`, `storedValue()`, `delta()`, `store()`. `driver()` identifies which driver owns it. `isTriggered()` in `Payload` is set when `|value − storedValue| > delta`.
- **`XCalibratedEntry`** (`kame/analyzer/analyzer.h`) — derives a new scalar from an existing `XScalarEntry` via an `XCalibrationCurve`. Exposes a proxy `XScalarEntry` inserted into `XScalarEntryList` only when both source and curve are valid. The proxy uses the **source's driver** so `XTextWriter` and `XEntryListConnector` handle it like any other entry. Proxy is recreated whenever source changes to carry the correct driver reference.
- **`XCalibrationCurve`** (`kame/thermometer/thermometer.h`) — base for all calibration curves. Virtuals `useLogScaleRaw()` / `useLogScaleOutput()` control whether the raw/output axes use log-space for cspline interpolation and the calibration table graph. `XResistanceThermometer` returns `true` for both; `XGenericCalibration` returns `false` for both. `XCSplineCalibrationX<Base>` respects these virtuals.

### Signal/Listener Pattern

Nodes communicate via `Talker<T>` / `Listener<T>` (in `kame/xnode.h` area). Listeners connect to a `Talker` and receive callbacks within the STM framework, maintaining consistency with snapshots.

### Scripting

- Python scripting uses pybind11; entry point in `kame/script/xpythonsupport.cpp`; runtime support in `kame/script/xpythonsupport.py`
- Ruby scripting entry point in `kame/script/xrubysupport.cpp`
- The `kame` module is **pre-imported** in all script contexts — use `Root()`, `Snapshot()`, `Transaction()` directly (not `kame.Root()`)
- `Root()` returns the measurement root node
- Primary drivers subclass `XPythonDriver<T>` (`kame/driver/pythondriver.h`); secondary drivers subclass `XPythonSecondaryDriver`
- Register Python driver classes with `MyClass.exportClass("TypeName", MyClass, "Label")` — makes them appear in the driver list exactly like compiled drivers
- `Payload.local()` returns a transaction-isolated `dict` deep-copied per snapshot; use it to store Python-side state with the same consistency guarantees as C++ Payload fields
- Payload GC uses a deferred deque + mutex to avoid holding the GIL during snapshot cleanup (GIL-enabled builds); Python 3.13 free-threading (`Py_GIL_DISABLED`) is also supported
- `modules/python/pydrivers.py` — built-in Python driver examples: `Test4Res` (simple 4-terminal resistance with polarity switching) and `Py4Res` (multi-current variant); good reference for writing new Python drivers
- Python interpreter runs in its own OS thread; Qt UI operations must be dispatched via `kame_mainthread(closure)`
- Jupyter/IPython kernel embedding is supported when `ipykernel` is available
- **Startup sequence:** only `xpythonsupport.py` is exec'd immediately; `pytestdriver.py` and `pydrivers.py` are collected as deferred scripts via `kame_deferred_scripts()` and executed on the first `kame_pybind_one_iteration()` tick (after the IPython kernel is up). Optional extension files that are absent produce a stderr warning, not a UI error.
- Script thread launch no longer has fixed `sleep()` delays; deferred scripts execute in the global namespace via `exec(script, globals())`
- **GIL startup synchronization** — `FrmKameMain` constructor creates `XMeasure` immediately, which starts the Python thread. Driver modules (including Python modules whose `PyDriverExporter`/`PyXNodeExporter` global constructors need the GIL) are loaded afterward in `main.cpp`. To prevent the main thread from blocking on `gil_scoped_acquire` while the Python thread holds the GIL during heavy imports, `XPython::execute()` releases the GIL and waits on `m_modules_loaded` before running `xpythonsupport.py`. `main.cpp` calls `form->signalAllModulesLoaded()` after the `lt_dlopenext` loop to unblock it.

### Serialization (.kam files)

`.kam` files are Ruby scripts generated by `XRubyWriter` (root entry point `kame/script/xrubywriter.cpp`) and loaded by `XRuby` / `XRubySupport`.

- Nodes with `runtime=true` are written as **comments** and are not restored on load. A node's `runtime=true` flag propagates to all descendants — even `runtime=false` children are commented if any ancestor is `runtime=true`.
- `XListNodeBase` children with `runtime=false` are serialized via `x.last.create(typename, name)` and recreated on load via `createByTypename(type, name)`. The `typename` comes from `getTypename()` and must match the registered key in `XTypeHolder`.
- `XAliasListNode` children are navigated by name (`x.last["name"]` → `getChild(name)`), not re-created.
- **`getTypename()` pitfall for template aliases**: the default `getTypename()` uses `typeid(*this).name()`. For template instantiation aliases (e.g. `using XFoo = XFooX<Functor>`), this returns a mangled name that won't match the registered `XTypeHolder` key. Override `getTypename()` or call `setStoredTypename(type)` inside `createByTypename()` and return it from the override.
- **Python loader (preferred)**: when `USE_PYBIND11` is defined, `.kam` files are routed to `XPython` (`loadKam` in `xpythonsupport.py`) instead of `XRuby`. A minimal translation (strip indentation, `x = Array.new` → `x = _KamStack()`, `x.last` → `x[-1]`, `x.pop` → `x.pop()`) is applied before `exec`. `_KamNode.create()` dispatches to the main thread via `kame_mainthread()` for non-thread-safe lists. This is faster than the Ruby path because `kame_mainthread` uses a dedicated condition variable rather than the Qt event queue. Falls back to Ruby when Python is unavailable.
- **`_KamNode` / `_KamFakeNode`**: `_KamNode` wraps `XNode` for `.kam` access; `_KamFakeNode` silently absorbs operations on nodes that are absent or not downcast to `XListNodeBase` (version-skew tolerance). `cast_to_pyobject` only downcasts to types explicitly registered via `export_xnode<T, XListNodeBase>` — unregistered list types come back as `kame.XNode` and are handled gracefully via `hasattr` guards in `_KamNode.create()`.

## Qt UI (.ui files)

- **No `stretch` property on layouts** — `<property name="stretch">` on a `QBoxLayout` is not valid in Qt's UI parser and silently breaks layout. Control stretch per-widget via `sizePolicy`:
  ```xml
  <property name="sizePolicy">
    <sizepolicy hsizetype="Preferred" vsizetype="Expanding">
      <horstretch>0</horstretch>
      <verstretch>1</verstretch>
    </sizepolicy>
  </property>
  ```
- **Bare layout in QVBoxLayout** — a bare `QLayout` (no wrapping widget) cannot have a size policy set on it; if placed alongside a `QWidget` in a `QVBoxLayout` it will consume all vertical space. Wrap in a `QWidget` if size policy control is needed.
- **Prefer `setEnabled` over `setVisible`** for optional form sections — hiding a widget collapses the layout and makes the form look broken; disabling keeps layout stable and signals inactivity.

## Code Conventions

- All exported symbols use `DECLSPEC_KAME` macro
- Node payload fields are public members of the nested `Payload` struct inside each node class
- Prefer `iterate_commit` / `iterate_commit_if` over manual retry loops for transactions
- Time-stamping: use `XTime` from `kame/xtime.h`; `m_recordTime` is set by the driver when data is captured
- **Safe list release** — before calling `list->release(node)`, guard with `Snapshot shot(*list); if(shot.isUpperOf(*node))` to prevent double-release crashes at shutdown (the list may have already cleared the node before the owning object's destructor runs).
- **Snapshot containment check** — use `shot.isUpperOf(*node)` to test whether a node is in a snapshot, **not** `try { shot.at(*node) } catch (NodeNotFoundError &)`. An inner catch masks any `NodeNotFoundError` that an outer catch block was meant to handle. `isUpperOf` is also the correct semantic: it is a direct O(1) containment predicate, not an exception-driven fallback.
- **`QPointer` for widget references in async callbacks** — store `QPointer<QWidget>` (not a raw pointer) when a widget reference is held across asynchronous callbacks (e.g. `Talker`/`Listener` or `TalkerOnce` signals). `QPointer` auto-nullifies when Qt destroys the widget; a raw pointer becomes dangling. Check `if(!m_widget)` before use. Example: `XWaveNGraph::m_btnMathTool` was changed from `QToolButton *` to `QPointer<QToolButton>` to fix a crash during `.kam` loading under concurrent driver creation.
- **`XCalibratedEntry` is a proper `XDriver`** — it lives in `XCalibratedEntryList` (not `XDriverList`) and owns its `XScalarEntry` as a direct child. In `onVisualization` callbacks, `shot.isUpperOf(*entry)` is always true for calibrated entries; no fallback snapshot is needed.
