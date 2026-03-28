# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KAME is a scientific instrument control and measurement software framework written in C++11/Qt. It provides a plugin-based architecture for controlling laboratory instruments (oscilloscopes, lock-in amplifiers, temperature controllers, magnet power supplies, etc.) with Python and Ruby scripting support. Version 7.8.1.

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
- **`Transaction<XN>`** — optimistic write access with copy-on-write on `operator[]`; commits can fail and are retried:
  - `iterate_commit(lambda)` — retries unconditionally on conflict
  - `iterate_commit_if(lambda)` — retries on conflict **only if** the lambda returns `false`; returning `true` means "give up"
- **Online insertion** — `insert(tr, child, true)` commits the child to the live tree immediately and makes it accessible via `tr[*child]` within the same transaction. Without the `true` flag, the child is only visible after the transaction commits.
- **`XNode::Payload`** — the data a node holds; subclass this in each driver/node to add fields
- `atomic_shared_ptr` (in `kame/atomic_smart_ptr.h`) is the lock-free primitive underpinning snapshots and commits

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
| `kame/analyzer/` | `XAnalyzer`, `XScalarEntry` — extract scalar values from driver records |
| `kame/math/` | FFT, AR, spectral analysis helpers |
| `kame/script/` | Python (pybind11) and Ruby scripting integration |
| `kame/graph/` | Plotting/graphing framework |
| `modules/charinterface/` | Hardware communication (serial/TCP/GPIB) |
| `modules/<instrument>/` | Per-instrument driver plugins |
| `tests/` | STM/concurrency unit tests |

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

### Serialization (.kam files)

`.kam` files are Ruby scripts generated by `XRubyWriter` (root entry point `kame/script/xrubywriter.cpp`) and loaded by `XRuby` / `XRubySupport`.

- Nodes with `runtime=true` are written as **comments** and are not restored on load. A node's `runtime=true` flag propagates to all descendants — even `runtime=false` children are commented if any ancestor is `runtime=true`.
- `XListNodeBase` children with `runtime=false` are serialized via `x.last.create(typename, name)` and recreated on load via `createByTypename(type, name)`. The `typename` comes from `getTypename()` and must match the registered key in `XTypeHolder`.
- `XAliasListNode` children are navigated by name (`x.last["name"]` → `getChild(name)`), not re-created.
- **`getTypename()` pitfall for template aliases**: the default `getTypename()` uses `typeid(*this).name()`. For template instantiation aliases (e.g. `using XFoo = XFooX<Functor>`), this returns a mangled name that won't match the registered `XTypeHolder` key. Override `getTypename()` or call `setStoredTypename(type)` inside `createByTypename()` and return it from the override.

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
