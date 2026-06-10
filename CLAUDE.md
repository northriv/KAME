# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KAME is a scientific instrument control and measurement software framework written in C++11/Qt. It provides a plugin-based architecture for controlling laboratory instruments (oscilloscopes, lock-in amplifiers, temperature controllers, magnet power supplies, etc.) with Python and Ruby scripting support. Version 8.0.

**Platforms:** macOS, Windows (64-bit). Linux support is discontinued.

## Build System

The primary build method on macOS is via **Qt Creator** using `kame.pro` (qmake).

**Adding new files to the build:**
- Core framework / scripts: edit `kame/kame.pro`
- Driver modules: edit `modules/<name>/<name>.pro`
- In `kame/kame.pro`: `SOURCES`/`HEADERS` for C++; `scriptfile.files` for files deployed to `Contents/Resources` (macOS); `DISTFILES` in `else { }` block for Windows

**macOS dependencies** (via MacPorts under `/opt/local`):
- `gsl`, `fftw3`, `libtool-ltdl`, `zlib`, `libusb`, `eigen3`, `pybind11` (no boost)
- Use genuine Qt (not from MacPorts); Qt5 compatibility module required for Qt 6
- Do NOT enable "Add build library search path..." in Qt Creator's executable environment pane — this causes crashes

Tests live in `kamestm/tests/` and cover the core STM framework: `atomic_shared_ptr_test`, `atomic_scoped_ptr_test`, `atomic_queue_test`, `mutex_test`, `transaction_test`, `transaction_negotiation_test`, `transaction_dynamic_node_test`, and the `transaction_payload_integrity_*` family. Pool-allocator tests live in `kamepoolalloc/tests/` (`alloc_*`).

**Memory-model verification (C++-derived)** (`kamestm/tests/cds_atomic_shared_ptr/`): GenMC model-checker tests derived from the C++ implementation in `kamepoolalloc/atomic_smart_ptr.h` (relocated from `kamestm/`). Three tests cover `load_shared_`/`release_tag_ref_` concurrency, `load_shared_` vs `compareAndSwap_` races, and multi-thread `compareAndSet` contention. Verifies reference counting safety but not payload values. Requires GenMC v0.16+ built against LLVM 20; see `Makefile` for build instructions. Run with `make run` from the test directory.

**Memory-model verification (TLA+-derived)** (`kamestm/tests/tlaplus/test_*.c`): GenMC tests mechanically generated from the TLA+ specifications. `test_atomic_shared_ptr.c` (Layer 0), `test_stm_commit.c` (Layer 1), `test_bundle_2level.c` and `test_bundle_3level.c` (Layer 2). These verify that the formal specs are realizable under the RC11 memory model.

## Architecture

### Software Transactional Memory (STM) Framework

The core abstraction is a lock-free, snapshot-based STM in `kamestm/transaction.h` (spun out as the standalone [`kamestm/`](kamestm/) dual-licensed library — see its README; `kame/` accesses it via INCLUDEPATH). All data nodes inherit from `Transactional::Node<XNode>` (via `XNode` in `kame/xnode.h`).

- **`Snapshot<XN>`** — immutable, consistent read view of a subtree, taken in O(1). A `Snapshot` taken *inside* a transaction reads the last **committed** state, not the in-progress `tr` state — use `tr[*node]` to see uncommitted writes within the same transaction. **Caution:** Taking a Snapshot inside a transaction can trigger bundling on the snapshot target's subtree. If the bundle/unbundle changes the PacketWrapper that the transaction's CAS will compare against, the transaction will always fail and retry. This occurs in two situations: (1) taking a Snapshot of a node that is an ancestor of the transaction target (bundling the ancestor absorbs the target's packet), and (2) when a hard link exists (a child with two or more parents), taking a Snapshot or Transaction on one parent triggers unbundle of the other parent's subtree, causing seemingly unrelated transactions to fail. In both cases, no data corruption occurs — the CAS simply never succeeds. Use `tr[*node]` to read within the same transaction scope instead. The hard-link case is formally modelled in `kamestm/tests/tlaplus/BundleUnbundle_hardlink_*.tla` (sibling-parents and root-with-intermediate self-collision variants); see `kamestm/tests/VERIFICATION.md` §5. A bundle-Phase-3 fix (skip child wrappers whose `local.subpackets[c] == nullptr`) is required for correctness when hard-link references coexist with `is_bundle_root` bundles.
- **`Snapshot::isOlderThan(other)`** — temporal ordering via a Lamport clock embedded in `m_serial`. Serial layout: 48-bit monotonic counter in the upper bits, 16-bit thread ID in the lower 16 bits; `SerialGenerator::gen(last_serial)` advances the thread-local counter past `last_serial` before incrementing (Lamport step). `snapshot()` reads the node's PacketWrapper before calling `gen()` and passes its `m_bundle_serial` so the snapshot serial is always greater than the committed state it observes. `bundle()` additionally advances against each child's bundle serial (propagated recursively), and `unbundle()` carries the super-node's serial into the new PacketWrapper. All comparisons use unsigned subtraction reinterpreted as signed, correct as long as the true counter difference is well below 2^47. To compare snapshots at different levels of the hierarchy, project the super-node snapshot down to the sub-node first: `Snapshot<XN>(nodeB, shotA).isOlderThan(shotB)`.
- **`Transaction<XN>`** — optimistic write access with copy-on-write on `operator[]`; commits can fail and are retried:
  - `iterate_commit(lambda)` — retries unconditionally on conflict
  - `iterate_commit_if(lambda)` — commits only when the lambda returns `true`; returning `false` skips the commit and retries unconditionally
  - `iterate_commit_while(lambda)` — retries as long as the lambda returns `true`; returning `false` aborts the loop (gives up)
- **Online insertion** — `insert(tr, child, true)` commits the child to the live tree immediately and makes it accessible via `tr[*child]` within the same transaction. Without the `true` flag, the child is only visible after the transaction commits.
- **`XNode::Payload`** — the data a node holds; subclass this in each driver/node to add fields. Payloads are **copy-on-write**: every `Transaction::operator[]` deep-copies the Payload, and every `Snapshot` retains its copy until released. For small scalar fields this is fine, but **large or variable-size data** (images, waveforms, masks, buffers) should be stored as `shared_ptr<const T>` (or `shared_ptr<T>` if mutation is needed before sharing). This way snapshots share the data cheaply — only the pointer is copied. When writing new data, create a *new* `shared_ptr` (`make_shared<T>(...)`) so older snapshots keep their immutable view. Example: `shared_ptr<std::vector<uint8_t>> m_mask` in `XGraph2DMathTool::Payload`.
- `atomic_shared_ptr` (in `kamepoolalloc/atomic_smart_ptr.h` — relocated there as the single home shared by the STM and the pool allocator) is the lock-free primitive underpinning STM snapshots/commits **and** the pool allocator's lock-free orphan-chunk reclaim chain (a chunk orphaned by an exited thread is pushed onto an `atomic_shared_ptr` chain, adopted/sweep-reclaimed; a re-owned chunk holds a self-referential owner-ref — see `kamepoolalloc/`). It uses a **tagged-pointer** scheme: the lower bits of the heap pointer (guaranteed zero by allocator alignment) store a small local reference counter. `LOCAL_REF_CAPACITY` is the alignment value — it defines the pointer mask, the refcnt mask, and the max refcnt. **Do not use `alignas(N)` or `alignof(Ref)` for this constant.** Use `sizeof(intptr_t)` (non-intrusive) or `sizeof(double)` (intrusive) instead: both equal 8 on 64-bit and reflect the minimum alignment any conforming allocator guarantees. `alignas(N > max_align_t)` on the struct does *not* force `operator new` to honour that alignment pre-C++17, so the lower bits may not be zero — causing silent pointer corruption and rare crashes.
- **Control block modes** (via marker inheritance on `T`):
  - **Default (no marker)**: separate alloc (`new T` + `new gref_<T>`), `weak_refcnt` included — `local_weak_ptr<T>` works out of the box.
  - **`atomic_strictrefonly`**: separate alloc, refcnt only (no `weak_refcnt`). `local_weak_ptr<T>` is a compile error. Saves 8 bytes per control block.
  - **`atomic_emplaced`**: combined alloc (`T` embedded in control block via `make_local_shared<T>`), `weak_refcnt` included.
  - **`atomic_weakable`** (extends `atomic_emplaced`): backward-compat alias, identical to `atomic_emplaced`.
  - All weak-capable control blocks (`gref_<T>`, `gref_weakable_<T>`) inherit from `gref_weak_base_`, which provides the type-erased `refcnt + weak_refcnt + try_promote()` layout used by `local_weak_ptr<T>`.

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

- **`XDriver`** (`kame/driver/driver.h`) — base for all instrument drivers; holds a timestamped `Payload` with `time()` (when the phenomenon occurred) and `timeAwared()` (acquisition start time — when the phenomenon started being measured; for non-real-time analysis, when the record was read); emits `onRecord` and `onVisualization` signals
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
- **`XCalibratedEntry`** (`kame/analyzer/analyzer.h`) — derives a new scalar from an existing `XScalarEntry` via an `XCalibrationCurve`. Exposes a proxy `XScalarEntry` inserted into `XScalarEntryList` only when both source and curve are valid. The proxy uses **`XCalibratedEntry` itself** as its driver (since `XCalibratedEntry` derives from `XDriver`), so `XTextWriter` receives the proxy value via the `CalibratedEntry`'s `onVisualization` signal. Proxy is recreated whenever source or curve changes.
- **`XCalibrationCurve`** (`kame/thermometer/thermometer.h`) — base for all calibration curves. Virtuals `useLogScaleRaw()` / `useLogScaleOutput()` control whether the raw/output axes use log-space for cspline interpolation and the calibration table graph. `XResistanceThermometer` returns `true` for both; `XGenericCalibration` returns `false` for both. `XCSplineCalibrationX<Base>` respects these virtuals.

### Graph Math Tools

- **`XGraphMathTool`** (`kame/graph/graphmathtool.h`) — base for on-graph measurement tools. 1D tools (`XGraph1DMathTool`) operate on axis ranges; 2D tools (`XGraph2DMathTool`) operate on rectangular image regions. Both produce `XScalarEntry` results. Tools are registered via `REGISTER_TYPE(XGraph1DMathToolList, ...)` / `REGISTER_TYPE(XGraph2DMathToolList, ...)`.
- **`XGraph2DMathTool` mask shapes** — each 2D tool has an `XComboNode maskType()` selecting `MaskShape::Rectangle` (default) or `MaskShape::Ellipse` (inscribed within the rectangle). The mask is generated by `XGraph2DMathTool::generateMask()` as a `std::vector<uint8_t>` (empty = no mask / all included). Functors receive the mask as a final parameter; `pixels()` computes the masked pixel count from selection coordinates and mask type.
- **`MaskShape::Arbitrary`** — the mask is stored in `XGraph2DMathTool::Payload::m_mask` and set externally (e.g. by Python via `payload.setMask(list)` / `payload.mask()`). `regenerateMask(tr)` is called automatically when selection coordinates or mask type change; for `Arbitrary`, it is a no-op (preserves the externally-set mask). `pixels()` counts from the stored mask for all shapes (including Ellipse).
- **Adding new mask shapes**: add a value to the `MaskShape` enum, extend `generateMask()`, add the label string in the `XGraph2DMathTool` constructor's `add()` call and in `XQGraph2DMathToolConnector::menuOpenActionActivated()`'s `maskLabels` array, and add an OSO drawing case to `OnScreenRectObject::drawNative()` (see `EllipseTool` for reference).
- **Python 2D math tool functors** receive 7 arguments: `(matrix, width, stride, numlines, coefficient, offset, mask)` where `mask` is a `(numlines, width)` uint8 numpy array. For `Rectangle` mask, an all-ones matrix is passed. Both are zero-copy views of transient C++ buffers, valid only during the call — `np.array(..., copy=True)` anything retained. These functors (and the C++ Sum/Average tools' scalar entries) are the **only quantitative pixel access** from Python; `X2DImagePlot::Payload::to_png()` is the rendered display image (gamma-encoded) and must not be decoded for analysis.
- **On-screen objects** — `OnScreenRectObject::Type::EllipseTool` draws an inscribed ellipse (48 line segments). The OSO type is chosen at creation time in `XGraph2DMathTool::createAdditionalOnScreenObjects()` based on the current mask type; changing the mask clears and recreates OSOs. The highlight overlay uses `OnPlotMaskObject`, which renders the actual mask bitmap as horizontal row-spans (quads), correctly visualising Rectangle, Ellipse, and Arbitrary masks.
- **Connector UI** — `XQGraph2DMathToolConnector` shows a "Mask Shape" submenu for each existing 2D tool with the current selection marked `*`. Mask actions are stored in `m_maskActions` and handled in `toolActivated()`.

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
- **MCP server** (`kame/script/kame_mcp_server.py`) — connects to the embedded IPython kernel via `jupyter_client`, providing AI assistants (Claude Code, etc.) with 8 tools: `kame_api`, `kame_manual` (user's manual TOC / per-section retrieval), `execute_code` (returns text + matplotlib plots as MCP ImageContent), `execute_code_async`/`get_result`/`stop_job` (background thread for long experiments, with `mcp_checkpoint()` progress reporting and cooperative stop), `tree` (recursive node browser with configurable depth, compact indented output), `kame_status`. Previous helper tools (`read_node`, `set_node`, `read_scalar`, `list_children`, `list_scalars`) were removed as redundant — `execute_code` handles all read/write operations directly. Kernel connection is reused across calls; `%matplotlib inline` is set automatically on first connect. `execute_code_async` runs code in a daemon thread on the kernel — KAME STM operations are thread-safe, but Python-level shared variables should not be read until the job completes. Auto-configured when launching a Jupyter notebook: `xpythonsupport.py` writes `.mcp.json` and `~/.kame_kernel_connection.json` in `launchJupyterConsole()` (notebook path only); both files are cleaned up on exit. Tool-generated code uses IPython expression results (bare last-line evaluation) instead of `print()`, because KAME's `MYDEFOUT` wraps print output in HTML via `display(IPython.display.HTML(...))` when a notebook is connected. The `_execute` message handler also filters out `display_data` messages containing HTML object reprs. API reference in `kame/script/kame_python_api.md` is served by the `kame_api` tool. The user's manual lives at `doc/manual/kame-8-en.md` (converted from the official docx; images in `doc/manual/media/`) and is served section-wise by the `kame_manual` tool; the md is deployed next to the server script via `scriptfile.files` in `kame.pro`, with a source-tree fallback path.

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
- Time-stamping: use `XTime` from `kamestm/xtime.h`; `m_recordTime` is set by the driver when data is captured
- **Safe list release** — before calling `list->release(node)`, guard with `Snapshot shot(*list); if(shot.isUpperOf(*node))` to prevent double-release crashes at shutdown (the list may have already cleared the node before the owning object's destructor runs).
- **Snapshot containment check** — use `shot.isUpperOf(*node)` to test whether a node is in a snapshot, **not** `try { shot.at(*node) } catch (NodeNotFoundError &)`. An inner catch masks any `NodeNotFoundError` that an outer catch block was meant to handle. `isUpperOf` is also the correct semantic: it is a direct O(1) containment predicate, not an exception-driven fallback.
- **`QPointer` for widget references in async callbacks** — store `QPointer<QWidget>` (not a raw pointer) when a widget reference is held across asynchronous callbacks (e.g. `Talker`/`Listener` or `TalkerOnce` signals). `QPointer` auto-nullifies when Qt destroys the widget; a raw pointer becomes dangling. Check `if(!m_widget)` before use. Example: `XWaveNGraph::m_btnMathTool` was changed from `QToolButton *` to `QPointer<QToolButton>` to fix a crash during `.kam` loading under concurrent driver creation.
- **`XCalibratedEntry` is a proper `XDriver`** — it lives in `XCalibratedEntryList` (not `XDriverList`) and owns its `XScalarEntry` as a direct child. In `onVisualization` callbacks, `shot.isUpperOf(*entry)` is always true for calibrated entries; no fallback snapshot is needed.
- **`XPointerItemNode::lsnOnListChanged` uses a fresh list snapshot** — when forwarding `onListChanged` to item-level listeners (e.g. combo box connectors with `FLAG_AVOID_DUP`), the method takes a fresh `Snapshot(*list)` instead of reusing the `shot` parameter from the triggering transaction. This is necessary because immediate listeners earlier in the same `onListChanged` dispatch loop may insert new entries (e.g. `XCalibratedEntry` proxy creation during pending-label resolution); the original `shot` would not contain those entries, and `FLAG_AVOID_DUP` deduplication would discard the newer event that does.
- **`weak_ptr::lock()` null check** — always check the return value of `weak_ptr::lock()` before dereferencing. In particular, `m_entries.lock()` in `XValGraph` and `XCalibratedEntry` can return `nullptr` during shutdown; dereference without a guard crashes.
- **`kame_mainthread` GIL scope** — the GIL release in `kame_mainthread` must be scoped to only the blocking wait (`talk` + condition wait). Python object operations (`status.is()`, `PyErr_SetString`) before and after the wait require the GIL.
- **Nested transaction for child-node init in constructors** — when a node constructor creates child nodes with non-transactional `create<T>(name, runtime)` and needs to initialize them (e.g. populating `XComboNode` items), do **not** use the outer `tr_meas` transaction passed from `createByTypename()`. The child is inserted outside that transaction's scope and `tr_meas` cannot see it, causing the transaction to stall (especially during `.kam` loading). Instead, use a nested `iterate_commit` on the node's own subtree:
  ```cpp
  // WRONG — stalls during .kam loading:
  MyNode::MyNode(..., Transaction &tr_meas, ...) :
      m_combo(create<XComboNode>("Combo", false)) {
      tr_meas[ *m_combo].add({"A", "B"});  // tr_meas can't see m_combo
  }
  // CORRECT — use own transaction:
  MyNode::MyNode(..., Transaction &tr_meas, ...) :
      m_combo(create<XComboNode>("Combo", false)) {
      iterate_commit([=](Transaction &tr){
          tr[ *m_combo].add({"A", "B"});
      });
  }
  ```

## Ohtaka (ISSP supercomputer) operating rules

When this Claude session is running on the Ohtaka login node (System B at ISSP, U-Tokyo), the login node is a **shared resource** and benchmarks/long jobs must NOT execute there directly. The pool allocator and STM stress / soak tests are the workloads most likely to harm shared interactivity if mis-run.

- **Working directory** — confine all work to `~/kame-claude/` (clone, builds, scratch). Do not touch other users' files or the system module tree.
- **Never run bench/test workloads on the login node** — wrap with `srun --exclusive` (or the appropriate `sbatch` job script) on a compute node. This applies to `ctest`, `bench_compare.sh`, `alloc_*_test`, the STM transaction tests, and any multi-threaded soak. The login node is fine for `cmake -B`, `cmake --build` (small parallelism), `git`, file edits, and reading test output.
- **Benchmark methodology** — for any number reported back to the user or written into a README:
  - Run on a compute node via `srun --exclusive` (one job per measurement).
  - Take **median of 5+ runs**; record min/max alongside.
  - Confirm node sanity before measuring: `lscpu --extended`, `dmesg | tail`, `numactl --hardware`.
  - Note the SLURM partition, node ID, governor (`/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`) and turbo state in the commit message or README footnote — Ohtaka results are only meaningful with that provenance.
- **perf / PMU** — Ohtaka compute nodes have a real PMU; `perf stat -e cycles,instructions,...` and `perf record -e cycles` work. Run profiling **inside** the `srun --exclusive` allocation. Long sampling runs (`perf record` for minutes) belong on compute nodes only.
- **No destructive ops on shared paths** — never `rm -rf` outside `~/kame-claude/`. No `module load`/`module unload` chains that leak environment to the user's shell — wrap each session in its own subshell.
- **Multi-Claude coordination** — when an Ohtaka Claude session runs in parallel with a cloud Claude session, the Ohtaka session specializes in: (a) absolute-number benchmarks, (b) perf-event profiling, (c) long soak / many-thread STM tests, (d) updating READMEs/tables with the resulting numbers. The cloud session keeps doing code edits, TLA+, GenMC, doc structure. Both rebase frequently from `origin/master` to keep merges trivial; flag any in-flight feature branch in the commit message.
- **Local environment specifics** (discovered 2026-06-10):
  - The benchmark checkout lives at `~/kame` (a symlink to `~/KAME`); the main test/bench build tree is `~/kame/build/tests` (configured from `tests/`, producing `kamestm-tests/` and `kamepoolalloc-tests/` subdirs).
  - That tree is built with **clang** from `~/llvm-install/bin/clang++` (CMAKE_BUILD_TYPE=RelWithDebInfo, `-O3 -DNDEBUG`) — NOT GCC. The system `/usr/bin/g++` is GCC 8 (el8) and too old for this codebase (`[[unlikely]]` etc. fail). Any "Linux benchmark" result from this tree is a **clang** number; don't attribute layout effects to GCC.
  - SLURM partition for benches: `i8cpu` (AMD EPYC 7702, 128 cores, 8 NUMA nodes, `THP=always`, no cpufreq governor exposed; node names like `c15uNNnN`). Typical invocation from `build/tests`: `srun -p i8cpu --time=00:30:00 --exclusive ../../kamepoolalloc/tests/bench/bench_compare.sh --build-dir kamepoolalloc-tests --mimalloc ~/mimalloc-bench/extern/mi3/out/release/libmimalloc.so --jemalloc ~/mimalloc-bench/extern/je/lib/libjemalloc.so --threads 128`.
  - `srun` is wrapped by `/home/system/bin/check_usage`, which mangles multi-line `bash -c '...'` arguments (syntax error at `do`). Put multi-command payloads in a script file and `srun` the file.
  - 16 KiB results vary up to ±20% across nodes/runs even with `--exclusive`; for any A/B claim, run both arms **interleaved in one srun job on one node** (same bench binary, alternate preloads) and compare medians.
  - `git push` to GitHub fails on the login node (no credential helper, no gh, no SSH key) — leave commits local and tell the user to push.