# Windows MinGW pool handoff

Handoff context for a Claude Code session running locally on a MinGW64 +
lld Windows machine.  Linux / macOS sessions have been driving the
kamepoolalloc work; the Windows-only failures below need on-target debug
access (build the EXE, attach a debugger, see actual error addresses /
stack traces, iterate quickly).

This document is intentionally terse — read it once, then start
debugging.  Nothing in here is end-user documentation.

## Reported symptoms (current master, MinGW64 + lld)

1. **`kame.exe` crashes at `kame/xnodeconnector.h:87` when the user adds
   the test driver from the UI.**  Line 87 is just the `struct QForm`
   declaration; the actual fault is somewhere inside QForm's constructor
   body (lines 89–93) — `setupUi(this)` or `installEventFilter`.
   Compiler line attribution lies for templates.

   The pool DOES activate in kame.exe (the `Reserve swap space ...`
   message from `mmap_new_region` appears).  So the activator runs and
   chunk-claim works in the EXE itself.  Something in the alloc / dealloc
   or freed-memory state corrupts the QForm allocation.

2. **Test binaries (link `libkamepoolalloc.dll`) run to completion but
   the `Reserve swap space` message never appears.**  This means
   `cold_first_access` is always falling through to `std::malloc` —
   `g_sys_image_loaded` is `false` at every call.  Inside the DLL the
   auto-activator constructor isn't taking effect.

## What's been tried (don't redo these)

| Commit | Attempt | Result |
|---|---|---|
| `bd92d0c7` | Drop `tls_model("initial-exec")` on Windows (allocator_prv.h) — hypothesis: IE-TLS attribute breaks across the EXE/DLL boundary on Windows ABI. | No change. |
| `004b8aea` | Two diagnostic levers: (a) `[[gnu::used]]` + file-scope global `KamePoolAutoActivator` as a static-init backup for the dylib auto-activator, in case lld drops the `__attribute__((constructor(101)))` record.  (b) `KAME_POOL_DISABLE_PREFILL=1` env-var to disable §29 freelist pre-fill at runtime. | (a) no change — tests still skip pool.  (b) no change — kame.exe still crashes. |

So we know:
- §29 pre-fill is NOT the crash cause (still crashes with `KAME_POOL_DISABLE_PREFILL=1`).
- IE-TLS attribute was not the (only) issue.
- Static-init in the DLL is silently not flipping `g_sys_image_loaded`,
  OR `g_sys_image_loaded` is flipping but the DLL's `operator new` /
  `kame_pool_*` are never called from the test binary.

## Reproduction

```sh
# build via qmake (MinGW64 shell)
cd <repo>
qmake -r
mingw32-make -j

# kame.exe crash repro
./kame.app/kame.exe   # or wherever it lands on win
# UI → Add test driver → SEGV at xnodeconnector.h:87
```

```sh
# test binaries
cd kamepoolalloc/tests/<build dir>
./alloc_stress_test.exe 2>&1 | head -5
# Expected on Linux/macOS:  "Reserve swap space starting @ 0x... w/ len. of 0x2000000B (node 0)."
# On Windows MinGW currently:  no Reserve message → test runs in libc fallback mode.
```

## First-cut diagnostic steps

### (1) Confirm whether the DLL static-init runs at all

Add a debug print to the auto-activator (commit 004b8aea defined both
the `__attribute__((constructor(101)))` and a backup global `static
KamePoolAutoActivator` — at least one of them should run):

```cpp
// kamepoolalloc/allocator.cpp around line 736
[[gnu::used]] __attribute__((constructor(101)))
static void kamepoolalloc_auto_activate() noexcept {
    fprintf(stderr, "[kamepool] ctor(101) activated\n");  // ADD
    g_sys_image_loaded = true;
}
```

And the backup ~line 750:

```cpp
struct KamePoolAutoActivator {
    KamePoolAutoActivator() noexcept {
        fprintf(stderr, "[kamepool] static-init activated\n");  // ADD
        g_sys_image_loaded = true;
    }
};
```

Run any test binary.  Three possible outcomes:

- **Both messages print** ⇒ activation IS happening; the bug is that
  the test binary's allocations don't reach the DLL's `operator new` /
  `kame_pool_malloc`.  See (2).
- **Only the static-init message prints** ⇒ lld drops the `(constructor)`
  attribute record.  Activation still works via the backup — but the
  user-side allocations still bypass the pool.  See (2).
- **Neither message prints** ⇒ DLL static-init isn't running.  Check
  DllMain emission (`-Wl,--enable-auto-import`, qmake `CONFIG += plugin`
  vs `CONFIG += shared`).

### (2) Confirm whether `operator new` is being routed through the DLL

A test binary's `new T()` call should hit the DLL's
`operator new(size_t)` (allocator.cpp:4209).  Add a print:

```cpp
__attribute__((noinline))
void* operator new(std::size_t size) {
    static bool s_seen = false;
    if(!s_seen) { fprintf(stderr, "[kamepool] op new size=%zu\n", size); s_seen = true; }
    KAME_HISTO_REC(size);
    if(void *p = kame_alloc_with_handler(size)) return p;
    throw std::bad_alloc();
}
```

If the print never appears, the test binary's allocations go to
libstdc++.dll's default `operator new`, not the DLL's override.  That
points to the C++ replacement-function linkage issue on Windows PE/COFF
(strong-symbol override across DLL boundaries does NOT work the way
ELF / Mach-O symbol interposition does — each loaded module has its
own import table for `operator new`, bound at module load time to
libstdc++.dll's default).

If that's the failure: tests need to be **inline-compiled with
allocator.cpp**, not linked against the DLL.  See `alloc_stress_test.pro`
for the standalone-build pattern (`else { ... }` branch); making the
qmake test path on Windows take that same inline route is one option.
The cmake test scaffold already builds a DLL though, so this needs to
be solved properly for cross-build parity.

### (3) The kame.exe crash

kame.exe's pool activates (Reserve prints), so the `operator new`
override there IS being called for at least some allocations.  The
crash happens with the test driver's QForm construction.

A focused diagnostic — add prints to QForm's constructor as a single
diff:

```cpp
// kame/xnodeconnector.h:87 area
template <class FRM, class UI>
struct QForm : public FRM, public UI {
    template <typename...Args>
    QForm(Args&&...args) : FRM(std::forward<Args>(args)...), UI() {
        fprintf(stderr, "[kame] QForm@%p ctor entry, sizeof(QForm)=%zu\n",
                (void*)this, sizeof(*this));
        this->setupUi(this);
        fprintf(stderr, "[kame] QForm@%p setupUi done\n", (void*)this);
        if(g_pFrmMain && this->isWindow())
            this->installEventFilter(g_pFrmMain);
        fprintf(stderr, "[kame] QForm@%p ctor done\n", (void*)this);
    }
};
```

The print(s) before the SEGV pinpoint the failing step.  Most likely
candidates:
- `setupUi(this)` — Qt's UI-generated form code does many small allocs;
  if any go to a mismatched alloc/free path (libc-alloced, pool-freed
  or vice versa via the radix lookup), the heap corrupts.
- `g_pFrmMain` access — if `DECLSPEC_KAME` import doesn't resolve right,
  this is a wild read.

### (4) Hot suspects (after the prints narrow things down)

- **Mismatched `operator new` / `operator delete`** across the
  libstdc++.dll boundary.  Windows PE/COFF doesn't share the
  replacement override across DLLs the way ELF/Mach-O does.  This is the
  classic Windows gotcha for `operator new` overrides.
- **TLS layout mismatch** between kame.exe's inline-compiled
  `g_thread_freelist_ptr[]` and any reader from a Qt worker thread.
  Currently `__thread` (no IE attribute on Windows since bd92d0c7).
- **`__sync_*` builtins on lld** — verify they emit correct LOCK-
  prefixed instructions, not just unordered loads/stores.

### (5) Fallback: revert Windows to `USE_STD_ALLOCATOR`

Pre-folder-move (`a80af9de`), Windows used `USE_STD_ALLOCATOR` for ALL
toolchains.  Restoring that one-line carve-out gives KAME stability on
Windows at the cost of no pool benefits there.  Tests on Windows would
also degrade to std::allocator (so Windows tests stop exercising the
pool — pool validation remains via Linux x86-64 / Linux m32 / macOS
ctest).

The patch is trivial (allocator.h:97):

```cpp
#if (defined(_WIN32) || defined(WINDOWS))
    // Pool not yet validated on Windows MinGW + lld; see this handoff.
    #define USE_STD_ALLOCATOR
#endif
```

If on-target debug doesn't converge in a session or two, recommend this
as the pragmatic stop-bleeding path.  KAME on Windows worked this way
for 20+ years.

## Files / line numbers index

| File | Lines | Purpose |
|---|---|---|
| `kamepoolalloc/allocator.h` | 84–99 | `USE_STD_ALLOCATOR` arch carve-out |
| `kamepoolalloc/allocator_prv.h` | 122–157 | `ALLOC_TLS` / `ALLOC_TLS_IE` macros (Windows path landed `bd92d0c7`) |
| `kamepoolalloc/allocator.cpp` | 722–771 | `g_sys_image_loaded` activation paths (DYLIB ctor + static-init backup) |
| `kamepoolalloc/allocator.cpp` | 843–931 | §29 pre-fill (toggle via `KAME_POOL_DISABLE_PREFILL`) |
| `kamepoolalloc/allocator.cpp` | 4209+ | `operator new` / `operator new[]` overrides |
| `kame/xnodeconnector.h` | 86–94 | `QForm` template — Windows crash site |
| `kame/main.cpp` | 32–39 | `kame_pool_set_realtime_mode(1)` call gated by `#ifndef USE_STD_ALLOCATOR` |
| `kame/kame.pro` | 172 | `SOURCES += ../kamepoolalloc/allocator.cpp` (inline-compile, all platforms) |
| `kamepoolalloc/kamepoolalloc.pro` | 56–66 | `win32-*g++` `-Wl,--export-all-symbols -Wl,--out-implib,...` |

## Working environment notes

- Master is the development branch (declared by the user, sister sessions
  push there).  Always `git pull --rebase origin master` before push.
- ctest scaffold is at `kamepoolalloc/tests/CMakeLists.txt` — Linux /
  macOS only (cmake hasn't been validated on Windows here).
- qmake builds for Windows go through `kame.pro` → SUBDIRS chain.

Good luck.
