# Windows MinGW pool — RESOLVED (free-family IAT redirect)

**Status: RESOLVED** (2026-06-01, commit `1a4f6ee3` + test/doc follow-up).
The kame.exe crash and the pool-coexistence problem on MinGW64 + lld are
fixed by a runtime free-family IAT redirect in
`kamepoolalloc/allocator.cpp` (§31).  This doc records the root cause, the
fix, and what remains.  The original "here are the symptoms, try these
diagnostic prints, else fall back to `USE_STD_ALLOCATOR`" content is
superseded — see git history (commit `83f865fb`) if you need it.

## Root cause

PE/COFF has **no cross-module interposition** of `operator new` /
`operator delete` / `free` — the ELF strong-symbol and Mach-O
`__DATA,__interpose` mechanisms KAME's pool relies on elsewhere simply do
not exist on Windows.  Each prebuilt DLL (Qt6*.dll, libc++.dll) binds
`free` / `operator delete` to the UCRT at *its own* link time.  kame.exe
inline-compiles `allocator.cpp`, so ITS `new`/`delete` are the pool's —
but Qt and libc++ keep calling the UCRT.

So a widget kame.exe pool-allocates and hands to Qt is freed by Qt via the
UCRT's `free()` → heap corruption (a pool pointer is not a CRT pointer).

Confirmed on-target with lldb: the crash was **not** in QForm construction
(this doc's original guess) but in **`~QForm` destruction** of the
Create-Driver dialog.  Its child widgets, pool-allocated in kame.exe, are
deleted by `QObjectPrivate::deleteChildren()` *inside Qt6Core.dll* →
libc++ `operator delete` → `free()` → `int3` in ntdll's heap path.
Backtrace spine: `XQButtonConnector::onClick` (xnodeconnector.cpp:150) →
`XDriverListConnector::onCreateTouched` (driverlistconnector.cpp:227) →
`qshared_ptr<QForm>` dtor → `sharedPtrQDeleter_` `delete` → `~QForm`.  The
faulting object and the clobbered return address both sat inside the
pool's reserved region.

Symptom #2 (test binaries never printed "Reserve swap space"): they linked
`libkamepoolalloc.dll`, whose `operator new` PE/COFF won't let consumers
bind to — so they ran entirely on libc++ and never touched the pool.

## The fix

1. **Free-family IAT redirect** — `allocator.cpp` §31, all under
   `#if defined(_WIN32)...`.  At pool activation, patch the `free` /
   `realloc` / `_msize` import slots of every loaded UCRT-family module so
   every `free()` in the process funnels through the pool's existing
   pool-or-foreign dispatcher (`kame_free`).  Pool pointers get pool-freed
   wherever Qt frees them; genuine CRT pointers forward to the real
   ucrtbase `free`.  This is the PE/COFF analogue of the ELF strong-symbol
   `free` shim / Mach-O `__interpose` — the approach `mimalloc-redirect.dll`
   takes, scoped to the deallocation family.
   - **Installed from BOTH activation paths**, so it fires no matter how a
     binary activates the pool: `activateAllocator()` (inline-compiled
     kame.exe) AND the dylib `__attribute__((constructor))` auto-activator
     (`KAMEPOOLALLOC_DYLIB` builds — the standalone/inline tests, and the
     real DLL).  Install hooks `free` BEFORE flipping `g_sys_image_loaded`,
     so the first pool pointer handed out is already safe to free anywhere.
     Idempotent (once-guard), so the two paths can't double-install.
   - **Module enumeration walks the PEB loader list directly**, NOT
     `EnumProcessModules` — the latter re-enters the loader and deadlocks
     when the auto-activator runs inside the DLL's DllMain (loader lock
     held).  The PEB walk takes no lock, so install is safe from both
     CRT-startup (EXE) and DllMain (DLL) contexts.
   - `malloc` is deliberately **not** redirected: crash-safety only needs
     *frees* reconciled.  KAME's hot-path allocations keep coming from the
     pool (kame.exe `operator new`), Qt's keep coming from the CRT.
   - Later-loaded DLLs are patched via `LdrRegisterDllNotification`.
   - Kill switch: `KAME_POOL_WIN_REDIRECT=0`.  Verbose:
     `KAME_POOL_VERBOSE=1` prints the patched-slot count (otherwise quiet,
     but always warns if it patched 0 slots).
   - Two hazards handled during bring-up: **(a) recursion** — the pool's
     own foreign-forward paths (`libsystem_*_for_pool`) and region release
     (`free_munmap`) call genuine `g_real_*` pointers, not `std::free`,
     since kame.exe's own `free` import is patched; **(b) cross-CRT** —
     only `ucrtbase` / `api-ms-win-crt-heap` modules are patched, never
     `msvcr*` (Ruby's `x64-msvcrt-ruby340.dll` lives on the legacy msvcrt
     heap; freeing an msvcrt pointer with ucrtbase `free` corrupts).

2. **Symptom #2** — `kamepoolalloc/tests/alloc_stress_test.pro` now takes
   the inline-compile path on Windows (the same one its standalone branch
   uses) instead of linking the DLL, so the allocator test gets its own
   pool `operator new` and actually exercises the pool.

## Verified on-target (MinGW64 + lld, Qt 6.10.1 llvm-mingw)

- kame.exe: clean startup; redirect patched 66 slots across 58 modules
  (identical via PEB walk and the earlier EnumProcessModules); pool
  activates ("Reserve swap space").  Adding the test driver no longer
  crashes; a manual load test ran crash-free; process exits status 0.
- `alloc_stress_test` (now inline-compiled on Windows): redirect installs
  from the auto-activator (5 slots / 3 modules), pool activates, and the
  full stress **PASSES** — 2000 threads, 42.6 M ops, 3.94 M ops/s,
  allocs==frees, 0 sentinel failures, chunk count drains 1059→4.
- `transaction_test` (links the DLL): passes with the rebuilt DLL — the
  DllMain-context install path does not deadlock.

## Still open

- **STM tests** (`kamestm/tests/*`) still link the DLLs and run on libc++
  on Windows (their `operator new` binds to libc++, not the DLL's pool —
  the same PE/COFF export limitation).  They test STM logic, not the pool,
  so this is low priority.  To make them exercise the pool, inline-compile
  `allocator.cpp` into them via a `tests/tests.pri` `win32` branch (as
  `alloc_stress_test.pro` now does); the auto-activator would then install
  the free-redirect automatically to reconcile the `kamestm.dll` boundary.
  Deferred — and verified harmless to leave as-is (`transaction_test`
  passes against the rebuilt DLL).
- **Aligned family** (`_aligned_free` / `_aligned_realloc` /
  `_aligned_msize`) is not redirected.  A pool-allocated *over-aligned*
  object freed by Qt via `_aligned_free` would still mismatch — but the
  pool's C-API rejects over-aligned requests on Windows
  (`kame_pool_aligned_alloc`), so this shouldn't arise.  First place to
  look if a teardown crash appears with heavy SIMD/over-aligned Qt types.
- cmake test scaffold (`tests/CMakeLists.txt`) remains Linux/macOS only.

## Files / line index

| File | What |
|---|---|
| `kamepoolalloc/allocator.cpp` (§31 blocks, `#if _WIN32`) | g_real_* bypass + resolver; `kame_iat_free/realloc/msize`; `kame_patch_one_module` / `kame_patch_all_modules`; `LdrRegisterDllNotification` hook; `kame_pool_win_install_redirect` |
| `kamepoolalloc/allocator.cpp` `activateAllocator()` | calls `kame_pool_win_install_redirect()` before flipping `g_sys_image_loaded` |
| `kamepoolalloc/allocator.cpp` `libsystem_*_for_pool` / `free_munmap` | Windows branches call `g_real_*` (recursion bypass) |
| `kamepoolalloc/tests/alloc_stress_test.pro` | win32 → inline-compile path |
| `kame/xnodeconnector.h` `QForm` / `sharedPtrQDeleter_` | the crash site (dialog teardown) |

## Build & on-target debug (this machine)

```sh
# Build kame.exe from the Bash tool (SHELL=cmd.exe is REQUIRED — the qmake
# recipes contain parens in -D defines that Git Bash's /usr/bin/sh chokes
# on; cmd.exe handles them).
export PATH="/c/Qt/Tools/llvm-mingw1706_64/bin:/c/Qt/Tools/mingw1310_64/bin:/c/Qt/6.10.1/llvm-mingw_64/bin:$PATH"
cd build/Desktop_Qt_6_10_1_llvm_mingw_64_bit-Debug/kame
mingw32-make -f Makefile.Debug SHELL=cmd.exe
```

```sh
# Debug a GUI crash with lldb: a stop-hook auto-prints the backtrace on any
# stop (plain `-b -o run` drops to a prompt before the queued bt runs).
cd build/Desktop_Qt_6_10_1_llvm_mingw_64_bit-Debug   # cwd = build root (modules/, resources/)
# PATH per kame.bat: Qt llvm-mingw bin + tools bin + mingw_64 bin
lldb -o 'target stop-hook add --one-liner "thread backtrace all"' -o run -- ./kame.exe
# then add the test driver from the UI; lldb captures the crash backtrace.
```

Notes: `kame.exe <file.kam>` loads a measurement at startup, but `openMes`
runs *before* driver modules load, so a `.kam` referencing a module driver
is a no-op at load — the crash needs the driver added interactively.  Run
allocator tests from `kamepoolalloc/tests/<build>/`.
