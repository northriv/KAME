# kamepoolalloc/tests/CMakeLists.txt
#
# Self-contained build for kamepoolalloc's tests.  When this directory
# is the top-level of a CMake build, the file builds the kamepoolalloc
# SHARED library and every test in this folder.  When included via
# `add_subdirectory(...)` from a parent project (KAME's
# tests/CMakeLists.txt), the parent's USE_KAME_ALLOCATOR /
# kamepoolalloc / Threads decisions propagate and we only build the
# tests themselves.

cmake_minimum_required(VERSION 3.14)
if(NOT PROJECT_NAME)
    project(kamepoolalloc_tests LANGUAGES CXX C)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Reuse parent USE_KAME_ALLOCATOR if defined; otherwise compute the
# default (same logic as the historic top-level tests/CMakeLists.txt).
if(NOT DEFINED USE_KAME_ALLOCATOR)
    # Default ON wherever the pool builds — matches production (kame.pri
    # removes USE_STD_ALLOCATOR; the pool ships on every platform, Windows
    # included).  See tests/CMakeLists.txt for the full rationale.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
        set(USE_KAME_ALLOCATOR OFF)   # pre-GCC-10 lacks the pool's C++17 atomics
    else()
        set(USE_KAME_ALLOCATOR ON)
    endif()
    message(STATUS "kamepoolalloc/tests: USE_KAME_ALLOCATOR = ${USE_KAME_ALLOCATOR}")
endif()

# Build the `kamepoolalloc` SHARED library target if it doesn't already
# exist (e.g. when this directory is built standalone).  The library
# lives one level up (`../allocator.cpp`).
if(USE_KAME_ALLOCATOR AND NOT TARGET kamepoolalloc)
    add_library(kamepoolalloc SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/../allocator.cpp)
    target_include_directories(kamepoolalloc PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_compile_definitions(kamepoolalloc PUBLIC KAMEPOOLALLOC_DYLIB)
    # Hot-path codegen: hide the (inline, COMDAT/weak) internal functions so
    # intra-DSO self-calls bind DIRECTLY instead of through the PLT.  Without
    # `-fvisibility-inlines-hidden`, `kame_free`'s call to the inline
    # `PoolAllocatorBase::deallocate` goes `call …@plt` on every free; with it
    # the symbol drops to local and the call is direct.  `-fno-semantic-
    # interposition` does the same for the non-inline self-calls
    # (`new_redirected_large`, `radix_lookup_slow`).  Neither changes which
    # symbols are exported — `malloc`/`free`/… stay global and interposable,
    # so the LD_PRELOAD drop-in semantics are untouched.
    target_compile_options(kamepoolalloc PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fvisibility-inlines-hidden;-fno-semantic-interposition>
        # MSVC on Japanese Windows (CP932 system code page) misreads UTF-8
        # box-drawing / Japanese characters in comments, potentially swallowing
        # newlines and hiding #define directives (symptom: C2065 on macros
        # defined just before a box-drawing art block).  /utf-8 fixes this.
        $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
    set_target_properties(kamepoolalloc PROPERTIES
        INSTALL_NAME_DIR "@rpath"
        BUILD_WITH_INSTALL_RPATH TRUE)
    # CMAKE_DL_LIBS: empty on glibc >= 2.34 (dlsym in libc); -ldl on
    # older glibc — needed by the Linux malloc_usable_size co-interpose's
    # RTLD_NEXT (allocator.cpp:5376).  Mirrors the same link in the
    # standalone kamepoolalloc/CMakeLists.txt:54.
    target_link_libraries(kamepoolalloc PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()
if(NOT USE_KAME_ALLOCATOR)
    add_compile_definitions(DISABLE_POOL_ALLOCATOR)
endif()

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Headers: only the library itself (`../`).  kamepoolalloc's tests are
# self-contained — they no longer pull kamestm's Qt-free test harness
# (`support_standalone.*` / `threadlocal.cpp` / `atomic_smart_ptr.h`); the
# includes were vestigial (no symbol from them was actually used), so the
# standalone repo builds with nothing but this directory.
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(test_link_libs Threads::Threads)
if(USE_KAME_ALLOCATOR)
    list(APPEND test_link_libs kamepoolalloc)
endif()

set(kamepoolalloc_test_names
    alloc_stress_test
    alloc_minimal_bench
    alloc_bucket34_repro)

foreach(t IN LISTS kamepoolalloc_test_names)
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} ${test_link_libs})
endforeach()

# (§36b) Intrusive atomic_shared_ptr + custom-disposer mode test.  Header-only
# (atomic_smart_ptr.h, relocated here) + std::thread; the intrusive mode
# allocates NO control block, so it needs neither libkamepoolalloc nor the pool.
add_executable(atomic_intrusive_dispose_test atomic_intrusive_dispose_test.cpp)
target_link_libraries(atomic_intrusive_dispose_test Threads::Threads)

# (§36b) Self-referential INTRUSIVE node in a lock-free orphan list — the
# prototype for the pool's orphan-chunk reuse.  Proves the explicit ref_traits
# specialisation forces intrusive on a self-referential type and that push /
# pop / mid-unlink are memory-safe under SMR + the custom disposer.  Header-only
# + std::thread (no libkamepoolalloc, no pool).
add_executable(atomic_intrusive_chain_test atomic_intrusive_chain_test.cpp)
target_link_libraries(atomic_intrusive_chain_test Threads::Threads)

# Pure-C smoke test of the kame_pool.h public ABI.  No C++ support
# files — exercises only the C-linkage public surface.
if(USE_KAME_ALLOCATOR)
    add_executable(c_api_test c_api_test.c)
    target_link_libraries(c_api_test kamepoolalloc Threads::Threads)
    set_property(TARGET c_api_test PROPERTY C_STANDARD 99)

    # (§27) Huge-alloc (> 32 MiB) round-trip / no-over-satisfy / MT test.
    # Uses the kame_pool_* C API directly, so it is meaningful only with the
    # pool linked in (DISABLE_POOL_ALLOCATOR has no kame_pool_malloc symbol).
    add_executable(alloc_huge_test alloc_huge_test.cpp)
    target_link_libraries(alloc_huge_test kamepoolalloc Threads::Threads)

    # Thread-churn / orphan-reuse repro for "driver start/stop -> reserved grows".
    add_executable(alloc_thread_churn_test alloc_thread_churn_test.cpp)
    target_link_libraries(alloc_thread_churn_test kamepoolalloc Threads::Threads)

    # Thread-EXIT free leak: frees a pooled buffer from a pthread_key destructor
    # during thread teardown (KAME's XThreadLocal pattern).  Asserts on
    # units_live/chunks_live (madvise-INDEPENDENT), valid on macOS.  Built BOTH
    # ways — the two linkages exercise genuinely different, both-shipping paths:
    #
    #   * STATIC (`alloc_thread_exit_free_test`) — allocator.cpp compiled IN,
    #     matching PRODUCTION (kame.app/kame.exe static-link the allocator so its
    #     operator-new override interposes ALL allocations).  Faithful repro of
    #     the owner-id==0 teardown leak (335c53b8): scenario B grows 340->1696
    #     unfixed, plateaus ~58 fixed; the in-body control stays flat.
    #
    #   * DYNAMIC (`alloc_thread_exit_free_test_dynamic`) — links the shared
    #     libkamepoolalloc (the LD_PRELOAD / macOS interpose drop-in, e.g. used by
    #     bench_compare.sh).  This linkage surfaced a SEPARATE, real dylib bug —
    #     NOT a test artifact: with malloc interposed, macOS instantiating the
    #     dylib's OWN per-thread thread_local TLV block (~32 KiB) went through the
    #     pool and leaked ~8 units/thread at thread exit (lldb root cause:
    #     kame_page_cold -> _tlv_get_addr -> dyld instantiateVariable -> malloc ->
    #     kame_pool_malloc).  Fixed in allocator.cpp (kame_page_cold parks the
    #     fast-TSD slot at the teardown sentinel so that re-entrant TLV-block
    #     malloc falls to the real heap).  This variant guards the fix: units now
    #     plateau (was unbounded, ~4 GB over hours of driver start/stop churn).
    add_executable(alloc_thread_exit_free_test
        alloc_thread_exit_free_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../allocator.cpp)
    target_compile_definitions(alloc_thread_exit_free_test PRIVATE KAMEPOOLALLOC_DYLIB)
    target_include_directories(alloc_thread_exit_free_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_link_libraries(alloc_thread_exit_free_test Threads::Threads ${CMAKE_DL_LIBS})

    add_executable(alloc_thread_exit_free_test_dynamic alloc_thread_exit_free_test.cpp)
    target_link_libraries(alloc_thread_exit_free_test_dynamic kamepoolalloc Threads::Threads)

    # Narrow-gap regression for the post-3145e139 L1 teardown guard.
    # `alloc_thread_exit_free_test` scenario B covers the SAME-thread
    # teardown free (the freer also allocated), so it arms `tls_l1` first
    # and `s_l1_drained` alone (the 30ea1daa guard) suffices.  This test
    # covers the second case: a consumer thread that NEVER armed its L1
    # recycle cache frees a CROSS-THREAD-origin > 32 KiB block from its own
    # pthread_key destructor.  Pre-3145e139 the dying consumer's
    # `recycle_push → l1_push` armed a fresh L1 that nothing drains →
    # +1 chunk/cycle permanent stranding.  Both linkages exercised, same
    # rationale as the sibling test.
    add_executable(alloc_thread_exit_unarmed_test
        alloc_thread_exit_unarmed_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../allocator.cpp)
    target_compile_definitions(alloc_thread_exit_unarmed_test PRIVATE KAMEPOOLALLOC_DYLIB)
    target_include_directories(alloc_thread_exit_unarmed_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_link_libraries(alloc_thread_exit_unarmed_test Threads::Threads ${CMAKE_DL_LIBS})

    add_executable(alloc_thread_exit_unarmed_test_dynamic alloc_thread_exit_unarmed_test.cpp)
    target_link_libraries(alloc_thread_exit_unarmed_test_dynamic kamepoolalloc Threads::Threads)

    # (§28) Cap-eviction (kame_pool_set_large_cache_cap synchronous shrink) test.
    # Verifies the two-phase K-then-N reduction, MT race-safety, and round-trip
    # behaviour.  Pool-only (uses the kame_pool_* C API directly).
    add_executable(alloc_evict_test alloc_evict_test.cpp)
    target_link_libraries(alloc_evict_test kamepoolalloc Threads::Threads)

    # macOS 16 KiB-page madvise-straddle regression (fix f0d6a487).  Forces the
    # otherwise-lazy MADV_FREE reclaim with memory pressure, then checks that a
    # released chunk's reclaim never zeroes a live neighbour chunk's header.
    # Meaningful only where page size > ALLOC_CHUNK_K_MAX (Apple arm64 = 16 KiB);
    # on Linux (4 KiB == K_MAX) it self-skips and trivially passes.
    # POSIX-only: uses sysconf(_SC_PAGESIZE) + madvise, so it does not compile
    # on Windows — exclude it there (the add_test below is already APPLE-gated).
    if(NOT WIN32)
        add_executable(alloc_madvise_straddle_repro alloc_madvise_straddle_repro.cpp)
        target_link_libraries(alloc_madvise_straddle_repro kamepoolalloc Threads::Threads)
    endif()

    # (§28.2) kame_pool_get_stats v2 tier-attribution fields (cache_bytes,
    # dedicated_chunk_bytes, large_alloc_count/bytes) instrumentation test.
    add_executable(alloc_stats_test alloc_stats_test.cpp)
    target_link_libraries(alloc_stats_test kamepoolalloc Threads::Threads)

    # (§30) kame_pool_set_realtime_mode() preset toggle — verifies the
    # bundled lazy-drain / auto-tune / thread-exit-reclaim knobs flip
    # together and restore on enable=0.
    add_executable(alloc_realtime_mode_test alloc_realtime_mode_test.cpp)
    target_link_libraries(alloc_realtime_mode_test kamepoolalloc Threads::Threads)

    # (§31/§32) Strong-symbol / interpose override correctness — asserts that
    # a bare libc malloc/calloc/realloc/free (and C++ operator new/delete)
    # actually route through the pool when the library is linked.  Oracle is
    # kame_pool_malloc_usable_size (non-zero ⇔ pool-owned).  Self-adapts to
    # platforms where plain malloc isn't interposed (musl): the C-malloc
    # assertions are gated on a runtime probe, operator-new + C-API ones run
    # unconditionally.  This is the only correctness gate for the override
    # layer that the bench/README head-to-head numbers depend on.
    add_executable(malloc_intercept_test malloc_intercept_test.cpp)
    target_link_libraries(malloc_intercept_test kamepoolalloc Threads::Threads)

    # (contrib) C++17 Allocator concept conformance for
    # `kame::pool_allocator<T>` — exercises rebind, propagate_on_*,
    # equality, allocate_shared, and STL containers (vector / list /
    # map / string).  Same Allocator surface rclcpp uses; passing here
    # is a structural pre-flight before any ROS 2 integration.
    add_executable(test_ros2_allocator ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/test_ros2_allocator.cpp)
    target_link_libraries(test_ros2_allocator kamepoolalloc Threads::Threads)

    # (contrib) `std::pmr::memory_resource` adaptor — the C++17 standard
    # interface; one adaptor covers every PMR-aware STL container plus
    # any third-party library (Boost.Container, Folly, abseil, …) that
    # consumes `std::pmr::memory_resource *`.  Auto-skips on toolchains
    # without `<memory_resource>` via `__has_include`.
    add_executable(test_pmr_resource ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/test_pmr_resource.cpp)
    target_link_libraries(test_pmr_resource kamepoolalloc Threads::Threads)

    # (contrib) Over-aligned C++17 Allocator (`kame::pool_aligned_allocator
    # <T, Align>`) for Eigen / SIMD / cacheline-aligned buffer use.
    # Drop-in for `Eigen::aligned_allocator<T>` on POSIX; auto-skips the
    # over-aligned section on Windows (current `kame_pool_aligned_alloc`
    # limitation) and still exercises the Align=16 default path.
    add_executable(test_aligned_allocator ${CMAKE_CURRENT_SOURCE_DIR}/../contrib/test_aligned_allocator.cpp)
    target_link_libraries(test_aligned_allocator kamepoolalloc Threads::Threads)

    # Tuning-diagnostic — run on the target machine to characterise VMM
    # cost / TLB shootdown / kamepoolalloc throughput, and print a
    # recommendation block.  NOT registered as a ctest (long-running,
    # machine-specific output meant for human reading) — invoke directly:
    #     ./alloc_tune_report [seconds-per-bench, default 2]
    add_executable(alloc_tune_report alloc_tune_report.cpp)
    target_link_libraries(alloc_tune_report kamepoolalloc Threads::Threads)

    # Third-party-style microbenches (mimalloc-bench / glibc-bench shapes).
    # NOT registered as ctests — perf-only.  See bench/README.md for usage.
    #
    # `bench_loop`         — single-thread hot malloc/free.
    # `bench_loop_pool`    — same loop but calls `kame_pool_malloc` directly.
    #                        This is the Windows "kame" route for the
    #                        bench_compare table: PE/COFF has no LD_PRELOAD and
    #                        llvm-mingw resolves `malloc` statically, so the
    #                        plain bench_loop stays at libc speed on Windows.
    # `bench_xthread`      — cross-thread producer/consumer (xmalloc-test
    #                        shape) routed via `malloc`/`free`.  Picks up
    #                        whatever allocator the linker resolves first
    #                        (LD_PRELOAD overrides apply).
    # `bench_xthread_pool` — same workload but calls `kame_pool_malloc`
    #                        directly.  Use the two side-by-side to
    #                        attribute deltas to the strong-symbol override
    #                        layer vs the pool core.
    add_executable(bench_loop bench/bench_loop.c)
    set_property(TARGET bench_loop PROPERTY C_STANDARD 99)
    target_link_libraries(bench_loop Threads::Threads)

    add_executable(bench_loop_pool bench/bench_loop.c)
    set_property(TARGET bench_loop_pool PROPERTY C_STANDARD 99)
    target_compile_definitions(bench_loop_pool PRIVATE LOOP_USE_KAME_POOL)
    target_link_libraries(bench_loop_pool kamepoolalloc Threads::Threads)

    add_executable(bench_xthread bench/bench_xthread.cpp)
    target_link_libraries(bench_xthread Threads::Threads)

    add_executable(bench_xthread_pool bench/bench_xthread.cpp)
    target_compile_definitions(bench_xthread_pool PRIVATE XTHREAD_USE_KAME_POOL)
    target_link_libraries(bench_xthread_pool kamepoolalloc Threads::Threads)
endif()

# --- CTest registration ---
enable_testing()
if(USE_KAME_ALLOCATOR)
    add_test(NAME c_api_test COMMAND c_api_test)
    add_test(NAME malloc_intercept_test COMMAND malloc_intercept_test)
    add_test(NAME alloc_huge_test COMMAND alloc_huge_test)
    add_test(NAME atomic_intrusive_dispose_test COMMAND atomic_intrusive_dispose_test)
    add_test(NAME atomic_intrusive_chain_test COMMAND atomic_intrusive_chain_test)
    # alloc_thread_churn scenario 2 asserts the reserved-region count PLATEAUS
    # across thread-churn cycles.  `reserved` = populated_region_count × 32 MiB
    # and is a monotonic high-water mark (regions are not munmap'd at runtime),
    # so the plateau only holds where freed chunk pages are reclaimed EAGERLY —
    # i.e. on Linux with MADV_DONTNEED.  On macOS the chunk-release path uses
    # MADV_FREE (lazy: pages return only under memory pressure, by design — it
    # avoids the reuse-heavy RSS blow-up MADV_DONTNEED would cost there), so the
    # region high-water keeps climbing in this synthetic churn and the assertion
    # trips even though there is NO real leak (the pages ARE reclaimable on
    # demand).  Windows has no madvise equivalent in this path.  So register the
    # ctest on Linux only; the executable is still BUILT everywhere (compile
    # coverage + manual `./alloc_thread_churn_test`).  See README §36 / the
    # chunk-tier reserved-drain note.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME alloc_thread_churn_test COMMAND alloc_thread_churn_test)
    endif()
    # Valid on all platforms: asserts on madvise-INDEPENDENT units_live/chunks_live.
    add_test(NAME alloc_thread_exit_free_test COMMAND alloc_thread_exit_free_test)
    add_test(NAME alloc_thread_exit_free_test_dynamic COMMAND alloc_thread_exit_free_test_dynamic)
    add_test(NAME alloc_thread_exit_unarmed_test COMMAND alloc_thread_exit_unarmed_test)
    add_test(NAME alloc_thread_exit_unarmed_test_dynamic COMMAND alloc_thread_exit_unarmed_test_dynamic)
    add_test(NAME alloc_evict_test COMMAND alloc_evict_test)
    add_test(NAME alloc_stats_test COMMAND alloc_stats_test)
    add_test(NAME alloc_realtime_mode_test COMMAND alloc_realtime_mode_test)
    add_test(NAME test_ros2_allocator COMMAND test_ros2_allocator)
    add_test(NAME test_pmr_resource COMMAND test_pmr_resource)
    add_test(NAME test_aligned_allocator COMMAND test_aligned_allocator)
    if(APPLE)
        # macOS-only (16 KiB pages): forces MADV_FREE reclaim via memory
        # pressure to make the straddle deterministic.  Args: objects, pressure_MiB.
        add_test(NAME alloc_madvise_straddle_repro
                 COMMAND alloc_madvise_straddle_repro 800000 4096)
    endif()
endif()
