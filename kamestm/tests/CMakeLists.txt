# kamestm/tests/CMakeLists.txt
#
# Self-contained build for kamestm's tests.  Mirrors
# kamepoolalloc/tests/CMakeLists.txt — works standalone (top-level
# project) or via add_subdirectory() from a parent.

cmake_minimum_required(VERSION 3.14)
if(NOT PROJECT_NAME)
    project(kamestm_tests LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC on a Japanese Windows (CP932) misreads the UTF-8 box-drawing and
# Japanese characters in the sources' comments: the trailing byte of some
# sequences is a valid CP932 lead byte, which swallows the following newline
# and hides the next preprocessor directive inside the comment.  Symptom:
# `error C2065: 'ALLOC_CHUNK_HEADER': undeclared identifier`.  Must be set
# BEFORE the first add_library/add_executable -- add_compile_options only
# reaches targets created after it -- and directory-wide, because every target
# that includes these headers is exposed, not just the library.
if(MSVC)
    add_compile_options(/utf-8)
endif()


# Reuse parent USE_KAME_ALLOCATOR if defined; otherwise compute default.
if(NOT DEFINED USE_KAME_ALLOCATOR)
    # Default ON wherever the pool builds — matches production (kame.pri
    # removes USE_STD_ALLOCATOR; the pool ships on every platform, Windows
    # included).  See tests/CMakeLists.txt for the full rationale.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
        set(USE_KAME_ALLOCATOR OFF)   # pre-GCC-10 lacks the pool's C++17 atomics
    else()
        set(USE_KAME_ALLOCATOR ON)
    endif()
    message(STATUS "kamestm/tests: USE_KAME_ALLOCATOR = ${USE_KAME_ALLOCATOR}")
endif()

# kamepoolalloc library target.  If the parent already provided it
# (via kamepoolalloc/tests/CMakeLists.txt), reuse.  Otherwise build
# our own copy so this directory is independently buildable.
if(USE_KAME_ALLOCATOR AND NOT TARGET kamepoolalloc)
    add_library(kamepoolalloc SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/../../kamepoolalloc/allocator.cpp)
    target_include_directories(kamepoolalloc PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../kamepoolalloc)
    target_compile_definitions(kamepoolalloc PUBLIC KAMEPOOLALLOC_DYLIB)
    # MSVC emits no import library for a DLL that exports nothing, so every
    # consumer fails with `LNK1181: cannot open input file kamepoolalloc.lib`.
    # The pool's public surface is annotated with DECLSPEC_KAMEPOOLALLOC, which
    # only kamepoolalloc.pro defines; rather than duplicate that here, take the
    # CMake equivalent of the `-Wl,--export-all-symbols` the .pro already uses
    # for MinGW.  No effect off Windows.
    set_target_properties(kamepoolalloc PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
        INSTALL_NAME_DIR "@rpath"
        BUILD_WITH_INSTALL_RPATH TRUE)
    # CMAKE_DL_LIBS: empty on glibc >= 2.34 (dlsym in libc); -ldl on
    # older glibc — needed by the Linux malloc_usable_size co-interpose's
    # RTLD_NEXT (allocator.cpp:5376).  Mirrors the same link in the
    # standalone kamepoolalloc/CMakeLists.txt:54.  Threads pulled in
    # for the Linux pthread_atfork / pthread_key_create path.
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(kamepoolalloc PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()
if(NOT USE_KAME_ALLOCATOR)
    add_compile_definitions(DISABLE_POOL_ALLOCATOR)
endif()

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Headers:
#   - kamestm/tests/ (support_standalone.h — the Qt-free harness;
#                     preincluded via -include below)
#   - kamestm/ (atomic.h, transaction.h, etc.)
#   - kamepoolalloc/ (allocator.h)
# Deliberately NO ../../kame on the path: kamestm sits BELOW kame in
# the dependency stack, and any `#include "support.h"` here must
# resolve to the standalone harness (preinclude), never to the
# Qt-aware kame/support.h.
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../kamepoolalloc)

include(CheckCXXSourceCompiles)
# The force-include is load-bearing, not a convenience: several tests open with
# `#define msecsleep(x) (x)` on line 1, and unless the header is processed
# FIRST that macro eats its own declaration
# (`void msecsleep(unsigned int)` -> `void (unsigned int)`).  MSVC spells the
# flag /FI and rejects -include, so ask it in its own dialect rather than
# silently dropping the option.
if(MSVC)
    set(KAMESTM_FORCE_INCLUDE_FLAG "/FI${CMAKE_CURRENT_SOURCE_DIR}/support_standalone.h")
else()
    set(KAMESTM_FORCE_INCLUDE_FLAG "-include${CMAKE_CURRENT_SOURCE_DIR}/support_standalone.h")
endif()
set(CMAKE_REQUIRED_FLAGS "${KAMESTM_FORCE_INCLUDE_FLAG}")
check_cxx_source_compiles("int main(){return 0;}" KAMESTM_TESTS_HAVE_FORCE_INCLUDE)
unset(CMAKE_REQUIRED_FLAGS)
if(KAMESTM_TESTS_HAVE_FORCE_INCLUDE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${KAMESTM_FORCE_INCLUDE_FLAG}>)
else()
    message(FATAL_ERROR
        "kamestm/tests: the compiler rejected ${KAMESTM_FORCE_INCLUDE_FLAG}; "
        "support_standalone.h must be force-included (see above).")
endif()

set(support_SRCS
    support_standalone.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../threadlocal.cpp)

set(test_link_libs Threads::Threads)
if(USE_KAME_ALLOCATOR)
    list(APPEND test_link_libs kamepoolalloc)
endif()

# Does this target have a lock-free 64-bit atomic?  Mirrors exactly the test
# transaction_definitions.h uses to auto-enable KAME_STM_COMPACT_STATE, so the
# answer here and the layout the library compiles cannot disagree.  False on
# i386/i486, ARMv5/v6, RV32, MIPS32 (and on -march=i486: CMPXCHG8B is
# i586/Pentium+).  See tools/audit/check_no_dcas.sh, which guards the library
# side of the same invariant.
check_cxx_source_compiles("
#include <atomic>
#if ATOMIC_LLONG_LOCK_FREE != 2
#error no lock-free 64-bit atomic
#endif
int main(){return 0;}" KAMESTM_HAVE_LOCKFREE_64)

if(NOT KAMESTM_HAVE_LOCKFREE_64)
    # Compact mode seals the stamp's PRIO field (STAMP_HIGHEST_MASK == 0), so
    # `is_highest_stamp` / `is_priv_stamp` are constant-false and Rules 0 /
    # 0c / 0d never fire: HIGHEST is simply not identifiable, and a HIGHEST Tx
    # can be preempted by a NORMAL one.  That is the documented trade for
    # running without DCAS -- data integrity is unaffected (the PacketWrapper
    # CAS below does not consult the stamp), only the negotiation heuristics
    # are.  These four assert the privilege machinery end-to-end, so they are
    # testing a mechanism this target does not have.
    set(kamestm_tests_need_prio
        transaction_starvation_test
        transaction_priv_strip_test
        transaction_highest_older_wins_test
        transaction_priv_expiry_test)
    # These carry 64-bit atomics of their OWN (latency sums, per-thread
    # progress counters) and so do not even link here.  Left as-is on purpose:
    # a bench legitimately wants a 64-bit tally, the same call already made for
    # g_alloc_size_histo in the allocator.  The library must be no-DCAS clean;
    # its instruments need not be.
    set(kamestm_tests_need_atomic64
        transaction_priv_pin_test
        transaction_priority_mixed_test
        transaction_latency_bench)
    message(STATUS
        "kamestm/tests: no lock-free 64-bit atomic (KAME_STM_COMPACT_STATE) -- "
        "excluding priority tests: ${kamestm_tests_need_prio}; "
        "and 64-bit-atomic tests: ${kamestm_tests_need_atomic64}")
endif()

set(kamestm_test_names
    fast_vector_test
    atomic_shared_ptr_test
    atomic_scoped_ptr_test
    atomic_queue_test
    mutex_test
    xwaitcell_test
    transaction_test
    transaction_dynamic_node_test
    transaction_negotiation_test
    transaction_payload_integrity_test
    transaction_payload_integrity_3level_test
    transaction_payload_integrity_mixed_test
    transaction_payload_integrity_3level_mixed_test
    transaction_wait_budget_test
    transaction_starvation_test
    transaction_sleep_in_tx_test
    transaction_priv_strip_test
    transaction_highest_older_wins_test
    transaction_priv_expiry_test
    transaction_reanchor_test
    transaction_priv_pin_test
    transaction_priority_mixed_test
    # Per-commit LATENCY tail (not throughput).  Pure observation: it times
    # iterate_commit from outside and changes nothing in kamestm, so it can
    # neither regress the library nor bias what it measures.  Not a ctest --
    # absolute latencies are machine-specific; run it deliberately.
    transaction_latency_bench
    # The control for the two above: the same clock and the same histogram
    # with the STM removed, so a tail can be attributed to the machine before
    # anyone instruments commit().  Not a ctest for the same reason
    # transaction_latency_bench is not — the numbers are the host's.
    latency_floor)

if(NOT KAMESTM_HAVE_LOCKFREE_64)
    list(REMOVE_ITEM kamestm_test_names
        ${kamestm_tests_need_prio} ${kamestm_tests_need_atomic64})
endif()

foreach(t IN LISTS kamestm_test_names)
    add_executable(${t} ${t}.cpp ${support_SRCS})
    target_link_libraries(${t} ${test_link_libs})
endforeach()

# NOTE: every add_test() / target_*() below for a test named in
# kamestm_tests_need_prio or kamestm_tests_need_atomic64 must sit inside an
# `if(TARGET ...)`, since compact-mode targets were dropped from the list
# above and referencing one is a hard configure error.

# Stage `with_pmqos` next to the binaries it wraps.  It lives in the source
# tree, every absolute latency number from latency_floor /
# transaction_latency_bench / transaction_priority_mixed_test is only
# quotable *with* it (see the measured floor table in the script itself),
# and reaching back across trees for it has twice produced a silent
# `sudo: './with_pmqos': command not found` -- silent because the usual
# invocation pipes stderr into a grep.  Copied rather than symlinked so a
# build tree survives the source tree moving; copy_if_different keeps the
# exec bit and refreshes on edit.
add_custom_target(with_pmqos ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/with_pmqos
            ${CMAKE_CURRENT_BINARY_DIR}/with_pmqos
    COMMENT "Staging with_pmqos next to the latency binaries")

# --- CTest registration ---
#
# Three configurations are meaningful and all three have been run (2026-08-12):
#   Release          40/40.
#   Debug            40/40 -- every library assertion live across every test,
#                    including STRICT_assert, the CAS-fail-twice invariant and
#                    the 3L rebuild bound.  Worth running after any change to
#                    the negotiation: Release compiles all of them out.
#   ASan + UBSan     -DUSE_KAME_ALLOCATOR=OFF (the pool interposes malloc and
#                    cannot coexist with ASan), 20/22 with ZERO sanitizer
#                    diagnostics.  The two failures are timeouts, not findings:
#                    ASan costs ~10x, so transaction_dynamic_node_test needs
#                    well over ten minutes, and transaction_nosyscall_highest_test
#                    cannot run under ASan at all (see its note below).
enable_testing()
# Union discipline of fast_vector<T,N>: the inline m_array and the heap
# m_vector are union'd, so any method reaching for the inactive member is
# UB.  Pins the inverted is_fixed() test in shrink_to_fit() that produced a
# sporadic SIGSEGV via Talker::disconnect.  Byte-pattern dependent by
# nature -- run under -fsanitize=address,undefined for full strength.
add_test(NAME fast_vector_test COMMAND fast_vector_test)
add_test(NAME atomic_shared_ptr_test COMMAND atomic_shared_ptr_test)
add_test(NAME atomic_scoped_ptr_test COMMAND atomic_scoped_ptr_test)
add_test(NAME atomic_queue_test COMMAND atomic_queue_test)
add_test(NAME mutex_test COMMAND mutex_test)
# The backend behind negotiate_sleep's parking is chosen at compile time
# (__ulock / futex / mutex+condvar) and nothing else exercises it directly.
add_test(NAME xwaitcell_test COMMAND xwaitcell_test)
add_test(NAME transaction_test COMMAND transaction_test)
add_test(NAME transaction_dynamic_node_test COMMAND transaction_dynamic_node_test)
add_test(NAME transaction_negotiation_test COMMAND transaction_negotiation_test)
add_test(NAME transaction_payload_integrity_test COMMAND transaction_payload_integrity_test)
add_test(NAME transaction_payload_integrity_3level_test COMMAND transaction_payload_integrity_3level_test)
# Asserts that a ScopedWaitBudget commit finishes within budget + slack.  The
# slack (default 10 ms, KAME_WB_TEST_SLACK_US) covers OS scheduling and
# post-expiry retries — everything the library deliberately does NOT model.
add_test(NAME transaction_wait_budget_test COMMAND transaction_wait_budget_test)
# Starvation bound on revocable priorities.  Built at 2 ms rather than the
# production 1000 ms so a 4 s arm reliably reaches it: measured, 20 ms fired only
# 1 of 3 runs while 2 ms — the value the latency bench fired at consistently —
# is dependable.  The production value is exercised by NOT firing, which the
# uncontended arm covers.
if(TARGET transaction_starvation_test)
    add_test(NAME transaction_starvation_test COMMAND transaction_starvation_test)
    target_compile_definitions(transaction_starvation_test PRIVATE
        KAME_STM_LOWPRIO_STARVE_MS=2 KAME_STM_LOWPRIO_STARVE_MIN_RETRIES=2)
endif()

# Debug-only detector for msecsleep() inside a Transaction.  -UNDEBUG is not
# optional: this tree is configured Release, and without it the whole detector
# and the whole test compile to nothing and the test passes having checked
# nothing.  Ordered after the definitions above so the flag applies to
# ${support_SRCS} compiled into this target too.
add_test(NAME transaction_sleep_in_tx_test COMMAND transaction_sleep_in_tx_test)
target_compile_options(transaction_sleep_in_tx_test PRIVATE -UNDEBUG)

# White-box test of tag_as_contender's Rule 0 (HIGHEST strips a stuck foreign
# non-HIGHEST privilege stamp).  -fno-access-control because it must plant a
# synthetic Reserved stamp on Node's private Linkage: privilege claims are
# probe-gated and cannot be manufactured deterministically through the public
# API, and the benchmarks prove only that the rule stays silent when healthy.
if(TARGET transaction_priv_strip_test)
    add_test(NAME transaction_priv_strip_test COMMAND transaction_priv_strip_test)
    target_compile_options(transaction_priv_strip_test PRIVATE -fno-access-control)
endif()

# HIGHEST-vs-HIGHEST older-wins has teeth (2026-08-11): the loser defers to a
# live privileged HIGHEST peer on the same predicate a NORMAL loser sleeps on,
# spinning instead of parking.  Same white-box need as the strip test.
if(TARGET transaction_highest_older_wins_test)
    add_test(NAME transaction_highest_older_wins_test
             COMMAND transaction_highest_older_wins_test)
    target_compile_options(transaction_highest_older_wins_test
                           PRIVATE -fno-access-control)
endif()

# HIGHEST + NORMAL + UI_DEFERRABLE role mix (acquisition / demoted downstream
# with a 20 ms budget / main-thread snapshots+transactions+structural churn),
# with a stall watchdog: any thread stuck >5 s = livelock = FAIL.  10 s here;
# soak with KAME_MIX_SECS=60+ when hunting.
if(TARGET transaction_priority_mixed_test)
    add_test(NAME transaction_priority_mixed_test
             COMMAND transaction_priority_mixed_test)
endif()
# The acquisition tier must not reach the kernel, and two of the three risks
# are invisible in the source: `now_us()` is a vDSO read only while the host
# clocksource is `tsc` (switch it to `hpet` and every commit traps, with
# nothing in the code having changed), and the pool allocator is syscall-free
# only while warm.  Neither is a property of the program text.  This arm puts
# a per-thread seccomp filter on the acquisition thread and FAILS on any call
# trapped while the thread is at Priority::HIGHEST -- the demoted NORMAL
# phase of the same thread is counted separately and allowed, since only the
# HIGHEST path carries the requirement.  Short by design: it is a per-commit
# property, so it does not need a soak.
# NOT sanitizer-compatible, by construction: ASan issues syscalls of its own
# (mmap/madvise and friends) that this filter traps and SKIPS, so the process
# wedges rather than reporting anything.  Under -fsanitize=address run the
# rest of the suite and leave this arm out; it is checking a property of the
# production binary anyway.
# Same binary as transaction_priority_mixed_test above, so it goes with it.
if(TARGET transaction_priority_mixed_test)
    add_test(NAME transaction_nosyscall_highest_test
             COMMAND transaction_priority_mixed_test)
    set_tests_properties(transaction_nosyscall_highest_test PROPERTIES
        ENVIRONMENT "KAME_MIX_NOSYSCALL=1;KAME_MIX_SECS=6"
        TIMEOUT 120)
endif()

# Field-crash repro (2026-07-30 T1Mode SIGABRT): a stalling ms-analysis Tx
# holding never-expiring NORMAL privilege pins third parties into the HANG
# watchdog.  FAILS without the privilege-expiry fix; the fix bounds the pin.
# Deterministic half of the same fix: the expiry rules on the negotiation
# predicates themselves, both agreeing consumers.  White-box, like the strip
# test and for the same reason.
if(TARGET transaction_priv_expiry_test)
    add_test(NAME transaction_priv_expiry_test COMMAND transaction_priv_expiry_test)
    target_compile_options(transaction_priv_expiry_test PRIVATE -fno-access-control)
endif()

# newTransactionUsingSnapshotFor must not orphan planted stamps when it
# re-anchors the snapshot base (the 2026-07-31 field freeze).  White-box.
add_test(NAME transaction_reanchor_test COMMAND transaction_reanchor_test)
target_compile_options(transaction_reanchor_test PRIVATE -fno-access-control)

if(TARGET transaction_priv_pin_test)
    add_test(NAME transaction_priv_pin_test COMMAND transaction_priv_pin_test)
endif()

# Long-running tests
set_tests_properties(
    transaction_test
    transaction_dynamic_node_test
    transaction_negotiation_test
    transaction_payload_integrity_test
    transaction_payload_integrity_3level_test
    PROPERTIES TIMEOUT 300)
