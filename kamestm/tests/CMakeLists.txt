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
    set_target_properties(kamepoolalloc PROPERTIES
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
set(CMAKE_REQUIRED_FLAGS "-include ${CMAKE_CURRENT_SOURCE_DIR}/support_standalone.h")
check_cxx_source_compiles("int main(){return 0;}" KAMESTM_TESTS_HAVE_FORCE_INCLUDE)
unset(CMAKE_REQUIRED_FLAGS)
if(KAMESTM_TESTS_HAVE_FORCE_INCLUDE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-include${CMAKE_CURRENT_SOURCE_DIR}/support_standalone.h>)
endif()

set(support_SRCS
    support_standalone.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../threadlocal.cpp)

set(test_link_libs Threads::Threads)
if(USE_KAME_ALLOCATOR)
    list(APPEND test_link_libs kamepoolalloc)
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

foreach(t IN LISTS kamestm_test_names)
    add_executable(${t} ${t}.cpp ${support_SRCS})
    target_link_libraries(${t} ${test_link_libs})
endforeach()

# --- CTest registration ---
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
add_test(NAME transaction_starvation_test COMMAND transaction_starvation_test)
target_compile_definitions(transaction_starvation_test PRIVATE
    KAME_STM_LOWPRIO_STARVE_MS=2 KAME_STM_LOWPRIO_STARVE_MIN_RETRIES=2)

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
add_test(NAME transaction_priv_strip_test COMMAND transaction_priv_strip_test)
target_compile_options(transaction_priv_strip_test PRIVATE -fno-access-control)

# HIGHEST + NORMAL + UI_DEFERRABLE role mix (acquisition / demoted downstream
# with a 20 ms budget / main-thread snapshots+transactions+structural churn),
# with a stall watchdog: any thread stuck >5 s = livelock = FAIL.  10 s here;
# soak with KAME_MIX_SECS=60+ when hunting.
add_test(NAME transaction_priority_mixed_test
         COMMAND transaction_priority_mixed_test)

# Field-crash repro (2026-07-30 T1Mode SIGABRT): a stalling ms-analysis Tx
# holding never-expiring NORMAL privilege pins third parties into the HANG
# watchdog.  FAILS without the privilege-expiry fix; the fix bounds the pin.
# Deterministic half of the same fix: the expiry rules on the negotiation
# predicates themselves, both agreeing consumers.  White-box, like the strip
# test and for the same reason.
add_test(NAME transaction_priv_expiry_test COMMAND transaction_priv_expiry_test)
target_compile_options(transaction_priv_expiry_test PRIVATE -fno-access-control)

# newTransactionUsingSnapshotFor must not orphan planted stamps when it
# re-anchors the snapshot base (the 2026-07-31 field freeze).  White-box.
add_test(NAME transaction_reanchor_test COMMAND transaction_reanchor_test)
target_compile_options(transaction_reanchor_test PRIVATE -fno-access-control)

add_test(NAME transaction_priv_pin_test COMMAND transaction_priv_pin_test)

# Long-running tests
set_tests_properties(
    transaction_test
    transaction_dynamic_node_test
    transaction_negotiation_test
    transaction_payload_integrity_test
    transaction_payload_integrity_3level_test
    PROPERTIES TIMEOUT 300)
