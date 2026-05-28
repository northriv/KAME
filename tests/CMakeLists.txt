cmake_minimum_required(VERSION 3.14)
project(kame_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../kame
    ${CMAKE_CURRENT_SOURCE_DIR}/../kamepoolalloc)

# Standalone support: replaces support.h + xtime.h with std::chrono.
# -include preinclude so include guards block the Qt-dependent originals.
# AOCC 4.0 crashes on -include; fall back to requiring each .cpp to
# #include "support_standalone.h" as its first include (its guards block
# the real support.h/xtime.h).
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-include ${CMAKE_CURRENT_SOURCE_DIR}/support_standalone.h")
check_cxx_source_compiles("int main(){return 0;}" HAVE_FORCE_INCLUDE)
unset(CMAKE_REQUIRED_FLAGS)
if(HAVE_FORCE_INCLUDE)
    add_compile_options(-include ${CMAKE_CURRENT_SOURCE_DIR}/support_standalone.h)
else()
    message(STATUS "Compiler does not support -include; each .cpp must #include support_standalone.h first")
endif()

set(support_SRCS
    support_standalone.cpp
    # `detail::tls_storage()` is the type-erased TLS dispatcher every
    # `XThreadLocal<T, Tag>` calls into.  In the real build this lives
    # in kame.dll; the standalone test binary needs its own copy.
    ${CMAKE_CURRENT_SOURCE_DIR}/../kame/threadlocal.cpp)

# KAME pool allocator: ON by default on Linux and macOS (incl. Apple
# Silicon — the pool now uses __builtin_ctzll and the ARM8 dmb barriers
# instead of the x86 bsf inline-asm). OFF on Windows (Qt build provides
# it) and on GCC <10 where std feature support is patchy.
# Override with: cmake .. -DUSE_KAME_ALLOCATOR=ON (or OFF)
if(NOT DEFINED USE_KAME_ALLOCATOR)
    if(WIN32)
        set(USE_KAME_ALLOCATOR OFF)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
        set(USE_KAME_ALLOCATOR OFF)
    else()
        set(USE_KAME_ALLOCATOR ON)
    endif()
endif()
message(STATUS "USE_KAME_ALLOCATOR = ${USE_KAME_ALLOCATOR}")
if(USE_KAME_ALLOCATOR)
    # Build the pool allocator as a SHARED library (kamepoolalloc).
    # Rationale:
    #   1. dyld processes `__DATA,__interpose` sections only from
    #      MH_DYLIB images, so the `free` interpose that catches
    #      libcxx/libsystem_pthread thread-exit destructions cannot
    #      live in the test executable — it must live in a dylib.
    #   2. Future standalone publication: kamepoolalloc is a
    #      self-contained allocator dylib, mirrors mimalloc/jemalloc
    #      packaging.
    # The tests link against the dylib via `target_link_libraries`;
    # the activator (tests/allocator.cpp) just calls
    # `activateAllocator()` from a static-init object so the pool is
    # enabled before main().
    add_library(kamepoolalloc SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/../kamepoolalloc/allocator.cpp)
    target_include_directories(kamepoolalloc PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../kamepoolalloc)
    # PUBLIC: define propagates to every test linking against the dylib.
    # In dylib mode the allocator is auto-activated via a
    # `__attribute__((constructor))` inside libkamepoolalloc itself (runs
    # at dylib load, before the main exe's static init), so the explicit
    # `activateAllocator()` call site / `tests/allocator.cpp` activator
    # shim becomes obsolete.  `KamePooledAllocGuard` (in allocator.h)
    # also degrades to a true no-op under this macro.
    target_compile_definitions(kamepoolalloc PUBLIC KAMEPOOLALLOC_DYLIB)
    # macOS: rpath so test binaries find the dylib next to themselves.
    set_target_properties(kamepoolalloc PROPERTIES
        INSTALL_NAME_DIR "@rpath"
        BUILD_WITH_INSTALL_RPATH TRUE)
else()
    add_compile_definitions(DISABLE_POOL_ALLOCATOR)
endif()

#! On Linux, prefer the `-pthread` compiler/linker flag over plain `-lpthread`.
#! `-pthread` survives `--as-needed` (the default in many distros' gcc)
#! whereas a bare `-lpthread` placed late on the command line gets
#! dropped when `threadlocal.cpp` is the only TU referencing
#! pthread_setspecific etc., causing the linker error
#!     "undefined reference to symbol 'pthread_setspecific@@GLIBC_*'"
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# --- Test targets ---
#
# `test_link_libs` collects libraries every test executable links
# against.  `kamepoolalloc` is appended when USE_KAME_ALLOCATOR is on
# (see above) so the shared-library `operator new` / `delete` and the
# `__DATA,__interpose` `free` override apply to every test binary.
set(test_link_libs Threads::Threads)
if(USE_KAME_ALLOCATOR)
    list(APPEND test_link_libs kamepoolalloc)
endif()

set(test_names
    atomic_shared_ptr_test
    atomic_scoped_ptr_test
    atomic_queue_test
    mutex_test
    transaction_test
    transaction_dynamic_node_test
    transaction_negotiation_test
    transaction_payload_integrity_test
    transaction_payload_integrity_3level_test
    transaction_payload_integrity_mixed_test
    transaction_payload_integrity_3level_mixed_test
    alloc_stress_test
    alloc_minimal_bench
    alloc_bucket34_repro)

foreach(t IN LISTS test_names)
    add_executable(${t} ${t}.cpp ${support_SRCS})
    target_link_libraries(${t} ${test_link_libs})
endforeach()

# --- CTest registration ---

enable_testing()

add_test(NAME atomic_shared_ptr_test COMMAND atomic_shared_ptr_test)
add_test(NAME atomic_scoped_ptr_test COMMAND atomic_scoped_ptr_test)
add_test(NAME atomic_queue_test COMMAND atomic_queue_test)
add_test(NAME mutex_test COMMAND mutex_test)
add_test(NAME transaction_test COMMAND transaction_test)
add_test(NAME transaction_dynamic_node_test COMMAND transaction_dynamic_node_test)
add_test(NAME transaction_negotiation_test COMMAND transaction_negotiation_test)
add_test(NAME transaction_payload_integrity_test COMMAND transaction_payload_integrity_test)
add_test(NAME transaction_payload_integrity_3level_test COMMAND transaction_payload_integrity_3level_test)

# Long-running tests
set_tests_properties(
    transaction_test
    transaction_dynamic_node_test
    transaction_negotiation_test
    transaction_payload_integrity_test
    transaction_payload_integrity_3level_test
    PROPERTIES TIMEOUT 300)
