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
    transaction_payload_integrity_3level_mixed_test)

foreach(t IN LISTS kamestm_test_names)
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
