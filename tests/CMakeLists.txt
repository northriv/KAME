cmake_minimum_required(VERSION 3.14)
project(kame_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../kame)

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

set(support_SRCS support_standalone.cpp)

# KAME pool allocator: OFF by default on macOS/Windows (Qt build provides it),
# ON on Linux unless compiler is too old (GCC < 10).
# Override with: cmake .. -DUSE_KAME_ALLOCATOR=ON (or OFF)
if(NOT DEFINED USE_KAME_ALLOCATOR)
    if(APPLE OR WIN32)
        set(USE_KAME_ALLOCATOR OFF)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
        set(USE_KAME_ALLOCATOR OFF)
    else()
        set(USE_KAME_ALLOCATOR ON)
    endif()
endif()
if(NOT USE_KAME_ALLOCATOR)
    add_compile_definitions(DISABLE_POOL_ALLOCATOR)
endif()

find_package(Threads REQUIRED)

# --- Test targets ---

add_executable(atomic_shared_ptr_test atomic_shared_ptr_test.cpp ${support_SRCS})
target_link_libraries(atomic_shared_ptr_test Threads::Threads)

add_executable(atomic_scoped_ptr_test atomic_scoped_ptr_test.cpp ${support_SRCS})
target_link_libraries(atomic_scoped_ptr_test Threads::Threads)

add_executable(atomic_queue_test atomic_queue_test.cpp ${support_SRCS})
target_link_libraries(atomic_queue_test Threads::Threads)

add_executable(mutex_test mutex_test.cpp ${support_SRCS})
target_link_libraries(mutex_test Threads::Threads)

add_executable(transaction_test transaction_test.cpp ${support_SRCS})
target_link_libraries(transaction_test Threads::Threads)

add_executable(transaction_dynamic_node_test transaction_dynamic_node_test.cpp ${support_SRCS})
target_link_libraries(transaction_dynamic_node_test Threads::Threads)

add_executable(transaction_negotiation_test transaction_negotiation_test.cpp ${support_SRCS})
target_link_libraries(transaction_negotiation_test Threads::Threads)

add_executable(transaction_payload_integrity_test transaction_payload_integrity_test.cpp ${support_SRCS})
target_link_libraries(transaction_payload_integrity_test Threads::Threads)

add_executable(transaction_payload_integrity_3level_test transaction_payload_integrity_3level_test.cpp ${support_SRCS})
target_link_libraries(transaction_payload_integrity_3level_test Threads::Threads)

add_executable(transaction_payload_integrity_mixed_test transaction_payload_integrity_mixed_test.cpp ${support_SRCS})
target_link_libraries(transaction_payload_integrity_mixed_test Threads::Threads)

add_executable(transaction_payload_integrity_3level_mixed_test transaction_payload_integrity_3level_mixed_test.cpp ${support_SRCS})
target_link_libraries(transaction_payload_integrity_3level_mixed_test Threads::Threads)

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
