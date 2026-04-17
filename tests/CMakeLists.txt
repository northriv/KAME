cmake_minimum_required(VERSION 3.14)
project(kame_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../kame)

# Standalone support: replaces support.h + xtime.h with std::chrono.
# Preincluded so include guards block the Qt-dependent originals.
add_compile_options(-include ${CMAKE_CURRENT_SOURCE_DIR}/support_standalone.h)

set(support_SRCS support_standalone.cpp)

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
