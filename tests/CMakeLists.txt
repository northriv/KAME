include_directories(
    ${CMAKE_SOURCE_DIR}/kame )

set(support_SRCS
    support.cpp
 )

add_executable(allocator_test allocator_test.cpp ${support_SRCS})
target_link_libraries(allocator_test pthread)
add_executable(atomic_shared_ptr_test atomic_shared_ptr_test.cpp ${support_SRCS})
target_link_libraries(atomic_shared_ptr_test pthread)
add_executable(atomic_scoped_ptr_test atomic_scoped_ptr_test.cpp ${support_SRCS})
target_link_libraries(atomic_scoped_ptr_test pthread)
add_executable(atomic_queue_test atomic_queue_test.cpp ${support_SRCS})
target_link_libraries(atomic_queue_test pthread)
add_executable(mutex_test mutex_test.cpp ${support_SRCS})
target_link_libraries(mutex_test pthread)
add_executable(transaction_test transaction_test.cpp xtime.cpp ${support_SRCS})
target_link_libraries(transaction_test pthread)
add_executable(transaction_negotiation_test transaction_negotiation_test.cpp xtime.cpp ${support_SRCS})
target_link_libraries(transaction_negotiation_test pthread)
add_executable(transaction_dynamic_node_test transaction_dynamic_node_test.cpp xtime.cpp ${support_SRCS})
target_link_libraries(transaction_dynamic_node_test pthread)

add_test(allocator_test allocator_test)
add_test(atomic_shared_ptr_test atomic_shared_ptr_test)
add_test(atomic_scoped_ptr_test atomic_scoped_ptr_test)
add_test(atomic_queue_test atomic_queue_test)
add_test(mutex_test mutex_test)
add_test(transaction_test transaction_test)
add_test(transaction_dynamic_node_test transaction_dynamic_node_test)
add_test(transaction_negotioation_test transaction_negotiation_test)
