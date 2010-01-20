include_directories(
    ${CMAKE_SOURCE_DIR}/kame )

set(support_SRCS
    support.cpp
 )

add_executable(atomic_shared_ptr_test atomic_shared_ptr_test.cpp ${support_SRCS})
target_link_libraries(atomic_shared_ptr_test pthread)
add_executable(atomic_scoped_ptr_test atomic_scoped_ptr_test.cpp ${support_SRCS})
target_link_libraries(atomic_scoped_ptr_test pthread)
add_executable(atomic_queue_test atomic_queue_test.cpp ${support_SRCS})
target_link_libraries(atomic_queue_test pthread)
add_executable(mutex_test mutex_test.cpp ${support_SRCS})
target_link_libraries(mutex_test pthread)
add_executable(transaction_test transaction_test.cpp transaction.cpp ${support_SRCS})
target_link_libraries(transaction_test pthread)
add_executable(xnode_test xnode.cpp transaction.cpp ${support_SRCS})
target_link_libraries(xnode_test pthread)

add_test(atomic_shared_ptr_test atomic_shared_ptr_test)
add_test(atomic_scoped_ptr_test atomic_scoped_ptr_test)
add_test(atomic_queue_test atomic_queue_test)
add_test(mutex_test mutex_test)
add_test(transaction_test transaction_test)
