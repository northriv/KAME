include_directories(
    ${CMAKE_SOURCE_DIR}/kame )

set(support_SRCS
    support.cpp
 )

add_executable(atomic_shared_ptr_test atomic_shared_ptr_test.cpp ${support_SRCS})
add_executable(atomic_scoped_ptr_test atomic_scoped_ptr_test.cpp ${support_SRCS})
add_executable(atomic_queue_test atomic_queue_test.cpp ${support_SRCS})
add_executable(mutex_test mutex_test.cpp ${support_SRCS})

add_test(atomic_shared_ptr_test atomic_shared_ptr_test)
add_test(atomic_scoped_ptr_test atomic_scoped_ptr_test)
add_test(atomic_queue_test atomic_queue_test)
add_test(mutex_test mutex_test)
