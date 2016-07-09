TEMPLATE = subdirs

CONFIG += testcase

SUBDIRS += \
    atomic_shared_ptr_test\
    atomic_scoped_ptr_test\
    atomic_queue_test\
    mutex_test\
    transaction_test\
    transaction_dynamic_node_test\
    transaction_negotiation_test

atomic_shared_ptr_test.file = atomic_shared_ptr_test.pro
atomic_scoped_ptr_test.file = atomic_scoped_ptr_test.pro
atomic_queue_test.file = atomic_queue_test.pro
mutex_test.file = mutex_test.pro
transaction_test.file = transaction_test.pro
transaction_dynamic_node_test.file = transaction_dynamic_node_test.pro
transaction_negotiation_test.file = transaction_negotiation_test.pro
