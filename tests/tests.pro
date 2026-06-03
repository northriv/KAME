TEMPLATE = subdirs

CONFIG += testcase

# Test sources now live in each library's own tests/ subdirectory.
# This top-level qmake project lists each test by its new path so
# `make tests` builds all of them in one pass.

SUBDIRS += \
    atomic_shared_ptr_test\
    atomic_scoped_ptr_test\
    atomic_queue_test\
    mutex_test\
    transaction_test\
    transaction_dynamic_node_test\
    transaction_negotiation_test\
    transaction_payload_integrity_test\
    transaction_payload_integrity_3level_test\
    transaction_payload_integrity_mixed_test\
    transaction_payload_integrity_3level_mixed_test\
    alloc_stress_test\
    malloc_intercept_test\
    xnode_typename_test

# kamestm tests (../kamestm/tests/*.pro)
atomic_shared_ptr_test.file                       = ../kamestm/tests/atomic_shared_ptr_test.pro
atomic_scoped_ptr_test.file                       = ../kamestm/tests/atomic_scoped_ptr_test.pro
atomic_queue_test.file                            = ../kamestm/tests/atomic_queue_test.pro
mutex_test.file                                   = ../kamestm/tests/mutex_test.pro
transaction_test.file                             = ../kamestm/tests/transaction_test.pro
transaction_dynamic_node_test.file                = ../kamestm/tests/transaction_dynamic_node_test.pro
transaction_negotiation_test.file                 = ../kamestm/tests/transaction_negotiation_test.pro
transaction_payload_integrity_test.file           = ../kamestm/tests/transaction_payload_integrity_test.pro
transaction_payload_integrity_3level_test.file    = ../kamestm/tests/transaction_payload_integrity_3level_test.pro
transaction_payload_integrity_mixed_test.file     = ../kamestm/tests/transaction_payload_integrity_mixed_test.pro
transaction_payload_integrity_3level_mixed_test.file = ../kamestm/tests/transaction_payload_integrity_3level_mixed_test.pro

# kamepoolalloc tests (../kamepoolalloc/tests/*.pro)
alloc_stress_test.file = ../kamepoolalloc/tests/alloc_stress_test.pro
malloc_intercept_test.file = ../kamepoolalloc/tests/malloc_intercept_test.pro

# kame-specific tests (stay in tests/ — XNode lives in kame/)
xnode_typename_test.file = xnode_typename_test.pro
