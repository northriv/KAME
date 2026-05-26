TARGET = transaction_payload_integrity_3level_mixed_test

include(tests.pri)

HEADERS += \
    ../kame/atomic_smart_ptr.h \
    ../kame/transaction.h \
    ../kame/transaction_impl.h \
    ../kame/transaction_definitions.h \
    ../kame/transaction_signal.h \
    ../kame/transaction_negotiation.h \
    ../kame/transaction_neg_impl.h

SOURCES += \
    transaction_payload_integrity_3level_mixed_test.cpp
