TARGET = transaction_negotiation_test

include(../../tests/tests.pri)

HEADERS += \
    ../kame/atomic_smart_ptr.h \
    ../kame/xthread.h \
    ../kame/transaction.h \
    ../kame/transaction_impl.h \
    ../kame/transaction_definitions.h \
    ../kame/transaction_signal.h \
    ../kame/transaction_negotiation.h \
    ../kame/transaction_neg_impl.h

SOURCES += \
    transaction_negotiation_test.cpp
