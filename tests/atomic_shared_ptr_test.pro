TARGET = atomic_shared_ptr_test

include(tests.pri)

HEADERS += \
    support.h \
    ../kame/allocator.h

SOURCES += \
    atomic_shared_ptr_test.cpp \
    support.cpp
