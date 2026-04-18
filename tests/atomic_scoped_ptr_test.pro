TARGET = atomic_scoped_ptr_test

include(tests.pri)

HEADERS += \
    support_standalone.h \
    ../kame/allocator.h

SOURCES += \
    atomic_scoped_ptr_test.cpp \
    support_standalone.cpp
