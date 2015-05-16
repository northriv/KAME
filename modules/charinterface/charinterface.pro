PRI_DIR = ../
include($${PRI_DIR}/modules-shared.pri)

HEADERS += \
    chardevicedriver.h \
    charinterface.h \
    dummyport.h \
    gpib.h \
    oxforddriver.h \
    serial.h \
    tcp.h

SOURCES += \
    charinterface.cpp \
    dummyport.cpp \
    gpib.cpp \
    oxforddriver.cpp \
    serial.cpp \
    tcp.cpp

win32 {
    INCLUDEPATH += "C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C\include"
    INCLUDEPATH += "C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\include"
    DEFINES += HAVE_NI488

    LIBS += -lWS2_32
}
