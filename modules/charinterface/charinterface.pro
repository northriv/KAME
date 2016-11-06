PRI_DIR = ../
include($${PRI_DIR}/modules-shared.pri)

HEADERS += \
    chardevicedriver.h \
    charinterface.h \
    dummyport.h \
    gpib.h \
    oxforddriver.h \
    serial.h \
    tcp.h \
    modbusrtuinterface.h

SOURCES += \
    charinterface.cpp \
    dummyport.cpp \
    gpib.cpp \
    oxforddriver.cpp \
    serial.cpp \
    tcp.cpp \
    modbusrtuinterface.cpp

win32 {
    INCLUDEPATH += "C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C"
    INCLUDEPATH += "C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C\include"
    INCLUDEPATH += "C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\include"
    DEFINES += HAVE_NI4882

    LIBS += -lWS2_32
}

macx: exists("/Library/Frameworks/NI4882.framework") {
    INCLUDEPATH += /Library/Frameworks/NI4882.framework/Headers
    LIBS += -F/Library/Frameworks -framework NI4882
    DEFINES += HAVE_NI4882
}
else {
    message("Missing library for NI488.2")
}
