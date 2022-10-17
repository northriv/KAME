PRI_DIR = ../
include($${PRI_DIR}/modules-shared.pri)

HEADERS += \
    chardevicedriver.h \
    charinterface.h \
    dummyport.h \
    gpib.h \
    oxforddriver.h \
    pfeifferprotocol.h \
    serial.h \
    tcp.h \
    modbusrtuinterface.h

SOURCES += \
    charinterface.cpp \
    dummyport.cpp \
    gpib.cpp \
    oxforddriver.cpp \
    pfeifferprotocol.cpp \
    serial.cpp \
    tcp.cpp \
    modbusrtuinterface.cpp

unix {
    exists("/opt/local/include/libusb-1.0/libusb.h") {
        LIBS += -lusb-1.0
        HEADERS += \
            cyfxusb.h \
            cyfxusbinterface_impl.h \

        SOURCES += \
            cyfxusb.cpp \
            cyfxusb_libusb.cpp \

        DEFINES += USE_FX_USB
    }
    else {
        message("Missing library for libusb-1.0")
    }
}

win32 {
    INCLUDEPATH += "C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C"
    INCLUDEPATH += "C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C\include"
    INCLUDEPATH += "C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\include"
    DEFINES += HAVE_NI4882

    HEADERS += \
        cyfxusb.h \
        cyfxusb_win32.h \
        cyfxusbinterface_impl.h \

    SOURCES += \
        cyfxusb.cpp \
        cyfxusb_win32.cpp \

    DEFINES += USE_FX_USB

    LIBS += -lWS2_32
    LIBS += -lsetupapi #for USB
}

macx{
    exists("/Library/Frameworks/NI4882.framework") {
        INCLUDEPATH += /Library/Frameworks/NI4882.framework/Headers
        LIBS += -F/Library/Frameworks -framework NI4882
        DEFINES += HAVE_NI4882
    }
    else {
        message("Missing library for NI488.2")
    }
}
