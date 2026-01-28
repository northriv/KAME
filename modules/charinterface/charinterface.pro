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
        cyfxusbinterface_impl.h \

    SOURCES += \
        cyfxusb.cpp

#    exists("c:/msys64/mingw64/include/libusb-1.0/libusb.h") {
#        SOURCES += \
#            cyfxusb_libusb.cpp
#        LIBS += -lusb-1.0
#        DEFINES += USE_LIBUSB_WHEN_WINCYFX_UNDETECTED
#        message("Using libusb-1.0")
#    }
    HEADERS += \
        cyfxusb_win32.h
    SOURCES += \
        cyfxusb_win32.cpp

    DEFINES += USE_FX_USB

    LIBS += -lWS2_32
    LIBS += -lsetupapi #for USB
}

macx{
    contains(QMAKE_HOST.arch, x86) | contains(QMAKE_HOST.arch, x86_64) {
        exists("/Library/Frameworks/NI4882.framework") {
            INCLUDEPATH += /Library/Frameworks/NI4882.framework/Headers
            LIBS += -F/Library/Frameworks -framework NI4882
            DEFINES += HAVE_NI4882
            message("Using NI488.2 for GPIB")
        }
    }
}
