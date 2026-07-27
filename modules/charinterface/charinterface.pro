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
    # macOS finds libusb under MacPorts' prefix; elsewhere on Unix it is a
    # normal pkg-config package.  Probing only the MacPorts path meant Linux
    # silently lost the whole Cypress FX2/FX3 USB interface.
    macx: HAS_LIBUSB = $$exists("/opt/local/include/libusb-1.0/libusb.h")
    else: HAS_LIBUSB = $$system(pkg-config --exists libusb-1.0 && echo 1)
    !isEmpty(HAS_LIBUSB):!equals(HAS_LIBUSB, false) {
        macx: LIBS += -lusb-1.0
        else: PKGCONFIG += libusb-1.0
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
    exists("C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C\include") |
    exists("C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\include") {
        INCLUDEPATH += "C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C"
        INCLUDEPATH += "C:\Program Files\National Instruments\Shared\ExternalCompilerSupport\C\include"
        INCLUDEPATH += "C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\include"
        DEFINES += HAVE_NI4882
        message("Using NI488.2 for GPIB")
    }

    HEADERS += \
        cyfxusb.h \
        cyfxusbinterface_impl.h \

    SOURCES += \
        cyfxusb.cpp

    exists("c:/msys64/mingw64/include/libusb-1.0/libusb.h") {
        SOURCES += \
            cyfxusb_libusb.cpp
        LIBS += -lusb-1.0
        DEFINES += USE_LIBUSB_WITH_WINCYFX
        message("Using libusb-1.0")
    }
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
# Linux/BSD: use the REAL linux-gpib kernel driver when its headers are
# present.  `HAVE_LINUX_GPIB` is read by gpib.h / gpib.cpp (which is where
# XNIGPIBPort's ib.h implementation lives) but was defined by nothing since
# the autotools build went away, so the native GPIB path was dead code.
unix:!macx {
    system(pkg-config --exists libgpib) {
        PKGCONFIG += libgpib
        DEFINES += HAVE_LINUX_GPIB
        message("Using linux-gpib for GPIB (pkg-config).")
    }
    else:exists("/usr/include/gpib/ib.h") {
        LIBS += -lgpib
        DEFINES += HAVE_LINUX_GPIB
        message("Using linux-gpib for GPIB (/usr/include/gpib/ib.h).")
    }
    else {
        message("linux-gpib not found — GPIB support disabled.")
    }
}

# Usermode NI USB-GPIB driver.  It exists because macOS and Windows have no
# kernel GPIB module; on Linux the kernel driver handled above is the right
# path, and building this here is actively wrong — compat.h routes every
# non-_WIN32 target to osx_compat.h, whose kernel-style `min`/`max` FUNCTION
# MACROS then collide with <limits> ("macro \"min\" requires 2 arguments"),
# and nothing links libusb for it either.  Restrict it to the two platforms
# it was written for.
macx|win32:!contains(DEFINES, HAVE_NI4882) {
    DEFINES += HAVE_USERMODE_NI_GPIB
    INCLUDEPATH += usermode-linux-gpib usermode-linux-gpib/linux-gpib
    QMAKE_CFLAGS += -Wno-unused-function -Wno-visibility
    HEADERS += nigpibport.h  \
        nigpibport.h \
        usermode-linux-gpib/compat.h            \
        usermode-linux-gpib/osx_compat.h        \
        usermode-linux-gpib/win_compat.h        \
        usermode-linux-gpib/NiGpibDriver.h      \
        usermode-linux-gpib/linux-gpib/ni_usb_gpib.h       \
        usermode-linux-gpib/linux-gpib/gpib.h              \
        usermode-linux-gpib/linux-gpib/gpib_user.h          \
        usermode-linux-gpib/linux-gpib/gpib_proto.h        \
        usermode-linux-gpib/linux-gpib/gpib_ioctl.h        \
        usermode-linux-gpib/linux-gpib/gpib_types.h        \
        usermode-linux-gpib/linux-gpib/gpibP.h             \
        usermode-linux-gpib/linux-gpib/nec7210.h           \
        usermode-linux-gpib/linux-gpib/tnt4882_registers.h \
        usermode-linux-gpib/linux-gpib/gpib_state_machines.h \

    SOURCES += \
        usermode-linux-gpib/NiGpibDriver.cpp \
        nigpibport.cpp \
        usermode-linux-gpib/linux-gpib/ni_usb_gpib.c \
        usermode-linux-gpib/gpib_stubs.c
    message("Using usermode NI USB-GPIB driver")
}
