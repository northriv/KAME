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
    # HAS_LIBUSB is set to 1 or left UNSET — never to the string "false",
    # which is non-empty and would read as "yes" to !isEmpty() below.
    macx:exists("/opt/local/include/libusb-1.0/libusb.h"): HAS_LIBUSB = 1
    !macx:system(pkg-config --exists libusb-1.0): HAS_LIBUSB = 1
    !isEmpty(HAS_LIBUSB) {
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
        message("linux-gpib not found — falling back to the usermode NI USB-GPIB driver.")
    }
}

# Usermode NI USB-GPIB driver — the fallback for every platform that has no
# kernel GPIB module available: macOS and Windows always, and Linux when
# linux-gpib is not installed (checked just above).  It talks to NI USB-B /
# USB-HS / USB-HS+ / KUSB-488A / MC USB-488 through libusb, so it needs
# libusb and nothing else; `osx_compat.h` keeps its historical name but is
# plain POSIX (see its header comment), so no Linux-specific shim is needed.
# macOS and Windows have no kernel GPIB module at all, so they always want it
# (their libusb comes from MacPorts / msys64 and is assumed, as before).
# Linux only wants it when linux-gpib is absent AND libusb is actually there.
macx|win32: USERMODE_NI_GPIB = 1
unix:!macx:!contains(DEFINES, HAVE_LINUX_GPIB):!isEmpty(HAS_LIBUSB): USERMODE_NI_GPIB = 1
!contains(DEFINES, HAVE_NI4882):!isEmpty(USERMODE_NI_GPIB) {
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
