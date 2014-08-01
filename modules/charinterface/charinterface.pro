PRI_DIR = ../
include($${PRI_DIR}/modules-shared.pri)

QT       += serialport

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
    INCLUDEPATH += "C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\include"
#    QMAKE_PRE_LINK = dlltool -d $${_PRO_FILE_PWD_}/ni4882.def -l$${DESTDIR}/libni4882.a
##    LIBS += -L"C:\Program Files (x86)\National Instruments\Shared\ExternalCompilerSupport\C\lib32\msvc" -lgpib-32
#    LIBS += -L$${DESTDIR} -lni4882
    DEFINES += HAVE_NI488
}
