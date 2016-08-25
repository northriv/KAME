PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame/graph\

win32 {
    exists("C:\Program Files*\Digilent\WaveFormsSDK\inc") {
        HEADERS += dwfdso.h
        SOURCES += dwfdso.cpp
        INCLUDEPATH += "C:\Program Files\Digilent\WaveFormsSDK\inc"
        INCLUDEPATH += "C:\Program Files (x86)\Digilent\WaveFormsSDK\inc"
    #    LIBS += -L"C:\Program Files\Digilent\WaveFormsSDK\lib\x86"
    #    LIBS += -L"C:\Program Files (x86)\Digilent\WaveFormsSDK\lib\x64"
    }
    else {
        message("Missing library for Digilent WaveForms SDK")
    }
}

macx: exists("/Library/Frameworks/dwf.framework") {
    INCLUDEPATH += /Library/Frameworks/dwf.framework/Headers
    LIBS += -F/Library/Frameworks -framework dwf
    HEADERS += dwfdso.h
    SOURCES += dwfdso.cpp
}

win32:LIBS += -ldsocore

INCLUDEPATH += $$PWD/../dso/core
DEPENDPATH += $$PWD/../dso/core
