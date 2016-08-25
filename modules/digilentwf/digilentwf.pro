PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame/graph\

win32 {
    exists("C:/Program Files/Digilent/WaveFormsSDK/inc/dwf.h") {
        INCLUDEPATH += "C:\Program Files\Digilent\WaveFormsSDK\inc"
        LIBS += -L"C:\Program Files\Digilent\WaveFormsSDK\lib\x86" -ldwf
        dwf_found = true
    }
    exists("C:/Program Files (x86)/Digilent/WaveFormsSDK/inc/dwf.h") {
        INCLUDEPATH += "C:\Program Files (x86)\Digilent\WaveFormsSDK\inc"
        contains(QMAKE_HOST.arch, x86_64) {
            LIBS += -L"C:\Program Files (x86)\Digilent\WaveFormsSDK\lib\x64" -ldwf
        }
        else {
            LIBS += -L"C:\Program Files (x86)\Digilent\WaveFormsSDK\lib\x86" -ldwf
        }
        dwf_found = true
    }
}

macx: exists("/Library/Frameworks/dwf.framework") {
    INCLUDEPATH += /Library/Frameworks/dwf.framework/Headers
    LIBS += -F/Library/Frameworks -framework dwf
    dwf_found = true
}

equals(dwf_found, true) {
    HEADERS += dwfdso.h
    SOURCES += dwfdso.cpp
}
else {
    message("Missing library for Digilent WaveForms SDK")
}

win32:LIBS += -ldsocore

INCLUDEPATH += $$PWD/../dso/core
DEPENDPATH += $$PWD/../dso/core
