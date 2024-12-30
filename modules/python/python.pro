PRI_DIR = ../
include($${PRI_DIR}/modules.pri)

INCLUDEPATH += \
    $${_PRO_FILE_PWD_}/../../kame/graph\
    $${_PRO_FILE_PWD_}/../../kame/script\

HEADERS += \

SOURCES += \
    basicdrivers.cpp \
    pybindnmr.cpp \
    pybindodmr.cpp

FORMS +=

RESOURCES += \
    pydrivers.qrc

macx {
  QMAKE_LFLAGS += -all_load  -undefined dynamic_lookup
}

win32: LIBS += -L$${PRI_DIR}/modules/

win32:LIBS += -lcharinterface

INCLUDEPATH += $$PWD/../charinterface
DEPENDPATH += $$PWD/../charinterface

win32:LIBS +=  -lnmrpulsercore

INCLUDEPATH += $$PWD/../nmr/pulsercore
DEPENDPATH += $$PWD/../nmr/pulsercore

win32:LIBS += -lsgcore

INCLUDEPATH += $$PWD/../sg/core
DEPENDPATH += $$PWD/../sg/core

win32:LIBS += -ldsocore

INCLUDEPATH += $$PWD/../dso/core
DEPENDPATH += $$PWD/../dso/core

win32:LIBS += -lmotorcore

INCLUDEPATH += $$PWD/../motor/core
DEPENDPATH += $$PWD/../motor/core

win32:LIBS += -lnetworkanalyzercore

INCLUDEPATH += $$PWD/../networkanalyzer/core
DEPENDPATH += $$PWD/../networkanalyzer/core

win32:LIBS += -ldmmcore

INCLUDEPATH += $$PWD/../dmm/core
DEPENDPATH += $$PWD/../dmm/core

win32:LIBS += -lmagnetpscore

INCLUDEPATH += $$PWD/../magnetps/core
DEPENDPATH += $$PWD/../magnetps/core

win32:LIBS += -lqdcore

INCLUDEPATH += $$PWD/../qd/core
DEPENDPATH += $$PWD/../qd/core

win32:LIBS += -ldcsourcecore

INCLUDEPATH += $$PWD/../dcsource/core
DEPENDPATH += $$PWD/../dcsource/core

win32:LIBS += -lnmr -lnmrpulser

INCLUDEPATH += $$PWD/../nmr
DEPENDPATH += $$PWD/../nmr

win32:LIBS += -lopticscore

INCLUDEPATH += $$PWD/../optics/core
DEPENDPATH += $$PWD/../optics/core

win32:LIBS += -loptics

INCLUDEPATH += $$PWD/../optics
DEPENDPATH += $$PWD/../optics

win32:LIBS += -lliacore

INCLUDEPATH += $$PWD/../lia/core
DEPENDPATH += $$PWD/../lia/core

#pybind11
macx {
    greaterThan(QT_MAJOR_VERSION, 5) {
        pythons="python3" $$files("/opt/local/bin/python3*") $$files("/usr/local/bin/python3*")
        for(PYTHON, pythons) {
            system("$${PYTHON} -m pybind11 --includes") {
                QMAKE_CXXFLAGS += $$system("$${PYTHON} -m pybind11 --includes")
#                QMAKE_LFLAGS += $$system("$${PYTHON}-config --embed --ldflags")
                DEFINES += USE_PYBIND11
                message("Python scripting support enabled.")
                break()
            }
        }
    }
}
win32-g++ {
    greaterThan(QT_MAJOR_VERSION, 5) {
        pythons="c:/msys64/mingw64/bin/python.exe"
        for(PYTHON, pythons) {
            system("$${PYTHON} -m pybind11 --includes") {
                QMAKE_CXXFLAGS += $$system("$${PYTHON} -m pybind11 --includes")
        #        QMAKE_LFLAGS += $$system("set PATH=c:/msys64/usr/bin;c:/msys64/mingw64/bin;%PATH% & c:/msys64/usr/bin/sh -c \"c:/msys64/mingw64/bin/python-config --embed --ldflags\"")
                LIBS += $$files(c:/msys64/mingw64/lib/libpython3*)
                DEFINES += USE_PYBIND11
                message("Python scripting support enabled.")
                break()
            }
        }
    }
}
