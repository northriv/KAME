TARGET = xnode_typename_test

TEMPLATE = app
CONFIG += exceptions rtti console testcase
CONFIG -= app_bundle qt

greaterThan(QT_MAJOR_VERSION, 4) {
    CONFIG += c++17
} else {
    QMAKE_CXXFLAGS += -std=c++17
}

SOURCES += \
    xnode_typename_test.cpp
