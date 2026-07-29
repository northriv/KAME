TARGET = xnode_typename_test

TEMPLATE = app
CONFIG += exceptions rtti console testcase
# See tests/tests.pri: `testcase` alone would make `make install` deploy this
# unit-test binary into $$[QT_INSTALL_TESTS].
CONFIG += no_testcase_installs
CONFIG -= app_bundle qt

greaterThan(QT_MAJOR_VERSION, 4) {
    CONFIG += c++17
} else {
    QMAKE_CXXFLAGS += -std=c++17
}

SOURCES += \
    xnode_typename_test.cpp
