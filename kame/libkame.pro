TARGET = libkame
TEMPLATE = lib

CONFIG += static

CONFIG += qt exceptions
CONFIG += sse2 rtti

greaterThan(QT_MAJOR_VERSION, 4) {
	CONFIG += c++11
}
else {
# for g++ with C++0x spec.
	QMAKE_CXXFLAGS += -std=c++0x -Wall
#	 -stdlib=libc++
}

QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

FORMS += \
    forms/caltableform.ui \
    forms/drivercreate.ui \
    forms/drivertool.ui \
    forms/graphtool.ui \
    forms/interfacetool.ui \
    forms/nodebrowserform.ui \
    forms/recordreaderform.ui \
    forms/rubythreadtool.ui \
    forms/scalarentrytool.ui \
    graph/graphdialog.ui \
    graph/graphform.ui \
    graph/graphnurlform.ui



