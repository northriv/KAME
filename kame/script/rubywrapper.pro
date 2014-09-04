TEMPLATE = lib

win32: CONFIG += shared
else: CONFIG += static

CONFIG += exceptions
CONFIG += sse sse2 rtti

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

greaterThan(QT_MAJOR_VERSION, 4) {
	CONFIG += c++11
}
else {
# for g++ with C++0x spec.
	QMAKE_CXXFLAGS += -std=c++0x -Wall
#	 -stdlib=libc++
}


HEADERS += \
    rubywrapper.h \

SOURCES += \
    rubywrapper.cpp \

DESTDIR=$$OUT_PWD/../../

macx {
    INCLUDEPATH += /System/Library/Frameworks/Ruby.framework/Versions/1.8/Headers
    LIBS += -framework Ruby
}
else:unix {
    INCLUDEPATH += /usr/lib/ruby/1.8/i386-linux/
    LIBS += -lruby
}
win32-g++ {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/include
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby/.ext/include/i386-mingw32
    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby -lmsvcrt-ruby210
}
win32-msvc* {
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby-2.1.2/include
    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby-2.1.2/.ext/include/i386-mswin32_120
    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby-2.1.2 -lmsvcr120-ruby210 #-static -lWS2_32 -lAdvapi32 -lShell32 -limagehlp -lShlwapi -lIphlpapi
#    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby-2.0.0-p481/include
#    INCLUDEPATH += $${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby-2.0.0-p481/.ext/include/i386-mswin32_120
#    LIBS += -L$${_PRO_FILE_PWD_}/$${PRI_DIR}../ruby-2.0.0-p481 -lmsvcr120-ruby200 #-static -lWS2_32 -lAdvapi32 -lShell32 -limagehlp -lShlwapi
}

win32-g++ {
  QMAKE_LFLAGS += -Wl,--export-all-symbols -Wl,--out-implib,$${TARGET}.a
}
win32-msvc* {
    DEFINES += DECLSPEC_RUBY=__declspec(dllexport)
}


