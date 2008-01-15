noinst_LIBRARIES = libmath.a

INCLUDES = -I$(top_srcdir)/kame \
	$(all_includes)

libmath_a_SOURCES = \
	cspline.cpp cspline.h \
	fir.cpp fir.h \
	fft.cpp fft.h \
	mem.cpp mem.h \
	rand.cpp rand.h \
	roots.cpp roots.h

SUBDIRS =
METASOURCES = AUTO

#AM_CXXFLAGS = $(CXXFLAGS) \
#	-O6  \
#	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
#	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
