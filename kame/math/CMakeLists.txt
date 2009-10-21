include_directories(
    ${GSL_INCLUDE_DIR} ${FFTW3_INCLUDE_DIR})

########### next target ###############
set(kamemath_SRCS
	cspline.cpp
	fir.cpp
	fft.cpp
	ar.cpp
	freqest.cpp
	freqestleastsquare.cpp
	spectrumsolver.cpp
	rand.cpp
	matrix.cpp
 )

add_library(kamemath STATIC ${kamemath_SRCS})

########### install files ###############


#
##AM_CXXFLAGS = $(CXXFLAGS) \
##	-O6  \
##	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
##	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
