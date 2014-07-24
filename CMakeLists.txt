#KAME compilation with KDE4
#KAME only with QT library should be handled by qmake instead.  
cmake_minimum_required (VERSION 2.8.4)
project( kame )

find_package(KDE4 REQUIRED)

add_definitions(${QT_DEFINITIONS} -DWITH_KDE ${KDE4_DEFINITIONS} -DVERSION="\\"4.0\\"")

include(KDE4Defaults)
SET(QT_USE_QTOPENGL TRUE)

include(MacroLibrary)

#include(ManualStuff.cmake)

#include(ConfigureChecks.cmake)

#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)

include(FindGettext)
#find_package( LAPACK REQUIRED)
find_package( Boost 1.34.0 REQUIRED)
message(STATUS "Boost found: ${Boost_INCLUDE_DIR}")
find_package( ZLIB REQUIRED)
message(STATUS "zlib found: ${ZLIB_INCLUDE_DIR} ${ZLIB_LIBRARIES}")
find_package( Ruby REQUIRED)
message(STATUS "Ruby found: ${RUBY_INCLUDE_PATH} ${RUBY_LIBRARY}")

find_program(GSL_FOUND "gsl-config")
if(GSL_FOUND)
    execute_process(COMMAND gsl-config --prefix OUTPUT_VARIABLE GSL_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_path( GSL_INCLUDE_DIR
    	gsl/gsl_math.h "${GSL_PREFIX}/include")
    execute_process(COMMAND gsl-config --libs OUTPUT_VARIABLE GSL_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "gsl found: ${GSL_INCLUDE_DIR} ${GSL_LIBRARY}")
else(1)
    message(FATAL_ERROR "gsl not found: ${GSL_INCLUDE_DIR} ${GSL_LIBRARY}")    
endif(GSL_FOUND)

find_path( FFTW3_INCLUDE_DIR
	fftw3.h /usr/include /usr/local/include /sw/include /opt/local/include)
find_library( FFTW3_LIBRARY NAMES fftw3 PATHS /usr/lib /usr/local/lib /sw/lib /opt/local/lib)
if( FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY )
    message(STATUS "FFTW3 found: ${FFTW3_INCLUDE_DIR} ${FFTW3_LIBRARY}")
else(1)
    message(FATAL_ERROR "FFTW3 not found: ${FFTW3_INCLUDE_DIR} ${FFTW3_LIBRARY}")
endif( FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY )

find_library( CLAPACK_LIBRARY NAMES clapack PATHS /usr/lib/atlas-sse2 /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib /opt/local/lib)
find_library( ATLAS_LIBRARY NAMES atlas PATHS /usr/lib/atlas-sse2 /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib /opt/local/lib)
find_library( CBLAS_LIBRARY NAMES cblas PATHS /usr/lib/atlas-sse2 /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib /opt/local/lib)
if( CLAPACK_LIBRARY AND ATLAS_LIBRARY AND CBLAS_LIBRARY )
    execute_process(COMMAND grep zheevr ${CLAPACK_LIBRARY} OUTPUT_VARIABLE ZHEEVR_FOUND_IN_CLAPACK OUTPUT_STRIP_TRAILING_WHITESPACE)
endif( CLAPACK_LIBRARY AND ATLAS_LIBRARY AND CBLAS_LIBRARY )
if(ZHEEVR_FOUND_IN_CLAPACK)
    set(LAPACK_LIBRARIES ${CLAPACK_LIBRARY} ${ATLAS_LIBRARY} ${CBLAS_LIBRARY})
    message(STATUS "zheevr_ found in CLAPACK/ATLAS: ${LAPACK_LIBRARIES}")
else(1)
    find_library( LAPACK_LIBRARY NAMES lapack PATHS /usr/lib/atlas-sse2 /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib /opt/local/lib)
    find_library( BLAS_LIBRARY NAMES blas ptcblas PATHS /usr/lib/atlas-sse2 /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib /opt/local/lib)
    find_library( F77BLAS_LIBRARY NAMES f77blas ptf77blas PATHS /usr/lib/atlas-sse2 /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib /opt/local/lib)
    if( LAPACK_LIBRARY AND ATLAS_LIBRARY AND BLAS_LIBRARY AND F77BLAS_LIBRARY )
        set(LAPACK_LIBRARIES ${LAPACK_LIBRARY} ${BLAS_LIBRARY} ${ATLAS_LIBRARY} ${F77BLAS_LIBRARY} gfortran)
        message(STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
    endif( LAPACK_LIBRARY AND ATLAS_LIBRARY AND BLAS_LIBRARY AND F77BLAS_LIBRARY )
endif(ZHEEVR_FOUND_IN_CLAPACK)

if( LAPACK_LIBRARIES )
else(1)
    message(FATAL_ERROR "LAPACK not found: ${LAPACK_LIBRARIES}")
endif( LAPACK_LIBRARIES )

find_path( LTDL_INCLUDE_DIR
	ltdl.h /usr/include /usr/local/include /sw/include /opt/local/include)
find_library( LTDL_LIBRARY NAMES ltdl PATHS /usr/lib /usr/local/lib /sw/lib /opt/local/lib)
if( LTDL_INCLUDE_DIR AND LTDL_LIBRARY )
    message(STATUS "LTDL found: ${LTDL_INCLUDE_DIR} ${LTDL_LIBRARY}")
else(1)
    message(FATAL_ERROR "LTDL not found: ${LTDL_INCLUDE_DIR} ${LTDL_LIBRARY}")
endif( LTDL_INCLUDE_DIR AND LTDL_LIBRARY )

find_library( GL_LIBRARY NAMES GL PATHS /usr/lib /usr/local/lib /sw/lib /opt/local/lib)
if( GL_LIBRARY )
    message(STATUS "GL found: ${GL_LIBRARY}")
else(1)
    message(FATAL_ERROR "GL not found: ${GL_LIBRARY}")
endif( GL_LIBRARY )

find_package( GLU REQUIRED)
message(STATUS "GLU found: ${GLU_INCLUDE_PATH} ${GLU_LIBRARY}")

#find_package( GLUT REQUIRED)
#message(STATUS "GLUT found: ${GLUT_INCLUDE_DIR} ${GLUT_LIBRARIES}")

find_path( NIDAQMX_INCLUDE_DIR
	NIDAQmx.h /usr/include /usr/local/include /sw/include /opt/local/include)
find_library( NIDAQMX_LIBRARY nidaqmx /usr/lib /usr/local/lib /sw/lib /opt/local/lib)
if(NIDAQMX_INCLUDE_DIR AND NIDAQMX_LIBRARY)
    add_definitions(-DHAVE_NI_DAQMX)
    message(STATUS "NIDAQMX found: ${NIDAQMX_INCLUDE_DIR} ${NIDAQMX_LIBRARY}")
else(1)
    set(NIDAQMX_INCLUDE_DIR "")
    set(NIDAQMX_LIBRARY "")
    message(STATUS "NIDAQMX not found: ${NIDAQMX_INCLUDE_DIR} ${NIDAQMX_LIBRARY}")
endif(NIDAQMX_INCLUDE_DIR AND NIDAQMX_LIBRARY)

find_path( LINUXGPIB_INCLUDE_DIR
	gpib/ib.h /usr/include /usr/local/include /sw/include /opt/local/include)
find_library( LINUXGPIB_LIBRARY gpib /usr/lib /usr/local/lib /sw/lib /opt/local/lib)
if( LINUXGPIB_INCLUDE_DIR AND LINUXGPIB_LIBRARY )
    add_definitions(-DHAVE_LINUX_GPIB)
    message(STATUS "LINUXGPIB found: ${LINUXGPIB_INCLUDE_DIR} ${LINUXGPIB_LIBRARY}")
else(1)
    set(LINUXGPIB_INCLUDE_DIR "")
    set(LINUXGPIB_LIBRARY "")
    message(STATUS "LINUXGPIB not found: ${LINUXGPIB_INCLUDE_DIR} ${LINUXGPIB_LIBRARY}")
endif( LINUXGPIB_INCLUDE_DIR AND LINUXGPIB_LIBRARY )

find_path( NI488_INCLUDE_DIR
	ni488.h /usr/include /usr/local/include)
find_library( NI488_LIBRARY NI488 /usr/lib /usr/local/lib)
if( NI488_INCLUDE_DIR AND NI488_LIBRARY )
    add_definitions(-DHAVE_NI488)
    message(STATUS "NI488 found: ${NI488_INCLUDE_DIR} ${NI488_LIBRARY}")
else(1)
    set(NI488_INCLUDE_DIR "")
    set(NI488_LIBRARY "")
    message(STATUS "NI488 not found: ${NI488_INCLUDE_DIR} ${NI488_LIBRARY}")
endif( NI488_INCLUDE_DIR AND NI488_LIBRARY )

set(GPIB_INCLUDE_DIR "${NI488_INCLUDE_DIR}${LINUXGPIB_INCLUDE_DIR}")
set(GPIB_LIBRARY "${NI488_LIBRARY}${LINUXGPIB_LIBRARY}")
#TEST_BIG_ENDIAN(__BIGENDIAN__)

include_directories(${KDE4_INCLUDES} ${KDE4_INCLUDE_DIR} ${QT_INCLUDES} 
 ${Boost_INCLUDE_DIR})

#add more include direcories, using INCLUDE_DIRECTORIES()
#add more link libraries, using TARGET_LINK_LIBRARIES()
#add some compile switches, using ADD_DEFINITIONS()

#add some "configure" checks, see How To Do Platform Checks and How To Find Installed Software
#take special care of former libtool convenience libraries. 
#They are not supported by cmake, instead there will be a file ConvenienceLibs.cmake created. 
#In this file you will find for every convenience lib a variable, which contains all source files of this convenience lib. 
#For the targets which linked to this convenience lib, just add the variable to the sources.
#a file AdditionalInfo.txt will be created. There you will see all *.in and *.in.in files of your project. 
#The stuff done in these files will have to be converted manually to cmake.

set_property(GLOBAL PROPERTY ENABLE_EXPORTS TRUE)
if(CMAKE_COMPILER_IS_GNUCXX)
  set (CMAKE_CXX_FLAGS "-std=c++0x -Wall -fexceptions")
else(1)
#assuming clang++
  set (CMAKE_CXX_FLAGS "-std=c++0x -stdlib=libc++ -Wall -fexceptions")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(KAME_LIB_INSTALL_DIR "${LIB_INSTALL_DIR}")
set(KAME_MODULE_DIR_SURFIX "/kame/modules")
add_definitions(-DKAME_MODULE_DIR_SURFIX="\"${KAME_MODULE_DIR_SURFIX}\"")
set(KAME_MODULE_INSTALL_DIR "${LIB_INSTALL_DIR}/${KAME_MODULE_DIR_SURFIX}")

if(APPLE)
#Apple's ld
  set(CMAKE_MODULE_LINKER_FLAGS "-std=c++0x -all_load  -undefined dynamic_lookup" ) 
  set(CMAKE_SHARED_LINKER_FLAGS "-std=c++0x -all_load  -undefined dynamic_lookup" ) 
  set(CMAKE_EXE_LINKER_FLAGS "-std=c++0x -all_load -dynamic")
#  set(CMAKE_MODULE_LINKER_FLAGS "-flat_namespace -std=c++0x -all_load  -undefined dynamic_lookup" ) 
#  set(CMAKE_SHARED_LINKER_FLAGS "-flat_namespace -std=c++0x -all_load  -undefined suppress" ) 
#  set(CMAKE_EXE_LINKER_FLAGS "-flat_namespace -std=c++0x -all_load -dynamic")
else(1)
#GNU libtool
  set(CMAKE_MODULE_LINKER_FLAGS "--module --shared")
  set(CMAKE_SHARED_LINKER_FLAGS "--shared")
  set(CMAKE_EXE_LINKER_FLAGS "-dlpreopen self -export-dynamic")
  set(CMAKE_CXX_LINK_EXECUTABLE "libtool --mode=link --tag=CXX ${CMAKE_CXX_LINK_EXECUTABLE}")
  set(CMAKE_CXX_CREATE_SHARED_MODULE "libtool --mode=link --tag=CXX ${CMAKE_CXX_CREATE_SHARED_MODULE}")
endif(APPLE)
message(STATUS "CMAKE_MODULE_LINKER_FLAGS: ${CMAKE_MODULE_LINKER_FLAGS}")
message(STATUS "CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_CXX_LINK_EXECUTABLE: ${CMAKE_CXX_LINK_EXECUTABLE}")
message(STATUS "CMAKE_CXX_CREATE_SHARED_MODULE: ${CMAKE_CXX_CREATE_SHARED_MODULE}")

add_subdirectory(kame)
add_subdirectory(modules)
add_subdirectory(po)
add_subdirectory(doc)
add_subdirectory(tests)

########### install files ###############

install(FILES  AUTHORS COPYING ChangeLog README DESTINATION ${DATA_INSTALL_DIR}/kame)

