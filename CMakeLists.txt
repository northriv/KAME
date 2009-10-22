cmake_minimum_required (VERSION 2.6)
project( kame )

message(STATUS "CMAKE_MODULE_LINKER_FLAGS: ${CMAKE_MODULE_LINKER_FLAGS}")
message(STATUS "CMAKE_MODULE_LINKER_FLAGS: ${CMAKE_MODULE_LINKER_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
find_package(KDE4 REQUIRED)

add_definitions(${QT_DEFINITIONS} ${KDE4_DEFINITIONS} -DVERSION="\\"2.9\\"")

include(KDE4Defaults)

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
	fftw3.h /usr/include /usr/local/include /sw/include)
find_library( FFTW3_LIBRARY NAMES fftw3 PATHS /usr/lib /usr/local/lib /sw/lib)
if( FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY )
    message(STATUS "FFTW3 found: ${FFTW3_INCLUDE_DIR} ${FFTW3_LIBRARY}")
else(1)
    message(FATAL_ERROR "FFTW3 not found: ${FFTW3_INCLUDE_DIR} ${FFTW3_LIBRARY}")
endif( FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY )

find_library( CLAPACK_LIBRARY NAMES clapack PATHS /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib)
find_library( ATLAS_LIBRARY NAMES atlas PATHS /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib)
find_library( CBLAS_LIBRARY NAMES cblas PATHS /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib)
if( CLAPACK_LIBRARY AND ATLAS_LIBRARY AND CBLAS_LIBRARY )
    execute_process(COMMAND grep zheevr ${CLAPACK_LIBRARY} OUTPUT_VARIABLE ZHEEVR_FOUND_IN_CLAPACK OUTPUT_STRIP_TRAILING_WHITESPACE)
endif( CLAPACK_LIBRARY AND ATLAS_LIBRARY AND CBLAS_LIBRARY )
if(ZHEEVR_FOUND_IN_CLAPACK)
    set(LAPACK_LIBRARIES ${CLAPACK_LIBRARY} ${ATLAS_LIBRARY} ${CBLAS_LIBRARY})
    message(STATUS "zheevr_ found in CLAPACK/ATLAS: ${LAPACK_LIBRARIES}")
else(1)
    find_library( LAPACK_LIBRARY NAMES lapack PATHS /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib)
    find_library( BLAS_LIBRARY NAMES blas PATHS /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib)
    find_library( F77BLAS_LIBRARY NAMES f77blas PATHS /usr/lib/sse2 /usr/lib /usr/local/lib/sse2 /usr/local/lib /sw/lib/sse2 /sw/lib)
    if( LAPACK_LIBRARY AND BLAS_LIBRARY AND ATLAS_LIBRARY AND F77BLAS_LIBRARY )
        set(LAPACK_LIBRARIES ${LAPACK_LIBRARY} ${BLAS_LIBRARY} ${ATLAS_LIBRARY} ${F77BLAS_LIBRARY} gfortran)
        message(STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
    endif( LAPACK_LIBRARY AND BLAS_LIBRARY AND ATLAS_LIBRARY AND F77BLAS_LIBRARY )
endif(ZHEEVR_FOUND_IN_CLAPACK)

if( LAPACK_LIBRARIES )
else(1)
    message(FATAL_ERROR "LAPACK not found: ${LAPACK_LIBRARIES}")
endif( LAPACK_LIBRARIES )

find_path( LTDL_INCLUDE_DIR
	ltdl.h /usr/include /usr/local/include /sw/include)
find_library( LTDL_LIBRARY NAMES ltdl PATHS /usr/lib /usr/local/lib /sw/lib)
if( LTDL_INCLUDE_DIR AND LTDL_LIBRARY )
    message(STATUS "LTDL found: ${LTDL_INCLUDE_DIR} ${LTDL_LIBRARY}")
else(1)
    message(FATAL_ERROR "LTDL not found: ${LTDL_INCLUDE_DIR} ${LTDL_LIBRARY}")
endif( LTDL_INCLUDE_DIR AND LTDL_LIBRARY )

find_library( ICONV_LIBRARY NAMES iconv PATHS /usr/lib /usr/local/lib /sw/lib)
if( ICONV_LIBRARY )
    message(STATUS "iconv found: ${ICONV_LIBRARY}")
else(1)
    message(STATUS "iconv not found, use system iconv() instead")
    set(ICONV_LIBRARY "")
endif( ICONV_LIBRARY )

find_path( FTGL_INCLUDE_DIR
	FTGL/FTGL.h /usr/include /usr/local/include /sw/include)
find_library( FTGL_LIBRARY ftgl /usr/lib /usr/local/lib /sw/lib)
if( FTGL_INCLUDE_DIR AND FTGL_LIBRARY )
    message(STATUS "FTGL found: ${FTGL_INCLUDE_DIR} ${FTGL_LIBRARY}")
else(1)
    message(FATAL_ERROR "FTGL not found: ${FTGL_INCLUDE_DIR} ${FTGL_LIBRARY}")
endif( FTGL_INCLUDE_DIR AND FTGL_LIBRARY )

find_library( GL_LIBRARY NAMES GL PATHS /usr/lib /usr/local/lib /sw/lib)
if( GL_LIBRARY )
    message(STATUS "GL found: ${GL_LIBRARY}")
else(1)
    message(FATAL_ERROR "GL not found: ${GL_LIBRARY}")
endif( GL_LIBRARY )

find_package( GLU REQUIRED)
message(STATUS "GLU found: ${GLU_INCLUDE_PATH} ${GLU_LIBRARY}")

#find_package( GLUT REQUIRED)
#message(STATUS "GLUT found: ${GLUT_INCLUDE_DIR} ${GLUT_LIBRARIES}")

#find_package(freetype REQUIRED)
find_program(FREETYPE_FOUND freetype-config)
if( FREETYPE_FOUND )
    execute_process(COMMAND freetype-config --prefix OUTPUT_VARIABLE FREETYPE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_path( FREETYPE_INCLUDE_DIR
    	freetype/freetype.h "${FREETYPE_PREFIX}/include/freetype2" "${FREETYPE_PREFIX}/include")
    execute_process(COMMAND freetype-config --libs OUTPUT_VARIABLE FREETYPE_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "FREETYPE found: ${FREETYPE_INCLUDE_DIR} ${FREETYPE_LIBRARY}")
else(1)
    message(FATAL_ERROR "FREETYPE not found: ${FREETYPE_INCLUDE_DIR} ${FREETYPE_LIBRARY}")
endif( FREETYPE_FOUND )
set(FREETYPE_LIBRARIES ${FREETYPE_LIBRARY})
#message(STATUS "Freetype found: ${FREETYPE_INCLUDE_DIR} ${FREETYPE_LIBRARIES}")

find_path( NIDAQMX_INCLUDE_DIR
	NIDAQmx.h /usr/include /usr/local/include /sw/include)
find_library( NIDAQMX_LIBRARY nidaqmx /usr/lib /usr/local/lib /sw/lib)
if(NIDAQMX_INCLUDE_DIR AND NIDAQMX_LIBRARY)
    add_definitions(-DHAVE_NI_DAQMX)
    message(STATUS "NIDAQMX found: ${NIDAQMX_INCLUDE_DIR} ${NIDAQMX_LIBRARY}")
else(1)
    set(NIDAQMX_INCLUDE_DIR "")
    set(NIDAQMX_LIBRARY "")
    message(STATUS "NIDAQMX not found: ${NIDAQMX_INCLUDE_DIR} ${NIDAQMX_LIBRARY}")
endif(NIDAQMX_INCLUDE_DIR AND NIDAQMX_LIBRARY)

find_path( LINUXGPIB_INCLUDE_DIR
	gpib/ib.h /usr/include /usr/local/include /sw/include)
find_library( LINUXGPIB_LIBRARY gpib /usr/lib /usr/local/lib /sw/lib)
if( LINUXGPIB_INCLUDE_DIR AND LINUXGPIB_LIBRARY )
    add_definitions(-DHAVE_LINUX_GPIB)
    message(STATUS "LINUXGPIB found: ${LINUXGPIB_INCLUDE_DIR} ${LINUXGPIB_LIBRARY}")
else(1)
    set(LINUXGPIB_INCLUDE_DIR "")
    set(LINUXGPIB_LIBRARY "")
    message(STATUS "LINUXGPIB not found: ${LINUXGPIB_INCLUDE_DIR} ${LINUXGPIB_LIBRARY}")
endif( LINUXGPIB_INCLUDE_DIR AND LINUXGPIB_LIBRARY )

#TEST_BIG_ENDIAN(__BIGENDIAN__)

include_directories(${KDE4_INCLUDES} ${KDE4_INCLUDE_DIR} ${QT_INCLUDES} 
 ${Boost_INCLUDE_DIR})

#add more include direcories, using INCLUDE_DIRECTORIES()
#add more link libraries, using TARGET_LINK_LIBRARIES()
#add some compile switches, using ADD_DEFINITIONS()
add_definitions (-DQT3_SUPPORT -DQT3_SUPPORT_WARNINGS)

#add some "configure" checks, see How To Do Platform Checks and How To Find Installed Software
#take special care of former libtool convenience libraries. 
#They are not supported by cmake, instead there will be a file ConvenienceLibs.cmake created. 
#In this file you will find for every convenience lib a variable, which contains all source files of this convenience lib. 
#For the targets which linked to this convenience lib, just add the variable to the sources.
#a file AdditionalInfo.txt will be created. There you will see all *.in and *.in.in files of your project. 
#The stuff done in these files will have to be converted manually to cmake.

set_property(GLOBAL PROPERTY ENABLE_EXPORTS TRUE)
set (CMAKE_CXX_FLAGS "-fexceptions")
if(APPLE)
  set(CMAKE_MODULE_LINKER_FLAGS "-Wl,-undefined,dynamic_lookup" ) 
  set(CMAKE_EXE_LINKER_FLAGS "-Wl,-all_load -Wl,-dynamic")
else(1)
  set(CMAKE_MODULE_LINKER_FLAGS "-module")
  set(CMAKE_EXE_LINKER_FLAGS "-dlpreopen self -export-dynamic")
  set(CMAKE_CXX_LINK_EXECUTABLE "libtool --mode=link --tag=CXX ${CMAKE_CXX_LINK_EXECUTABLE}")
  set(CMAKE_CXX_CREATE_SHARED_MODULE "libtool --mode=link --tag=CXX ${CMAKE_CXX_CREATE_SHARED_MODULE}")
endif(APPLE)
message(STATUS "CMAKE_MODULE_LINKER_FLAGS: ${CMAKE_MODULE_LINKER_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")

add_subdirectory(kame)
add_subdirectory(modules)
add_subdirectory(po)
add_subdirectory(doc)

########### install files ###############

install(FILES  AUTHORS COPYING ChangeLog README DESTINATION ${DATA_INSTALL_DIR}/kame)

