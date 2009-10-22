add_subdirectory(pulsercore)

########### next target ###############
include_directories(
    ${CMAKE_SOURCE_DIR}/modules/dmm/core
     ${CMAKE_SOURCE_DIR}/modules/dso/core 
     ${CMAKE_SOURCE_DIR}/modules/magnetps/core
     ${CMAKE_SOURCE_DIR}/modules/charinterface
     ${CMAKE_SOURCE_DIR}/kame/math
     ${CMAKE_SOURCE_DIR}/kame/graph
     ${CMAKE_BINARY_DIR}/kame/graph
     ${CMAKE_SOURCE_DIR}/modules/nmr/pulsercore
     ${CMAKE_SOURCE_DIR}/modules/sg/core)

set(nmr_SRCS
	nmrpulse.cpp 
	nmrspectrumsolver.cpp 
	nmrspectrum.cpp 
	nmrfspectrum.cpp 
	nmrrelaxfit.cpp 
	nmrrelax.cpp 
 )
set(nmrpulser_SRCS
	pulserdriversh.cpp 
	pulserdriverh8.cpp 
 )

kde4_add_ui_files(nmr_SRCS
    nmrfspectrumform.ui
    nmrpulseform.ui
    nmrrelaxform.ui
    nmrspectrumform.ui
)
 
kde4_add_library(nmr MODULE ${nmr_SRCS} ${nmrpulser_SRCS})
target_link_libraries(nmr charinterface 
     dmmcore dsocore sgcore magnetpscore nmrpulsercore)
########### install files ###############
install(TARGETS nmr LIBRARY  DESTINATION ${LIB_INSTALL_DIR}/kame/modules)

##AM_CXXFLAGS = $(CXXFLAGS) \
##	-O6  \
##	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
##	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
