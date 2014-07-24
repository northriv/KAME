add_subdirectory(pulsercore)

########### next target ###############
include_directories(
    ${CMAKE_SOURCE_DIR}/modules/dmm/core
     ${CMAKE_SOURCE_DIR}/modules/dso/core 
     ${CMAKE_SOURCE_DIR}/modules/motor/core 
     ${CMAKE_SOURCE_DIR}/modules/magnetps/core
     ${CMAKE_SOURCE_DIR}/modules/networkanalyzer/core
     ${CMAKE_SOURCE_DIR}/modules/charinterface
     ${CMAKE_SOURCE_DIR}/kame/math
     ${CMAKE_SOURCE_DIR}/kame/graph
     ${CMAKE_SOURCE_DIR}/kame/icons
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
    pulseanalyzer.cpp
    autolctuner.cpp
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
  	autolctunerform.ui
 )
 
kde4_add_library(nmr MODULE ${nmr_SRCS})
target_link_libraries(nmr kame_nmrpulsercore kame_magnetpscore kame_sgcore kame_dmmcore kame_dsocore 
	kame_networkanalyzercore kame_motorcore
	${MODULE_LINKER_FLAGS})
kde4_add_library(nmrpulser MODULE ${nmrpulser_SRCS})
target_link_libraries(nmrpulser kame_nmrpulsercore
	 kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS nmr nmrpulser LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

##AM_CXXFLAGS = $(CXXFLAGS) \
##	-O6  \
##	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
##	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations
