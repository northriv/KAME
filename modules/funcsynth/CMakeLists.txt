include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface)

########### next target ###############
set(funcsynth_SRCS
    funcsynth.cpp
    userfuncsynth.cpp
 )

kde4_add_ui_files(funcsynth_SRCS
  funcsynthform.ui  )

kde4_add_library(funcsynth MODULE ${funcsynth_SRCS})
target_link_libraries(funcsynth kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS funcsynth LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})

