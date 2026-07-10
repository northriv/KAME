set(relaycore_SRCS
    relaydriver.cpp
 )

kde4_add_ui_files(relaycore_SRCS
  relayform.ui  )

kde4_add_library(kame_relaycore SHARED ${relaycore_SRCS})
target_link_libraries(kame_relaycore ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS kame_relaycore LIBRARY  DESTINATION ${KAME_LIB_INSTALL_DIR})
