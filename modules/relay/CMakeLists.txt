add_subdirectory(core)

include_directories(
    ${CMAKE_SOURCE_DIR}/modules/charinterface
    ${CMAKE_SOURCE_DIR}/modules/motor/core
    ${CMAKE_SOURCE_DIR}/modules/relay/core)

########### next target ###############
set(relay_SRCS
    userrelay.cpp
 )

add_library(relay MODULE ${relay_SRCS})
target_link_libraries(relay kame_relaycore kame_motorcore kame_charinterface ${MODULE_LINKER_FLAGS})
########### install files ###############
install(TARGETS relay LIBRARY  DESTINATION ${KAME_MODULE_INSTALL_DIR})
