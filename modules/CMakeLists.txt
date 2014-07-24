include_directories(
    ${CMAKE_SOURCE_DIR}/kame 
    ${CMAKE_SOURCE_DIR}/kame/analyzer
    ${CMAKE_SOURCE_DIR}/kame/driver)

add_subdirectory(testdriver)

add_dependencies(testdriver kame)
########### install files ###############
