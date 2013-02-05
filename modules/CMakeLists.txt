include_directories(
    ${CMAKE_SOURCE_DIR}/kame 
    ${CMAKE_SOURCE_DIR}/kame/analyzer
    ${CMAKE_SOURCE_DIR}/kame/driver)

add_subdirectory(charinterface)
add_subdirectory(counter)
add_subdirectory(dmm)
add_subdirectory(dcsource)
add_subdirectory(tempcontrol)
add_subdirectory(magnetps)
add_subdirectory(lia)
add_subdirectory(funcsynth)
add_subdirectory(testdriver)
add_subdirectory(dso)
add_subdirectory(levelmeter)
add_subdirectory(sg)
add_subdirectory(fourres)
add_subdirectory(nmr)
add_subdirectory(montecarlo)
add_subdirectory(nidaq)
add_subdirectory(networkanalyzer)
add_subdirectory(motor)
add_subdirectory(flowcontroller)

add_dependencies(testdriver kame)
########### install files ###############
