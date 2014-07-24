include_directories(${CMAKE_SOURCE_DIR}/kame) 

add_subdirectory(forms)
add_subdirectory(math)
add_subdirectory(script)
add_subdirectory(driver)
add_subdirectory(graph)
add_subdirectory(thermometer)
add_subdirectory(analyzer)
add_subdirectory(icons)

include_directories(
    ${CMAKE_SOURCE_DIR}/kame/forms
    ${CMAKE_BINARY_DIR}/kame/forms
    ${CMAKE_SOURCE_DIR}/kame/script
    ${CMAKE_SOURCE_DIR}/kame/driver
    ${CMAKE_SOURCE_DIR}/kame/analyzer
    ${CMAKE_SOURCE_DIR}/kame/thermometer
    ${CMAKE_BINARY_DIR}/kame/thermometer
    ${CMAKE_SOURCE_DIR}/kame/math
    ${CMAKE_SOURCE_DIR}/kame/icons
    ${RUBY_INCLUDE_PATH} ${LTDL_INCLUDE_DIR})

########### next target ###############

set(kame_SRCS
    allocator.cpp
    measure.cpp
    kame.cpp
    xnodeconnector.cpp
    xsignal.cpp
    xscheduler.cpp
    xthread.cpp
    threadlocal.cpp
    support.cpp
    xtime.cpp
    xnode.cpp
    xlistnode.cpp
    xitemnode.cpp
    main.cpp
    )

set(kame_RESOURCES
	kame.qrc
	)
	
QT4_ADD_RESOURCES(kame_RESOURCES_RCC ${kame_RESOURCES})

kde4_add_executable(kame ${kame_SRCS} ${kame_RESOURCES_RCC})
target_link_libraries(kame forms thermometer kamemath script analyzer driver icons graph 
    m
    ${GLU_LIBRARY} 
    ${RUBY_LIBRARY} ${GSL_LIBRARY} ${FFTW3_LIBRARY}
    ${LAPACK_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${LTDL_LIBRARY}
    ${KDE4_KDEUI_LIBS} ${KDE4_KIO_LIBS} ${QT_QTOPENGL_LIBRARY})
 
set_target_properties(kame PROPERTIES ENABLE_EXPORTS ON)

########### install files ###############
install(TARGETS kame ${INSTALL_TARGETS_DEFAULT_ARGS})

install(FILES  kame.desktop DESTINATION ${XDG_APPS_INSTALL_DIR})
install(FILES  DESTINATION ${DATA_INSTALL_DIR}/kame)

kde4_install_icons(${ICON_INSTALL_DIR})

