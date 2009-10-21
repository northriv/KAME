include_directories(${CMAKE_SOURCE_DIR}/kame) 

add_subdirectory(forms)
add_subdirectory(math)
add_subdirectory(script)
add_subdirectory(driver)
add_subdirectory(graph)
add_subdirectory(thermometer)
add_subdirectory(analyzer)
add_subdirectory(icons)
add_subdirectory(mikachanfont)

include_directories(
    ${CMAKE_SOURCE_DIR}/kame/forms
    ${CMAKE_BINARY_DIR}/kame/forms
    ${CMAKE_SOURCE_DIR}/kame/script
    ${CMAKE_SOURCE_DIR}/kame/driver
    ${CMAKE_SOURCE_DIR}/kame/analyzer
    ${CMAKE_SOURCE_DIR}/kame/thermometer
    ${CMAKE_BINARY_DIR}/kame/thermometer
    ${CMAKE_SOURCE_DIR}/kame/icons
    ${RUBY_INCLUDE_PATH} ${LTDL_INCLUDE_DIR})

########### next target ###############

set(kame_SRCS
    measure.cpp
    kame.cpp
    xnodeconnector.cpp
    xsignal.cpp
    xscheduler.cpp
    thread.cpp
    threadlocal.cpp
    rwlock.cpp
    support.cpp
    xtime.cpp
    xnode.cpp
    xlistnode.cpp
    xitemnode.cpp
    main.cpp
    )

kde4_add_executable(kame ${kame_SRCS})
target_link_libraries(kame thermometer forms kamemath script analyzer driver icons graph 
    m
    ${GLU_LIBRARY} ${FTGL_LIBRARY} ${FREETYPE_LIBRARIES}
    ${RUBY_LIBRARY} ${GSL_LIBRARY} ${FFTW3_LIBRARY}
    ${LAPACK_LIBRARIES}
    ${ICONV_LIBRARY} ${ZLIB_LIBRARIES}
    ${LTDL_LIBRARY}
    ${KDE4_KDEUI_LIBS} ${KDE4_KIO_LIBS} ${QT_QT3SUPPORT_LIBRARY} ${QT_QTOPENGL_LIBRARY})
 
set_target_properties(kame PROPERTIES ENABLE_EXPORTS ON)

########### install files ###############
install(TARGETS kame ${INSTALL_TARGETS_DEFAULT_ARGS})

install(FILES  kame.desktop DESTINATION ${XDG_APPS_INSTALL_DIR})
install(FILES  DESTINATION ${DATA_INSTALL_DIR}/kame)

kde4_install_icons(${ICON_INSTALL_DIR})

