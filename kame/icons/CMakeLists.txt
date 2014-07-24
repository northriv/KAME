########### next target ###############
set(icons_SRCS
	icon.cpp
	kame-24x24-png.c)

kde4_add_library(icons STATIC ${icons_SRCS})


########### install files ###############

install(FILES  cdrom.png graph.png ruby.png DESTINATION ${DATA_INSTALL_DIR}/kame/pics)
#kde4_install_icons(${ICON_INSTALL_DIR})
