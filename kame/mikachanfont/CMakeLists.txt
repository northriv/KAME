
include_directories(${KDE4_INCLUDES} ${KDE4_INCLUDE_DIR} ${QT_INCLUDES} )


########### install files ###############

install(FILES  ChangeLog COPYRIGHT COPYRIGHT.ja README README.ja mikachan.ttf DESTINATION ${DATA_INSTALL_DIR}/kame/mikachan)



#original Makefile.am contents follow:

#EXTRA_DIST = ChangeLog COPYRIGHT COPYRIGHT.ja README README.ja mikachan.ttf
#
#rcdir = $(kde_datadir)/kame/mikachan
#rc_DATA = ChangeLog COPYRIGHT COPYRIGHT.ja README README.ja mikachan.ttf
