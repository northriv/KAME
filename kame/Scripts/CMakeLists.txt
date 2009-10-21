
include_directories(${KDE4_INCLUDES} ${KDE4_INCLUDE_DIR} ${QT_INCLUDES} )


########### install files ###############

install(FILES  test.seq DESTINATION ${DATA_INSTALL_DIR}/kame/Sequences)



#original Makefile.am contents follow:

#EXTRA_DIST = \
#	test.seq
#
#rcdir = $(kde_datadir)/kame/Sequences
#rc_DATA = \
#	test.seq
#	