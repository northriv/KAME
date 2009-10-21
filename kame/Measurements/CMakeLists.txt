
include_directories(${KDE4_INCLUDES} ${KDE4_INCLUDE_DIR} ${QT_INCLUDES} )


########### install files ###############

install(FILES  thermometers_ku.kam  	new_thermometer_sample.kam  	testdriver.kam DESTINATION ${DATA_INSTALL_DIR}/kame/Measurements)



#original Makefile.am contents follow:

#EXTRA_DIST = \
#	thermometers_ku.kam \
#	new_thermometer_sample.kam \
#	testdriver.kam
#
#rcdir = $(kde_datadir)/kame/Measurements
#rc_DATA = \
#	thermometers_ku.kam \
#	new_thermometer_sample.kam \
#	testdriver.kam
#	