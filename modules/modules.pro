TEMPLATE = subdirs

SUBDIRS += testdriver\


dcsourcecore.file = dcsource/core/dcsourcecore.pro
dcsource.depends += dcsourcecore
dmmcore.file = dmm/core/dmmcore.pro
dmm.depends += dmmcore charinterface
flowcontrollercore.file = flowcontroller/core/flowcontrollercore.pro
flowcontroller.depends += flowcontrollercore charinterface
levelmetercore.file = levelmeter/core/levelmetercore.pro
levelmeter.depends += levelmetercore charinterface
magnetpscore.file = magnetps/core/magnetpscore.pro
magnetps.depends += magnetpscore charinterface
motorcore.file = motor/core/motorcore.pro
motor.depends += motorcore charinterface
sgcore.file = sg/core/sgcore.pro
sg.depends += sgcore charinterface
dsocore.file = dso/core/dsocore.pro
dsocore.depends += sgcore
dso.depends += dsocore charinterface
networkanalyzercore.file = networkanalyzer/core/networkanalyzercore.pro
networkanalyzer.depends += networkanalyzercore charinterface
nmrpulsercore.file = nmr/pulsercore/nmrpulsercore.pro
nmrpulser.file = nmr/nmrpulser.pro
nmrpulser.depends += nmrpulsercore charinterface
nmr.depends += nmrpulsercore dmmcore dsocore sgcore magnetpscore motorcore networkanalyzercore
fourres.depends += dmmcore dcsourcecore
nidaq.depends += nmrpulsercore dsocore
tempcontrol.depends += dcsourcecore flowcontrollercore

