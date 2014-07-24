TEMPLATE = subdirs

SUBDIRS += testdriver\
    charinterface\
    counter\
    dcsourcecore\
    dcsource\
    dmmcore\
    dmm\
    dsocore\
    dso\
    flowcontrollercore\
    flowcontroller\
    fourres\
    funcsynth\
    levelmetercore\
    levelmeter\
    lia\
    magnetpscore\
    magnetps\
    montecarlo\
    motorcore\
    motor\
    networkanalyzercore\
    networkanalyzer\
    nidaq\
    nmrpulsercore\
    nmr\
    sgcore\
    sg\
    tempcontrol

dcsourcecore.file = dcsource/core/dcsourcecore.pro
dcsource.depends += dcsourcecore
dmmcore.file = dmm/core/dmmcore.pro
dmm.depends += dmmcore
flowcontrollercore.file = flowcontroller/core/flowcontrollercore.pro
flowcontroller.depends += flowcontrollercore
levelmetercore.file = levelmeter/core/levelmetercore.pro
levelmeter.depends += levelmetercore
magnetpscore.file = magnetps/core/magnetpscore.pro
magnetps.depends += magnetpscore
motorcore.file = motor/core/motorcore.pro
motor.depends += motorcore
sgcore.file = sg/core/sgcore.pro
sg.depends += sgcore
dsocore.file = dso/core/dsocore.pro
dsocore.depends += sgcore
dso.depends += dsocore
networkanalyzercore.file = networkanalyzer/core/networkanalyzercore.pro
networkanalyzer.depends += networkanalyzercore
nmrpulsercore.file = nmr/pulsercore/nmrpulsercore.pro
nmr.depends += nmrpulsercore dmmcore dsocore sgcore magnetpscore motorcore networkanalyzercore
fourres.depends += dmmcore dcsourcecore
nidaq.depends += nmrpulsercore dsocore
tempcontrol.depends += dcsourcecore flowcontrollercore

