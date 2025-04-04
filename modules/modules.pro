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
    liacore\
    lia\
    magnetpscore\
    magnetps\
    motorcore\
    motor\
    networkanalyzercore\
    networkanalyzer\
    nidaq\
    digilentwf\
    nmrpulsercore\
    nmrpulser\
    thamway\
    nmr\
    sgcore\
    sg\
    tempcontrol\
    qdcore\
    qd\
    gauge\
    pumpcontroller\
    opticscore\
    optics\
    arbfunc\
    twoaxis\
    python

unix: SUBDIRS +=    montecarlo\

counter.depends += charinterface
dcsourcecore.file = dcsource/core/dcsourcecore.pro
dcsource.depends += dcsourcecore
dmmcore.file = dmm/core/dmmcore.pro
dmm.depends += dmmcore charinterface
funcsynth.depends += charinterface
flowcontrollercore.file = flowcontroller/core/flowcontrollercore.pro
flowcontroller.depends += flowcontrollercore charinterface
levelmetercore.file = levelmeter/core/levelmetercore.pro
levelmeter.depends += levelmetercore charinterface
magnetpscore.file = magnetps/core/magnetpscore.pro
magnetps.depends += magnetpscore charinterface
motorcore.file = motor/core/motorcore.pro
motor.depends += motorcore charinterface
liacore.file = lia/core/liacore.pro
lia.depends += liacore charinterface
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
nmr.depends += nmrpulsercore dmmcore dsocore sgcore magnetpscore motorcore networkanalyzercore qdcore
fourres.depends += dmmcore dcsourcecore
nidaq.depends += nmrpulsercore dsocore
digilentwf.depends += dsocore
tempcontrol.depends += dcsourcecore flowcontrollercore
thamway.file = nmr/thamway/thamway.pro
thamway.depends += nmrpulsercore sgcore networkanalyzercore
qdcore.file = qd/core/qdcore.pro
qd.depends += qdcore charinterface
gauge.depends += charinterface
pumpcontroller.depends += charinterface
arbfunc.depends += charinterface
twoaxis.depends += motorcore
opticscore.file = optics/core/opticscore.pro
optics.depends += opticscore sgcore liacore motorcore charinterface
python.depends += nmrpulsercore dmmcore dsocore sgcore magnetpscore
python.depends += motorcore networkanalyzercore qdcore opticscore
python.depends += sgcore liacore tempcontrol levelmetercore charinterface
python.depends += nmr optics
