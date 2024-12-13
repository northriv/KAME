#KAME extention by python
import time
import datetime
import inspect
for i in range(50):
    time.sleep(0.3) #todo wainting for module loading.
    import kame #imports kame drivers.
    if hasattr(kame, "XDMM"):
        break
time.sleep(0.5) #todo wainting for module loading.
from kame import *

#Defines KAME driver class inside python.
class Test4Res(XPythonSecondaryDriver):
    def __init__(self, name, runtime, tr, meas):
        XPythonSecondaryDriver.__init__(self, name, runtime, tr, meas) #pybind11 requires this., otherwise TypeError throws.

        #setups Qt UIs
        form = self.loadUIFile(':/script/formpytestdriver.ui')
        #adds nodes
        self.insert(XDoubleNode("Wait", False))
        self.insert(XBoolNode("Control", False))
        self.insert(XUIntNode("DMMChannel", False))

        entry = XScalarEntry("Resistance", False, self, "%.3g")
        self.insert(entry)  
        meas["ScalarEntries"].insert(tr, entry)
        #Driver selecters
        self.insert(XDMMItemNode("DMM", False, tr, meas["Drivers"], True)) # choosing XDMM-based class from the driver list.
        self.insert(XDCSourceItemNode("DCSource", False, tr, meas["Drivers"], True)) # choosing XDCSource-based class from the driver list.
        #stores UI connectors during the lifetime.
        self.conns = [
            #synchronizes nodes and UIs.
            XQDoubleSpinBoxConnector(self["Wait"], form.findChildWidget("Wait")),
            XQToggleButtonConnector(self["Control"], form.findChildWidget("Control")),
            XQSpinBoxUnsignedConnector(self["DMMChannel"], form.findChildWidget("DMMChannel")),
            XQComboBoxConnector(self["DMM"], form.findChildWidget("DMM"), tr),
            XQComboBoxConnector(self["DCSource"], form.findChildWidget("DCSource"), tr),
            XQLCDNumberConnector(self["Resistance"]["Value"], form.findChildWidget("lcdResistance")),
            ]
        #setups link btw this driver and a selected driver.
        self.connect(self["DMM"])
        self.connect(self["DCSource"])
        return

    #Pickups valid snapshots before going to analyze().
    # shot_self: Snapshot for self.
    # shot_emitter: Snapshot for the event emitting driver.
    # shot_others: Snapshot for the other connected dirvers.
    def checkDependency(self, shot_self, shot_emitter, shot_others, emitter):
        dmm = shot_this[self["DMM"]].get() #selected driver.
        dcsrc = shot[self["DCSource"]].get() #selected driver.
        if emitter == dmm:
            shot_dcsrc = shot_others
            shot_dmm = shot_emitter
            wait = float(tr[self["Wait"]]) * 1e-3 #[s]
            if shot_dmm[dmm].timeAwared() < shot_dcsrc[dcsrc].time() + wait:
                return False
            return True #Good, approved
        if not bool(shot_self[self["Control"]]) and emitter == dcsrc:
            #dc source is controled by others.
            return True #Good, approved
        return False #skipping this record.

    #Analyzes data acquired by the connected drivers.
    #Never include I/O operations, because transaction might repreat many times.
    def analyze(self, tr, shot_emitter, shot_others, emitter):
        if emitter == dcsrc:
            shot_dmm = shot_others
            shot_dcsrc = shot_emitter
        else:
            shot_dcsrc = shot_others
            shot_dmm = shot_emitter
        
        shot_dmm = shot_others
        shot_dcsrc = shot_emitter
        dmm = tr[self["DMM"]].get() #selected driver.
        dcsrc = tr[self["DCSource"]].get() #selected driver.
        dmmch = int(tr[self["DMMChannel"]])
        volt = shot_dmm[dmm].value(dmmch)
        curr = float(shot_dcsrc[dcsrc]["Value"])
 
        storage = tr[self].local() #dict for tr[self]
        try:
            recent = storage["Recent"]
        except KeyError:
            storage[ "Recent"] = [] #for the first time
            recent = storage["Recent"]

        if emitter == dcsrc:
            #this driver is NOT in charge of switching dc source polarity.
            #eliminating bad events (dc source changed during the measurement).
            recent = [x for x in recent if x['dmm_start'] < shot_dcsrc[dcsrc].timeAwared()]
            storage["Recent"] = recent
        else:        
            recent.append({'dmm_start':shot_emitter[dmm].timeAwared(),
                'dmm_fin':shot_emitter[dmm].time(),
                'curr':curr, 'vold':volt})
            if recent[-1]['start'] - recent[0]['start'] > 30:
                del recent[0] #erase too old record.

            if not bool(tr[self["Control"]]):
                #this driver is NOT in charge of switching dc source polarity.
                skipRecord()
            
        if curr < 0:
            skipRecord() #waits for positive current.

        #searching for the newest record with +curr.
        for idx in range(len(recent) - 1, 0, -1):
            if recent[idx]['curr'] == curr:
                volt = recent[idx]['volt']
                break

        #searching for the newest record with -curr.
        for idx in range(len(recent) - 1, 0, -1):
            if recent[idx]['curr'] == -curr:
                res = (volt - recent[idx]['volt']) / curr / 2
                self["Resistance"].value(tr, res)
                return
        skipRecord() #no valid record.

    #may perform I/O ops or graph ops using the snapshot after analyze().
    def visualize(self, shot):
        if bool(tr[self["Control"]]):
            #this driver is in charge of switching dc source polarity.
            dcsrc = shot[self["DCSource"]].get() #selected driver.
            try:
                recent = storage["Recent"]
            except KeyError:
                return
            curr = recent["curr"]
            #switching polarity.
            dcsrc.changeValue(0, -curr)
        return

#Declares that python-side driver to C++ driver list.
XPythonSecondaryDriver.exportClass("Test4Res", Test4Res, "Test python-based driver: 4-Terminal Resistance Measumrent")
