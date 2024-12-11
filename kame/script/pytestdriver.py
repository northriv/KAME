#KAME extention by python
import time
import datetime
import inspect
for i in range(50):
    time.sleep(0.3) #todo wainting for module loading.
    import kame #imports kame drivers.
    Classes = str([y[0] for y in inspect.getmembers(kame, inspect.isclass)])
    if "XDMM" in Classes:
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
    # shot_this: Snapshot for self.
    # shot_emitter: Snapshot for the event emitting driver.
    # shot_others: Snapshot for the other connected dirvers.
    def checkDependency(shot_this, shot_emitter, shot_others, emitter):
        dmm = shot_this[self["DMM"]].get() #selected driver.
        dcsource = shot[self["DCSource"]].get() #selected driver.
        if emitter != dmm:
            return False #skipping this record.
        wait = float(shot_this[self["Wait"]]) * 1e-3 #[s]
        if shot_emitter[dmm].timeAwared() < shot_others[dcsource].time() + wait:
            return False #Bad case, DC source has been changed after dmm reading.
        return True #Good, approved

    #Analyzes data acquired by the connected drivers.
    #Never include I/O operations, because transaction might repreat many times.
    def analyze(tr, shot_emitter, shot_others, emitter):
        dmm = tr[self["DMM"]].get() #selected driver.
        dcsource = tr[self["DCSource"]].get() #selected driver.
        dmmch = int(tr[self["DMMChannel"]])
        curr = float(shot_others[dcsource]["Value"])
        try:
            recent = tr["Recent"]
        except ValueError:
            tr[ "Recent"]
            skipRecord()
        except IndexError:
            skipRecord()
        if prev_curr != -curr:
            skipRecord()
        
        tr["Current"] = curr
        volt = shot_emitter[dmm].value(dmmch)
        tr["Resistance"] = volt
        res = volt / curr
        self["Resistance"].value(tr, res)
        return

    #may perform I/O ops or graph ops using the snapshot after analyze().
    def visualize(shot):
        dcsource = shot[self["DCSource"]].get() #selected driver.
        curr = float(shot_others[dcsource]["Value"])
        dcsource.changeValue(0, -curr)
        return

#Declares that python-side driver to C++ driver list.
XPythonSecondaryDriver.exportClass("Test4Res", Test4Res, "Test 4res terminal meas.")
