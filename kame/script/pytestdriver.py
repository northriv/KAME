#KAME extention by python
import datetime
from kame import *

#Defines KAME driver class inside python.
class Test4Res(XPythonSecondaryDriver):
    def __init__(self, name, runtime, tr, meas):
        XPythonSecondaryDriver.__init__(self, name, runtime, tr, meas) #pybind11 requires this., otherwise TypeError throws.
        form = self.loadUIFile(':/script/formpytestdriver.ui') #setups UI
        #adds nodes
        self.insert(XDoubleNode("Wait", False))
        self.insert(XBoolNode("Control", False))
        entry = XScalarEntry("Resistance", False, self, "%.3g")
        self.insert(entry)  
        meas["ScalarEntries"].insert(tr, entry)
        self.insert(XDriverItemNode("Driver1", False, tr, meas["Drivers"], True))
#        self.connect()
        #stores UI connectors during the lifetime.
        self.conns = [
            XQDoubleSpinBoxConnector(self["Wait"], form.findChildWidget("Wait")),
            XQToggleButtonConnector(self["Control"], form.findChildWidget("Control")),
            XQComboBoxConnector(self["Driver1"], form.findChildWidget("Driver1"), tr),
            ]
        return

    def checkDependency(shot_this, shot_emitter, shot_others, emitter):
        return True

    def analyze(tr, shot_emitter, shot_others, emitter):
        return

    def visualize(shot):
        return

#Declares that python-side driver to C++ driver list.
XPythonSecondaryDriver.exportClass("Test4Res", Test4Res, "Test 4res terminal meas.")
