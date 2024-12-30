#KAME extention by python
import time
import datetime
import inspect
for i in range(50):
    time.sleep(0.3) #waiting for module loading.
    import kame #imports kame drivers.
    if hasattr(kame, "XDMM"):
        break

#Once KAME modules have been successfully loaded, the above startup codes are not needed.
from kame import *

#sa pydrives.py

#The first example:
#Defines KAME driver class inside python.
#XPythonCharDeviceDriverWithThread, the base abstract class for KAME driver which has one free-running thread and one standard I/O interface.
class TestRandom(XPythonCharDeviceDriverWithThread):
    def __init__(self, name, runtime, tr, meas):
        XPythonCharDeviceDriverWithThread.__init__(self, name, runtime, tr, meas) #super().__init__ cannot be used.
        self["Interface"]["Device"] = "DUMMY" #this example does not need an external device.

        entry = XScalarEntry("X", False, self, "%.5f")
        self.insert(entry) #self does NOT belong to tr yet inside the constructor. Never use insert(tr,...).
        meas["ScalarEntries"].insert(tr, entry) #tr: transaction obj. during the creation.
        entry = XScalarEntry("Y", False, self, "%.3f [V]")
        self.insert(entry)
        meas["ScalarEntries"].insert(tr, entry) #tr: transaction obj. during the creation.

    def analyzeRaw(self, reader, tr):
        x = reader.pop_double() #reading FIFO, from execute(), or RAW data file.
        y = reader.pop_double()
        storage = tr[self].local() #dict for tr[self]
        storage["X"] = x #update the dict, to be used for further analysis.
        self["X"].value(tr, x) #update the scalar entry.
        storage["Y"] = y
        self["Y"].value(tr, y)

    def visualize(self, shot):
        return

    def execute(self, is_terminated):
        while not is_terminated():
            time.sleep(0.01)
            writer = RawData() #FIFO storage
            writer.push_double(float(np.random.random()))
            writer.push_double(float(np.random.random()))
            self.finishWritingRaw(writer, datetime.datetime.now(), datetime.datetime.now())
        return

#Declares that python-side driver to C++ driver list.
XPythonCharDeviceDriverWithThread.exportClass("TestRandom", TestRandom, "Test python-based driver: Random Number Generation")

#Second example, 1DMathTool
class Test1DMathTool(XPythonGraph1DMathTool):
    def __init__(self, name, runtime, tr, entries, driver, plot, entryname):
        XPythonGraph1DMathTool.__init__(self, name, runtime, tr, entries, driver, plot, entryname)  #super().__init__ cannot be used.
        self.setFunctor(lambda x, y: np.sum(np.array(y)))

XPythonGraph1DMathTool.exportClass("NumPySum", Test1DMathTool, "NumPy Sum")

#Third example, 2DMathTool
class Test2DMathTool(XPythonGraph2DMathTool):
    def __init__(self, name, runtime, tr, entries, driver, plot, entryname):
        XPythonGraph2DMathTool.__init__(self, name, runtime, tr, entries, driver, plot, entryname)  #super().__init__ cannot be used.
        self.setFunctor(lambda matrix, width, stride, numlines, coefficient: coefficient * np.sum(matrix))

XPythonGraph2DMathTool.exportClass("NumPySum", Test2DMathTool, "NumPy Sum")
