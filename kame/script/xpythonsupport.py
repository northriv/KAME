#KAME xpyhonsuppport start-up code
import time
import sys
import threading
import traceback
import inspect
import datetime
import ctypes
import numpy as np
from kame import *
STDOUT = sys.stdout
STDERR = sys.stderr
STDIN = sys.stdin

print("Hello! KAME Python support.")

#Thread-monitor
MONITOR_PERIOD=0.2

TLS = threading.local()
TLS.xscrthread = None# XScriptingThreads()[0]
TLS.logfile = None
class MyDefIO:
	def write(s):
		if TLS.xscrthread:
			if s[-1] == '\n':
				s = s[0:-1]
			for l in s.splitlines():
				if len(l) and l[0] == "#":
					l = l.replace("&", "&amp;")
					l = l.replace("<", "&lt;")
					l = l.replace(">", "&gt;")
					l = "<font color=#008800>" + l + "</font" 
				my_defout(TLS.xscrthread, l)
			if TLS.logfile:
				TLS.logfile.write(str(datetime.datetime.now()) + ":" + s + '\n')
		else:
			STDERR.write(s)

	def readline():
		if TLS.xscrthread:
			return my_defin(TLS.xscrthread)
		else:
			STDIN.readline()
		
class MyDefOErr:
	def write(s):
		STDERR.write(s)
		if s[-1] == '\n':
			s = s[0:-1]
		if TLS.xscrthread:
			s = s.replace("&", "&amp;")
			s = s.replace("<", "&lt;")
			s = s.replace(">", "&gt;")
			for l in s.splitlines():
				l = "<font color=#ff0000>" + l + "</font" 
				my_defout(TLS.xscrthread, l)
#this does not work why?
#			if TLS.logfile:
#				TLS.logfile.write("Err:" + str(datetime.datetime.now()) + ":" + s + '\n')

sys.stdout = MyDefIO
sys.stderr = MyDefOErr
sys.stdin = MyDefIO

event = threading.Event()

#do not use time.sleep() please.
def sleep(sec):
	if sec < 1.2:
		time.sleep(sec)
		return
	start = time.time()
	while True:
		remain = sec - (time.time() - start)
		if TLS.xscrthread:
			xpythread = TLS.xscrthread
			if str(xpythread["Action"]) == "kill":
				xpythread["Action"] = ""
				raise RuntimeError("Kill")
			if str(xpythread["Action"]) == "wakeup":
				xpythread["Action"] = ""
				xpythread["Status"] = "run"
				return #ignores remaining time
			if str(xpythread["Action"]) == "suspend":
				xpythread["Action"] = ""
				sec = 1e10
			if remain > 1e9:
				xpythread["Status"] = "sleep"
			else:
				xpythread["Status"] = "{}s @{}".format(int(remain), inspect.currentframe().f_back)
		if remain < 0:
			break
		event.wait(min([remain, 0.33]))
	if TLS.xscrthread:
		xpythread = TLS.xscrthread
		xpythread["Status"] = "run"

def loadSequence():
	TLS.xscrthread = xpythread #thread-local-storage
	TLS.logfile = None
	try:
		xpythread["ThreadID"] = str(threading.current_thread().native_id)
		xpythread["Status"] = "run"
		if "lineshell" in filename:
			print("#KAME Python interpreter>")
			exec(open(filename).read())
		else:
			with open(filename + ".log", mode='a') as logfile:
				TLS.logfile = logfile
				print("#" + str(threading.current_thread()) + " started.")
				exec(open(filename).read())
				print(str(threading.current_thread()) + " Finished.")
				TLS.logfile = None
	except Exception as inst:
		sys.stderr.write(str(traceback.format_exc()))
		pass
	TLS.xscrthread["Status"] = ""

print("#testing python interpreter.")
while not is_main_terminated():
	time.sleep(0.1)
	try:
		time.sleep(MONITOR_PERIOD)
		for xpythread in XScriptingThreads():
			xpythread_status = xpythread["Status"]
			xpythread_action = xpythread["Action"]
			xpythread_threadid = xpythread["ThreadID"]
			xpythread_filename = xpythread["Filename"]
			threadlist = [str(x) for x in threading.enumerate()]
			action = str(xpythread_action)
			if str(xpythread_threadid) in threadlist:
				pass
			else:
				if action == "starting":
					time.sleep(0.5)
					xpythread_action.set("")
					print("Starting a new thread")
					filename = str(xpythread_filename)
					print("Loading "+ filename)
					thread = threading.Thread(daemon=True, target=loadSequence)
					thread.start()
					time.sleep(0.3)
				if action == "kill":
					time.sleep(0.5)
					if action == "kill":
						#cannot be killed by timer.
						ctypes.pythonapi.PyThreadState_SetAsyncExc(
							ctypes.c_long(int(str(xpythread_threadid))), 
							ctypes.py_object(SystemExit)
						)
	except EOFError:
		pass
	except Exception as inst:
		sys.stderr.write(str(traceback.format_exc()))
		pass
for thread in threading.enumerate():
	thread.join()

print("bye!")

