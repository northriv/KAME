#KAME xpyhonsuppport start-up code
#fundamental imports
import time
import sys
import threading
import traceback
import inspect
import datetime
import os
import multiprocessing
if os.name == 'nt':
	#needed to import system modules.
	for p in os.environ['PATH'].split(os.pathsep):
		if os.path.isdir(p):
			os.add_dll_directory(p)	
else:
	try:
		multiprocessing.set_start_method('fork') #needed for Apple silicon
	except Exception:
		pass #allowed only once.

HasIPython = False
try:
	#optional imports.
	import ctypes
	import numpy as np
	import pdb

	from ipykernel.eventloops import register_integration
	import IPython
	HasIPython = True
#	import matplotlib
#	matplotlib.use('Agg') #GUI does not work yet
#	import matplotlib.pyplot as plt
except (ImportError, ModuleNotFoundError):
	pass
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

import io

class MyDefIO:
	@staticmethod
	def write(s):
		if hasattr(TLS, 'xscrthread') and TLS.xscrthread:
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
			STDERR.write(s) #redirecting to terminal, for debug purpose.

	@staticmethod
	def readline():
		if hasattr(TLS, 'xscrthread') and TLS.xscrthread:
			while not is_main_terminated():		
				ret = my_defin(TLS.xscrthread)
				if ret:
					break #no input detected.
				time.sleep(0.2)
			return ret
		else:
			return STDIN.readline() #redirecting to terminal, for debug purpose.

	@staticmethod
	def read():
		return stdio.readline()
	@staticmethod
	def flush():
		pass
	@staticmethod
	def fileno():
		return STDOUT.fileno()
	@staticmethod
	def isatty():
		return True
	@property
	def encoding():
		return STDOUT.encoding
	@property
	def buffer():
		return io.BytesIO()

class MyDefOErr(MyDefIO):
	@staticmethod
	def write(s):
		STDERR.write(s) #redirecting to terminal, for debug purpose.
		if s[-1] == '\n':
			s = s[0:-1]
		if hasattr(TLS, 'xscrthread') and TLS.xscrthread:
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

def loadSequence(xpythread, filename):
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
	except Exception:
		sys.stderr.write(str(traceback.format_exc()))
	TLS.xscrthread["Status"] = ""

def kame_pybind_one_iteration():
	try:
		#For node browser pane
		PyInfoForNodeBrowser().set(str([y[0] for y in inspect.getmembers(LastPointedByNodeBrowser(), inspect.ismethod)]))

		for xpythread in XScriptingThreads():
			xpythread_status = xpythread["Status"]
			xpythread_action = xpythread["Action"]
			xpythread_threadid = xpythread["ThreadID"]
			xpythread_filename = xpythread["Filename"]
			threadlist = [str(pythread) for pythread in threading.enumerate()]
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
					thread = threading.Thread(daemon=True, target=loadSequence, args=(xpythread, filename))
					thread.start()
					time.sleep(0.3)
				if action == "kill":
					if os.name == 'posix':
						time.sleep(0.5)
						if action == "kill":
							#cannot be killed by timer.
							ctypes.pythonapi.PyThreadState_SetAsyncExc(
							    ctypes.c_long(int(str(xpythread_threadid))),
								ctypes.py_object(SystemExit)
							)
	except EOFError:
		pass
	except Exception:
		sys.stderr.write(str(traceback.format_exc()))

def findExecutables(prog):
	import glob
	paths = os.environ['PATH'].split(os.pathsep)
	if os.name == 'posix':
		paths.extend(['/opt/homebrew/bin', '/opt/local/bin'])
	ret = []
	for p in paths:
		if os.path.isdir(p):
			ret.extend(glob.glob(p + os.sep + prog))
			ret.extend(glob.glob(p + os.sep + prog + os.extsep + "*"))
			ret.extend(glob.glob(p + os.sep + prog + "-[3-9]*"))
	return ret

def listOfJupyterPrograms():
	return findExecutables('jupyter')

def launchJupyterConsole(prog, console):
#	import ipykernel
#	import re
#	json = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group()
	from ipykernel.kernelapp import IPKernelApp
	app = IPKernelApp.instance()
	json = app.connection_file
	if not len(json):
		sys.stderr.write("IPython kernel could not be started.")
	print("Using existing kernel = " + json)
	args = [repr(prog), '--existing', json,] #repr is to disable escape sequence.
	#multiprocessing.Process is insane.
	# p = multiprocessing.Process(target = mylauncher, args=(console, args,))
	#p.start()
	import subprocess
	args.insert(1, console)
	if console == 'console':
		subprocess.Popen(args, stdout=STDOUT, stderr=STDERR, stdin=STDIN)
	else:
		subprocess.Popen(args, stdout=STDOUT, stderr=STDERR, stdin=STDIN)

#import linecache
#linecache.clearcache()

if not HasIPython:
	print("#testing python interpreter.")
	#kame_pybind_main_loop
	while not is_main_terminated():
		time.sleep(MONITOR_PERIOD)
		kame_pybind_one_iteration()
else:

	@register_integration('kamepybind11')
	def loop_kamepysupport(kernel):
		import asyncio
		import nest_asyncio
		nest_asyncio.apply()

		poll_interval = kernel._poll_interval
		class Timer:
			def __init__(self, func):
				self.stdout = sys.stdout
				self.stderr = sys.stderr
				self.stdin = sys.stdin

				print(str(func))
				self.func = func

			def on_timer(self):
				sys.stdout = self.stdout
				sys.stderr = self.stderr
				sys.stdin = self.stdin

				loop = asyncio.get_event_loop()
				try:
					loop.run_until_complete(self.func())
				except Exception:
					kernel.log.exception("Error in message handler")

				self.stdout = sys.stdout
				self.stderr = sys.stderr
				self.stdin = sys.stdin
				sys.stdout = MyDefIO
				sys.stderr = MyDefOErr
				sys.stdin = MyDefIO

				# if not is_main_terminated():
				kame_pybind_one_iteration()
				time.sleep(poll_interval)

			def start(self):
				self.on_timer()  # Call it once to get things going.
				print("start\n")
				while not is_main_terminated():
					self.on_timer()
				print(str([y[0] for y in inspect.getmembers(kernel, inspect.ismethod)]))
				# from ipykernel.kernelapp import IPKernelApp
				# app = IPKernelApp.instance()
				# app.close()
				print("finish\n")

		kernel.timer = Timer(kernel.do_one_iteration)
		kernel.timer.start()

	# @loop_kamepysupport.exit
	# def loop_kamepysupport_exit(kernel):
	# 	try:
	# 		print("exit\n")
	# 		del kernel.timer
	# 	except (RuntimeError, AttributeError):
	# 		pass

	# First create a config object from the traitlets library
	from traitlets.config import Config
	c = Config()

	c.InteractiveShellApp.exec_lines = [
	    '%gui kamepybind11'
	]
#	c.InteractiveShell.colors = 'LightBG'
#	c.TerminalIPythonApp.display_banner = False
    #c.InteractiveShellApp.gui = 'kamepybind11' #does not work

	sys.stdout = STDOUT
	sys.stderr = STDERR
	sys.stdin = STDIN

	# Now starting ipython kernel.
	IPython.embed_kernel(config=c)

sys.stderr.write("bye")

sys.stdout = STDOUT
sys.stderr = STDERR
sys.stdin = STDIN

for thread in threading.enumerate():
	try:
		if thread != threading.current_thread():
			thread.join(timeout=0.5)
	except Exception as inst:
		sys.stderr.write(str(traceback.format_exc()))

sys.stderr.write("bye!\n")
