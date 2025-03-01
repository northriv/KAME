#KAME xpyhonsuppport start-up code
#fundamental imports
import time
import sys
import html
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
	import IPython #this import hinders from freeing XPython/XMeasure normally.
	from IPython.display import display
	# import ipywidgets
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

#For osx, module files are in kame.app/Contents/MacOSX/../Resources
#For win, module files are in Resources
KAME_ResourceDir = os.path.join(os.path.dirname(sys.executable), '../Resources' if sys.platform == 'darwin' else 'Resources')
sys.path.insert(0, KAME_ResourceDir) #adds resource folder for importable modules.

print("Hello! KAME Python support.")

#Thread-monitor
MONITOR_PERIOD=0.2

TLS = threading.local()
if HasIPython:
	XScriptingThreads()[0].setLabel("IPython kernel")
	XScriptingThreads()[0]["Action"] = ""
	XScriptingThreads()[0]["Status"] = "No connection"
	XScriptingThreads()[0]["Filename"] = "#Launch Jupyter client from \"Script\" menu."
	TLS.xscrthread = XScriptingThreads()[0]
else:
	TLS.xscrthread = None
TLS.logfile = None

import io

class MyDefIO:
	def write_html(self, s):
		if hasattr(TLS, 'xscrthread') and TLS.xscrthread:
			my_defout(TLS.xscrthread, s)
			if s and TLS.logfile:
				TLS.logfile.write(str(datetime.datetime.now()) + ":" + s + '\n')
				TLS.logfile.flush()
			return len(s)
		else:
			return STDERR.write(s) #redirecting to terminal, for debug purpose.
	def write_internal(self, s, flush = True, color = None, stderr = False):
		if not s:
			return 0
		if hasattr(TLS, 'xscrthread') and TLS.xscrthread:
			if flush:
				self.flush()
			if s[-1] == '\n':
				s = s[:-1] #both QTextBrowser and display(HTML) adds an extra linebreak at the end.
			escaped_s = html.escape(s) #to HTML
			escaped_s = escaped_s.replace('\r\n', '<br>').replace('\r', '<br>').replace('\n', '<br>') #linebreaks
			color_l = color
			if stderr:
				color_l = '#ff0000'
			elif len(s) and s[0] == "#":
				color_l = '#008800'
			if color_l:
				escaped_s = "<font color=\"{}\">".format(color_l) + escaped_s + "</font>" 
			else:
				escaped_s = "<font>" + escaped_s + "</font>" 
			if HasIPython and XScriptingThreads()[0] == TLS.xscrthread:
				if not NOTEBOOK_TOKEN:
					if stderr:
						STDERR.write(s)  #for console/qtconsole
					else:
						STDOUT.write(s)  #for console/qtconsole
				else:
					#redirecting to area beneath the cell, for jupyter notebook.
					display(IPython.display.HTML(escaped_s))
			my_defout(TLS.xscrthread, escaped_s)
			if s and TLS.logfile:
				TLS.logfile.write(str(datetime.datetime.now()) + ":" + s + '\n')
				TLS.logfile.flush()
			return len(s)
		else:
			return STDERR.write(s) #redirecting to terminal, for debug purpose.
	def write(self, s):
		return self.write_internal(s)
	def readline(self):
		if hasattr(TLS, 'xscrthread') and TLS.xscrthread:
			while not is_main_terminated():		
				ret = my_defin(TLS.xscrthread)
				if ret:
					break #no input detected.
				time.sleep(0.2)
			return ret
		else:
			return STDIN.readline() #redirecting to terminal, for debug purpose.

	def read(self):
		return STDIN.readline()
	def flush(self):
		self.write_internal(self.buffer.getvalue(), flush=False)
		self.buffer.truncate(0)
		self.buffer.seek(0)
	def fileno(self):
		return STDOUT.fileno()
	def isatty(self):
		return False
	@property
	def encoding(self):
		return STDOUT.encoding
	@property
	def buffer(self):
		if not hasattr(TLS, 'buffer'):
			TLS.buffer = io.StringIO()
		return TLS.buffer

class MyDefOErr(MyDefIO):
	def write(self, s):
		STDERR.write(s) #redirecting to terminal, for debug purpose.
		return self.write_internal(s, stderr=True)

MYDEFOUT = MyDefIO()
MYDEFIN = MyDefIO()
MYDEFERR = MyDefOErr()
sys.stdout = MYDEFOUT
sys.stderr = MYDEFERR
sys.stdin = MYDEFIN

event = threading.Event()

#do not use time.sleep() please.
def sleep(sec):
	start = time.time()
	while True:
		remain = sec - (time.time() - start)
		if TLS.xscrthread:
			xpythread = TLS.xscrthread
			if str(xpythread["Action"]) == "kill":
				xpythread["Action"] = ""
				xpythread["Status"] = "killed @{}s @{}".format(int(remain), str(fback))
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
				fback = inspect.currentframe().f_back
				if HasIPython and 'ipykernel' in fback.f_code.co_filename: #sleep() in IPython kernel
					fback = "Cell In[{}]:line {} in {}".format(get_ipython().execution_count, fback.f_lineno, fback.f_code.co_name)
				xpythread["Status"] = "{}s sleep @{}".format(int(remain), str(fback))
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
					STDERR.write("Starting a new thread")
					filename = str(xpythread_filename)
					STDERR.write("Loading "+ filename)
					thread = threading.Thread(daemon=True, target=loadSequence, args=(xpythread, filename))
					thread.start()
					time.sleep(0.3)
				if action == "kill":
					if os.name == 'posix':
						time.sleep(0.5)
						if action == "kill":
							#cannot be killed by timer.
							ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(int(str(xpythread_threadid))), ctypes.py_object(SystemExit))
	except EOFError:
		pass
	except Exception:
		STDERR.write(str(traceback.format_exc()))

def findExecutables(prog):
	import glob
	paths = os.environ['PATH'].split(os.pathsep)
	if os.name == 'posix':
		paths.extend(['/opt/homebrew/bin', '/opt/local/bin'])
	ret = []
	for p in paths:
		if os.path.isdir(p):
			ret.extend(glob.glob(os.path.join(p, prog)))
			ret.extend(glob.glob(os.path.join(p, prog + os.extsep + "*")))
			ret.extend(glob.glob(os.path.join(p, prog + "-[3-9]*")))
	return ret

def listOfJupyterPrograms():
	return findExecutables('jupyter')

NOTEBOOK_TOKEN = None
NOTEBOOK_PROC = None

def launchJupyterConsole(prog, argv):
	if not HasIPython:
		raise RuntimeError('IPython not properly installed.') #, ipywidgets?
	global NOTEBOOK_TOKEN
	global NOTEBOOK_PROC
	from ipykernel.kernelapp import IPKernelApp
	app = IPKernelApp.instance()
	json = app.connection_file
	if not len(json):
		sys.stderr.write("IPython kernel could not be started.")
	print("Using existing kernel = " + json)
	args = [prog, '--existing', json,]

	import subprocess
	console = argv.split()
	args.insert(1, console[0])

	if console[0] == 'console':
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	elif console[0] == 'qtconsole':
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	elif console[0] == 'notebook':
		import ipykernel
		connection_file = ipykernel.connect.get_connection_file()
		import binascii
		token = binascii.hexlify(os.urandom(24)).decode('ascii')
		NOTEBOOK_TOKEN = token #for later identification in server list.
		env = dict(os.environ)
		env['PYTHONPATH'] = os.pathsep.join((KAME_ResourceDir, env.get('PYTHONPATH', '')))
		env['KAME_NOTEBOOK_SERVER_TOKEN'] = token
		env['KAME_IPYTHON_CONNECTION_FILE'] = connection_file
		args = [prog, console[0], '--config=' + os.path.join(KAME_ResourceDir, 'jupyter_notebook_config.py')]
		print("Launching jupyter notebook: ", *args)
		proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=console[1])
	else:
		raise RuntimeError('Unknown console.')
	
	time.sleep(0.5)
	ret = proc.poll()
	if ret:
		outs, errs = proc.communicate() #Lauching failed.
		raise RuntimeError(outs)
	NOTEBOOK_PROC = proc

	XScriptingThreads()[0]["Filename"] = ' '.join(args)

import linecache
linecache.clearcache() #suppress lengthy traceback inside REPL.

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
				try:
					from jupyter_server import serverapp as app
					self.serverapp = app
				except ImportError:
					self.serverapp = None
				import ipykernel
				connection_file = ipykernel.connect.get_connection_file()
				MYDEFOUT.write("#KAME IPython binding")
				MYDEFOUT.write("#Use sleep() instead of time.sleep().")
				self.logfilename = os.path.splitext(connection_file)[0] + "-log" + os.extsep + "txt"
				MYDEFOUT.write_html(r'<font color="#008800">Logging console output to <a href="file:///'
						+ self.logfilename + r'">' + html.escape(self.logfilename) + '</a></font>')
				TLS.logfile = open(self.logfilename, mode='a')
				self.func = func

			def on_timer(self):
				loop = asyncio.get_event_loop()
				try:
					loop.run_until_complete(self.func())
					if self.serverapp:
						s = ''
						for server in list(self.serverapp.list_running_servers()):
							if server['token'] == NOTEBOOK_TOKEN:
								url = r'{}?token={}'.format(server['url'], server['token'])
								s = r'notebook in {}: <a href="{}">{}</a>'.format(server['root_dir'], url, html.escape(url))
								break
						if s:
							if str(XScriptingThreads()[0]["Filename"]) != s:
								#detected connection to notebook.
								XScriptingThreads()[0]["Filename"] = s
								XScriptingThreads()[0]["Status"] = ''
								TLS.logfile.close()
								from ipykernel.kernelapp import IPKernelApp
								app = IPKernelApp.instance()
								json = app.connection_file
								self.logfilename = os.path.join(server['root_dir'], os.path.splitext(json)[0]) + '-log' + os.extsep + 'txt'
								TLS.logfile = open(self.logfilename, mode='a')
								MYDEFOUT.write_html(r'<font color="#008800">' + s + '</font>')
								MYDEFOUT.write_html(r'<font color="#008800">Changing logfile to <a href="file:///'
									 + self.logfilename + r'">' + html.escape(self.logfilename) + '</a></font>')
				except Exception:
					sys.stderr.write(str(traceback.format_exc()))

				sys.stdout = MYDEFOUT
				sys.stderr = MYDEFERR
				sys.stdin = MYDEFIN

				# if not is_main_terminated():
				kame_pybind_one_iteration()
				time.sleep(poll_interval)

			def start(self):
				self.on_timer()  # Call it once to get things going.
				while not is_main_terminated():
					self.on_timer()

				TLS.logfile.close()
				TLS.logfile = None

				sys.stdout = STDOUT
				sys.stderr = STDERR
				sys.stdin = STDIN

				if NOTEBOOK_PROC:
					get_ipython().run_line_magic('save', '-a ' + os.path.splitext(self.logfilename)[0] + "-save")
					NOTEBOOK_PROC.terminate() #stops Jupyter client
					NOTEBOOK_PROC.terminate() #again
				#print(str([y[0] for y in inspect.getmembers(kernel, inspect.ismethod)]))

				# from ipykernel.kernelapp import IPKernelApp
				# app = IPKernelApp.instance()
				# app.close()
				task = asyncio.create_task(self.func())
				task.cancel()
				sys.stderr.write("sys.exit(0) from python.\n")
				sys.exit(0) #I could not find better way to exit normally.
				# raise IPython.terminal.embed.KillEmbedded('') #exits loop, magic %exit_raise no more exists.

		kernel.timer = Timer(kernel.do_one_iteration)
		kernel.timer.start()

	@loop_kamepysupport.exit
	def loop_kamepysupport_exit(kernel):
		try:
			sys.stderr.write("exit\n")
			del kernel.timer
		except (RuntimeError, AttributeError):
			pass

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

	try:
		# Now starting ipython kernel.
		IPython.embed_kernel(config=c)
	except Exception:
		sys.stderr.write(str(traceback.format_exc()))

#With IPython, these lines cannot be reached.
sys.stdout = STDOUT
sys.stderr = STDERR
sys.stdin = STDIN
for thread in threading.enumerate():
	try:
		if thread != threading.current_thread():
			thread.join(timeout=0.3)
	except Exception as inst:
		sys.stderr.write(str(traceback.format_exc()))

sys.stderr.write("bye!\n")
