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
import signal
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
_deferred_done = False
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
	fback = ""
	while True:
		remain = sec - (time.time() - start)
		if TLS.xscrthread:
			xpythread = TLS.xscrthread
			if str(xpythread["Action"]) == "kill":
				xpythread["Action"] = ""
				xpythread["Status"] = "killed @{}s @{}".format(int(remain), str(fback))
				if str(xpythread["ThreadID"]) == "-1":
					#probably sleep() in IPython kernel; raise KeyboardInterrupt
					#so it behaves the same as Jupyter's built-in stop button
					raise KeyboardInterrupt
				else:
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
		try:
			event.wait(min([remain, 0.33]))
		except KeyboardInterrupt:
			#Jupyter stop button (or any SIGINT) interrupted the wait;
			#update KAME status and re-raise so the cell stops normally
			if TLS.xscrthread:
				xpythread = TLS.xscrthread
				remain = sec - (time.time() - start)
				xpythread["Status"] = "killed @{}s @{}".format(int(remain), str(fback))
			raise
	if TLS.xscrthread:
		xpythread = TLS.xscrthread
		xpythread["Status"] = "run"

class _KamFakeNode:
	"""Silent placeholder for nodes missing due to version skew."""
	def __init__(self, key=''): self._key = key
	def __getitem__(self, key): return _KamFakeNode(key)
	def create(self, *a): return _KamFakeNode(a)
	def load(self, v): STDERR.write("KamFakeNode[{}].load({}) ignored\n".format(self._key, v))

class _KamNode:
	"""Wraps XNode for .kam loading: chained [] access, .load(), and .create() with
	main-thread dispatch for non-thread-safe lists (e.g. XDriverList)."""
	# Aliases for backward-compatible .kam loading (old name → new name).
	_aliases = {
		"Begin": "First", "End": "Last",
		"BeginX": "FirstX", "BeginY": "FirstY",
		"EndX": "LastX", "EndY": "LastY",
	}
	def __init__(self, node): self._node = node
	def __getitem__(self, key):
		child = self._node[key]
		if child is None:
			# Try alias for backward compatibility with old .kam files.
			alias = self._aliases.get(key)
			if alias:
				child = self._node[alias]
		if child is None:
			return _KamFakeNode(key)
		return _KamNode(child)
	def create(self, type_name, name=''):
		if not hasattr(self._node, 'createByTypename'):
			STDERR.write("_KamNode.create({!r},{!r}): node not downcast to XListNodeBase, skipped\n".format(type_name, name))
			return _KamFakeNode(type_name)
		thread_safe = getattr(self._node, 'isThreadSafeDuringCreationByTypename', lambda: False)()
		if thread_safe:
			child = self._node.createByTypename(type_name, name)
		else:
			child = kame_mainthread(lambda: self._node.createByTypename(type_name, name))
		if child is None:
			STDERR.write("_KamNode.create({!r},{!r}): createByTypename returned None\n".format(type_name, name))
			return _KamFakeNode(type_name)
		return _KamNode(child)
	def load(self, value):
		try: self._node.load(str(value))
		except Exception as e: STDERR.write(str(e) + '\n')
	def getName(self): return self._node.getName()

class _KamStack(list):
	def __lshift__(self, val):
		if val is not None:
			self.append(val)
		return self

def loadKam(xpythread, filename):
	"""Execute a .kam measurement configuration file using Python."""
	import re
	TLS.xscrthread = xpythread
	TLS.logfile = None
	try:
		xpythread["ThreadID"] = str(threading.current_thread().native_id)
		xpythread["Status"] = "run"
		with open(filename, 'r', encoding='utf-8') as f:
			src = f.read()
		# Minimal Ruby→Python translation: x.last→x[-1], x.pop→x.pop()
		# Strip leading whitespace — .kam indentation is cosmetic; Python exec rejects it.
		src = '\n'.join(line.lstrip() for line in src.splitlines())
		# Replace only outside of string literals
		def _replace_outside_strings(line):
			line = line.replace('x = Array.new', 'x = _KamStack()')
			line = line.replace('x.last', 'x[-1]')
			line = re.sub(r'\bx\.pop\b(?!\s*\()', 'x.pop()', line)
			return line
		src = '\n'.join(_replace_outside_strings(line) if not line.lstrip().startswith('#') else line for line in src.splitlines())
		root = Root()
		rname = root.getName()
		rname = rname[0].upper() + rname[1:]
		globs = {'x': _KamStack(), '_KamStack': _KamStack, rname: _KamNode(root)}
		exec(compile(src, filename, 'exec'), globs)
		print(filename + " loaded.")
	except Exception:
		sys.stderr.write(str(traceback.format_exc()))
	finally:
		TLS.xscrthread["Status"] = ""

def loadSequence(xpythread, filename):
	TLS.xscrthread = xpythread #thread-local-storage
	TLS.logfile = None
	try:
		xpythread["ThreadID"] = str(threading.current_thread().native_id)
		xpythread["Status"] = "run"
		if "lineshell" in filename:
			print("#KAME Python interpreter>")
			exec(open(filename, 'r', encoding="utf-8").read())
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
	global _deferred_done
	if not _deferred_done:
		_deferred_done = True
		for _script in kame_deferred_scripts():
			try:
				exec(_script, globals())
			except Exception:
				STDERR.write(str(traceback.format_exc()))
	try:
		#For node browser pane
		PyInfoForNodeBrowser().set(str([y[0] for y in inspect.getmembers(LastPointedByNodeBrowser(), inspect.ismethod)]))

		for xpythread in XScriptingThreads():
			xpythread_status = xpythread["Status"]
			xpythread_action = xpythread["Action"]
			xpythread_threadid = xpythread["ThreadID"]
			xpythread_filename = xpythread["Filename"]
			threadlist = [str(pythread.native_id) for pythread in threading.enumerate() if pythread.native_id is not None]
			action = str(xpythread_action)
			if str(xpythread_threadid) in threadlist:
				pass
			else:
				if action == "starting":
					xpythread_action.set("")
					STDERR.write("Starting a new thread")
					filename = str(xpythread_filename)
					STDERR.write("Loading "+ filename)
					target = loadKam if filename.endswith('.kam') else loadSequence
					thread = threading.Thread(daemon=True, target=target, args=(xpythread, filename))
					thread.start()
				if action == "kill":
					if str(xpythread_threadid) == "-1":
						pass
					else:
						if os.name == 'posix':
							time.sleep(0.5)
							if action == "kill":
								STDERR.write("Could not kill by timer.")
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
NOTEBOOK_MCP_JSON = None

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
		env['KAME_PID'] = str(os.getpid())
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

	# Write MCP config for Claude Code in the notebook workspace.
	if console[0] == 'notebook':
		global NOTEBOOK_MCP_JSON
		import json as _json
		# Write kernel connection info for the MCP server.
		import ipykernel
		_conn_file = ipykernel.connect.get_connection_file()
		_kame_conn_info = os.path.join(os.path.expanduser('~'), '.kame_kernel_connection.json')
		try:
			with open(_kame_conn_info, 'w') as _f:
				_json.dump({'connection_file': _conn_file, 'pid': os.getpid()}, _f)
		except OSError:
			pass
		# Write .mcp.json pointing to the MCP server script.
		mcp_server_script = os.path.join(KAME_ResourceDir, 'kame_mcp_server.py')
		# Find a Python that can run the MCP server (needs mcp & jupyter_client).
		# sys.executable is the KAME binary when Python is embedded, so search for
		# a Python interpreter that has the required packages.
		import subprocess as _sp, shutil as _sh
		python_cmd = None
		_candidates = []
		# 1. Python next to the jupyter executable
		_bin_dir = os.path.dirname(prog)
		for _name in ('python3', 'python'):
			_c = os.path.join(_bin_dir, _name)
			if os.path.isfile(_c):
				_candidates.append(_c)
		# 2. Common venv location for KAME MCP (sibling of the build directory)
		import platform as _pf
		_venv_subdir = ('Scripts', 'python.exe') if _pf.system() == 'Windows' else ('bin', 'python3')
		for _depth in range(3, 7):  # search up from Resources to find kame-mcp-venv
			_venv_base = os.path.join(KAME_ResourceDir, *(['..'] * _depth), 'kame-mcp-venv', *_venv_subdir)
			_venv_py = os.path.normpath(_venv_base)
			if os.path.isfile(_venv_py):
				_candidates.insert(0, _venv_py)  # prefer venv
				break
		# 3. System python3
		_sys_py = _sh.which('python3')
		if _sys_py:
			_candidates.append(_sys_py)
		for _c in _candidates:
			try:
				_sp.check_call([_c, '-c', 'import mcp, jupyter_client'],
					stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=5)
				python_cmd = _c
				break
			except Exception:
				continue
		if not python_cmd:
			print("Warning: No Python with 'mcp' and 'jupyter_client' found for MCP server.", file=sys.stderr)
			python_cmd = _candidates[0] if _candidates else 'python3'
		mcp_json_path = os.path.join(console[1], '.mcp.json')
		try:
			with open(mcp_json_path, 'w') as _f:
				_json.dump({'mcpServers': {'kame': {
					'command': python_cmd,
					'args': [mcp_server_script]
				}}}, _f, indent=2)
			NOTEBOOK_MCP_JSON = mcp_json_path
			print("MCP config written to " + mcp_json_path)
		except OSError:
			pass

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
				self._initial_logfilename = self.logfilename
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

				# Remove MCP files created for Claude Code.
				if NOTEBOOK_MCP_JSON:
					try:
						os.remove(NOTEBOOK_MCP_JSON)
					except OSError:
						pass
					try:
						os.remove(os.path.join(os.path.expanduser('~'), '.kame_kernel_connection.json'))
					except OSError:
						pass
				# Delete the log file if Jupyter was never launched and no
				# server/notebook was ever connected (logfilename unchanged).
				if not NOTEBOOK_PROC and self.logfilename == self._initial_logfilename:
					try:
						os.remove(self.logfilename)
					except OSError:
						pass

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
		IPython.embed_kernel(config=c) #, interrupt_mode='signal'
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
