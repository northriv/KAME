#KAME xpyhonsuppport start-up code
#fundamental imports
import time
import sys
import html
import re
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

#Convenience imports for user scripts.  Kept SEPARATE from the IPython probe
#below on purpose: `np` is not referenced anywhere in this file, yet having it
#in the same try meant that a host merely missing numpy reported "no IPython"
#— which silently disables the Jupyter menu actions, the notebook and the MCP
#server, while the error text tells you to install ipykernel, which is already
#there.  One missing optional package should not take the others down with it.
try:
	import ctypes
	import pdb
except (ImportError, ModuleNotFoundError) as e:
	sys.stderr.write("KAME: a stdlib module is unavailable (%s)\n" % e)
try:
	import numpy as np
except (ImportError, ModuleNotFoundError):
	sys.stderr.write("KAME: numpy is not installed; scripts using `np` will fail.\n")

HasIPython = False
try:
	from ipykernel.eventloops import register_integration
	import IPython #this import hinders from freeing XPython/XMeasure normally.
	from IPython.display import display
	# import ipywidgets
	HasIPython = True
#	import matplotlib
#	matplotlib.use('Agg') #GUI does not work yet
#	import matplotlib.pyplot as plt
except (ImportError, ModuleNotFoundError) as e:
	#Name the module that actually failed — "IPython is not installed" when it
	#IS installed sends you looking in the wrong place.
	sys.stderr.write(
		"KAME: IPython/Jupyter support disabled (%s).\n"
		"      Install into THIS interpreter: %s -m pip install ipykernel ipython jupyter\n"
		% (e, sys.executable))
from kame import *
_deferred_done = False
STDOUT = sys.stdout
STDERR = sys.stderr
STDIN = sys.stdin

#Where the deployed script files (kame_mcp_server.py, the notebook config,
#the user's manual, pythonlineshell.py) live, relative to the executable:
#  osx  : kame.app/Contents/MacOS/../Resources
#  win  : Resources\
#  linux: next to the binary in a build tree, or $prefix/share/kame once
#         installed (matching kame.pro's `scriptfile.path`).
#Probe the candidates instead of assuming, and key on a file that is always
#deployed, so a layout that does not exist is skipped rather than silently
#becoming an unusable sys.path entry.
def _kame_resource_dir():
	_exedir = os.path.dirname(sys.executable)
	if sys.platform == 'darwin':
		_cands = [os.path.join(_exedir, '../Resources')]
	elif sys.platform.startswith('win'):
		_cands = [os.path.join(_exedir, 'Resources')]
	else:
		_cands = [_exedir,
		          os.path.join(_exedir, 'Resources'),
		          os.path.join(_exedir, '../share/kame'),
		          '/usr/local/share/kame', '/usr/share/kame']
	for _c in _cands:
		if os.path.isfile(os.path.join(_c, 'kame_mcp_server.py')):
			return os.path.normpath(_c)
	return os.path.normpath(_cands[0])   #keep the platform default

KAME_ResourceDir = _kame_resource_dir()
sys.path.insert(0, KAME_ResourceDir) #adds resource folder for importable modules.

print("Hello! KAME Python support.")

#Thread-monitor
MONITOR_PERIOD=0.2

#Where someone setting MCP up can read how.  It has to travel IN the message:
#the manual is served THROUGH MCP, so it is unreachable until this works, and
#by then they do not need it.
MCP_SETUP_URL = 'https://github.com/northriv/KAME#ai-assisted-experiment-automation-mcp'


class _KameTLS(threading.local):
	"""Per-thread script context, with every field readable from any thread.

	threading.local calls a subclass __init__ once per thread that touches it,
	which is exactly the semantics wanted here: a thread that is not a script
	thread reads None instead of raising AttributeError.  Plain
	threading.local() left every access a landmine for code running anywhere
	but the kernel thread -- an MCP async job (execute_code_async runs the
	code in a daemon thread) hit it on the first `TLS.xscrthread`, and the
	guards scattered through this file exist only because of it."""
	def __init__(self):
		self.xscrthread = None
		self.logfile = None
		self.cell_status = None

TLS = _KameTLS()
if HasIPython:
	XScriptingThreads()[0].setLabel("IPython kernel")
	XScriptingThreads()[0]["Action"] = ""
	XScriptingThreads()[0]["Status"] = "idle"
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
	# Resolve the scripting-thread context once. It is absent on threads
	# KAME did not start as scripting threads (e.g. execute_code_async
	# worker threads); there sleep() degrades to a plain interruptible
	# wait instead of raising AttributeError on a missing TLS.xscrthread.
	xthr = getattr(TLS, 'xscrthread', None)
	while True:
		remain = sec - (time.time() - start)
		if xthr:
			xpythread = xthr
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
				xpythread["Status"] = getattr(TLS, 'cell_status', None) or "run"
				return #ignores remaining time
			if str(xpythread["Action"]) == "suspend":
				xpythread["Action"] = ""
				sec = 1e10
			if remain > 1e9:
				xpythread["Status"] = "sleep"
			else:
				fback = inspect.currentframe().f_back
				if HasIPython and 'ipykernel' in fback.f_code.co_filename: #sleep() in IPython kernel
					#During cell N, IPython has already advanced execution_count to N+1.
					fback = "Cell In[{}]:line {} in {}".format(get_ipython().execution_count - 1, fback.f_lineno, fback.f_code.co_name)
				xpythread["Status"] = "{}s sleep @{}".format(int(remain), str(fback))
		if remain < 0:
			break
		try:
			event.wait(min([remain, 0.33]))
		except KeyboardInterrupt:
			#Jupyter stop button (or any SIGINT) interrupted the wait;
			#update KAME status and re-raise so the cell stops normally
			if xthr:
				xpythread = xthr
				remain = sec - (time.time() - start)
				xpythread["Status"] = "killed @{}s @{}".format(int(remain), str(fback))
			raise
	if xthr:
		xpythread = xthr
		xpythread["Status"] = getattr(TLS, 'cell_status', None) or "run"

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
				#Run the script with ONE dict serving as both globals and
				#locals.  A bare exec(src) inside this function hands the
				#script THIS function's locals, so a script-level `x = ...`
				#lands in a namespace the script's own top-level `def`s
				#cannot see -- every helper that touches a script-level
				#variable then dies with NameError.  Seeded from our globals
				#so Root()/Snapshot()/Transaction()/sleep() stay available,
				#and copied so a script cannot clobber them for everyone
				#else.  compile() with the real path also puts the filename
				#and correct lines in tracebacks instead of "<string>".
				_src = open(filename, encoding="utf-8").read()
				_ns = dict(globals())
				_ns["__name__"] = "__main__"
				_ns["__file__"] = filename
				exec(compile(_src, filename, "exec"), _ns)
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
	"""Return paths to executables named `prog` reachable via PATH (+
	well-known locations on POSIX). De-duplicated by canonical path so
	the menu doesn't list the same binary multiple times when (a)
	several glob patterns match (e.g. `jupyter`, `jupyter-3.13`), or
	(b) PATH already contains the well-known dirs we extend, or (c) a
	symlink points at a binary already discovered."""
	import glob
	# Dedupe path entries first so each directory is scanned once.
	paths = list(dict.fromkeys(os.environ['PATH'].split(os.pathsep)))
	if os.name == 'posix':
		for extra in ('/opt/homebrew/bin', '/opt/local/bin'):
			if extra not in paths:
				paths.append(extra)
	seen = set()
	ret = []
	for p in paths:
		if not os.path.isdir(p):
			continue
		for pattern in (prog,
		                prog + os.extsep + "*",
		                prog + "-[3-9]*"):
			for c in glob.glob(os.path.join(p, pattern)):
				try:
					real = os.path.realpath(c)
				except OSError:
					continue
				if real in seen:
					continue
				# Filter out matches that aren't actually executable
				# (e.g. `jupyter-3.13.dist-info` entries on Windows
				# venvs sometimes bleed into the wildcard).
				if not os.access(c, os.X_OK):
					continue
				seen.add(real)
				ret.append(c)
	return ret

def listOfJupyterPrograms():
	return findExecutables('jupyter')

NOTEBOOK_TOKEN = None
NOTEBOOK_PROC = None
NOTEBOOK_MCP_JSON = None
NOTEBOOK_MCP_HTTP_PROC = None
NOTEBOOK_MCP_URL_FILE = None
NOTEBOOK_MCP_HTTP_LOG = None
NOTEBOOK_LOG_TAIL = None	#last lines of the server's output, for diagnostics
NOTEBOOK_ATEXIT_DONE = False


def runningNotebookURL():
	"""URL of the notebook server this KAME already launched, or None.

	NOTEBOOK_PROC holds one process, so a second launch used to overwrite it
	and orphan the first for good -- and the orphan kept its port, which is
	why a run of clicks left servers on 8888, 8889, 8890...  One server per
	KAME is the right invariant anyway: they would all serve the same embedded
	kernel.
	"""
	if not NOTEBOOK_TOKEN or NOTEBOOK_PROC is None or NOTEBOOK_PROC.poll() is not None:
		return None
	try:
		from jupyter_server import serverapp
	except ImportError:
		return None
	for _s in serverapp.list_running_servers():
		if _s.get('token') == NOTEBOOK_TOKEN:
			return '{}?token={}'.format(_s['url'], _s['token'])
	return None


def stopNotebookServer():
	"""Stop the Jupyter server KAME launched.  Idempotent.

	terminate() alone was optimistic: it was called twice in a row and then
	followed straight by sys.exit(0), so nothing established that the server
	had actually gone.  Wait for it, and escalate if it will not.
	"""
	global NOTEBOOK_PROC
	proc, NOTEBOOK_PROC = NOTEBOOK_PROC, None
	if proc is None or proc.poll() is not None:
		return
	try:
		proc.terminate()
		proc.wait(timeout=5)
	except Exception:
		try:
			proc.kill()
			proc.wait(timeout=5)
		except Exception:
			pass


def _drainNotebookOutput(proc):
	"""Consume the server's stdout forever, keeping only the tail.

	Nothing read this pipe after the 0.5 s launch check, so the server would
	block in write() once the ~64 KB pipe buffer filled -- Jupyter logs every
	HTTP request, so that is a matter of browsing, not of days.  The tail is
	kept because a server that dies later leaves its reason there.
	"""
	global NOTEBOOK_LOG_TAIL
	import collections
	NOTEBOOK_LOG_TAIL = collections.deque(maxlen=200)
	def _run():
		try:
			for _line in iter(proc.stdout.readline, b''):
				NOTEBOOK_LOG_TAIL.append(
					_line.decode('utf-8', errors='replace').rstrip())
		except Exception:
			pass
		finally:
			try:
				proc.stdout.close()
			except Exception:
				pass
	threading.Thread(target=_run, name='kame-notebook-drain', daemon=True).start()

def launchJupyterConsole(prog, argv):
	if not HasIPython:
		raise RuntimeError(
			"IPython is not installed in KAME's embedded Python.\n"
			"Install it with one of:\n"
			f"  {sys.executable} -m pip install ipykernel ipython jupyter\n"
			"  /opt/local/bin/pip install ipykernel ipython jupyter\n"
			"Then restart KAME.")
	global NOTEBOOK_TOKEN
	global NOTEBOOK_PROC
	from ipykernel.kernelapp import IPKernelApp
	app = IPKernelApp.instance()
	json = app.connection_file
	if not len(json):
		raise RuntimeError(
			"KAME's embedded IPython kernel hasn't started yet "
			"(no connection file registered). The kernel usually comes up "
			"a few seconds after KAME launches; please retry shortly. "
			"If this persists, check stderr for kernel startup errors.")
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
		#Clicking the link again means "show me the notebook", not "give me a
		#second server": both would serve this one embedded kernel, and the
		#loser of the port race is what used to be left behind.  Reopen the
		#browser on the one already running.
		_url = runningNotebookURL()
		if _url:
			import webbrowser
			webbrowser.open(_url)
			MYDEFOUT.write_html('<font color="#008800">Notebook server already '
				'running: <a href="{}">{}</a></font>'.format(_url, html.escape(_url)))
			return
		#Alive but unlisted (its runtime file went missing) is still ours, and
		#a replacement is going up regardless: retire it rather than lose the
		#only handle to it.
		stopNotebookServer()
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
	if ret is not None:
		# Process already exited within 0.5 s → launch failed. Decode
		# captured output (bytes from subprocess.PIPE) and translate
		# common failure modes into actionable guidance.
		outs, _ = proc.communicate()
		raw = outs.decode('utf-8', errors='replace') if isinstance(outs, (bytes, bytearray)) else (outs or '')
		raw = raw.strip()
		sub = console[0]
		lines = [f"Failed to launch `{prog} {sub}` (exit code {ret})."]
		# Common pip packages required by each Jupyter subcommand —
		# Jupyter's CLI distributes these as separate pip packages,
		# so a minimal `jupyter` install may not have e.g. notebook.
		pkg_map = {
			'notebook': 'notebook',
			'qtconsole': 'qtconsole',
			'console': 'jupyter_console',
		}
		req = pkg_map.get(sub, sub)
		lower = raw.lower()
		if ('no module named' in lower
		    or 'subcommand' in lower and 'not available' in lower
		    or 'not a jupyter command' in lower):
			lines += [
				"",
				f"The `{sub}` Jupyter subcommand isn't installed against this Python.",
				f"Install it with one of:",
				f"  {sys.executable} -m pip install {req} ipykernel",
				f"  pip install {req} ipykernel",
				f"  /opt/local/bin/pip install {req} ipykernel    # MacPorts",
				f"  /opt/homebrew/bin/pip install {req} ipykernel  # Homebrew",
			]
		elif 'errno 2' in lower or 'no such file' in lower:
			lines += [
				"",
				f"Jupyter executable not found at: {prog}",
				"It may have been uninstalled or your PATH changed since",
				"KAME populated the menu. Re-open the Script menu to",
				"refresh the list.",
			]
		elif 'permission' in lower:
			lines += [
				"",
				f"Permission denied executing {prog}.",
				"Check file permissions or that the path isn't on a",
				"quarantined volume.",
			]
		else:
			lines += [
				"",
				"Common causes:",
				f"  • `{req}` package not installed (try: pip install {req})",
				"  • `ipykernel` missing in the same Python that runs jupyter",
				"  • Python version mismatch between jupyter and KAME's kernel",
			]
		if raw:
			lines += ["", "Captured output:", raw]
		raise RuntimeError("\n".join(lines))
	NOTEBOOK_PROC = proc
	# Past the 0.5 s failure check, so nobody is going to call communicate()
	# again: keep the pipe moving, or the server deadlocks on a full buffer.
	_drainNotebookOutput(proc)
	# Last-resort net for the paths that never reach finish(): a clean
	# interpreter shutdown still runs this, and the server's own watchdog
	# (jupyter_notebook_config.py) covers the rest, crashes included.
	global NOTEBOOK_ATEXIT_DONE
	if not NOTEBOOK_ATEXIT_DONE:
		import atexit
		atexit.register(stopNotebookServer)
		NOTEBOOK_ATEXIT_DONE = True
	# Show the launch command in the Script pane NOW, before the MCP setup
	# below, which takes seconds.  This same node is the duplicate guard of
	# the notebook-URL detector on the Python thread's Timer: when this write
	# happened at the END of the function, the detector could fire first,
	# get overwritten here, and fire again on the next tick — printing the
	# "notebook in ...: <url>" / "Changing logfile" pair exactly twice.
	XScriptingThreads()[0]["Filename"] = ' '.join(args)

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
				# notebook_token / notebook_dir let the MCP server locate the
				# notebook server (contents API) among Jupyter runtime files.
				# resource_dir lets an installed agent plugin (whose cache
				# holds only the plugin directory) find kame_mcp_server.py on
				# ANY layout, dev build trees included — the fixed
				# /Applications-style candidates in bin/kame-mcp-server miss
				# those.
				_json.dump({'connection_file': _conn_file, 'pid': os.getpid(),
							'notebook_token': token,
							'notebook_dir': os.path.abspath(console[1]),
							'resource_dir': KAME_ResourceDir}, _f)
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
		# 1. The interpreter that actually runs `prog` (the Jupyter
		#    launcher). Its shebang names the Python whose environment has
		#    jupyter_client — and almost always where the user pip-installed
		#    `mcp`. Try it FIRST: on Homebrew, dirname(prog)/python3 can be a
		#    DIFFERENT Python (e.g. python@3.14, no `mcp`) than jupyter's own
		#    shebang interpreter (python@3.13), and launching the MCP server
		#    against the wrong one makes it exit immediately at `import mcp`.
		def _shebang_interp(_path):
			try:
				with open(_path, 'rb') as _shf:
					_first = _shf.readline(512)
			except OSError:
				return None
			if not _first.startswith(b'#!'):
				return None  # e.g. a Windows .exe launcher — no shebang
			_parts = _first[2:].decode('utf-8', 'replace').split()
			if not _parts:
				return None
			# `#!/usr/bin/env python3` → resolve the named interpreter.
			if os.path.basename(_parts[0]) == 'env' and len(_parts) > 1:
				return _sh.which(_parts[1])
			return _parts[0] if os.path.isfile(_parts[0]) else None
		_jpy = _shebang_interp(prog)
		if _jpy:
			_candidates.append(_jpy)
		# 2. Python next to the jupyter executable
		_bin_dir = os.path.dirname(prog)
		for _name in ('python3', 'python', 'python3.exe', 'python.exe'):
			_c = os.path.join(_bin_dir, _name)
			if os.path.isfile(_c):
				_candidates.append(_c)
		# 3. Common venv location for KAME MCP (sibling of the build directory)
		import platform as _pf
		_venv_subdir = ('Scripts', 'python.exe') if _pf.system() == 'Windows' else ('bin', 'python3')
		# Search up from the resource dir.  Start at depth 1, not 3: the old
		# floor assumed the macOS bundle's `kame.app/Contents/Resources`
		# (2 levels in, so depth 3 is the build dir).  A Windows release keeps
		# `resources` only ONE level inside the unzipped folder, so depth 3
		# already lands on the folder's GRANDPARENT and `kame-mcp-venv` placed
		# next to kame.exe — the obvious spot — was never found.  Depths 1-2
		# are two extra isfile() checks and cannot false-positive (nothing
		# ships a kame-mcp-venv inside kame.app/Contents).
		for _depth in range(1, 7):  # search up from Resources to find kame-mcp-venv
			_venv_base = os.path.join(KAME_ResourceDir, *(['..'] * _depth), 'kame-mcp-venv', *_venv_subdir)
			_venv_py = os.path.normpath(_venv_base)
			if os.path.isfile(_venv_py):
				_candidates.insert(0, _venv_py)  # prefer an explicit venv
				break
		# 4. System python3
		_sys_py = _sh.which('python3')
		if _sys_py:
			_candidates.append(_sys_py)
		# 5. Versioned interpreters (python3.X). The unversioned `python3` in
		#    these directories can denote a different major.minor than the one
		#    the user pip-installed `mcp` into: Homebrew relinks python3 on
		#    upgrades, and MacPorts creates no unversioned link at all without
		#    `port select`. Observed live: jupyter launched from MacPorts'
		#    python3.14 (no mcp) while mcp lived in Homebrew's python3.11 —
		#    reachable only as a versioned name. Newest first; the import
		#    probe below rejects the losers.
		import glob as _glob, re as _re
		_vers = []
		for _d in {_bin_dir, '/opt/homebrew/bin', '/opt/local/bin',
				   '/usr/local/bin', '/usr/bin'}:
			_vers += [_p for _p in _glob.glob(os.path.join(_d, 'python3.*'))
					  if _re.search(r'python3\.\d+$', _p)]
		_candidates += sorted(_vers,
			key=lambda _p: int(_re.search(r'python3\.(\d+)$', _p).group(1)),
			reverse=True)
		# De-duplicate by resolved path, preserving priority order.
		_seen = set()
		_uniq = []
		for _c in _candidates:
			_rp = os.path.realpath(_c)
			if _rp not in _seen:
				_seen.add(_rp)
				_uniq.append(_c)
		_candidates = _uniq
		# Every candidate is a DIFFERENT Python than the one embedded in KAME,
		# so KAME's own interpreter environment must not leak into it.  On
		# Windows this is fatal rather than untidy: kame-msyspython.bat exports
		# PYTHONHOME=C:\msys64\mingw64 and a PYTHONPATH of MSYS2's stdlib, and a
		# real CPython told to use those loads mingw-built C extensions it
		# cannot open -- `ModuleNotFoundError: No module named '_socket'`, or a
		# uv venv trampoline failing outright with "No Python at ..." (exit
		# 103).  That killed both the probe (so the venv looked unusable and the
		# fallback picked it anyway) and the server itself.  VIRTUAL_ENV is
		# dropped for the same reason, PYTHONSTARTUP because it would run
		# KAME-oriented code in a foreign interpreter.
		_child_env = dict(os.environ)
		for _v in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'PYTHONSTARTUP'):
			_child_env.pop(_v, None)
		# Probe each candidate with the EXACT imports kame_mcp_server.py runs
		# at module load. A bare `import mcp` is not enough: a Python carrying
		# a stale/partial `mcp` (top-level import OK but neither server class)
		# would pass and then crash the server at startup. Either mcp line
		# counts, matching the server's own fallback -- requiring only the 1.x
		# module would reject a perfectly good `pip install mcp` (which serves
		# 2.x) and send the user hunting for a missing package.
		_probe = ('import jupyter_client, mcp.types\n'
				  'try:\n'
				  '    from mcp.server.fastmcp import FastMCP, Image\n'
				  'except ImportError:\n'
				  '    from mcp.server import MCPServer\n'
				  '    from mcp.server.mcpserver import Image\n')
		for _c in _candidates:
			try:
				_sp.check_call(
					[_c, '-c', _probe],
					stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=10,
					env=_child_env)
				python_cmd = _c
				break
			except Exception:
				continue
		if not python_cmd:
			print("Warning: No Python with 'mcp' and 'jupyter_client' found for MCP server.", file=sys.stderr)
			python_cmd = _candidates[0] if _candidates else 'python3'
		# launchJupyterConsole runs on the main thread via py::eval, so
		# TLS.xscrthread is not set → MYDEFOUT.write would fall back to
		# the OS terminal. Write progress messages directly to KAME's
		# Python script-thread log so they are visible in the GUI's
		# message area (and the Python pane).
		_pyscrthread = XScriptingThreads()[0]
		def _gui_log(msg, color='#008800'):
			"""Mirror text to KAME's GUI message area + STDERR fallback."""
			try:
				escaped = html.escape(msg)
				my_defout(_pyscrthread,
					f'<font color="{color}">{escaped}</font>')
			except Exception:
				pass
			# Always also write to terminal so headless launches log too.
			STDERR.write(msg + '\n')

		# Transport selection:
		#   Default → streamable-http with Bearer token. The HTTP server
		#   stays resident across Claude Code sessions, eliminating the
		#   per-session subprocess startup cost (~1 s on Windows,
		#   ~0.5 s on macOS for `import mcp`). Also lets external
		#   Claude Code sessions connect from any cwd by reading
		#   `~/.kame_mcp_url`.
		#   Override via env var KAME_MCP_TRANSPORT=stdio|http.
		global NOTEBOOK_MCP_HTTP_PROC
		global NOTEBOOK_MCP_URL_FILE
		_transport_env = os.environ.get('KAME_MCP_TRANSPORT', '').lower()
		if _transport_env in ('stdio', 'http'):
			_use_http = (_transport_env == 'http')
		else:
			_use_http = True  # default everywhere

		mcp_json_path = os.path.join(console[1], '.mcp.json')
		try:
			if _use_http:
				import secrets as _secrets, socket as _socket
				import subprocess as _sp
				import tempfile as _tf
				import signal as _signal
				# Stable endpoint across restarts: reuse the previous
				# port+token (or an explicit KAME_MCP_PORT / KAME_MCP_TOKEN)
				# when the port is reclaimable, so .mcp.json does not change
				# on every launch — a client that connected once keeps working
				# after a KAME restart without a manual MCP reconnect.
				_url_file = os.path.join(os.path.expanduser('~'), '.kame_mcp_url')
				_prev = {}
				try:
					with open(_url_file) as _pf2:
						_prev = _json.load(_pf2)
				except (OSError, ValueError):
					_prev = {}
				_env_port = os.environ.get('KAME_MCP_PORT', '').strip()
				_port = None
				_token = None
				if _env_port.isdigit():
					_port = int(_env_port)
					_token = (os.environ.get('KAME_MCP_TOKEN', '').strip()
							  or _prev.get('token') or _secrets.token_urlsafe(32))
				elif _prev.get('port') and _prev.get('token'):
					_port = int(_prev['port'])
					_token = _prev['token']
				# A stale MCP server from a previous KAME run still holds the
				# old port but points at a now-dead kernel. If we recorded its
				# pid and can confirm (POSIX) it is our server, terminate it so
				# the port/token can be reused — verifying the command line so
				# we never kill an unrelated, pid-recycled process.
				_prev_pid = _prev.get('server_pid')
				if _port and _prev_pid:
					try:
						_pid = int(_prev_pid)
						if _pf.system() == 'Windows':
							# No ps on Windows: confirm the pid is a python
							# process, then force-kill (best effort).
							_tl = _sp.check_output(
								['tasklist', '/FI', 'PID eq {}'.format(_pid),
								 '/FO', 'CSV', '/NH'], text=True, timeout=5)
							if 'python' in _tl.lower():
								_sp.run(['taskkill', '/F', '/PID', str(_pid)],
										stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
										timeout=5)
								time.sleep(0.4)
						else:
							# Verify the command line before SIGTERM so a
							# pid-recycled, unrelated process is never killed.
							_cmd = _sp.check_output(
								['ps', '-p', str(_pid), '-o', 'command='],
								text=True, timeout=3)
							if 'kame_mcp_server' in _cmd:
								os.kill(_pid, _signal.SIGTERM)
								time.sleep(0.4)
					except Exception:
						pass
				# Confirm the chosen port is bindable now; otherwise fall back
				# to a fresh OS-assigned port (and mint a token if we lack one).
				def _port_free(_p):
					_t = _socket.socket()
					try:
						_t.bind(('127.0.0.1', _p))
						return True
					except OSError:
						return False
					finally:
						_t.close()
				if not _port or not _port_free(_port):
					_sk = _socket.socket()
					_sk.bind(('127.0.0.1', 0))
					_port = _sk.getsockname()[1]
					_sk.close()
				if not _token:
					_token = _secrets.token_urlsafe(32)
				# Capture server stderr/stdout to a tempfile so we can
				# surface a real error message if Popen fails — DEVNULL
				# would leave the user staring at "MCP didn't start"
				# with no clue why.
				global NOTEBOOK_MCP_HTTP_LOG
				_logf = _tf.NamedTemporaryFile(
					prefix='kame_mcp_http_', suffix='.log',
					mode='w', delete=False)
				NOTEBOOK_MCP_HTTP_LOG = _logf.name
				# The token goes in the environment, never in argv: a process's
				# environment is private to its owner, while argv is readable by
				# every local user through `ps` for as long as the server runs
				# (and is copied into crash reports and support dumps). Those
				# local users are precisely who the token keeps from POSTing to
				# 127.0.0.1 and running Python inside KAME.
				# _child_env, not os.environ: the server runs a different
				# Python, so KAME's PYTHONHOME/PYTHONPATH must not follow it
				# (see the probe above -- on MSYS2 they break it outright).
				_mcp_env = dict(_child_env)
				_mcp_env['KAME_MCP_TOKEN'] = _token
				NOTEBOOK_MCP_HTTP_PROC = _sp.Popen(
					[python_cmd, mcp_server_script,
					 '--transport=http',
					 f'--port={_port}'],
					stdout=_logf, stderr=_sp.STDOUT, env=_mcp_env,
				)
				_logf.close()  # child still holds the fd
				# Confirm the server actually LISTENS before declaring
				# success. A single 0.5 s poll is too short: a cold
				# `import mcp` + uvicorn startup can exceed it, so a server
				# that crashes a moment later (wrong Python, missing `mcp`)
				# would be mis-reported as "started" and a .mcp.json written
				# pointing at a dead port.
				_deadline = time.time() + 8.0
				_ret = None
				_listening = False
				while time.time() < _deadline:
					_ret = NOTEBOOK_MCP_HTTP_PROC.poll()
					if _ret is not None:
						break  # process exited — read its log below
					_probe = _socket.socket()
					_probe.settimeout(0.25)
					try:
						_probe.connect(('127.0.0.1', _port))
						_listening = True
					except OSError:
						pass
					finally:
						_probe.close()
					if _listening:
						break
					time.sleep(0.2)
				if not _listening:
					# Either it exited, or it never bound the port in time.
					# Reap it (so it cannot linger as a <defunct> zombie that
					# KAME never waits on), then surface the captured output
					# and fall back to stdio.
					if _ret is None:
						try:
							NOTEBOOK_MCP_HTTP_PROC.terminate()
						except Exception:
							pass
					try:
						_ret = NOTEBOOK_MCP_HTTP_PROC.wait(timeout=3)
					except Exception:
						pass
					try:
						with open(NOTEBOOK_MCP_HTTP_LOG, 'r',
								  errors='replace') as _lf:
							_raw = _lf.read().strip()
					except OSError:
						_raw = ''
					_lower = _raw.lower()
					_lines = [
						f"MCP HTTP server failed to start "
						f"(exit code {_ret}). Falling back to stdio."
					]
					if 'no module named' in _lower:
						_lines += [
							"",
							"A required Python package is missing. Install with:",
							f"  {python_cmd} -m pip install mcp jupyter_client uvicorn starlette",
							f"  Setup instructions: {MCP_SETUP_URL}",
						]
					elif ('address already in use' in _lower
					      or 'errno 48' in _lower):
						_lines += [
							"",
							"Port conflict — another KAME instance is "
							"probably already running. Quit it, or set "
							"KAME_MCP_TRANSPORT=stdio to disable HTTP MCP.",
						]
					elif 'permission' in _lower:
						_lines += [
							"",
							f"Permission denied executing {python_cmd}. "
							"Check file permissions or quarantine status.",
						]
					else:
						_lines += [
							"",
							"Common causes:",
							"  • mcp / uvicorn / starlette package missing in this Python",
							"  • Python version mismatch with KAME's kernel",
							"  • Firewall blocking 127.0.0.1 listen socket (Windows)",
						]
					if _raw:
						_lines += ["", "Captured output:", _raw]
					_gui_log("\n".join(_lines), color='#cc0000')
					NOTEBOOK_MCP_HTTP_PROC = None
					# Fall through: write stdio config below.
					_use_http = False

			if _use_http:
				_url = f'http://127.0.0.1:{_port}/mcp'
				with open(mcp_json_path, 'w') as _f:
					_json.dump({'mcpServers': {'kame': {
						'type': 'http',
						'url': _url,
						'headers': {'Authorization': f'Bearer {_token}'}
					}}}, _f, indent=2)
				# Also publish to ~/.kame_mcp_url so external Claude
				# Code sessions outside the notebook dir can find it.
				NOTEBOOK_MCP_URL_FILE = os.path.join(
					os.path.expanduser('~'), '.kame_mcp_url')
				with open(NOTEBOOK_MCP_URL_FILE, 'w') as _f:
					_json.dump({'url': _url, 'token': _token,
								'port': _port, 'pid': os.getpid(), 'server_pid': NOTEBOOK_MCP_HTTP_PROC.pid}, _f)
				_gui_log(
					f"#MCP HTTP server started.\n"
					f"#  URL  : {_url}\n"
					f"#  Port : {_port}\n"
					f"#  PID  : {NOTEBOOK_MCP_HTTP_PROC.pid}\n"
					f"#  Token: {_token[:8]}\u2026 (full token in {mcp_json_path})\n"
					f"#  URL file: {NOTEBOOK_MCP_URL_FILE}\n"
					f"#  Notebook .mcp.json: {mcp_json_path}")
			else:
				with open(mcp_json_path, 'w') as _f:
					_json.dump({'mcpServers': {'kame': {
						'command': python_cmd,
						'args': [mcp_server_script]
					}}}, _f, indent=2)
				_gui_log(
					f"#MCP stdio config written.\n"
					f"#  Server script: {mcp_server_script}\n"
					f"#  Python      : {python_cmd}\n"
					f"#  .mcp.json   : {mcp_json_path}")
			NOTEBOOK_MCP_JSON = mcp_json_path
		except OSError:
			pass

def _kame_workspace_dir():
	"""Best-guess workspace dir holding .mcp.json (where Claude should start)."""
	_m = globals().get('NOTEBOOK_MCP_JSON')
	if _m:
		return os.path.dirname(_m)
	try:
		import json as _j
		with open(os.path.join(os.path.expanduser('~'), '.kame_kernel_connection.json')) as _f:
			return _j.load(_f).get('notebook_dir') or os.path.expanduser('~')
	except Exception:
		return os.path.expanduser('~')

def _open_linux_terminal(inner):
    """Run `inner` (a `bash -lc` string) in the first available terminal emulator.

    Returns True if one was spawned.  Shared by the Claude and Codex launchers so
    the terminal-probe list lives in exactly one place.
    """
    import shutil as _shutil, subprocess as _sp
    _cands = []
    if os.environ.get('TERMINAL'):
        _cands.append((os.environ['TERMINAL'], '-e'))
    _cands += [('x-terminal-emulator', '-e'), ('gnome-terminal', '--'),
               ('konsole', '-e'), ('xfce4-terminal', '-x'),
               ('kitty', '--'), ('alacritty', '-e'),
               ('wezterm', 'start'), ('foot', ''), ('xterm', '-e')]
    for _term, _flag in _cands:
        _path = _shutil.which(_term)
        if not _path:
            continue
        _argv = [_path] + ([_flag] if _flag else []) + ['bash', '-lc', inner]
        try:
            _sp.Popen(_argv)
            return True
        except OSError:
            continue
    return False


def _kame_codex_spec():
    """The MCP server KAME wired up this session, as a launch spec, or None.

    Single source of truth: the same `.mcp.json` / `~/.kame_mcp_url` KAME already
    wrote, so Codex is pointed at exactly the server this KAME started.  Returns
    {'type':'stdio','command','args'} or {'type':'http','url','token'}.
    """
    import json as _json
    _paths = []
    _m = globals().get('NOTEBOOK_MCP_JSON')
    if _m:
        _paths.append(_m)
    _paths.append(os.path.join(os.path.expanduser('~'), '.kame_mcp_url'))
    for _p in _paths:
        try:
            with open(_p) as _f:
                _d = _json.load(_f)
        except Exception:
            continue
        if not isinstance(_d, dict):
            continue
        _srv = (_d.get('mcpServers') or {}).get('kame')
        if isinstance(_srv, dict):
            if _srv.get('type') == 'http' or _srv.get('url'):
                _tok = ''
                _auth = (_srv.get('headers') or {}).get('Authorization', '')
                if _auth.lower().startswith('bearer '):
                    _tok = _auth[7:]
                return {'type': 'http', 'url': _srv.get('url'), 'token': _tok}
            if _srv.get('command'):
                return {'type': 'stdio', 'command': _srv['command'],
                        'args': list(_srv.get('args') or [])}
        if _d.get('url'):
            return {'type': 'http', 'url': _d['url'], 'token': _d.get('token', '')}
    return None


def _toml_quote(s):
    return '"' + str(s).replace('\\', '\\\\').replace('"', '\\"') + '"'


def _msix_aumid(name_glob):
    """AppUserModelID of an installed MSIX package, or None (Windows only).

    Both the Claude and Codex desktop apps ship as MSIX packages on Windows.
    Those are unreachable the ordinary ways -- not on PATH, no executable under
    Program Files that may be run directly, and nothing in App Paths or the
    Uninstall registry -- so `start "" Claude` opens nothing and a CLI-based
    launcher reports the CLI missing even though the app is installed.  A
    packaged app is launched by AppUserModelID via the AppsFolder shell
    namespace: `explorer.exe shell:AppsFolder\\<family>!<appid>`.

    Asked of the OS rather than hardcoded: the publisher hash in the family
    name is stable, but a plain-exe install has no AUMID at all and must keep
    using the caller's existing route.  Observed values here:
    Claude_pzs8sxrjxfjjc!Claude, OpenAI.Codex_2p2nqsd0c76g0!App.
    """
    import platform as _pf, subprocess as _sp
    if _pf.system() != 'Windows':
        return None
    try:
        _ps = _sp.run(
            ['powershell', '-NoProfile', '-Command',
             '$p=Get-AppxPackage {} | Select-Object -First 1;'
             'if($p){{$id=($p | Get-AppxPackageManifest).Package.'
             'Applications.Application.Id;'
             'if($id -is [array]){{$id=$id[0]}};'
             '"$($p.PackageFamilyName)!$id"}}'.format(name_glob)],
            capture_output=True, text=True, timeout=20)
        for _line in reversed((_ps.stdout or '').strip().splitlines()):
            _line = _line.strip()
            if '!' in _line:
                return _line
    except Exception:
        pass
    return None


def _launch_msix(aumid):
    """Open a packaged app by AUMID.  Caller checked it is non-None."""
    import subprocess as _sp
    _sp.Popen(['explorer.exe', 'shell:AppsFolder\\' + aumid])


def _resolve_cli(binary):
    """Find a CLI even when KAME was launched as a macOS GUI application.

    Finder/Qt-launched applications commonly inherit only the system PATH, so
    `shutil.which()` cannot see installers' usual destinations such as
    ~/.local/bin.  Try those explicitly, then ask the user's login shell as a
    last resort.  The returned path is absolute so the temporary Terminal
    script does not depend on the GUI process's PATH.
    """
    import shutil as _sh, shlex as _shlex, subprocess as _sp
    _p = _sh.which(binary)
    if _p:
        return _p
    for _d in (os.path.expanduser('~/.local/bin'),
               os.path.expanduser('~/.cargo/bin'),
               '/opt/homebrew/bin', '/usr/local/bin', '/opt/local/bin'):
        _p = os.path.join(_d, binary)
        if os.path.isfile(_p) and os.access(_p, os.X_OK):
            return _p
    if os.name == 'posix':
        _shell = os.environ.get('SHELL') or '/bin/zsh'
        try:
            _out = _sp.check_output(
                [_shell, '-lic', 'command -v -- ' + _shlex.quote(binary)],
                stderr=_sp.DEVNULL, text=True, timeout=5)
            for _line in reversed(_out.splitlines()):
                _line = _line.strip()
                if os.path.isabs(_line) and os.access(_line, os.X_OK):
                    return _line
        except Exception:
            pass
    return None


def _register_stdio_entry():
    """The persistent server entry to register, or None with a reason shown.

    A terminal client gets a session-scoped override at launch; a GUI app has
    no command line, so it needs an entry in its own configuration -- and that
    entry has to outlive this KAME process.  Hence the plugin's STDIO launcher
    and not the HTTP URL: the launcher reaches whichever kernel is current
    through ~/.kame_kernel_connection.json, whereas the HTTP port is assigned
    per launch and the entry would be stale by the next restart.  Registering
    while KAME is down is therefore harmless too -- the server starts and its
    tools report that KAME is not running.
    """
    _pd = _kame_plugin_dir()
    _launcher = os.path.join(_pd, 'bin', 'kame-mcp-server') if _pd else ''
    if not _launcher or not os.path.isfile(_launcher):
        _kame_gui_html('<font color="#cc0000">The plugin launcher was not found'
            '{} &mdash; rebuild/redeploy KAME.</font>'.format(
            ' at ' + html.escape(_launcher) if _launcher else ''))
        return None
    return _launcher


def _claude_desktop_config():
    """Path of Claude Desktop's config, if the app has a config dir here."""
    import platform as _pf
    _p = {
        'Darwin': '~/Library/Application Support/Claude/claude_desktop_config.json',
        'Windows': os.path.join(os.environ.get('APPDATA', '~'),
                                'Claude', 'claude_desktop_config.json'),
    }.get(_pf.system(), '~/.config/Claude/claude_desktop_config.json')
    _p = os.path.expanduser(_p)
    return _p if os.path.isdir(os.path.dirname(_p)) else None


def _google_agent_config():
    """(label, path) for Google's agent CLI, or None.

    Two products share the ~/.gemini directory.  Antigravity CLI (`agy`), the
    current one, keeps MCP servers in ~/.gemini/config/mcp_config.json, shared
    with the IDE and the SDK.  Gemini CLI, which it replaced for consumers on
    2026-06-18 but which enterprise licences still run, used the `mcpServers`
    key inside ~/.gemini/settings.json.  Same key, different file -- so the
    directory alone does not say which, and writing to the wrong one fails
    silently.  Prefer the current product, and fall back to the legacy file
    only when it is the only one present.

    Antigravity also has `agy mcp add`, which would be the better mechanism by
    the same argument that makes `codex mcp add` right for Codex -- the client
    owning its own format.  It is not used here because its exact invocation
    could not be verified against a real `agy`, and a guessed command line is
    worse than an edit to a documented path."""
    _d = os.path.expanduser('~/.gemini')
    if not os.path.isdir(_d):
        return None
    _agy = os.path.join(_d, 'config', 'mcp_config.json')
    if os.path.isfile(_agy) or os.path.isdir(os.path.dirname(_agy)):
        return ('Antigravity CLI', _agy)
    _legacy = os.path.join(_d, 'settings.json')
    if os.path.isfile(_legacy):
        return ('Gemini CLI (legacy)', _legacy)
    return ('Antigravity CLI', _agy)


def _lmstudio_deeplink():
    """(label, url-scheme) of an installed LM Studio-family app, or None.

    The scheme is per PRODUCT, not per family: Bionic's bundle declares
    `bionic`, LM Studio's declares `lmstudio`, and sending one to the other
    fails with kLSApplicationNotFoundErr -- which is how the first attempt
    here silently "succeeded" against Bionic while doing nothing at all."""
    for _p, _label, _scheme in (
            ('/Applications/Bionic.app', 'Bionic', 'bionic'),
            (os.path.expanduser('~/Applications/Bionic.app'), 'Bionic', 'bionic'),
            ('/Applications/LM Studio.app', 'LM Studio', 'lmstudio'),
            (os.path.expanduser('~/Applications/LM Studio.app'), 'LM Studio', 'lmstudio')):
        if os.path.exists(_p):
            return (_label, _scheme)
    return None


def _mcp_json_edit(label, path, entry, apply, plan, done, fail):
    """Plan, or apply, one `mcpServers.<name>` entry in a JSON config.

    Shared by every client whose configuration is JSON with that key and which
    offers no CLI of its own to do it.  Touches that one key and writes the
    rest back unchanged, after a backup -- these files hold the user's own
    settings, and for some clients (Gemini) that is most of the file."""
    import json as _json, shutil as _sh

    def _load(p):
        """Parse `p`, treating an EMPTY file as {}.

        A client that has a config path but has never written to it leaves a
        0-byte file -- Antigravity does, and json.load then dies with
        "Expecting value: line 1 column 1 (char 0)", which surfaced as a
        traceback on Apply. Empty means "no settings yet", so {} is the right
        reading. Malformed-but-non-empty still raises: that is somebody's real
        configuration this cannot safely rewrite."""
        with open(p) as _f:
            _text = _f.read()
        return _json.loads(_text) if _text.strip() else {}

    _old = None
    try:
        if os.path.isfile(path):
            _old = ((_load(path) or {}).get('mcpServers') or {}).get('kame')
    except Exception:
        _old = '(unreadable)'
    plan.append('<b>{}</b> &mdash; <tt>{}</tt><br/>'
        '&nbsp;&nbsp;<font color="#cc0000">-</font> mcpServers.kame: <tt>{}</tt><br/>'
        '&nbsp;&nbsp;<font color="#008800">+</font> mcpServers.kame: <tt>{}</tt><br/>'
        '&nbsp;&nbsp;(all other keys preserved; a <tt>.kame-backup</tt> copy is kept)'.format(
        html.escape(label), html.escape(path),
        html.escape(_json.dumps(_old) if _old is not None else '(absent)'),
        html.escape(_json.dumps(entry))))
    if not apply:
        return
    try:
        _conf = {}
        if os.path.isfile(path):
            _conf = _load(path) or {}
            _sh.copy2(path, path + '.kame-backup')
        #Only this one key is touched; everything else is written back.
        _conf.setdefault('mcpServers', {})['kame'] = entry
        #A client can be installed with its config directory not yet created;
        #without this the write fails with FileNotFoundError on the dir.
        _dir = os.path.dirname(path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        with open(path, 'w') as _f:
            _json.dump(_conf, _f, indent=2)
        done.append('{} ({})'.format(label, path))
    except Exception:
        fail.append(label + ': ' + traceback.format_exc(limit=1))


def _register_desktop_mcp(apply=False):
    """Add KAME to the desktop GUI clients found on this machine.

    Two steps on purpose: the first click only reports what would change --
    every target path, and for the file that gets edited, its current `kame`
    entry against the new one -- and the second applies it.  Persistent
    configuration of another application is not something to do behind a
    single click.

    Each client is reached the way that client supports.  LM Studio / Bionic
    take an `lmstudio://add_mcp` deeplink, so the app itself shows a
    confirmation and no path has to be guessed -- its mcp.json location is not
    derivable anyway (the bundle builds it at run time and carries both
    'mcp.json' and 'ng-mcp.json' strings; a guessed path would fail silently,
    which is the worst outcome).  Codex has `codex mcp add`, which owns the
    TOML format.  Claude Desktop and Google's agent CLI are edited as JSON --
    additively, after a backup.
    """
    import subprocess as _sp, platform as _pf
    _launcher = _register_stdio_entry()
    if not _launcher:
        return
    _entry = {'command': _launcher}
    _sys = _pf.system()
    _plan, _done, _fail = [], [], []

    _lm = _lmstudio_deeplink()
    _cc = _claude_desktop_config()
    _cx = _resolve_cli('codex')
    _agy = _resolve_cli('agy')
    #The CLI owns the file; hand-edit its JSON only when it is absent
    #(an enterprise Gemini Code Assist licence still runs the old CLI).
    _gc = None if _agy else _google_agent_config()

    if not (_lm or _cc or _gc or _cx or _agy):
        _kame_gui_html('<font color="#996600">No MCP client found to register '
            'with (looked for Bionic / LM Studio, Claude Desktop, and the '
            'codex and agy CLIs).</font>')
        return

    # --- Bionic / LM Studio: nothing to do, and nothing that can be checked -
    # Point one of its projects at the notebook workspace and it reads the
    # .mcp.json KAME already writes there -- confirmed working before any of
    # this existed.  An `<scheme>://add_mcp` deeplink was tried instead and
    # withdrawn: Bionic declares `bionic`, not `lmstudio`, and sending the
    # right scheme still produced no dialog and no entry.  `open` exits 0 once
    # the URL reaches the app, which says nothing about the app acting on it,
    # so there is no success to report and reporting one would be a lie.
    if _lm:
        _plan.append('<b>{}</b> &mdash; nothing to register: open the notebook '
                     'workspace as a project and it picks up the '
                     '<tt>.mcp.json</tt> KAME writes there.'.format(
                     html.escape(_lm[0])))

    # --- JSON-configured clients with no CLI of their own -------------------
    if _cc:
        _mcp_json_edit('Claude Desktop', _cc, _entry, apply, _plan, _done, _fail)
    if _gc:
        if apply:
            #config/ need not exist yet even when ~/.gemini does.
            os.makedirs(os.path.dirname(_gc[1]), exist_ok=True)
        _mcp_json_edit(_gc[0], _gc[1], _entry, apply, _plan, _done, _fail)

    # --- clients that own their config through a CLI of their own -----------
    # Always preferable to editing their file: the CLI knows fields we would
    # not think to write.  `agy mcp add` records "disabled": false alongside
    # the command, which a hand-built entry would omit.
    for _label, _bin, _argv, _where in (
            ('Codex', _cx, ['mcp', 'add', 'kame', '--', _launcher],
             '~/.codex/config.toml'),
            ('Antigravity CLI', _agy, ['mcp', 'add', 'kame', _launcher],
             '~/.gemini/config/mcp_config.json')):
        if not _bin:
            continue
        _plan.append('<b>{}</b> &mdash; runs <tt>{} {}</tt> (writes {}).'.format(
            html.escape(_label), html.escape(os.path.basename(_bin)),
            html.escape(' '.join(_argv)), html.escape(_where)))
        if apply:
            try:
                _r = _sp.run([_bin] + _argv, capture_output=True, text=True, timeout=30)
                if _r.returncode == 0:
                    _done.append('{} ({})'.format(_label, _where))
                else:
                    _fail.append(_label + ': '
                                 + (_r.stderr or _r.stdout or 'failed').strip())
            except Exception:
                _fail.append(_label + ': ' + traceback.format_exc(limit=1))

    if not apply:
        _kame_gui_html(
            '<font color="#0066cc">Registering KAME as a persistent MCP server '
            'would do the following:</font><br/>&nbsp;&nbsp;'
            + '<br/>&nbsp;&nbsp;'.join(_plan)
            + '<br/><font color="#0066cc">The entry runs the stdio launcher, so '
              'it stays valid across KAME restarts, and is inert while KAME is '
              'not running.&nbsp; '
              '<a href="kame:register-mcp-apply">&#9654; Apply</a></font>')
        return
    if _done:
        _kame_gui_html('<font color="#008800">Registered with: '
            + html.escape('; '.join(_done))
            + '.&nbsp; Restart the app if it was already running.</font>')
    if _fail:
        _kame_gui_html('<font color="#cc0000">Failed: '
            + html.escape('; '.join(_fail)) + '</font>')


def _codex_desktop_launch(binary='codex'):
    """One-click launch of the Codex Desktop app, the twin of `kame:claude-app`.

    `codex app` opens the installer when the app is missing, so this needs no
    existence check of its own -- unlike the Claude app link, which can only
    ask the OS to open a bundle that may not be there.

    The same ephemeral `-c mcp_servers.kame.*` overrides the terminal launcher
    builds are passed along (the subcommand accepts -c), and the bearer token
    goes through the environment rather than argv.  Whether the app itself
    honours a launcher-scoped override is not something this can verify -- if
    it does not, the overrides are simply ignored, and the supported route for
    a GUI client is to install the plugin once (`codex plugin add kame@kame`),
    whose stdio launcher reaches the kernel through
    ~/.kame_kernel_connection.json and so does not care about the HTTP port.
    """
    import subprocess as _sp
    _bin = _resolve_cli(binary)
    if not _bin:
        # The app can be installed while its CLI is not: on Windows Codex
        # ships as an MSIX package, and this launcher reached it only through
        # `codex app`.  Open the package directly instead of reporting the CLI
        # missing.  The -c MCP overrides below cannot come along that way, but
        # the docstring's caveat already applies to them -- and the plugin
        # route (`codex plugin add kame@kame`) does not need them.
        _aumid = _msix_aumid('*codex*')
        if _aumid:
            _launch_msix(_aumid)
            _kame_gui_log('#Launched the Codex app (MSIX package; the `codex` '
                          'CLI is not installed, so no MCP override was '
                          'passed -- use `codex plugin add kame@kame` once).')
            return
        _kame_gui_html('<font color="#cc0000">`{}` not found in PATH.</font>'.format(
            html.escape(binary)))
        return
    _wd = _kame_workspace_dir()
    _spec = _kame_codex_spec()
    _env, _ov = dict(os.environ), []
    if _spec and _spec['type'] == 'stdio':
        _ov += ['-c', 'mcp_servers.kame.command=' + _toml_quote(_spec['command'])]
        _ov += ['-c', 'mcp_servers.kame.args=['
                + ', '.join(_toml_quote(a) for a in _spec['args']) + ']']
    elif _spec:
        _ov += ['-c', 'mcp_servers.kame.url=' + _toml_quote(_spec['url'])]
        if _spec.get('token'):
            _ov += ['-c', 'mcp_servers.kame.bearer_token_env_var='
                    + _toml_quote('KAME_MCP_TOKEN')]
            _env['KAME_MCP_TOKEN'] = _spec['token']
    try:
        _sp.Popen([_bin, 'app'] + _ov + [_wd], env=_env)
    except Exception:
        _kame_gui_html('<font color="#cc0000">Launching the {} desktop app failed:'
            '<br/>{}</font>'.format(html.escape(binary),
            html.escape(traceback.format_exc())))
        return
    _kame_gui_log("#Launching the {} desktop app in {} ...".format(binary, _wd))
    if not _spec:
        _kame_gui_html('<font color="#996600">No KAME MCP config found &mdash; '
            'start the Jupyter notebook first to enable KAME tools.</font>')


def _codex_launch(binary):
    """One-click launch of codex / codex-fugu wired to KAME's MCP server.

    macOS/Linux: EPHEMERAL, session-scoped `-c mcp_servers.kame.*` overrides —
    nothing is written to ~/.codex/config.toml, so it cannot clobber the user's
    other servers and needs no cleanup (mirrors the throwaway `.mcp.json`).
    Windows: delegates to `codex mcp {remove,add}` so cmd.exe never has to quote
    a TOML array; the entry is re-registered (self-correcting) on each launch.

    Token hygiene: the bearer token must never appear in argv (visible to any
    local user via ps / /proc/*/cmdline for the life of the terminal), so both
    POSIX paths go through a 0o700 temporary launch script that exports it and
    deletes itself on first read; the .bat likewise self-deletes after codex
    exits.
    """
    import shlex as _shlex, platform as _pf, tempfile as _tf, subprocess as _sp
    _bin = _resolve_cli(binary)
    if not _bin:
        _kame_gui_html('<font color="#cc0000">`{}` not found in PATH.</font>'.format(
            html.escape(binary)))
        return
    _spec = _kame_codex_spec()
    if not _spec:
        _kame_gui_html('<font color="#996600">No KAME MCP config found &mdash; '
            'launching {} without KAME tools. Start the Jupyter notebook first '
            'to enable them.</font>'.format(html.escape(binary)))
    _wd = _kame_workspace_dir()
    _sys = _pf.system()
    try:
        if _sys == 'Windows':
            _lines = ['@echo off', 'cd /d "{}"'.format(_wd)]
            if _spec:
                _lines.append('"{}" mcp remove kame >nul 2>&1'.format(_bin))
                if _spec['type'] == 'stdio':
                    _a = ' '.join('"{}"'.format(a) for a in _spec['args'])
                    _lines.append('"{}" mcp add kame -- "{}" {}'.format(
                        _bin, _spec['command'], _a))
                else:
                    if _spec.get('token'):
                        _lines.append('set "KAME_MCP_TOKEN={}"'.format(_spec['token']))
                    _lines.append('"{}" mcp add kame --url "{}"{}'.format(
                        _bin, _spec['url'],
                        ' --bearer-token-env-var KAME_MCP_TOKEN'
                        if _spec.get('token') else ''))
            _lines += ['"{}"'.format(_bin), 'pause']
            # Self-delete (the token is inside): `(goto) 2>nul` aborts batch
            # reading with the already-parsed `& del` still executing, so no
            # "batch file not found" complaint. Skipped if the window is
            # closed during pause — %TEMP% is per-user, acceptable.
            _lines.append('(goto) 2>nul & del "%~f0"')
            _bat = _tf.NamedTemporaryFile('w', suffix='.bat', delete=False)
            _bat.write('\r\n'.join(_lines) + '\r\n')
            _bat.close()
            _sp.Popen(['cmd', '/c', 'start', '', _bat.name])
        else:
            _env, _ov = {}, []
            if _spec and _spec['type'] == 'stdio':
                _ov += ['-c', 'mcp_servers.kame.command=' + _toml_quote(_spec['command'])]
                _arr = '[' + ', '.join(_toml_quote(a) for a in _spec['args']) + ']'
                _ov += ['-c', 'mcp_servers.kame.args=' + _arr]
            elif _spec:
                _ov += ['-c', 'mcp_servers.kame.url=' + _toml_quote(_spec['url'])]
                if _spec.get('token'):
                    _ov += ['-c', 'mcp_servers.kame.bearer_token_env_var=' + _toml_quote('KAME_MCP_TOKEN')]
                    _env['KAME_MCP_TOKEN'] = _spec['token']
            _cmd = [_bin] + _ov
            # Owner-only, self-deleting launch script: `rm -f -- "$0"` on the
            # first line unlinks the file while bash keeps reading from the
            # already-open fd, so the exported token exists on disk only for
            # the instant between Popen and the shell starting.
            _lines = ['#!/bin/bash', 'rm -f -- "$0"',
                      'cd {}'.format(_shlex.quote(_wd))]
            for _k, _v in _env.items():
                _lines.append('export {}={}'.format(_k, _shlex.quote(_v)))
            if _sys == 'Darwin':
                _lines.append('exec {}'.format(' '.join(_shlex.quote(a) for a in _cmd)))
                _sc = _tf.NamedTemporaryFile('w', suffix='.command', delete=False)
                _sc.write('\n'.join(_lines) + '\n')
                _sc.close()
                os.chmod(_sc.name, 0o700)
                _sp.Popen(['open', '-a', 'Terminal', _sc.name])
            else:
                _lines.append(' '.join(_shlex.quote(a) for a in _cmd))
                _lines.append('exec bash')  # keep the window alive afterwards
                _sc = _tf.NamedTemporaryFile('w', suffix='.sh', delete=False)
                _sc.write('\n'.join(_lines) + '\n')
                _sc.close()
                os.chmod(_sc.name, 0o700)
                if not _open_linux_terminal(_shlex.quote(_sc.name)):
                    os.unlink(_sc.name)
                    _kame_gui_html('<font color="#cc0000">No terminal emulator found. '
                        'Set $TERMINAL, or run <tt>{}</tt> yourself in {}.</font>'.format(
                        html.escape(binary), html.escape(_wd)))
                    return
    except Exception:
        _kame_gui_html('<font color="#cc0000">Launching {} failed:<br/>{}</font>'.format(
            html.escape(binary), html.escape(traceback.format_exc())))
        return
    _kame_gui_log("#Launching {} (terminal) in {} ...".format(binary, _wd))


def _kame_gui_html(htmlmsg):
	"""Show raw html in KAME's Python pane, visibly, from ANY thread.

	kame: link handlers run on the MAIN thread via py::eval, where
	TLS.xscrthread is unset and MYDEFOUT falls back to the OS terminal — so
	every error they printed was invisible in the GUI ("clicking does
	nothing").  Write to the script thread's pane node directly, the same
	trick launchJupyterConsole's _gui_log uses."""
	try:
		my_defout(XScriptingThreads()[0], htmlmsg)
	except Exception:
		pass
	try:
		import re as _re
		STDERR.write(_re.sub(r'<[^>]+>', '', htmlmsg) + '\n')
	except Exception:
		pass


def _kame_gui_log(msg, color='#008800'):
	"""Plain-text variant of _kame_gui_html."""
	_kame_gui_html('<font color="{}">{}</font>'.format(color, html.escape(msg)))


PYAI_PYTHON_FILE = os.path.join(os.path.expanduser('~'), '.kame_pyai_python')
PYAI_AGENT_FILE = os.path.join(os.path.expanduser('~'), '.kame_pyai_agent')


def _warn_if_mcp_unavailable():
	"""Say, in the pane, when nothing can run the MCP server.

	The links printed above are the only place a user meets MCP, so a missing
	prerequisite has to be visible there rather than only when a click fails.
	Probed on a daemon thread: it spawns interpreters, and this thread goes on
	to pump KAME's event loop.

	Accuracy matters more than coverage here -- a false alarm on a working
	setup would be worse than silence -- so the candidate list mirrors the real
	search: the interpreter that runs Jupyter (whose environment is almost
	always where `mcp` was installed), then a kame-mcp-venv beside the
	installation, before the generic python3 names."""
	def _run():
		_extra = []
		try:
			_progs = listOfJupyterPrograms()
		except Exception:
			_progs = []
		if _progs:
			#The shebang of the jupyter launcher names the interpreter whose
			#environment has jupyter_client -- the primary candidate.
			try:
				with open(_progs[0], 'rb') as _f:
					_first = _f.readline(512)
				if _first.startswith(b'#!'):
					_parts = _first[2:].decode('utf-8', 'replace').split()
					if _parts:
						import shutil as _sh2
						_cand = (_sh2.which(_parts[1])
								 if os.path.basename(_parts[0]) == 'env' and len(_parts) > 1
								 else _parts[0])
						if _cand:
							_extra.append(_cand)
			except OSError:
				pass
		_sub = ('Scripts', 'python.exe') if os.name == 'nt' else ('bin', 'python3')
		for _d in range(1, 7):
			_v = os.path.join(KAME_ResourceDir, *(['..'] * _d), 'kame-mcp-venv', *_sub)
			if os.path.isfile(_v):
				_extra.append(_v)
				break
		if _find_python_with(('mcp', 'jupyter_client'), 'KAME_MCP_PYTHON', _extra):
			return
		_kame_gui_html('<font color="#996600">No Python with <tt>mcp</tt> and '
			'<tt>jupyter_client</tt> was found, so the MCP server cannot start '
			'and the AI links above will fail.&nbsp; Setup: <a href="{0}">{0}</a>'
			'</font>'.format(MCP_SETUP_URL))
	threading.Thread(target=_run, name='kame-mcp-precheck', daemon=True).start()


def _open_when_listening(url, host, port, timeout=90.0):
	"""Open the browser once something answers on host:port.

	`clai web` prints its URL and waits; nothing opens the browser, so the
	link looked like it had merely opened a terminal.  The wait has to happen
	off the main thread: kame: link handlers run there, and blocking it would
	freeze the GUI for as long as the server takes to come up."""
	def _run():
		import socket as _s, webbrowser as _wb
		_deadline = time.time() + timeout
		while time.time() < _deadline:
			try:
				with _s.create_connection((host, port), timeout=0.5):
					_wb.open(url)
					return
			except OSError:
				time.sleep(0.5)
		_kame_gui_html('<font color="#996600">The web UI did not come up on '
			'{} within {:.0f}s &mdash; see the terminal window for why.</font>'.format(
			html.escape(url), timeout))
	threading.Thread(target=_run, name='kame-pyai-web-open', daemon=True).start()


def _free_port():
	"""A port free right now, for handing to a child that will bind it."""
	import socket as _s
	_sk = _s.socket()
	try:
		_sk.bind(('127.0.0.1', 0))
		return _sk.getsockname()[1]
	finally:
		_sk.close()


def _pyai_agent(py):
	"""(clai --agent spec, directory to run in, ASGI app spec) for the links.

	Precedence: an agent the user picked in the dialog, then KAME_PYAI_AGENT
	for scripted setups, then the one shipped in Resources.  Choosing your own
	is the point: KAME owns the MCP endpoint, not your capability list, model
	roster or instructions -- an agent that also wants a Coder, memory or web
	search belongs in your module, with KAME's toolset added to it as one
	capability."""
	try:
		with open(PYAI_AGENT_FILE) as _f:
			_saved = _f.read().strip()
	except OSError:
		_saved = ''
	_spec = _saved or os.environ.get('KAME_PYAI_AGENT') or ''
	if not _spec:
		return ('kame_pydantic_ai:agent', None, '')
	#A spec file (clai reads .yml/.yaml/.json itself) is passed as a path;
	#a module spec needs its own directory as cwd so the import resolves.
	_parts = _spec.split('|')
	_path, _var = _parts[0], (_parts[1] if len(_parts) > 1 else '')
	_app = _parts[2] if len(_parts) > 2 else ''
	if _var:
		_mod = os.path.splitext(os.path.basename(_path))[0]
		return ('{}:{}'.format(_mod, _var), os.path.dirname(_path),
				'{}:{}'.format(_mod, _app) if _app else '')
	return (_spec, None, '')


def _pyai_pick_agent(py, path):
	"""Remember an agent module/spec the user picked, after checking it loads.

	A .py is only useful if it actually exposes an Agent, and which variable
	holds it is convention, not law -- so ask the interpreter rather than
	assume `agent` and fail later inside clai with nothing to go on."""
	import subprocess as _sp3
	if not path:
		#Cancel clears the choice and returns to the agent KAME ships.
		try:
			os.unlink(PYAI_AGENT_FILE)
			_kame_gui_log('#Using the agent KAME ships (kame_pydantic_ai:agent).')
		except OSError:
			_kame_gui_log('#Already using the agent KAME ships.')
		return
	if os.path.splitext(path)[1].lower() in ('.yml', '.yaml', '.json'):
		_record = path
		_note = 'agent spec file'
	else:
		# Report the Agent variable and, separately, any ASGI app the module
		# built with Agent.to_web(models=...) -- that app carries the model
		# roster the author chose, which `clai web` cannot see because it takes
		# an Agent, not an app.  Serving it directly is the only way their
		# model switcher works as written.
		_probe = ('import importlib.util as u\n'
				  'from pydantic_ai import Agent\n'
				  's = u.spec_from_file_location("_kame_pyai_probe", %r)\n'
				  'm = u.module_from_spec(s); s.loader.exec_module(m)\n'
				  'v = vars(m)\n'
				  'ns = [k for k, o in v.items() if isinstance(o, Agent)]\n'
				  'try:\n'
				  '    from starlette.applications import Starlette\n'
				  'except ImportError:\n'
				  '    Starlette = ()\n'
				  'apps = [k for k, o in v.items()\n'
				  '        if Starlette and isinstance(o, Starlette)]\n'
				  'print(("agent" if "agent" in ns else (ns[0] if ns else "")))\n'
				  'print(("app" if "app" in apps else (apps[0] if apps else "")))\n'
				  % path)
		try:
			_r = _sp3.run([py, '-c', _probe], capture_output=True, text=True,
						  timeout=120, cwd=os.path.dirname(path) or None)
		except Exception:
			_kame_gui_html('<font color="#cc0000">Could not run {} to check '
				'{}.</font>'.format(html.escape(py), html.escape(path)))
			return
		_lines = [_l.strip() for _l in (_r.stdout or '').strip().splitlines()]
		_var = _lines[0] if _lines else ''
		_app = _lines[1] if len(_lines) > 1 else ''
		if _r.returncode != 0 or not _var:
			_kame_gui_html('<font color="#cc0000">{} defines no '
				'<tt>pydantic_ai.Agent</tt>, so clai has nothing to run.'
				'{}</font>'.format(html.escape(os.path.basename(path)),
				'<br/><tt>' + html.escape((_r.stderr or '').strip().splitlines()[-1][:200])
				+ '</tt>' if (_r.stderr or '').strip() else ''))
			return
		_record = path + '|' + _var + '|' + _app
		_note = 'agent <tt>{}</tt>{}'.format(html.escape(_var),
			', web UI from <tt>{}</tt> (your own model list)'.format(html.escape(_app))
			if _app else ', web UI from clai')
	try:
		with open(PYAI_AGENT_FILE, 'w') as _f:
			_f.write(_record + '\n')
	except OSError:
		pass
	_kame_gui_html('<font color="#008800">Pydantic AI links will use {} ({}). '
		'Pick again and Cancel to go back to the one KAME ships.</font>'.format(
		html.escape(os.path.basename(path)), _note))


def _find_python_with(mods, env_override=None, extra=()):
	"""First interpreter that imports every name in `mods`, or None.

	The same lesson as the MCP-server search: unversioned python3 names are
	unreliable (Homebrew relinks them on upgrades, MacPorts makes none
	without `port select`), so versioned python3.X binaries are scanned too,
	newest first, each candidate proved by actually importing `mods`."""
	import shutil as _sh, subprocess as _sp, glob as _glob, re as _re
	_cands = []
	if env_override and os.environ.get(env_override):
		_cands.append(os.environ[env_override])
	_cands += [_e for _e in extra if _e]
	for _n in ('python3', 'python'):
		_p = _sh.which(_n)
		if _p:
			_cands.append(_p)
	_vers = []
	for _d in ('/opt/homebrew/bin', '/opt/local/bin', '/usr/local/bin', '/usr/bin'):
		_vers += [_p for _p in _glob.glob(os.path.join(_d, 'python3.*'))
				  if _re.search(r'python3\.\d+$', _p)]
	_cands += sorted(_vers,
		key=lambda _p: int(_re.search(r'python3\.(\d+)$', _p).group(1)),
		reverse=True)
	_probe = ';'.join('import ' + _m for _m in mods)
	# Every candidate is a DIFFERENT interpreter from the one embedded in KAME,
	# so KAME's own Python environment must not follow it into the probe.  On
	# Windows this decides the answer rather than merely tidying it up:
	# kame-msyspython.bat exports PYTHONHOME=C:\msys64\mingw64 and MSYS2's
	# PYTHONPATH, and a real CPython told to use those loads mingw-built C
	# extensions it cannot open -- `ModuleNotFoundError: No module named
	# '_socket'` -- so EVERY candidate fails and the caller reports "no Python
	# with mcp and jupyter_client" while the very same venv starts the server
	# fine from launchJupyterConsole, which does strip them.
	_env = dict(os.environ)
	for _v in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'PYTHONSTARTUP'):
		_env.pop(_v, None)
	_seen = set()
	for _c in _cands:
		_rp = os.path.realpath(_c)
		if _rp in _seen:
			continue
		_seen.add(_rp)
		try:
			_sp.check_call([_c, '-c', _probe],
						   stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=10,
						   env=_env)
			return _c
		except Exception:
			continue
	return None


def _venv_python(d):
	"""Interpreters worth trying for a folder picked in the venv dialog.

	People pick the PROJECT, not the venv, and they are not wrong to: uv,
	poetry and pdm all keep the interpreter in a `.venv` that macOS's file
	dialog does not even show by default, so the venv root is not a thing the
	user can conveniently point at.  Accepting only <picked>/bin/python turned
	the natural choice into an error message.

	Also accept the interpreter itself and a `bin`/`Scripts` directory, since
	the same dialog reaches those just as easily.  Returns every existing
	candidate, most specific first; the caller picks the one that imports what
	it needs, so a project holding several venvs resolves to the usable one."""
	if os.path.isfile(d) and os.access(d, os.X_OK):
		return [d]
	_bin = 'Scripts' if os.name == 'nt' else 'bin'
	_exe = ('python.exe', 'python3.exe') if os.name == 'nt' else ('python', 'python3')
	_cands = []
	for _root in (d,) + tuple(os.path.join(d, _v) for _v in ('.venv', 'venv', 'env')):
		_cands += [os.path.join(_root, _bin, _e) for _e in _exe]
	#The dialog lands inside the venv's bin/ as readily as on the venv root.
	_cands += [os.path.join(d, _e) for _e in _exe]
	_seen, _out = set(), []
	for _c in _cands:
		_r = os.path.realpath(_c)
		if os.path.isfile(_c) and _r not in _seen:
			_seen.add(_r)
			_out.append(_c)
	return _out


def _kame_plugin_dir():
	"""The Claude Code plugin shipped with this KAME, or None.

	Deployed builds carry it at <Resources>/plugin (kame.pro); a source build
	that has not deployed it is found by the same upward search the MCP venv
	uses.  Handing this to `claude --plugin-dir` loads the kame skill and MCP
	server without any /plugin install ceremony, and always the version this
	KAME shipped with."""
	_c = os.path.join(KAME_ResourceDir, 'plugin')
	if os.path.isfile(os.path.join(_c, '.claude-plugin', 'plugin.json')):
		return _c
	for _depth in range(3, 7):
		_c = os.path.normpath(os.path.join(
			KAME_ResourceDir, *(['..'] * _depth), 'kame', 'script', 'plugin'))
		if os.path.isfile(os.path.join(_c, '.claude-plugin', 'plugin.json')):
			return _c
	return None


def kame_handle_link(action):
	"""Dispatch clicks on kame: links in the IPython/script pane (Jupyter / Claude launch)."""
	import subprocess as _sp, shlex as _shlex, platform as _pf
	try:
		if action == 'notebook':
			_progs = listOfJupyterPrograms()
			if not _progs:
				_kame_gui_html('<font color="#cc0000">No Jupyter program found '
					'(install jupyter, or use the Script menu).</font>')
				return
			_wd = _kame_workspace_dir()
			_kame_gui_log("#Launching Jupyter notebook ({}) in {} ...".format(_progs[0], _wd))
			launchJupyterConsole(_progs[0], 'notebook ' + _wd)
		elif action == 'claude-app':
			_sys = _pf.system()
			if _sys == 'Darwin':
				_sp.Popen(['open', '-a', 'Claude'])
			elif _sys == 'Windows':
				# The Windows Claude app ships as an MSIX package, and `start
				# "" Claude` cannot open one: packaged apps are not on PATH,
				# have no Program Files exe and no App Paths / Uninstall
				# registry entry, so cmd has nothing to resolve.  They are
				# launched by AppUserModelID through the AppsFolder shell
				# namespace instead.  Ask the OS for the ID rather than
				# hardcoding it -- the publisher hash in the family name is
				# stable, but a plain-exe install has no AUMID at all, and
				# that install is exactly what the `start` fallback handles.
				_aumid = _msix_aumid('-Name Claude')
				if _aumid:
					_launch_msix(_aumid)
				else:
					_sp.Popen(['cmd', '/c', 'start', '', 'Claude'])
			else:
				# There is no Claude desktop app on Linux.  Running the bare
				# `claude` CLI with inherited stdio and no tty (which is what
				# this used to do) does nothing visible, so fall through to the
				# terminal path instead of pretending it worked.
				return kame_handle_link('claude-cli')
			_kame_gui_log("#Launched Claude app.")
		elif action == 'claude-cli':
			_wd = _kame_workspace_dir()
			_sys = _pf.system()
			# Load the bundled plugin (kame skill + MCP server) in place, so a
			# session launched from KAME needs no /plugin install and always
			# gets the version this KAME shipped.  Not on Windows: the
			# plugin's server launcher is POSIX sh, so --plugin-dir there
			# would greet every session with an MCP startup error.
			_pd = _kame_plugin_dir() if _sys != 'Windows' else None
			_claude = 'claude --plugin-dir {}'.format(_shlex.quote(_pd)) if _pd else 'claude'
			if _sys == 'Darwin':
				_osa = 'tell application "Terminal" to do script "cd {} && {}"'.format(
					_shlex.quote(_wd), _claude)
				_sp.Popen(['osascript', '-e', _osa,
						   '-e', 'tell application "Terminal" to activate'])
			elif _sys == 'Windows':
				_sp.Popen(['cmd', '/c', 'start', 'cmd', '/k', 'cd /d "{}" && claude'.format(_wd)])
			else:
				_inner = 'cd {} && {}; exec bash'.format(_shlex.quote(_wd), _claude)
				if not _open_linux_terminal(_inner):
					_kame_gui_html('<font color="#cc0000">No terminal emulator found. '
						'Set $TERMINAL, or run <tt>claude</tt> yourself in {}.</font>'.format(
						html.escape(_wd)))
					return
			_kame_gui_log("#Launching Claude Code (terminal) in {}{} ...".format(
				_wd, " with the kame plugin" if _pd else ""))
		elif action == 'codex-cli':
			_codex_launch('codex')
		elif action == 'codex-fugu-cli':
			_codex_launch('codex-fugu')
		elif action == 'codex-app':
			_codex_desktop_launch('codex')
		elif action == 'register-mcp':
			_register_desktop_mcp(apply=False)
		elif action == 'register-mcp-apply':
			_register_desktop_mcp(apply=True)
		elif action.startswith('pyai-'):
			# Vendor-neutral client on Pydantic AI: any provider:model, local
			# models via an OpenAI-compatible endpoint. The wrapper connects
			# over HTTP from ~/.kame_mcp_url and carries the server's safety
			# instructions with the toolset.
			#
			# The usual install is a VENV (pip install pydantic-ai clai into
			# ~/somewhere/venv), which no PATH probe can see — so the GUI asks
			# for the venv folder on first use (kame.cpp, like the notebook
			# workspace dialog), passes it as 'pyai-cli?venv=<dir>', and the
			# choice is remembered in ~/.kame_pyai_python. A remembered
			# interpreter that stopped importing pydantic_ai is deleted so the
			# next click re-asks — self-healing, no manual cleanup.
			action, _, _agentfile = action.partition('?file=')
			_agentfile = _agentfile.strip()
			action, _, _venvdir = action.partition('?venv=')
			_venvdir = _venvdir.strip()
			_wd = _kame_workspace_dir()
			_script = os.path.join(KAME_ResourceDir, 'kame_pydantic_ai.py')
			if not os.path.isfile(_script):
				_kame_gui_html('<font color="#cc0000">kame_pydantic_ai.py not '
					'found in {} &mdash; rebuild/redeploy KAME.</font>'.format(
					html.escape(KAME_ResourceDir)))
				return
			import subprocess as _sp2
			_py = None
			if _venvdir:
				_cands = _venv_python(_venvdir)
				if not _cands:
					_kame_gui_html('<font color="#cc0000">No Python interpreter under '
						'{}.<br/>Looked for <tt>bin/python</tt> and '
						'<tt>.venv/</tt>, <tt>venv/</tt>, <tt>env/</tt> inside it '
						'(<tt>Scripts\\python.exe</tt> on Windows). Pick the project '
						'or venv folder, or the interpreter itself.</font>'.format(
						html.escape(_venvdir)))
					return
				_why = ''
				for _c in _cands:
					try:
						_r = _sp2.run([_c, '-c', 'import pydantic_ai'],
									  capture_output=True, text=True, timeout=15)
						if _r.returncode == 0:
							_py = _c
							break
						_why = _why or (_r.stderr or '')
					except Exception:
						continue
				if not _py and ('Operation not permitted' in _why
								or 'init_import_site' in _why):
					# EPERM, not EACCES: macOS privacy (TCC), not file modes.
					# The venv sits under a protected folder (Documents,
					# Desktop, Downloads, iCloud Drive) and the interpreter is
					# a CHILD of KAME, so it inherits KAME's authorisation and
					# is refused without any prompt being shown.  A build-tree
					# KAME is ad-hoc signed, so its cdhash changes on every
					# rebuild and a granted authorisation does not survive one.
					_kame_gui_html('<font color="#cc0000">macOS privacy protection '
						'blocked {} from reading its own venv.<br/>The venv is under '
						'a protected folder (Documents / Desktop / Downloads / iCloud '
						'Drive), and it runs as a child of KAME, so it is refused '
						'without a prompt.<br/>The venv has to live outside those four '
						'&mdash; anywhere else in your home directory will do; the '
						'project itself can stay where it is (with uv, '
						'<tt>UV_PROJECT_ENVIRONMENT=&lt;path&gt; uv sync</tt>).<br/>'
						'Granting KAME Full Disk Access in System Settings &rarr; '
						'Privacy &amp; Security also works, but a locally built KAME '
						'is ad-hoc signed, so that grant is lost on the next '
						'rebuild.</font>'.format(html.escape(_cands[0])))
					return
				if not _py:
					_kame_gui_html('<font color="#cc0000">{} lacks <tt>pydantic_ai</tt>. '
						'Install it there: <tt>{} -m pip install pydantic-ai clai</tt>'
						'{}{}</font>'.format(
						html.escape(_cands[0]), html.escape(_cands[0]),
						'<br/>(also tried: ' + html.escape(', '.join(_cands[1:])) + ')'
						if len(_cands) > 1 else '',
						'<br/><tt>' + html.escape(_why.strip().splitlines()[-1][:200]) + '</tt>'
						if _why.strip() else ''))
					return
				try:
					with open(PYAI_PYTHON_FILE, 'w') as _f:
						_f.write(_py + '\n')
					_kame_gui_log('#Remembered Pydantic AI interpreter in ' + PYAI_PYTHON_FILE)
				except OSError:
					pass
			if not _py:
				_saved = None
				try:
					with open(PYAI_PYTHON_FILE) as _f:
						_saved = _f.read().strip() or None
				except OSError:
					pass
				_extra = [_saved,
						  os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'bin', 'python')
						  if os.environ.get('VIRTUAL_ENV') else None,
						  os.path.join(_wd, '.venv', 'bin', 'python')]
				_py = _find_python_with(('pydantic_ai',), 'KAME_PYAI_PYTHON', _extra)
				if not _py:
					if _saved:
						# The remembered interpreter went stale (venv moved or
						# emptied); forget it so the next click re-opens the
						# folder dialog instead of failing the same way forever.
						try:
							os.unlink(PYAI_PYTHON_FILE)
						except OSError:
							pass
					_kame_gui_html('<font color="#cc0000">No Python with '
						'<tt>pydantic_ai</tt> found. Click the link again and pick '
						'the venv folder where you installed it '
						'(<tt>pip install pydantic-ai clai</tt>), or set '
						'KAME_PYAI_PYTHON.</font>')
					return
			if action == 'pyai-agent':
				#Picking is its own action: it needs the interpreter (to check
				#the file really exposes an Agent) but launches nothing.
				_pyai_pick_agent(_py, _agentfile)
				return
			_sys = _pf.system()
			# Hand the agent to the user's own clai rather than running the script
			# ourselves.  Imported as a module, kame_pydantic_ai exposes `agent` with
			# NO model bound -- only KAME's MCP toolset and its safety instructions --
			# and clai fills that in itself (`if agent.model is None or model_arg_set:
			# agent.model = infer_model(args.model or default_model)`).  So the model,
			# the API keys and the defaults all stay in the setup the user already
			# has, and KAME never has to ask which model to use.  Running the script
			# directly is the fallback, and that one does have to be told a model.
			_weburl = ''   #set below only for the clai web path
			_clai = os.path.join(os.path.dirname(_py),
								 'clai.exe' if _sys == 'Windows' else 'clai')
			_via_clai = os.path.isfile(_clai)
			if _via_clai:
				# Pass -m when the user has named a model, because clai's own
				# default is openai:gpt-5 and most people have no key for it:
				# without this the link dies on
				# `UserError: Set the OPENAI_API_KEY environment variable`,
				# which says nothing about what to do. With neither the env var
				# nor a key, clai's default and its error are the right owner
				# of the problem -- KAME still does not pick a model.
				_model = (os.environ.get('KAME_PYAI_MODEL')
						  or os.environ.get('PYDANTIC_AI_MODEL') or '')
				# Which agent: the one picked in the dialog (kame:pyai-agent),
				# else KAME_PYAI_AGENT for scripted setups, else the one shipped
				# in Resources.  KAME has no business owning the capability
				# list, the model roster or the instructions -- an agent that
				# also wants a Coder, memory or web search belongs in the user's
				# own module, needing only KAME's toolset added to it
				# (capabilities.MCP(url=..., authorization_token=...) read from
				# ~/.kame_mcp_url).  A picked module runs in its own directory
				# so the import resolves.
				_agent, _agentdir, _webapp = _pyai_agent(_py)
				if _agentdir:
					_wd = _agentdir
				#-m only for the agent KAME ships, which binds no model on
				#purpose.  clai overrides unconditionally when -m is present
				#(`if agent.model is None or model_arg_set`), so passing it to an
				#agent the user wrote would silently replace the model chosen in
				#their own module -- the opposite of the point of picking one.
				_own = _agent != 'kame_pydantic_ai:agent'
				#For the web UI, choose the port ourselves so the browser can be
				#opened on it: clai prints its URL and waits, and nothing was
				#opening it, which made the link look like it only ran a terminal.
				_webargs = []
				if action == 'pyai-web':
					_port = _free_port()
					_webargs = ['--host', '127.0.0.1', '--port', str(_port)]
					_weburl = 'http://127.0.0.1:{}'.format(_port)
				#Serve the module's own to_web() app when it has one: that is
				#where the author's model roster lives, and `clai web` cannot see
				#it because it takes an Agent, not an app -- which is why the UI
				#came up with nothing to choose from.  KAME_PYAI_MODEL may list
				#several (comma or space separated) for the clai path, where -m is
				#repeatable and builds the picker.
				_uvi = os.path.join(os.path.dirname(_py),
									'uvicorn.exe' if _sys == 'Windows' else 'uvicorn')
				if action == 'pyai-web' and _webapp and os.path.isfile(_uvi):
					_cmd = [_uvi, _webapp, '--host', '127.0.0.1', '--port', str(_port)]
				else:
					_models = [_x for _x in re.split(r'[,\s]+', _model) if _x] \
							  if _model and (not _own or action == 'pyai-web') else []
					_cmd = [_clai] + (['web'] if action == 'pyai-web' else []) \
						   + ['-a', _agent] \
						   + [_a for _x in _models for _a in ('-m', _x)] \
						   + _webargs
			else:
				_cmd = [_py, _script] + (['--web'] if action == 'pyai-web' else [])
			# The agent module ships with KAME, not in the user's venv.
			_cmdline = ('PYTHONPATH={} '.format(_shlex.quote(KAME_ResourceDir))
						if _via_clai else '') \
					   + ' '.join(_shlex.quote(a) for a in _cmd)
			if _sys == 'Darwin':
				_osa = 'tell application "Terminal" to do script "cd {} && {}"'.format(
					_shlex.quote(_wd), _cmdline)
				_sp.Popen(['osascript', '-e', _osa,
						   '-e', 'tell application "Terminal" to activate'])
			elif _sys == 'Windows':
				_sp.Popen(['cmd', '/c', 'start', 'cmd', '/k',
						   'cd /d "{}" && {}{}'.format(
							_wd,
							'set "PYTHONPATH={}" && '.format(KAME_ResourceDir)
							if _via_clai else '',
							' '.join('"{}"'.format(a) for a in _cmd))])
			else:
				_inner = 'cd {} && {}; exec bash'.format(_shlex.quote(_wd), _cmdline)
				if not _open_linux_terminal(_inner):
					_kame_gui_html('<font color="#cc0000">No terminal emulator found. '
						'Set $TERMINAL, or run <tt>{}</tt> yourself.</font>'.format(
						html.escape(_cmdline)))
					return
			if _weburl:
				_open_when_listening(_weburl, "127.0.0.1", _port)
			_kame_gui_log("#Launching Pydantic AI {} in {} ({}) ...".format(
				"web UI" if action == 'pyai-web' else "CLI", _wd,
				("via clai, agent " + _agent + ("; its own model" if _own
					else "; model from -m or clai's default"))
				if _via_clai else _py + "; needs --model or KAME_PYAI_MODEL"))
		else:
			_kame_gui_html('<font color="#cc0000">Unknown link action: {}</font>'.format(
				html.escape(str(action))))
	except Exception:
		_kame_gui_html('<font color="#cc0000">Link action "{}" failed:<br/>{}</font>'.format(
			html.escape(str(action)), html.escape(traceback.format_exc())))

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
				#Grouped by vendor: eight flat entries on one line stopped being
				#readable, and the terminal/desktop pair now repeats per vendor.
				MYDEFOUT.write_html(r'<font color="#0066cc">Quick launch:&nbsp; <a href="kame:notebook">&#9654; Jupyter notebook</a> &nbsp;&nbsp;|&nbsp;&nbsp; Claude: <a href="kame:claude-cli">&#9654; Code</a> &nbsp;<a href="kame:claude-app">&#9654; app</a> &nbsp;&nbsp;|&nbsp;&nbsp; Codex: <a href="kame:codex-cli">&#9654; CLI</a> &nbsp;<a href="kame:codex-fugu-cli">&#9654; fugu</a> &nbsp;<a href="kame:codex-app">&#9654; app</a> &nbsp;&nbsp;|&nbsp;&nbsp; Pydantic AI: <a href="kame:pyai-cli">&#9654; CLI</a> &nbsp;<a href="kame:pyai-web">&#9654; web</a> &nbsp;<a href="kame:pyai-agent">&#9881; agent</a></font>')
				#A client KAME does not launch gets no per-session override, so it
				#needs a one-time entry in its own config; this reports the change
				#first and only writes on the follow-up link.  The names must track
				#_register_desktop_mcp -- this label went stale when agy was added.
				_warn_if_mcp_unavailable()
				MYDEFOUT.write_html(r'<font color="#0066cc">One-time setup:&nbsp; <a href="kame:register-mcp">&#9654; Register KAME with your AI clients</a> <font color="#808080">(Codex / Antigravity / Claude Desktop)</font></font>')
				self.logfilename = os.path.splitext(connection_file)[0] + "-log" + os.extsep + "txt"
				self._initial_logfilename = self.logfilename
				MYDEFOUT.write_html(r'<font color="#008800">Logging console output to <a href="file:///'
						+ self.logfilename + r'">' + html.escape(self.logfilename) + '</a></font>')
				TLS.logfile = open(self.logfilename, mode='a')
				self.func = func
				# The kernel's shell ZMQ stream, used by start() to notice a
				# pending request and yield control back (see there).
				self.shell_stream = None
				try:
					from ipykernel.eventloops import get_shell_stream
					self.shell_stream = get_shell_stream(kernel)
				except Exception:
					self.shell_stream = getattr(kernel, 'shell_stream', None)

			def on_timer(self):
				loop = asyncio.get_event_loop()
				try:
					if self.func is not None:
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
					# YIELD BACK TO THE KERNEL when a shell message is waiting.
					#
					# A `%gui` loop hook owns the kernel's thread while it runs,
					# so it must hand control back for the kernel to service
					# requests -- that is what every stock loop_* in
					# ipykernel.eventloops does (`if shell_stream.flush(limit=1):
					# exit the toolkit main loop`).  ipykernel <= 6 let us get
					# away with never returning because `kernel.do_one_iteration()`
					# pumped one message per tick from inside the loop; ipykernel
					# 7 removed that method, so without this check an external
					# `jupyter console --existing` (and the MCP server, which
					# uses the same channel) never gets an answer.
					#
					# Returning is safe and cheap: ipykernel re-enters the hook
					# via enter_eventloop() after each message, and the teardown
					# below is guarded on is_main_terminated() so it only runs
					# when KAME is really quitting.
					if self.shell_stream is not None:
						try:
							if self.shell_stream.flush(limit=1):
								return
						except Exception:
							pass
				self.finish()

			def finish(self):
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
				# Tear down background HTTP MCP server (Windows path).
				if NOTEBOOK_MCP_HTTP_PROC is not None:
					try:
						NOTEBOOK_MCP_HTTP_PROC.terminate()
						NOTEBOOK_MCP_HTTP_PROC.wait(timeout=5)
					except Exception:
						try: NOTEBOOK_MCP_HTTP_PROC.kill()
						except Exception: pass
				if NOTEBOOK_MCP_URL_FILE:
					try:
						os.remove(NOTEBOOK_MCP_URL_FILE)
					except OSError:
						pass
				if NOTEBOOK_MCP_HTTP_LOG:
					try:
						os.remove(NOTEBOOK_MCP_HTTP_LOG)
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
					stopNotebookServer() #and wait for it to actually go
				#print(str([y[0] for y in inspect.getmembers(kernel, inspect.ismethod)]))

				# from ipykernel.kernelapp import IPKernelApp
				# app = IPKernelApp.instance()
				# app.close()
				task = asyncio.create_task(self.func())
				task.cancel()
				sys.stderr.write("sys.exit(0) from python.\n")
				sys.exit(0) #I could not find better way to exit normally.
				# raise IPython.terminal.embed.KillEmbedded('') #exits loop, magic %exit_raise no more exists.

		#Publish the executing cell into the kernel's XScriptingThread "Status",
		#reusing the very events that feed the iopub execute_input broadcast —
		#so the Script tab shows the running cell even when the code is not
		#inside sleep(). During cell N, IPython has already advanced
		#execution_count to N+1, hence the -1. The label is kept in
		#TLS.cell_status so sleep() can restore it on wakeup/exit.
		def _kame_pre_run_cell(info):
			try:
				if TLS.xscrthread:
					lines = (getattr(info, 'raw_cell', '') or '').strip().splitlines()
					head = lines[0][:60] if lines else ''
					label = "run Cell In[{}]: {}".format(get_ipython().execution_count - 1, head)
					TLS.cell_status = label
					TLS.xscrthread["Action"] = "" #discard stale wakeup/suspend/kill armed while idle
					TLS.xscrthread["Status"] = label
			except Exception:
				pass
		def _kame_post_run_cell(result):
			try:
				TLS.cell_status = None
				if TLS.xscrthread:
					ok = "done" if getattr(result, 'success', True) else "ERROR"
					n = getattr(result, 'execution_count', None)
					if n is None:
						n = get_ipython().execution_count - 1
					TLS.xscrthread["Status"] = "idle (Cell In[{}] {})".format(n, ok)
			except Exception:
				pass
		# ONE-TIME SETUP.  Everything from here to Timer() must run exactly once
		# per kernel, but loop_kamepysupport() itself is re-entered by ipykernel
		# after every message it processes (see Timer.start()).  Without this
		# guard the pre/post_run_cell hooks get registered again on each
		# re-entry, the startup banner is reprinted into the Script pane, and
		# the console log file is reopened -- all visible within seconds of the
		# first external client connecting.
		if getattr(kernel, 'timer', None) is None:
			# Status publishing is a convenience; it must never keep the timer
			# below from starting, since that timer is what pumps KAME's event
			# loop.
			try:
				kernel.shell.events.register('pre_run_cell', _kame_pre_run_cell)
				kernel.shell.events.register('post_run_cell', _kame_post_run_cell)
			except Exception:
				sys.stderr.write("KAME: executing-cell status unavailable: "
								 + str(traceback.format_exc()))
			# `Kernel.do_one_iteration` is the ipykernel <= 6 coroutine that
			# pumped one shell message per call.  ipykernel 7 removed it (the
			# kernel drives its own anyio task and a `%gui` hook is expected
			# only to pump the TOOLKIT), so calling it unconditionally raised
			# AttributeError from a tornado callback on every tick -- a fresh
			# `pip install ipykernel` gets 7.x.  Pass None and let on_timer()
			# skip it; the KAME-side work (kame_pybind_one_iteration, notebook
			# detection, stdout rebinding) still runs every tick, and
			# Timer.start() yields to the kernel when a message is pending.
			kernel.timer = Timer(getattr(kernel, 'do_one_iteration', None))
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

	# WHY A WATCHDOG THREAD IS NEEDED TO QUIT.
	#
	# Timer.start() polls is_main_terminated() and exits through finish(), but
	# it only gets to do so while the %gui hook is running, and the hook
	# returns to the kernel whenever a shell message is pending.  Once
	# ipykernel parks in its own event loop waiting for the next message
	# (blocked in kevent) the hook is not re-entered, so the flag is never read
	# again and embed_kernel() never returns.  XMeasure::terminate_all() then
	# blocks forever in m_python->join() and closing KAME hangs -- reproduced
	# twice on 2026-07-30, always when quitting after stopping a measurement
	# (FrmKameMain::closeEvent refuses to close while an interface is open, so
	# this is the only path that reaches it).
	#
	# Stopping the kernel's io_loop from another thread does not depend on the
	# hook at all.  add_callback() is thread-safe, and the loop thread has
	# released the GIL while it sits in kevent, so this thread runs.  The guard
	# is is_main_terminated() -- the same condition Timer.start() uses -- so
	# this can never fire while KAME is still running.
	def _kame_kernel_terminator():
		from ipykernel.kernelapp import IPKernelApp
		while not is_main_terminated():
			time.sleep(0.2)
		for _ in range(50):
			try:
				if IPKernelApp.initialized():
					loop = IPKernelApp.instance().io_loop
					if loop is not None:
						loop.add_callback(loop.stop)
						STDERR.write("kame: stopped the kernel event loop to quit.\n")
						return
			except Exception:
				pass
			time.sleep(0.2)
		STDERR.write("kame: could not reach the kernel event loop; "
					 "quitting may hang.\n")

	threading.Thread(target=_kame_kernel_terminator,
					 name="kame-kernel-terminator", daemon=True).start()

	try:
		# Now starting ipython kernel.
		IPython.embed_kernel(config=c) #, interrupt_mode='signal'
	except Exception:
		sys.stderr.write(str(traceback.format_exc()))

#Reached either through Timer.finish() (hook path) or after the watchdog above
#stopped the event loop.  The MCP hand-off files are removed here as well as in
#finish(), because a stale ~/.kame_kernel_connection.json left by a previous run
#points the MCP bridge at a dead kernel; removal is idempotent.
try:
	for _f in (NOTEBOOK_MCP_URL_FILE,
			   os.path.join(os.path.expanduser('~'), '.kame_kernel_connection.json')):
		if _f:
			try:
				os.remove(_f)
			except OSError:
				pass
	if NOTEBOOK_MCP_HTTP_PROC is not None:
		try:
			NOTEBOOK_MCP_HTTP_PROC.terminate()
			NOTEBOOK_MCP_HTTP_PROC.wait(timeout=5)
		except Exception:
			try: NOTEBOOK_MCP_HTTP_PROC.kill()
			except Exception: pass
except Exception:
	sys.stderr.write(str(traceback.format_exc()))

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
