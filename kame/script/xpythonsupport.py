#KAME xpyhonsuppport start-up code
import time
import sys
import numpy as np
from kame import *
STDOUT = sys.stdout
STDERR = sys.stderr
STDIN = sys.stdin

print("Hello! KAME Python support.")

#Thread-monitor
MONITOR_PERIOD=0.2

# class Node:
# 	def __getitem__(pos):
# 		shot = Snapshot(self)
# 		if isinstance(pos, int):
# 			return shot[pos]
# 		else:
# 			return [x.getName() for x in list(shot)]

class MyDefIO:
	def write(s):
		s = s.replace("&", "&amp;")
		s = s.replace("<", "&lt;")
		s = s.replace(">", "&gt;")
		if s[-1] == '\n':
			s = s[0:-1]
		for l in s.splitlines():
			if len(l) and l[0] == "#":
				l = "<font color=#008800>" + l + "</font" 
			my_defout(l)
	def readline():
		return my_defin()
class MyDefOErr:
	def write(s):
		s = s.replace("&", "&amp;")
		s = s.replace("<", "&lt;")
		s = s.replace(">", "&gt;")
		if s[-1] == '\n':
			s = s[0:-1]
		for l in s.splitlines():
			l = "<font color=#ff0000>" + l + "</font" 
			my_defout(l)

sys.stdout = MyDefIO
sys.stderr = MyDefOErr
sys.stdin = MyDefIO

print("#testing python interpreter.")
while not is_main_terminated():
	time.sleep(MONITOR_PERIOD)
	try:
		line = input()
		print(">>{}".format(line))
		print(eval(line))
	except EOFError:
		pass
	except Exception as inst:
		sys.stderr.write(str(type(inst)))
		sys.stderr.write(str(inst))
		pass

print("bye!")

