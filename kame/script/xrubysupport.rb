#xrubysupport.rb
#KAME xrubysuppport start-up code
# -*- encoding: utf-8 -*-
# -*- coding: utf-8 -*-

#redirect defout, deferr
class << $stdout
  def write(str)
    Thread.current[:logfile].puts(Time.now().strftime("%T") + ":" + str) if Thread.current[:logfile]
    str = str.gsub(/&/, "&amp;")
    str = str.gsub(/</, "&lt;")
    str = str.gsub(/>/, "&gt;")
#  	str = str.gsub(/"/, "&quot;")
		str.each_line {|line|
			line = line.gsub(/\n/, "")
			if /^\s*#/ =~ str then
				line = line.gsub(/^/, "<font color=#008800>")
				line = line.gsub(/$/, "</font>")
			end
			XScriptingThreads.my_rbdefout(line, Thread.current.inspect[/0x[0-9a-f]*/])
		}
  end
end
class << $stderr
  def write(str)
    Thread.current[:logfile].puts(Time.now().strftime("%T") + ":" + str) if Thread.current[:logfile]
    str = str.gsub(/&/, "&amp;")
    str = str.gsub(/</, "&lt;")
    str = str.gsub(/>/, "&gt;")
    #  	str = str.gsub(/"/, "&quot;")
    str.each_line {|line|
        line = line.gsub(/\n/, "")
        line = line.gsub(/^/, "<font color=#ff0000>")
        line = line.gsub(/$/, "</font>")
        XScriptingThreads.my_rbdefout(line, Thread.current.inspect[/0x[0-9a-f]*/])
    }
  end
end
class Mystdin
	def getc()
		raise RuntimeError.new("Unsupported function.")
	end
	def gets()
		while(1)
			line = XScriptingThreads.my_rbdefin(Thread.current.inspect[/0x[0-9a-f]*/])
			if !line then
				sleep(0.15)
				next
			end
			return line
		end
	end
	def readline()
		self.gets()
	end
end
$stdin = Mystdin.new

alias :_orig_sleep :sleep
# Redefinition of sleep to show status
def sleep(sec)
	if sec < 1 then
		Thread.current[:statusstring] = nil
		_orig_sleep(sec)
		return
	end
	start = Time.now()
	while (Time.now() - start < sec)
		rest = Integer(sec - (Time.now() - start))
		if /^(.+?):(\d+)(?::in `(.*)')?/ =~ caller.first
			Thread.current[:statusstring] = " #{rest}s @#{$3}:#{$2}"
	    end
		if /^(.+?):(\d+)(?::in `(.*)')?/ =~ caller[1]
			Thread.current[:statusstring] += " < #{$3}:#{$2}"
	    end
		x = Time.now()
 		_orig_sleep(0.5)
        break if(Time.now() - x < 0.5) #interrupted.
    end
	Thread.current[:statusstring] = nil
end

#function to dump exception info
def print_exception(exc)
	$stderr.print "Exception:", exc.message
	bt_shown = false
	exc.backtrace.each {|b|
		$stderr.print b unless b.include?(__FILE__)
		bt_shown = true
	}
	$stderr.print exc.backtrace[0] unless bt_shown
end

#useful modules
include Math

#Fake node
class XFakeNode
	def [](key)
		$stderr.print("[#{key}] Ignored\n")
		self
	end
	def []=(key,value)
		$stderr.print("[#{key}]=#{value} Ignored\n")
	end
	def value=(value)
		$stderr.print("value=#{value} Ignored\n")
	end
	def <<(value)
		$stderr.print("<< #{value} Ignored\n")
	end
	def value()
		$stderr.print("value() Ignored\n")
	end
	def set(value)
		$stderr.print("set(#{value}) Ignored\n")
	end
	def get()
		$stderr.print("get() Ignored\n")
	end
	def load(value)
		$stderr.print("load(#{value}) Ignored\n")
	end
	def create(*arg)
		$stderr.print("create(...) Ignored\n")
		self
	end
end

#impliment more functions
class XNode
	include Enumerable
	#iterator
	def each
		idx = 0
		while idx < self.count()
			yield(self[idx])
			idx += 1
		end
	end
	#implicit conversion to Array
	def to_ary
		ary = Array.new()
		self.each {|x| ary.push x }
		return ary
	end
	def [](key)
		if Thread.current[:load_kam] then
			begin
				self.child(key)
		    rescue RuntimeError
			     $! = RuntimeError.new("unknown exception raised") unless $!
			     print_exception($!)
			     XFakeNode.new()
		 	end
		 else
			self.child(key)
		 end
	end
	#element substitution
	def []=(key, value)
		self[key].set(value)
	end
end

class XValueNode
	#implicit conversion to Number
	def to_int
		return self.get()
	end
	def set(value)
		if Thread.current[:load_kam] then
			begin
				self.internal_load(value)
		    rescue RuntimeError
			     $! = RuntimeError.new("unknown exception raised") unless $!
			     print_exception($!)
			     nil
		 	end
		 else
			self.internal_set(value)
		 end
	end
	def get
		if Thread.current[:load_kam] then
			begin
				self.internal_get()
		    rescue RuntimeError
			     $! = RuntimeError.new("unknown exception raised") unless $!
			     print_exception($!)
			     nil
		 	end
		 else
			self.internal_get()
		 end
	end
	def load(value)
		if Thread.current[:load_kam] then
			begin
				self.internal_load(value)
		    rescue RuntimeError
			     $! = RuntimeError.new("unknown exception raised") unless $!
			     print_exception($!)
		 	end
		else
			self.internal_load(value)
		end
	end
	#alias to set()
	def value=(value)
		self.set(value)
	end
	#alias to set()
	def <<(value)
		self.set(value)
	end
	#alias to get()
	def value()
		self.get()
	end
end

class XListNode
	def create(*arg)
		type = ""
		name = ""
		type = arg[0] if arg.size >= 1
		name = arg[1] if arg.size >= 2
		if Thread.current[:load_kam] then
			begin
				self.internal_create(type, name)
		    rescue RuntimeError
			     $! = RuntimeError.new("unknown exception raised") unless $!
			     print_exception($!)
			     XFakeNode.new()
		 	end
		 else
			self.internal_create(type, name)
		 end
	end
end

include GC
print "Hello! KAME Ruby support.\n"
#GC.start
print "Ruby " + RUBY_VERSION + " " + RUBY_PLATFORM + " " + RUBY_RELEASE_DATE + "\n"

#Thread-monitor
MONITOR_PERIOD=0.2

begin
	while ( !XScriptingThreads.is_main_terminated() )
	 XScriptingThreads.each {|xrbthread|
	   xrbthread_status = xrbthread["Status"]
	   xrbthread_action = xrbthread["Action"]
	   xrbthread_threadid = xrbthread["ThreadID"]
	   xrbthread_filename = xrbthread["Filename"]
	   thread = Thread.list.find {|x| x.inspect[/0x[0-9a-f]*/] == xrbthread_threadid.value }
	   action = xrbthread_action.value
	   if thread then
	     if thread.status == "starting" then
	       raise "why starting?"
		 end
		 if thread.status == "sleep" && action == "wakeup" then
	       print "wakeup " +  thread.to_s + "\n"
		   thread.wakeup
	       print thread.inspect + "\n"
	     end
		 if thread.alive? && action == "kill" then
	       print "kill " +  thread.to_s + "\n"
	       thread.kill
		   print thread.inspect + "\n"
		 end
		 xrbthread_action.set("")
		 status = thread.status
	     if status then
		   status += thread[:statusstring] if thread[:statusstring] && (status == "sleep")
	       if xrbthread_status.value != status then
			 xrbthread_status.value = status
		   end
	     else
	       if xrbthread_status.value != "" then
		       xrbthread_status.value = ""
		   end
	     end
	   else
	      if action == "starting" then
       	  sleep(0.5)
		      xrbthread_action.value = ""
				  print "Starting a new thread\n"
			  	filename = xrbthread_filename.value()
			  	print "Loading #{filename}.\n"
		      thread = Thread.new do
#						$stderr.print("a2\n")
						Thread.pass
		          begin
		             print thread.inspect + "\n"
#								print "b\n"
								if /\.seq/i =~ filename then
									Thread.current[:logfile] = File.open(filename + ".log", "a")
									Thread.current[:logfile].sync = true
									Thread.current[:logfile].puts("Started @#{Time.now()} \n")
								else
									Thread.current[:load_kam] = true
								end
								Thread.current[:xrbthread] = xrbthread;
								load filename
								print thread.to_s + " Finished.\n"
							rescue ScriptError, StandardError, SystemExit
								$! = RuntimeError.new("unknown exception raised") unless $!
								print_exception($!)
							end
						ensure
							Thread.current[:logfile].close() if Thread.current[:logfile]
						end
		       if thread then
		            xrbthread_threadid.value = thread.inspect[/0x[0-9a-f]*/]
			        xrbthread_status.value = thread.status
		       else
		            xrbthread_action.value = "failure"
		            xrbthread_status.value = ""
		       end
		  else
		  	if xrbthread_status.value != "" then
			  	print "thread is dead.\n"
		  		xrbthread_status.value = ""
		    end
		  end
	   end
	 }
	
	 sleep(MONITOR_PERIOD)
	end
rescue ScriptError, StandardError, SystemExit
	$! = RuntimeError.new("unknown exception raised") unless $!
	print_exception($!)
end

print "bye!\n"
