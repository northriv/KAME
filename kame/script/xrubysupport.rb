#xrubysupport.rb
#KAME2 xrubysuppport start-up code

$KCODE = 'UTF8'

#redirect defout, deferr
class << $stdout
  def write(str)
    XRubyThreads.my_rbdefout(str, Thread.current.object_id)
  end
end
class << $stderr
  def write(str)
    XRubyThreads.my_rbdefout(str, Thread.current.object_id)
  end
end

#function to dump exception info
def print_exception(exc)
	print exc.message, "\n"
	bt_shown = false
	exc.backtrace.each {|b|
		print b, "\n" unless b.include?(__FILE__)
		bt_shown = true
	}
	print exc.backtrace[0], "\n" unless bt_shown
end

#useful modules
include Math

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

print "Hello! KAME Ruby support.\n"

#Thread-monitor
MONITOR_PERIOD=0.2

begin
	while ( !XRubyThreads.is_main_terminated() )
	 XRubyThreads.each {|xrbthread|
	   xrbthread_status = xrbthread["Status"]
	   xrbthread_action = xrbthread["Action"]
	   xrbthread_threadid = xrbthread["ThreadID"]
	   xrbthread_filename = xrbthread["Filename"]
	   thread = Thread.list.find {|x| x.object_id == xrbthread_threadid.value }
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
	     if action != "" then
		     xrbthread_action.set("")
		 end
	     if thread.status then
	       if xrbthread_status.value != thread.status then
		       xrbthread_status.value = thread.status
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
			  print "Loading #{xrbthread_filename.get()}.\n";
		      thread = Thread.new {
		          Thread.pass
		          begin
		             print thread.inspect + "\n"
		             load xrbthread_filename.value
		             print thread.to_s + " Finished.\n"
		          rescue ScriptError, StandardError
		    		     $! = RuntimeError.new("unknown exception raised") unless $!
		    		     print_exception($!)
		     	  end
		       }
		       if thread then
		            xrbthread_threadid.value = thread.object_id
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
rescue ScriptError, StandardError
	$! = RuntimeError.new("unknown exception raised") unless $!
	print_exception($!)
end

print "bye!\n"
