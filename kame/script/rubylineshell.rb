#rubyLine Shell.rb

greeting=["Your instruments are ready, Commander.\nAwaiting your orders.\n",\
"We are Borg. You will be assimilated.\nResistance is futile.\n"]
print greeting[Integer(rand()**2*greeting.size())]

lineno = 1
bind = binding()
while(!XRubyThreads.is_main_terminated())
	begin
		line = gets()
		print "##{lineno}>>#{line}"
	rescue ScriptError, StandardError
		break
	end
	begin
		p eval(line, bind, "User", lineno)
	rescue ScriptError, StandardError, SystemExit
		$! = RuntimeError.new("unknown exception raised") unless $!
		print_exception($!)
	end
	lineno+=1
end
