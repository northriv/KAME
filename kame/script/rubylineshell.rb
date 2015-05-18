#rubyLine Shell.rb
#coding: utf-8

greeting=[\
"Use the Source, Luke.\n",\
"404 Physics Not Found.\n",\
"How many fails(0-15)?\n",\
"We are Borg. You will be assimilated.\nResistance is futile.\n",\
"404 Research Not Funded.\n",\
"403 Forbidden Experiment.\n"]
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
