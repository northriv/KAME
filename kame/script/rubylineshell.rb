#rubyLine Shell.rb

print "Hello KAME Ruby Line Shell!\n"
print "Awaiting your command....\n"

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
