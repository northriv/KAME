#rubyinterrupter.rb

print "Hello KAME Ruby interrupter!\n"
print "Awaiting your command....\n"

lineno = 1
bind = binding()
while(1)
	begin
		line = gets()
		print "##{lineno}>>#{line}"
		p eval(line, bind, "User", lineno)
	rescue ScriptError, StandardError, SystemExit
		$! = RuntimeError.new("unknown exception raised") unless $!
		print_exception($!)
	end
	lineno+=1
end
