#rubyinterrupter.rb

print "KAME Ruby interrupter.\n"

no = 1
bind = binding()
while(1)
	begin
		line = gets()
		print "##{no}>>#{line}"
		p eval(line, bind, "User", no)
	rescue ScriptError, StandardError, SystemExit
		$! = RuntimeError.new("unknown exception raised") unless $!
		print_exception($!)
	end
	no+=1
end
