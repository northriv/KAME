#rubyLine Shell.rb
# -*- coding: utf-8 -*-

greeting=[\
"Use the Source, Luke.\n",\
"404 Physics Not Found.\n",\
"How many fails(0-15)?\n",\
"We are Borg. You will be assimilated.\nResistance is futile.\n",\
"404 Research Not Funded.\n",\
"403 Forbidden Experiment.\n"]
print greeting[Integer(rand()**2*greeting.size())]

#Usuful contants.
h = 6.626070040e-34
hbar = h / 2 / PI
c = 299792458.0
mu0 = 4e-7 * PI
epsilon0 = 8.854187817e-12
e = 1.6021766208e-19
g = 2.00231930436182
mub = 927.4009994e-26
kb = 1.38064852e-23
NA = 6.022140857e23

#Atomic weight
#IUPAC 2005
H=1.00794
Li=6.941
B=10.811
C=12.0107
N=14.0067
O=15.9994
F=18.9984032
Na=22.98976928
Mg=24.3050
Al=26.9815386
Si=28.0855
P=30.973762
S=32.065
Cl=35.453
K=39.0983
Ca=40.078
Sc=44.955912
Ti=47.867
V=50.9415
Cr=51.9961
Mn=54.938045
Fe=55.845
Co=58.933195
Ni=58.6934
Cu=63.546
Zn=65.409
Ga=69.723
Ge=72.64
As=74.92160
Se=78.96
Sr=87.62
Y=88.90585
Zr=91.224
In=114.818
Sn=118.710
Sb=121.760
Te=127.60
Cs=132.9054519
Ba=137.327
La=138.90547
Ce=140.116
Sm=150.36
Em=151.964
Yb=173.04
Lu=174.967
Hf=178.49
Pb=207.2
Bi=208.98040

#gyromagnetic ratio * 2 * PI [MHz/T]
gnuc = [g * mub / h * 1e-6,\
42.5763875, 6.5357348, 32.4344523, 0.0, 0.0, 6.2655778, 16.5468008, 0.0, -5.9827544,\
4.5742615, 13.6601971, 0.0, 10.7058412, 3.0767047, -4.3158692, 0.0, -5.7718497, 0.0, 40.0618307,\
0.0, -3.3611061, 0.0, 11.2622634, 0.0, -2.6063710, 0.0, 11.0940693, 0.0, -8.4587305,\
0.0, 17.2352376, 0.0, 3.2681635, 0.0, 4.1715957, 0.0, 3.4724131, 0.0, 1.9867731,\
-2.4702897, 1.0905111, 0.0, -2.8654032, 0.0, 10.3425483, 0.0, -2.4002583, 0.0, -2.4008982,\
4.2449974, 11.1988451, 0.0, -2.4066286, 0.0, 10.5543535, 0.0, 1.3785289, 0.0, 10.1021310,\
0.0, -3.8046477, 0.0, 11.2893305, 0.0, 12.0932664, 0.0, 2.6639207, 0.0, 10.2189095,\
0.0, 12.9843949, 0.0, -1.4851985, 0.0, 7.2901905, 0.0, 8.1199613, 0.0, 10.6670796,\
0.0, 11.4983998, 0.0, -1.6381691, 0.0, 4.1107259, 0.0, 0.0, 0.0, -2.0863273,\
0.0, -3.9580279, 0.0, 10.4210690, 0.0, -2.7746717, 0.0, -2.8329030, 0.0, -1.9607069,\
0.0, -2.1975245, 0.0, -1.3566740, 0.0, -1.9483381, 0.0, -1.7234151, 0.0, -1.9813062,\
0.0, -9.0327850, 0.0, 0.0, 0.0, 0.0, 0.0, -15.1709287, 0.0, -15.8770040,\
0.0, 10.1887752, 0.0, 0.0, 0.0, -13.4327519, 0.0, 8.5184648, 0.0, -11.8405726,\
0.0, 3.5099638, 0.0, 5.5843794, 0.0, 4.2297329, 0.0, 4.7314833, 5.6176563, 6.0141876,\
0.0, 13.037, 0.0, -2.320, 0.0, -1.431, 0.0, -1.775, 0.0, -1.465,\
0.0, 10.584, 0.0, 4.675, 0.0, -1.307, 0.0, -1.716, 0.0, 10.235,\
0.0, -1.465, 0.0, 2.052, 0.0, 9.086, 0.0, -1.226, 0.0, -3.53,\
0.0, 7.4505723, 0.0, -2.0526, 0.0, 4.8554, 3.4619, 1.706, 0.0, -1.0716,\
0.0, 5.1047386, 0.0, 1.7738971, 0.0, 9.5901610, 0.0, 0.0, 0.0, 3.3062268,\
0.0, 0.7315, 0.0, 0.7966, 0.0, 9.1525541, 0.0, 0.7361, 0.0, 7.6257810,\
0.0, -2.8149732, 0.0, 24.3209950, 0.0, 24.5596944, 0.0, 8.9072353, 0.0, 8.9072353
]

lineno = 1
bind = binding()
while(!XScriptingThreads.is_main_terminated())
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
