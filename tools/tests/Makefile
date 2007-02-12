CXX=g++
CFLAGS=-Wall -g3 -O1 -I/include/qt3 -I ../../kame -I/usr/local/include -I/usr/lib/qt-3.3/include -L/usr/local/lib -lnidaqmx

all : nidaqmx_cont_ao_test nidaqmx_cont_co_test
clean :
	rm -f nidaqmx_cont_ao_test nidaqmx_cont_co_test

nidaqmx_cont_ao_test : nidaqmx_cont_ao_test.cpp
	$(CXX) $(CFLAGS) nidaqmx_cont_ao_test.cpp -o nidaqmx_cont_ao_test

nidaqmx_cont_co_test : nidaqmx_cont_co_test.cpp
	$(CXX) $(CFLAGS) nidaqmx_cont_co_test.cpp -o nidaqmx_cont_co_test
	