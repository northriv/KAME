CXX=g++
CFLAGS=-I/sw/include/qt -I ../../kame -I/sw/include -I/usr/lib/qt-3.3/include -lpthread -DMACOSX  \
	-DSIZEOF_INT=4 -DSIZEOF_LONG=4 -DSIZEOF_VOID_P=4 -DSIZEOF_SHORT=2 -DSIZEOF_FLOAT=4 -DSIZEOF_DOUBLE=8 \
	-O6 -msse3 -mfpmath=sse -march=prescott   \
	-funroll-loops -fomit-frame-pointer -funswitch-loops -finline-functions -fgcse-after-reload \
	-ffast-math -fno-tree-vectorize -ftree-vectorizer-verbose=3 -fno-unsafe-math-optimizations \
	-D__sse2__

#	-g3 -O0

all : atomic_shared_ptr_test atomic_scoped_ptr_test atomic_queue_test mutex_test atomic_test
clean :
	rm -f atomic_shared_ptr_test atomic_scoped_ptr_test atomic_queue_test mutex_test atomic_test

support.o : support.cpp
	$(CXX) $(CFLAGS) -c support.cpp -o support.o
atomic_shared_ptr_test : support.o atomic_shared_ptr_test.cpp
	$(CXX) $(CFLAGS) support.o atomic_shared_ptr_test.cpp -o atomic_shared_ptr_test

atomic_scoped_ptr_test : support.o atomic_scoped_ptr_test.cpp
	$(CXX) $(CFLAGS) support.o atomic_scoped_ptr_test.cpp -o atomic_scoped_ptr_test
	
atomic_queue_test : support.o atomic_queue_test.cpp
	$(CXX) $(CFLAGS) support.o atomic_queue_test.cpp -o atomic_queue_test

mutex_test : support.o mutex_test.cpp
	$(CXX) $(CFLAGS) support.o mutex_test.cpp -o mutex_test

atomic_test : support.o atomic_test.cpp
	$(CXX) $(CFLAGS) support.o atomic_test.cpp -o atomic_test

check : atomic_shared_ptr_test atomic_scoped_ptr_test atomic_queue_test mutex_test atomic_test
	./atomic_shared_ptr_test && \
	./atomic_scoped_ptr_test && \
	./atomic_queue_test && \
	./mutex_test && \
	./atomic_test && \
	echo 'done.'
	
