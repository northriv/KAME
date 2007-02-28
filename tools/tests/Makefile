CXX=g++
CFLAGS=-g3 -O6 -msse3 -mfpmath=sse -march=prescott -g -I/sw/include/qt -I ../../kame -I/sw/include -I/usr/lib/qt-3.3/include -lpthread -DMACOSX

all : atomic_shared_ptr_test atomic_scoped_ptr_test atomic_queue_test
clean :
	rm -f atomic_shared_ptr_test atomic_scoped_ptr_test atomic_queue_test

atomic_shared_ptr_test : atomic_shared_ptr_test.cpp
	$(CXX) $(CFLAGS) atomic_shared_ptr_test.cpp -o atomic_shared_ptr_test
	$(CXX) $(CFLAGS) -DHAVE_CAS_2 atomic_shared_ptr_test.cpp -o atomic_shared_ptr_test_fake_cas2

atomic_scoped_ptr_test : atomic_scoped_ptr_test.cpp
	$(CXX) $(CFLAGS) atomic_scoped_ptr_test.cpp -o atomic_scoped_ptr_test
	$(CXX) $(CFLAGS) -DHAVE_CAS_2 atomic_scoped_ptr_test.cpp -o atomic_scoped_ptr_test_fake_cas2
	
atomic_queue_test : atomic_queue_test.cpp
	$(CXX) $(CFLAGS) atomic_queue_test.cpp -o atomic_queue_test
	$(CXX) $(CFLAGS) -DHAVE_CAS_2 atomic_queue_test.cpp -o atomic_queue_test_fake_cas2

check : atomic_shared_ptr_test atomic_scoped_ptr_test atomic_queue_test
	./atomic_shared_ptr_test && \
	./atomic_shared_ptr_test_fake_cas2 && \
	./atomic_scoped_ptr_test && \
	./atomic_scoped_ptr_test_fake_cas2 && \
	./atomic_queue_test && \
	./atomic_queue_test_fake_cas2 && \
	echo 'done.'
	