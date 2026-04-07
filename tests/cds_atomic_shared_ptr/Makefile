# Makefile for GenMC tests of atomic_shared_ptr
#
# Usage:
#   make run                # run all tests with GenMC
#   make run-test1          # run test 1 only
#
# Build GenMC (one-time):
#   brew install llvm@20 hwloc
#   cd genmc && mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm@20 \
#            -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang \
#            -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
#            -DCMAKE_EXE_LINKER_FLAGS="-L/opt/homebrew/lib" \
#            -DCMAKE_BUILD_TYPE=Release
#   make -j$(sysctl -n hw.ncpu)

GENMC ?= genmc/build/genmc

# --unroll=5: bound for CAS retry loops (increase if GenMC warns)
GMCFLAGS ?= --rc11 --unroll=5

SRCDIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

run: run-test1 run-test2 run-test3

run-test1:
	@echo "=== Test 1: load_shared_ / release_tag_ref_ safety ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_test_load.c
	@echo ""

run-test2:
	@echo "=== Test 2: load_shared_ + compareAndSwap_ race ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_test_cas.c
	@echo ""

run-test3:
	@echo "=== Test 3: multiple compareAndSwap_ race ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_test_multi_cas.c
	@echo ""

# Run with IMM memory model
run-imm: GMCFLAGS = --imm --unroll=5
run-imm: run

.PHONY: run run-test1 run-test2 run-test3 run-imm
