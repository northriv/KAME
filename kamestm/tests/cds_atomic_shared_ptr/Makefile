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

GENMC ?= genmc/build/bin/genmc

# --unroll=5: bound for CAS retry loops (increase if GenMC warns)
GMCFLAGS ?= --rc11 --unroll=5

SRCDIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

run: run-test1 run-test2 run-test3 run-test4 run-test5 run-test6 run-test7 run-test8 run-test9 run-test10 run-test11

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

run-test4:
	@echo "=== Test 4: swap + local_reset safety ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_test_swap.c
	@echo ""

run-test5:
	@echo "=== Test 5: multi CAS with release_tag_ref leave_excess ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_test_multi_cas_excess.c
	@echo ""

run-test6:
	@echo "=== Test 6: CAS+load with release_tag_ref leave_excess ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_test_cas_excess.c
	@echo ""

run-test7:
	@echo "=== Test 7: swap with release_tag_ref cas_rcnt ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_test_swap_excess.c
	@echo ""

run-test8:
	@echo "=== Test 8: compareAndSet_ (NOSWAP=true) no-acquire optimization ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_test_cas_noacquire.c
	@echo ""

run-test9:
	@echo "=== Test 9: scoped_atomic_view + compareAndSetWeak race ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_test_scoped_weak.c
	@echo ""

run-test10:
	@echo "=== Test 10: biased refcount private→shared DIRECT publish (KAME_LSP_BIASED) ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_test_biased_publish.c
	@echo ""

# Scenario 4 (three-way composition) is minutes-long, so `make run` covers
# scenarios 1-3 and the bug knob; run `make run-test11-full` deliberately.
run-test11:
	@echo "=== Test 11: release_tagheld_zeroreset_ (bulk TagHeld release) ==="
	$(GENMC) $(GMCFLAGS) -- -DSCEN=1 $(SRCDIR)cds_test_zeroreset.c
	$(GENMC) $(GMCFLAGS) -- -DSCEN=2 $(SRCDIR)cds_test_zeroreset.c
	$(GENMC) $(GMCFLAGS) -- -DSCEN=3 $(SRCDIR)cds_test_zeroreset.c
	@echo "--- bug knob: transfer-after-CAS swap MUST violate ---"
	-$(GENMC) $(GMCFLAGS) -- -DSCEN=3 -DSWAP_TRANSFER_AFTER_CAS $(SRCDIR)cds_test_zeroreset.c
	@echo ""

run-test11-full: run-test11
	@echo "=== Test 11 SCEN 4: zeroreset + load + swap composition ==="
	$(GENMC) $(GMCFLAGS) --disable-estimation -- -DSCEN=4 $(SRCDIR)cds_test_zeroreset.c
	@echo ""

# Run with IMM memory model
run-imm: GMCFLAGS = --imm --unroll=5
run-imm: run

.PHONY: run run-test1 run-test2 run-test3 run-test4 run-test5 run-test6 run-test7 run-test8 run-test9 run-test10 run-test11 run-test11-full run-imm
