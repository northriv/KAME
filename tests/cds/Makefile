# Makefile for GenMC model-checking of the §25/§26 large-recycle cache.
#
# Self-contained under kamepoolalloc/ (no dependency on kamestm's GenMC
# harness — kamepoolalloc is published as its own standalone subtree).
#
# Usage:
#   make run                # model-check the cache ownership/release protocol
#   make smoke              # concrete gcc build + run (sanity; no model check)
#
# Build GenMC (one-time), per kamestm/tests/cds_atomic_shared_ptr/Makefile:
#   brew install llvm@20 hwloc        # (or distro LLVM 15-20)
#   cd genmc && mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=<llvm@20> -DCMAKE_BUILD_TYPE=Release && make -j
#
# GenMC compiles the source directly through its LLVM frontend; pass an
# absolute path.  --rc11 selects the RC11 memory model; --unroll bounds
# loops (the cache's per-slot scans are bounded by NSLOT, so 5 is ample).

GENMC    ?= genmc/build/bin/genmc
GMCFLAGS ?= --rc11 --unroll=5
SRCDIR    = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

run: run-ownership run-radix run-dll

run-ownership:
	@echo "=== LRC: exclusive-ownership / no-premature-release (UAF/double-free) ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_lrc_ownership.c
	@echo ""

run-radix:
	@echo "=== radix L2 lazy-install: single-winner CAS, no UAF on loser leaf (INV-9) ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_radix_install.c
	@echo ""

run-dll:
	@echo "=== thread-exit DLL walk vs cross-thread free: 1-releaser + cached-next (INV-6, INV-21) ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_dll_exit_race.c
	@echo ""

# IMM is strictly stronger than RC11 for this code; offered for parity with
# the kamestm harness.
run-imm: GMCFLAGS = --imm --unroll=5
run-imm: run

# Concrete build + run: confirms the harness compiles and the happy-path
# logic holds.  Does NOT explore interleavings — use `make run` (GenMC) for
# the actual verification.
smoke: smoke-ownership smoke-radix smoke-dll
smoke-ownership:
	@cc -std=c11 -O2 -pthread -fsanitize=thread $(SRCDIR)cds_lrc_ownership.c -o /tmp/cds_lrc_ownership_smoke \
	  && for i in 1 2 3 4 5 6 7 8; do /tmp/cds_lrc_ownership_smoke || exit 1; done \
	  && echo "smoke OK: cds_lrc_ownership (8 concrete runs, TSAN-clean)"
smoke-radix:
	@cc -std=c11 -O2 -pthread -fsanitize=thread $(SRCDIR)cds_radix_install.c -o /tmp/cds_radix_install_smoke \
	  && for i in 1 2 3 4 5 6 7 8; do /tmp/cds_radix_install_smoke || exit 1; done \
	  && echo "smoke OK: cds_radix_install (8 concrete runs, TSAN-clean)"
smoke-dll:
	@cc -std=c11 -O2 -pthread -fsanitize=thread $(SRCDIR)cds_dll_exit_race.c -o /tmp/cds_dll_exit_race_smoke \
	  && for i in 1 2 3 4 5 6 7 8; do /tmp/cds_dll_exit_race_smoke || exit 1; done \
	  && echo "smoke OK: cds_dll_exit_race (8 concrete runs, TSAN-clean)"

.PHONY: run run-ownership run-radix run-dll run-imm smoke smoke-ownership smoke-radix smoke-dll
