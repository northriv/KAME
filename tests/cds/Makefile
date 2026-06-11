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

run: run-ownership run-radix run-dll run-batch run-seqlock run-handoff

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

run-batch:
	@echo "=== CrossDeallocBatch held-bit: chunk alive while any slot held; 1-releaser (INV-6) ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_crossbatch_heldbit.c
	@echo ""

run-seqlock:
	@echo "=== seqlock lookup re-read: reader acquire fence keeps the meta1/data/meta2 snapshot consistent (candidate epoch+seqlock recycle root-cure; INV_NoTornRead) ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_seqlock_recycle.c
	@echo ""

# NEGATIVE control: remove the reader's acquire fence (-DNO_READER_FENCE) — the
# exact weak-memory mistake the re-read guards against.  This target SUCCEEDS
# only when GenMC reports a torn-read Safety violation, proving the positive
# control above is non-vacuous and actually exercises the fence.  Not part of
# `run` (it is supposed to fail-find).
run-seqlock-nofence:
	@echo "=== seqlock NEGATIVE control: reader acquire fence REMOVED — MUST report a torn-read Safety violation ==="
	@if $(GENMC) $(GMCFLAGS) -- -DNO_READER_FENCE $(SRCDIR)cds_seqlock_recycle.c >/tmp/cds_seqlock_neg.log 2>&1; then \
	    echo "FAIL: expected a torn-read Safety violation, but GenMC reported none"; tail -5 /tmp/cds_seqlock_neg.log; exit 1; \
	  elif grep -q "Safety violation" /tmp/cds_seqlock_neg.log; then \
	    echo "OK: torn read detected — negative control behaves as expected (fence is load-bearing)"; \
	  else \
	    echo "FAIL: GenMC failed for a reason OTHER than the expected Safety violation"; tail -5 /tmp/cds_seqlock_neg.log; exit 1; \
	  fi
	@echo ""

run-handoff:
	@echo "=== alloc→handoff→cross-free: plain back_offset/palloc published via the app pointer hand-off (resolver's one weak-memory premise) ==="
	$(GENMC) $(GMCFLAGS) $(SRCDIR)cds_handoff_publish.c
	@echo ""

# NEGATIVE control: relaxed hand-off (-DNO_HANDOFF_SYNC) — violates the
# premise that a freed pointer was received via a release/acquire transfer.
# Succeeds only when GenMC reports a non-atomic race on the plain resolve
# metadata, proving the premise is load-bearing.  Not part of `run`.
run-handoff-norel:
	@echo "=== handoff NEGATIVE control: relaxed transfer — MUST report a Non-atomic race on the plain metadata ==="
	@if $(GENMC) $(GMCFLAGS) -- -DNO_HANDOFF_SYNC $(SRCDIR)cds_handoff_publish.c >/tmp/cds_handoff_neg.log 2>&1; then \
	    echo "FAIL: expected a Non-atomic race, but GenMC reported none"; tail -5 /tmp/cds_handoff_neg.log; exit 1; \
	  elif grep -qiE "Non-atomic race|Safety violation" /tmp/cds_handoff_neg.log; then \
	    echo "OK: race detected — the resolver's relaxed/plain reads are sound ONLY because the hand-off synchronises"; \
	  else \
	    echo "FAIL: GenMC failed for a reason OTHER than the expected race"; tail -5 /tmp/cds_handoff_neg.log; exit 1; \
	  fi
	@echo ""

# IMM is strictly stronger than RC11 for this code; offered for parity with
# the kamestm harness.
run-imm: GMCFLAGS = --imm --unroll=5
run-imm: run

# Concrete build + run: confirms the harness compiles and the happy-path
# logic holds.  Does NOT explore interleavings — use `make run` (GenMC) for
# the actual verification.
smoke: smoke-ownership smoke-radix smoke-dll smoke-batch smoke-seqlock smoke-handoff
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
smoke-batch:
	@cc -std=c11 -O2 -pthread -fsanitize=thread $(SRCDIR)cds_crossbatch_heldbit.c -o /tmp/cds_crossbatch_heldbit_smoke \
	  && for i in 1 2 3 4 5 6 7 8; do /tmp/cds_crossbatch_heldbit_smoke || exit 1; done \
	  && echo "smoke OK: cds_crossbatch_heldbit (8 concrete runs, TSAN-clean)"
smoke-seqlock:
	@cc -std=c11 -O2 -pthread -fsanitize=thread $(SRCDIR)cds_seqlock_recycle.c -o /tmp/cds_seqlock_recycle_smoke \
	  && for i in 1 2 3 4 5 6 7 8; do /tmp/cds_seqlock_recycle_smoke || exit 1; done \
	  && echo "smoke OK: cds_seqlock_recycle (8 concrete runs, TSAN-clean)"
smoke-handoff:
	@cc -std=c11 -O2 -pthread -fsanitize=thread $(SRCDIR)cds_handoff_publish.c -o /tmp/cds_handoff_publish_smoke \
	  && for i in 1 2 3 4 5 6 7 8; do /tmp/cds_handoff_publish_smoke || exit 1; done \
	  && echo "smoke OK: cds_handoff_publish (8 concrete runs, TSAN-clean)"

.PHONY: run run-ownership run-radix run-dll run-batch run-seqlock run-seqlock-nofence run-handoff run-handoff-norel run-imm smoke smoke-ownership smoke-radix smoke-dll smoke-batch smoke-seqlock smoke-handoff
