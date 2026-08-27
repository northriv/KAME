#!/bin/bash
# §13.208  The two-line gate §13.206 asks for, mechanised.
#
# §13.206 bisected a vanished signature to instrumentation added inside the flush
# region, and set the rule: on the failing target that region is codegen-sensitive
# enough that measuring inside it changes what it does, and an arm64 validation
# cannot detect the class.  So every build that is about to be READ must first
# prove it still has the phenomenon.
#
# Usage:  verify_shape_gate.sh <libkamepoolalloc.so> <reproducer> [cap]
# Exit 0 only if BOTH hold.  Anything else means the build is not a measurement
# platform, whatever its counters say.
set -u
SO=${1:?so}; REPRO=${2:?reproducer}; CAP=${3:-1}
want_syms=${KAME_WANT_FLUSH_SYMS:-4}
got=$(nm -C "$SO" 2>/dev/null | grep -c 'flush_impl<false>')
[ "$got" = 0 ] && got=$(nm -C "$SO" 2>/dev/null | grep -c 'CrossDeallocBatch::flush')
printf 'flush symbols: %s (want %s)\n' "$got" "$want_syms"
vio=$(KAME_BATCH_CAP=$CAP "$REPRO" 2>&1 | grep -oE 'bit_clear_bad=[0-9]+' | tail -1 | cut -d= -f2)
printf 'bit_clear_bad: %s (want > 0 on a platform that exhibits the fault)\n' "${vio:-<none>}"
[ "$got" = "$want_syms" ] || { echo 'FAIL: flush region changed shape'; exit 1; }
[ -n "${vio:-}" ] && [ "$vio" -gt 0 ] || { echo 'FAIL: no violations — nothing to measure'; exit 2; }
echo 'OK: shape intact and phenomenon present'
