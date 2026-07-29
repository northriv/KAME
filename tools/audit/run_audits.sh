#!/bin/sh
# Static crash-bug audits (see CLAUDE.md "Driver-authoring rules") plus the
# kamepoolalloc no-DCAS invariant.  Run from anywhere; exit 1 on any
# regression.  This is the full-audit entry point — CI calls this.
#
# The Python scans are static and instant (they cover kame/ and modules/).
# check_no_dcas.sh compiles kamepoolalloc/allocator.cpp twice (~20 s), so the
# pre-commit hook sets KAME_AUDIT_SKIP_NO_DCAS=1 unless kamepoolalloc/ is
# actually staged; set it yourself to skip that phase by hand.
cd "$(dirname "$0")/../.." || exit 1
status=0
python3 tools/audit/check_node_names.py kame modules || status=1
python3 tools/audit/check_stm_closures.py kame modules || status=1
python3 tools/audit/check_payload_const.py kame modules || status=1
python3 tools/audit/check_ui_listeners.py kame modules || status=1
[ -n "$KAME_AUDIT_SKIP_NO_DCAS" ] || tools/audit/check_no_dcas.sh || status=1
exit $status
