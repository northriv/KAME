#!/bin/sh
# Static crash-bug audits (see CLAUDE.md "Driver-authoring rules").
# Run from anywhere; scans kame/ and modules/. Exit 1 on any regression.
cd "$(dirname "$0")/../.." || exit 1
status=0
python3 tools/audit/check_node_names.py kame modules || status=1
python3 tools/audit/check_stm_closures.py kame modules || status=1
exit $status
