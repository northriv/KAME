#!/usr/bin/env python3
"""Detect sibling node-name collisions in create<>() calls.

Node name strings are the lookup keys for Python access (node["Name"]),
.kam serialization, and the Node Browser. Two siblings created with the
same name leave the later one silently unreachable, and when both are
runtime=false the .kam round-trip corrupts the first one (see CLAUDE.md
"Driver-authoring rules" #3; fixed exemplar: commit ebe414df6).

Heuristic: create<...>("Name") literals are grouped by the enclosing
column-0 definition (constructor/function) or column-0 class/struct
declaration -- sibling children are created within one constructor in
this codebase. A duplicate name within one group is an error.

Suppress a deliberate duplicate with `// audit-ok: <reason>` on the line
of the create<>() call.

Usage: check_node_names.py <dir-or-file> [...]
Exit status: 1 if any collision is found.
"""
import re
import sys
import pathlib

# (?<!\w) keeps xqcon_create out. Template args never contain ; { }.
# The name literal may sit on a continuation line (DOTALL), optionally
# after a leading `tr,` / `ref(tr),` argument.
CREATE_RE = re.compile(
    r'(?<!\w)create(?:Orphan)?\s*<[^;{}]{1,300}?>\s*\(\s*'
    r'(?:ref\(tr\w*\)\s*,\s*|tr\w*\s*,\s*)?"([^"\n]*)"',
    re.DOTALL)
ANCHOR_RE = re.compile(
    r'^(?:[A-Za-z_][\w:<>~,\s]*::[\w~]+\s*\(|(?:class|struct)\s+\w+)',
    re.MULTILINE)
SUPPRESS = 'audit-ok'


def scan(path: pathlib.Path):
    try:
        text = path.read_text(errors='replace')
    except OSError:
        return []
    anchors = [(m.start(), m.group(0).split('(')[0].strip())
               for m in ANCHOR_RE.finditer(text)]
    findings = []
    seen = {}  # (anchor_idx, name) -> lineno
    for m in CREATE_RE.finditer(text):
        name = m.group(1)
        if not name:
            continue  # anonymous list entries are index-accessed
        lineno = text.count('\n', 0, m.start()) + 1
        line = text.splitlines()[lineno - 1]
        if SUPPRESS in line:
            continue
        anchor_idx = -1
        for i, (pos, _label) in enumerate(anchors):
            if pos <= m.start():
                anchor_idx = i
            else:
                break
        key = (anchor_idx, name)
        if key in seen:
            scope = anchors[anchor_idx][1] if anchor_idx >= 0 else '<file top>'
            findings.append(
                f'{path}:{lineno}: duplicate node name "{name}" in {scope} '
                f'(first created at line {seen[key]})')
        else:
            seen[key] = lineno
    return findings


def main(argv):
    roots = [pathlib.Path(a) for a in argv[1:]] or [pathlib.Path('.')]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            for pat in ('*.cpp', '*.h'):
                files.extend(p for p in root.rglob(pat)
                             if 'tests' not in p.parts)
    findings = []
    for f in sorted(set(files)):
        findings.extend(scan(f))
    for line in findings:
        print(line)
    if findings:
        print(f'\ncheck_node_names: {len(findings)} collision(s) found.',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
