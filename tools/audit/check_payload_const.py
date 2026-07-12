#!/usr/bin/env python3
"""Detect non-const smart-pointer data members in Payload structs.

Payload clones are SHALLOW: shared_ptr members alias the same heap
object across the transaction clone and every live Snapshot. Heap data
in a Payload must therefore be pointer-to-const (publish a fresh
pointer, never mutate in place) — see CLAUDE.md, STM section.
Confirmed corruption exemplars: 83bb9ffaf (XNMRT1 pulse accumulation,
XODMR2DAnalysis rescale); latent conversions: e5ac4973a.

Allowlisted automatically:
  - pointees deriving (transitively) from XNode: such nodes manage
    their own isolation through their own Payloads (navigation handles)
  - Listener / Talker framework objects
Suppress a reviewed member with `// audit-ok: <reason>` on its line
(e.g. deep-cloning Payload copy ctor, manual copy-on-write).

Usage: check_payload_const.py <dir-or-file> [...]
Exit status: 1 on any unallowlisted non-const member.
"""
import re
import sys
import pathlib

PAYLOAD_RE = re.compile(
    r'\b(?:struct|class)\s+(?:DECLSPEC_\w+\s+)?Payload\b')
PTR_RE = re.compile(
    r'\b(?:std::)?(?:local_shared_ptr|shared_ptr|atomic_shared_ptr)\s*<\s*'
    r'(?!const\b)([A-Za-z_][\w:]*)')
CLASS_RE = re.compile(
    r'\b(?:class|struct)\s+(?:DECLSPEC_\w+\s+)?([A-Za-z_]\w*)\s*:\s*'
    r'((?:\s|public|protected|private|virtual|,|[\w:<>])+?)\s*\{')
ALLOW_NAMES = {'Listener', 'Talker', 'XListener'}
SUPPRESS = 'audit-ok'


def matching_brace(text, open_idx):
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        elif c == '/' and i + 1 < n:
            if text[i + 1] == '/':
                j = text.find('\n', i)
                i = j if j != -1 else n
            elif text[i + 1] == '*':
                j = text.find('*/', i + 2)
                i = j + 1 if j != -1 else n
        elif c == '"':
            i += 1
            while i < n and text[i] != '"':
                if text[i] == '\\':
                    i += 1
                i += 1
        i += 1
    return -1


def collect_files(roots):
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            for pat in ('*.h', '*.hpp', '*.cpp'):
                files.extend(p for p in root.rglob(pat)
                             if 'tests' not in p.parts)
    return sorted(set(files))


def xnode_closure(files):
    """Names of classes transitively deriving from XNode."""
    bases = {}  # name -> set(base names)
    for f in files:
        try:
            text = f.read_text(errors='replace')
        except OSError:
            continue
        for m in CLASS_RE.finditer(text):
            name = m.group(1)
            base_ids = set(re.findall(r'[A-Za-z_]\w*', m.group(2)))
            base_ids -= {'public', 'protected', 'private', 'virtual', 'std'}
            bases.setdefault(name, set()).update(base_ids)
    derived = {'XNode'}
    changed = True
    while changed:
        changed = False
        for name, bs in bases.items():
            if name not in derived and bs & derived:
                derived.add(name)
                changed = True
    return derived


def main(argv):
    roots = [pathlib.Path(a) for a in argv[1:]] or [pathlib.Path('.')]
    files = collect_files(roots)
    nodeclasses = xnode_closure(files)
    findings = []
    payloads = 0
    for f in files:
        try:
            text = f.read_text(errors='replace')
        except OSError:
            continue
        for m in PAYLOAD_RE.finditer(text):
            brace = text.find('{', m.end())
            semi = text.find(';', m.end())
            if brace == -1 or (semi != -1 and semi < brace):
                continue  # forward declaration
            close = matching_brace(text, brace)
            if close == -1:
                continue
            payloads += 1
            body = text[brace + 1:close]
            base_line = text.count('\n', 0, brace) + 1
            for ln_i, line in enumerate(body.split('\n')):
                if SUPPRESS in line:
                    continue
                # type aliases and conversion operators are not data members
                if re.search(r'\busing\b|\bTalker\s*<|\boperator\b', line):
                    continue
                for pm in PTR_RE.finditer(line):
                    pointee = pm.group(1).split('::')[-1]
                    if pointee in nodeclasses or pointee in ALLOW_NAMES:
                        continue
                    # skip function declarations returning pointers
                    # (member lines end with ; and contain no '(') —
                    # accessors like `shared_ptr<T> foo() const {...}`
                    if re.search(r'\w\s*\(', line.split('//')[0]):
                        continue
                    findings.append(
                        f'{f}:{base_line + 1 + ln_i}: non-const pointee '
                        f'"{pointee}" in Payload — snapshot-shared heap data '
                        f'must be pointer-to-const (or // audit-ok: <reason>)')
    for line in findings:
        print(line)
    if findings:
        print(f'\ncheck_payload_const: {len(findings)} finding(s) across '
              f'{payloads} Payload structs.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
