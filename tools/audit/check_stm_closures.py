#!/usr/bin/env python3
"""Detect non-rollbackable side effects inside iterate_commit closures,
and GIL-held STM entry in pybind11 bindings.

Check 1 -- iterate_commit closures must be idempotent (CLAUDE.md
"Driver-authoring rules" #1): the closure re-runs on every CAS retry,
so a free/print/sleep/interface-I/O inside executes once per retry.
A failed commit rolls back the transaction state but NOT the side
effect (fixed exemplars: 90b92913d double-free, 4dcf84649 latch).
Flagged tokens inside any iterate_commit/_if/_while lambda:
    fftw_free / fftw_destroy_plan / gErrPrint / gWarnPrint /
    msecsleep / interface()->

Check 2 -- pybind11 .def()/py::init() lambdas that enter STM
negotiation (Snapshot/Transaction construction on a node, commit,
iterate_commit, trans(), ***node reads) must release the GIL
(CLAUDE.md rule #4; fixed exemplar: d5aefdc46). Applied only under
modules/python/ and kame/script/. A binding is compliant if the .def
call or lambda body mentions gil_scoped_release.

Suppress a reviewed-and-legitimate hit with `// audit-ok: <reason>`
on the offending line (check 1) or anywhere in the binding (check 2).

Usage: check_stm_closures.py <dir-or-file> [...]
Exit status: 1 if any finding.
"""
import re
import sys
import pathlib

SUPPRESS = 'audit-ok'

# interface()-> is only a hazard for I/O verbs (they take the interface
# mutex); node accessors (device(), port(), softwareTriggerManager()...)
# are lock-free and legitimate inside a transaction.
SIDE_EFFECT_RE = re.compile(
    r'fftw_free|fftw_destroy_plan|gErrPrint|gWarnPrint|msecsleep\s*\('
    r'|interface\s*\(\s*\)\s*->\s*'
    r'(?:send|receive|query|write|readRegister|burstRead)\w*\s*\(')
ITERATE_RE = re.compile(r'\biterate_commit(?:_if|_while)?\s*\(')

STM_ENTRY_RE = re.compile(
    r'\bSnapshot\s+\w+\s*\(\s*\*|\bSnapshot\s*\(\s*\*'
    r'|\bTransaction\s+\w+\s*\(\s*\*|\bTransaction\s*\(\s*\*'
    r'|\.commit(?:OrNext)?\s*\(|\biterate_commit|\btrans\s*\(\s*\*|\*\*\*')
# NOT py::init: an init inside .def(py::init(...), py::call_guard<...>())
# would be scanned without its sibling call_guard argument and misreported;
# the .def( span already covers the whole argument list.
DEF_RE = re.compile(r'\.def(?:_static|_property\w*)?\s*\(')
GIL_OK = 'gil_scoped_release'
PYBIND_DIRS = ('modules/python', 'kame/script')


def balanced_span(text, open_pos):
    """Return end offset of the parenthesized region starting at open_pos."""
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        c = text[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return i
        elif c == '"':  # skip string literals
            i += 1
            while i < n and text[i] != '"':
                if text[i] == '\\':
                    i += 1
                i += 1
        i += 1
    return n - 1


def lineno_of(text, pos):
    return text.count('\n', 0, pos) + 1


def kind_of(token):
    """Normalize a matched token to a stable baseline key."""
    t = token.strip()
    for k in ('fftw_free', 'fftw_destroy_plan', 'gErrPrint', 'gWarnPrint',
              'msecsleep'):
        if t.startswith(k):
            return k
    if t.startswith('interface'):
        return 'interface-io'
    return t


def check_closures(path, text):
    findings = []  # (message, (relpath, kind))
    lines = text.splitlines()
    for m in ITERATE_RE.finditer(text):
        open_pos = text.index('(', m.end() - 1)
        end = balanced_span(text, open_pos)
        body = text[open_pos:end]
        for sm in SIDE_EFFECT_RE.finditer(body):
            ln = lineno_of(text, open_pos + sm.start())
            if SUPPRESS in lines[ln - 1]:
                continue
            findings.append((
                f'{path}:{ln}: "{sm.group(0).strip()}" inside an '
                f'iterate_commit closure (re-runs per CAS retry; '
                f'hoist it out or mark // audit-ok: <reason>)',
                (str(path), kind_of(sm.group(0)))))
    return findings


def check_pybind_gil(path, text):
    if not any(d in str(path).replace('\\', '/') for d in PYBIND_DIRS):
        return []
    findings = []
    for m in DEF_RE.finditer(text):
        open_pos = text.index('(', m.end() - 1)
        end = balanced_span(text, open_pos)
        body = text[open_pos:end]
        if GIL_OK in body or SUPPRESS in body:
            continue
        sm = STM_ENTRY_RE.search(body)
        if sm:
            ln = lineno_of(text, open_pos + sm.start())
            findings.append((
                f'{path}:{ln}: pybind binding enters STM negotiation '
                f'("{sm.group(0).strip()}") without gil_scoped_release '
                f'(GIL-vs-STM deadlock; add py::call_guard'
                f'<py::gil_scoped_release>() or mark // audit-ok: <reason>)',
                (str(path), 'gil')))
    return findings


BASELINE = pathlib.Path(__file__).with_name('stm_closures.baseline')


def load_baseline():
    counts = {}
    if BASELINE.exists():
        for line in BASELINE.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            cnt, path, kind = line.split('\t')
            counts[(path, kind)] = int(cnt)
    return counts


def save_baseline(counts):
    lines = ['# Grandfathered findings (ratchet): a (file, kind) pair may',
             '# not exceed its count here. Regenerate after fixing with:',
             '#   python3 tools/audit/check_stm_closures.py --update-baseline kame modules']
    for (path, kind), cnt in sorted(counts.items()):
        lines.append(f'{cnt}\t{path}\t{kind}')
    BASELINE.write_text('\n'.join(lines) + '\n')


def main(argv):
    args = argv[1:]
    update = '--update-baseline' in args
    if update:
        args.remove('--update-baseline')
    roots = [pathlib.Path(a) for a in args] or [pathlib.Path('.')]
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
        try:
            text = f.read_text(errors='replace')
        except OSError:
            continue
        findings.extend(check_closures(f, text))
        findings.extend(check_pybind_gil(f, text))

    counts = {}
    for _msg, key in findings:
        counts[key] = counts.get(key, 0) + 1

    if update:
        save_baseline(counts)
        print(f'baseline updated: {sum(counts.values())} finding(s) '
              f'across {len(counts)} (file, kind) pairs.')
        return 0

    baseline = load_baseline()
    regressions = {k: (c, baseline.get(k, 0))
                   for k, c in counts.items() if c > baseline.get(k, 0)}
    if regressions:
        for msg, key in findings:
            if key in regressions:
                print(msg)
        for (path, kind), (cur, base) in sorted(regressions.items()):
            print(f'REGRESSION: {path} [{kind}]: {cur} finding(s), '
                  f'baseline allows {base}.', file=sys.stderr)
        return 1
    improved = sum(base - counts.get(k, 0)
                   for k, base in baseline.items() if counts.get(k, 0) < base)
    if improved:
        print(f'check_stm_closures: OK ({improved} finding(s) below '
              f'baseline — consider --update-baseline to ratchet down).')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
