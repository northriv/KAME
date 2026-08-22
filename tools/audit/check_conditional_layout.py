#!/usr/bin/env python3
"""Detect class layout that depends on a project-defined macro.

A data member or virtual function behind `#ifdef SOMETHING` gives the
class two different layouts.  That is harmless only while every
translation unit agrees on SOMETHING — and project macros come from
.pro/.pri files, which do NOT have to agree: kame.pri handed USE_RUBY to
libkame and all 44 module targets, while only kame/kame.pro looked for
ruby.h and could answer "no".

Exemplar 8bb86a9b6: `shared_ptr<XRuby> m_ruby` behind #ifdef USE_RUBY in
measure.h split sizeof(XMeasure) by 16 bytes.  Every later member shifted,
the modules' m_interfaces landed on the app's m_drivers, and the first
driver added wrote an XInterface into the driver list — then a
static_pointer_cast<XDriver> walked it.  Adding any driver crashed.
A virtual behind a conditional does the same to the vtable.

Only macros the BUILD SYSTEM hands out can differ between targets, so
the risky set is derived from the .pro/.pri/CMakeLists files rather than
guessed: a macro #defined in a header is by construction identical in
every translation unit that includes it (USE_QTHREAD in support.h,
KAME_NEGSITE_ENABLED in transaction_definitions.h, SERIAL_POSIX in
serial.h are all fine for exactly that reason).  Suppress a reviewed
case with `// audit-ok: <reason>` on the #if line or the member line.

Usage: check_conditional_layout.py <dir-or-file> [...]
Exit status: 1 on any unexempt conditional member or virtual.
"""
import re
import sys
import pathlib

# Macros a .pro/.pri/CMakeLists can hand to one target and not another.
BUILD_DEF = re.compile(
    r'DEFINES\s*[-+]?=\s*(.+)|'
    r'add_definitions\s*\(([^)]*)\)|'
    r'target_compile_definitions\s*\(([^)]*)\)|'
    r'-D\s*([A-Za-z_]\w*)')
NAME = re.compile(r'[A-Za-z_]\w*')
# Vendored third-party trees are not ours to police.
SKIP_DIRS = ('genmc', 'contrib', 'extern')


def build_macros(root):
    """Every macro name a build file defines, i.e. the ones that can differ
    between targets.  A macro #defined in a header cannot."""
    out = set()
    for pat in ('*.pro', '*.pri', 'CMakeLists.txt', '*.cmake'):
        for f in pathlib.Path(root).rglob(pat):
            if any(d in f.parts for d in SKIP_DIRS):
                continue
            for line in f.read_text(errors='replace').splitlines():
                line = line.split('#')[0]
                m = BUILD_DEF.search(line)
                if not m:
                    continue
                blob = next((g for g in m.groups() if g), '')
                for n in NAME.findall(blob):
                    if n.isupper() or '_' in n:
                        out.add(n)
    return out

COND = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$')
MACRO = re.compile(r'\b([A-Za-z_]\w*)\b')
# KAME names every data member m_*; that convention is what makes this
# tractable without parsing C++.
MEMBER = re.compile(r'^\s*(?!static\b)(?!typedef\b)[\w:<>,\s\*&]+?\bm_\w+\s*'
                    r'(?:\[[^\]]*\])?\s*(?:=[^;]*)?;')
VIRTUAL = re.compile(r'^\s*virtual\b')
SUPPRESS = 'audit-ok'
CLASS_OPEN = re.compile(r'^\s*(?:template\s*<.*>\s*)?(?:class|struct)\b')


def macros_of(expr):
    """Macro names a #if expression depends on."""
    return [m for m in MACRO.findall(expr)
            if m not in ('defined', 'if', 'ifdef', 'ifndef', 'elif')]


def scan(path, risky_macros):
    findings = []
    lines = path.read_text(errors='replace').splitlines()
    # stack of (macro-list, line-no, suppressed, is_include_guard)
    stack = []
    guard_seen = False
    for i, line in enumerate(lines):
        m = COND.match(line)
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind in ('if', 'ifdef', 'ifndef'):
                names = macros_of(rest)
                # The file's own include guard wraps everything; it is not a
                # configuration choice.
                guard = (kind == 'ifndef' and not guard_seen and
                         i + 1 < len(lines) and
                         lines[i + 1].strip().startswith('#define'))
                if guard:
                    guard_seen = True
                dead = rest.strip() == '0'
                stack.append((names, i + 1, SUPPRESS in line, guard or dead))
            elif kind == 'endif':
                if stack:
                    stack.pop()
            continue
        if not stack:
            continue
        live = [f for f in stack if not f[3] and not f[2]]
        if not live:
            continue
        code = line.split('//')[0]
        if not (VIRTUAL.match(code) or MEMBER.match(code)):
            continue
        if SUPPRESS in line:
            continue
        klass = 0
        for j in range(i, 0, -1):
            if CLASS_OPEN.match(lines[j - 1]):
                klass = j
                break
        # A conditional that wraps the whole class is not a layout split: the
        # class either exists or it does not, and a target without the macro
        # fails to compile rather than silently disagreeing about offsets.
        # Only a frame opened INSIDE the class body can split the layout, so
        # test each enclosing frame separately -- pythondriver.h guards its
        # entire contents with USE_PYBIND11 and is fine.
        risky = sorted({n for f in live if f[1] > klass
                        for n in f[0] if n in risky_macros})
        if not risky:
            continue
        opened = max(f[1] for f in live if f[1] > klass
                     and any(n in risky_macros for n in f[0]))
        what = 'virtual function' if VIRTUAL.match(code) else 'data member'
        findings.append(
            f'{path}:{i + 1}: {what} behind #if {"/".join(risky)} '
            f'(opened at line {opened}) — that macro comes from a build file, '
            f'so a target that never sees it gets a different layout for '
            f'this class')
    return findings


def main(argv):
    findings = []
    risky_macros = build_macros(pathlib.Path(__file__).resolve().parents[2])
    for arg in argv[1:]:
        p = pathlib.Path(arg)
        files = ([p] if p.is_file() else
                 sorted(f for f in p.rglob('*.h')))
        for f in files:
            if f.suffix != '.h' or any(d in f.parts for d in SKIP_DIRS):
                continue
            findings += scan(f, risky_macros)
    for line in findings:
        print(line)
    if findings:
        print(f'\ncheck_conditional_layout: {len(findings)} finding(s).',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
