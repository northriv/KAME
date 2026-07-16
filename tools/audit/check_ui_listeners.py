#!/usr/bin/env python3
"""Detect listener callbacks that do Qt UI work without FLAG_MAIN_THREAD_CALL.

Talker/Listener callbacks run INLINE on the committing thread unless the
connection carries Listener::FLAG_MAIN_THREAD_CALL — and the committing
thread can be a driver thread or the Python/MCP scripting thread (any
scripted set()/touch() fires listeners on the Python thread). A callback
touching Qt widgets without the flag is a cross-thread Qt call (UB/crash).
See CLAUDE.md "Driver-authoring rules" #6; fixed exemplar: b6d5f7e6b
(tempcontrol SetupChannel connector rebuild, XMicroCAM m_txtCode access).

Heuristic: for every connectWeakly()/connect(&Class::method) call lacking
FLAG_MAIN_THREAD_CALL, locate Class::method's body (same file, plus
same-class helpers it calls, one level) and flag Qt-UI tokens: m_form->,
xqcon_create, QMessageBox/QFileDialog, setPalette, or widget-prefixed
member calls (m_btn*/m_txt*/m_cmb*/m_ed*/m_lcd*/m_ckb*/m_dsb*/m_tbl*->...).

Suppress a reviewed site with `// audit-ok: <reason>` inside the connect
call or on the connect line (e.g. a talker only ever fired from the main
thread, like XQGraph mouse-event tool selection).

Usage: check_ui_listeners.py <dir-or-file> [...]
Exit status: 1 on any finding.
"""
import re
import sys
import pathlib

CONNECT_RE = re.compile(r'\.connect(?:Weakly)?\s*\(')
# class part may be nested (XFoo::Loop::onBar) — capture greedily.
METHOD_RE = re.compile(r'&\s*([\w:]+)::(\w+)')
FLAG = 'FLAG_MAIN_THREAD_CALL'
SUPPRESS = 'audit-ok'
UI_RE = re.compile(
    r'm_form\s*->|m_pForm\s*->|xqcon_create|QMessageBox|QFileDialog'
    r'|->\s*setPalette\s*\('
    r'|\bm_(?:btn|txt|cmb|ed|lcd|ckb|dsb|tbl|spb|dbl)\w*\s*->')


def balanced_span(text, open_pos):
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
        elif c == '"':
            i += 1
            while i < n and text[i] != '"':
                if text[i] == '\\':
                    i += 1
                i += 1
        i += 1
    return n - 1


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


def method_body(text, cls, method):
    """Body of the qualified Class::method definition in this file.
    Deliberately direct-body only (no helper following, no unqualified
    in-class definitions): keeps the false-positive rate workable; the
    audited real findings all had UI tokens directly in the callback."""
    base = re.escape(cls.split('::')[-1])
    m = re.search(rf'\b{base}(?:<[\w,\s]*>)?::{re.escape(method)}\s*\(', text)
    if m:
        brace = text.find('{', m.start())
        if brace != -1:
            close = matching_brace(text, brace)
            if close != -1:
                return text[brace:close]
    return None


def scan_file(path, text):
    findings = []
    for m in CONNECT_RE.finditer(text):
        open_pos = text.index('(', m.end() - 1)
        end = balanced_span(text, open_pos)
        span = text[m.start():end]
        if FLAG in span or SUPPRESS in span:
            continue
        # also allow suppression on the line the connect starts on
        line_start = text.rfind('\n', 0, m.start()) + 1
        line_end = text.find('\n', m.start())
        if SUPPRESS in text[line_start:line_end if line_end != -1 else None]:
            continue
        mm = METHOD_RE.search(span)
        if not mm:
            continue  # listener-object reuse; flags ride the original
        cls, method = mm.group(1), mm.group(2)
        body = method_body(text, cls, method)
        if body is None:
            continue  # virtual hook or defined elsewhere; out of heuristic scope
        um = UI_RE.search(body)
        if um:
            ln = text.count('\n', 0, m.start()) + 1
            findings.append(
                f'{path}:{ln}: {cls}::{method} connected without '
                f'{FLAG} but touches Qt UI ("{um.group(0).strip()}") — '
                f'add Listener::FLAG_MAIN_THREAD_CALL or // audit-ok: <reason>')
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
        try:
            text = f.read_text(errors='replace')
        except OSError:
            continue
        findings.extend(scan_file(f, text))
    for line in findings:
        print(line)
    if findings:
        print(f'\ncheck_ui_listeners: {len(findings)} finding(s).',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
