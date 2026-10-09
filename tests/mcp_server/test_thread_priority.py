#!/usr/bin/env python3
"""SCRIPTING follows a thread into the threads it starts.

STM priority is per OS thread and a new thread starts at NORMAL, so the
trapdoor an MCP session is locked into was escaped by threading.Thread(...).
xpythonsupport.py wraps Thread.start to close that.  The file cannot be
imported outside KAME, so the one function is lifted out of it with `ast` and
driven against a fake per-thread priority that has the binding's trapdoor.
No KAME, no `mcp`:

    python3 -m unittest tests/mcp_server/test_thread_priority.py
"""
import ast
import concurrent.futures
import threading
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "kame" / "script" / "xpythonsupport.py"
_TREE = ast.parse(_SRC.read_text(encoding="utf-8"))


def _function(name):
    return next(n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)


NORMAL, SCRIPTING = 0, 4
_tls = threading.local()


def get():
    return getattr(_tls, "p", NORMAL)


def set_(p):
    if get() == SCRIPTING and p != SCRIPTING:
        raise RuntimeError("Priority::SCRIPTING is sticky")
    _tls.p = p


def setUpModule():
    ns = {}
    exec(compile(ast.Module([_function("_kame_inherit_scripting_priority")], []),
                 str(_SRC), "exec"), ns)
    ns["_kame_inherit_scripting_priority"](threading, get, set_, SCRIPTING)


def tearDownModule():
    threading.Thread.start = threading.Thread.start.__wrapped__


def on_fresh_thread(fn, scripting):
    """Runs fn on a new thread (started from the NORMAL test thread), locked
    first if asked; the lock is per thread, so the test thread stays free."""
    out = {}
    def body():
        try:
            if scripting:
                set_(SCRIPTING)
            out["v"] = fn()
        except BaseException as e:
            out["e"] = e
    t = threading.Thread(target=body)
    t.start()
    t.join(5)
    if "e" in out:
        raise out["e"]
    return out["v"]


def level_of_child(make_thread=None):
    seen = []
    t = (make_thread or (lambda f: threading.Thread(target=f)))(lambda: seen.append(get()))
    t.start()
    t.join(5)
    return seen[0]


class Inheritance(unittest.TestCase):
    def test_child_of_scripting_is_scripting(self):
        self.assertEqual(on_fresh_thread(level_of_child, scripting=True), SCRIPTING)

    def test_child_of_normal_is_untouched(self):
        self.assertEqual(on_fresh_thread(level_of_child, scripting=False), NORMAL)

    def test_grandchild_too(self):
        self.assertEqual(on_fresh_thread(
            lambda: on_fresh_thread(level_of_child, scripting=False), scripting=True), SCRIPTING)

    def test_subclass_overriding_run(self):
        timer = lambda f: threading.Timer(0, f)
        self.assertEqual(on_fresh_thread(lambda: level_of_child(timer), scripting=True), SCRIPTING)

    def test_executor_worker(self):
        def submit():
            with concurrent.futures.ThreadPoolExecutor(1) as ex:
                return ex.submit(get).result(5)
        self.assertEqual(on_fresh_thread(submit, scripting=True), SCRIPTING)

    def test_child_cannot_leave(self):
        def child_escapes():
            err = []
            def f():
                try:
                    set_(NORMAL)
                except RuntimeError as e:
                    err.append(e)
            t = threading.Thread(target=f)
            t.start()
            t.join(5)
            return err
        self.assertEqual(len(on_fresh_thread(child_escapes, scripting=True)), 1)

    def test_opt_out(self):
        def opted_out(f):
            t = threading.Thread(target=f)
            t._kame_inherit_priority = False
            return t
        self.assertEqual(on_fresh_thread(lambda: level_of_child(opted_out), scripting=True), NORMAL)

    def test_wrapper_is_dropped_after_run(self):
        def run_one():
            t = threading.Thread(target=lambda: None)
            t.start()
            t.join(5)
            return "run" in t.__dict__
        self.assertFalse(on_fresh_thread(run_one, scripting=True))

    def test_installing_twice_does_not_wrap_twice(self):
        ns = {}
        exec(compile(ast.Module([_function("_kame_inherit_scripting_priority")], []),
                     str(_SRC), "exec"), ns)
        before = threading.Thread.start
        ns["_kame_inherit_scripting_priority"](threading, get, set_, SCRIPTING)
        self.assertIs(threading.Thread.start, before)


class ScriptPaneLauncher(unittest.TestCase):
    def test_script_pane_runs_opt_out(self):
        """kame_pybind_one_iteration runs on the kernel thread, which is
        SCRIPTING once an MCP client has connected; the user's Script-pane run
        it starts must not be locked by that."""
        fn = _function("kame_pybind_one_iteration")
        self.assertTrue(any(
            isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == "_kame_inherit_priority"
                for t in n.targets)
            for n in ast.walk(fn)))


if __name__ == "__main__":
    unittest.main()
