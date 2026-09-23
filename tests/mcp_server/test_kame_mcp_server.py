#!/usr/bin/env python3
"""Regression tests for kame_mcp_server.py, with the kernel faked.

No KAME, no Jupyter: `_get_client` is replaced by a scripted client that
answers each execution the way IPython's iopub would, so the server's own
logic -- path handling, job ids, stop semantics, output unwrapping -- is
pinned without hardware.  Needs an interpreter that can import `mcp`:

    KAME_MCP_LOG_DIR=/tmp/x <venv>/bin/python -m unittest tests/mcp_server/test_kame_mcp_server.py
"""
import base64
import json
import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="kame-mcp-test-")
os.environ["KAME_MCP_LOG_DIR"] = _TMP          # before import: _LOG_DIR is read then
os.environ["KAME_MCP_NO_LOG"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kame" / "script"))
import kame_mcp_server as m  # noqa: E402


class FakeKernel:
    """Answers one execution with the given iopub messages.

    `reply(code)` returns a list of (msg_type, content) or raises to model a
    kernel that does not answer.  Records every code string it was sent."""
    def __init__(self, reply):
        self.reply, self.sent, self.msgs = reply, [], []

    def execute(self, code):
        self.sent.append(code)
        items = self.reply(code)
        self.msgs = [{"parent_header": {"msg_id": "m"}, "msg_type": t, "content": c}
                     for t, c in items]
        self.msgs.append({"parent_header": {"msg_id": "m"}, "msg_type": "status",
                          "content": {"execution_state": "idle"}})
        return "m"

    def get_iopub_msg(self, timeout=None):
        if not self.msgs:
            raise queue.Empty
        return self.msgs.pop(0)


def result(text_plain):
    return ("execute_result", {"data": {"text/plain": text_plain}})


def stream(text):
    return ("stream", {"text": text})


def error(ename, evalue):
    return ("error", {"ename": ename, "evalue": evalue, "traceback": []})


def use(kernel):
    m._get_client = lambda: kernel
    return kernel


class ExecuteText(unittest.TestCase):
    def test_single_string_literal_is_unwrapped(self):
        use(FakeKernel(lambda c: [result(repr("PID: 1\nDrivers (0):"))]))
        self.assertEqual(m._execute_text("x"), "PID: 1\nDrivers (0):")

    def test_json_string_comes_back_parseable(self):
        use(FakeKernel(lambda c: [result(repr('{"job_id": "_mcp_1"}'))]))
        self.assertEqual(json.loads(m._execute_text("x"))["job_id"], "_mcp_1")

    def test_mixed_output_is_left_alone(self):
        use(FakeKernel(lambda c: [stream("hello\n"), result("'x'")]))
        self.assertEqual(m._execute_text("x"), "hello\n\n'x'")   # joined with a newline, as before

    def test_non_string_repr_and_errors_are_left_alone(self):
        use(FakeKernel(lambda c: [result("42")]))
        self.assertEqual(m._execute_text("x"), "42")
        use(FakeKernel(lambda c: [error("ValueError", "bad")]))
        self.assertTrue(m._execute_text("x").startswith("ERROR: ValueError: bad"))

    def test_quote_shaped_but_not_a_literal_is_left_alone(self):
        use(FakeKernel(lambda c: [result("'unterminated")]))
        self.assertEqual(m._execute_text("x"), "'unterminated")


class Tree(unittest.TestCase):
    INJECT = 'NoSuchNode"] or int("x") or Root()["Drivers'

    def test_path_elements_reach_the_kernel_as_a_list_literal_only(self):
        k = use(FakeKernel(lambda c: [result(repr("a (T)"))]))
        m.tree(self.INJECT)
        code = k.sent[-1]
        self.assertIn(repr([self.INJECT]), code)      # as data ...
        self.assertNotIn('["NoSuchNode"]', code)      # ... never as source
        self.assertNotIn('int("x") or Root()', code.replace(repr([self.INJECT]), ""))

    def test_quote_in_a_name_is_just_a_name(self):
        k = use(FakeKernel(lambda c: [result(repr("ok"))]))
        m.tree('Drivers/He said "hi"')
        self.assertIn(repr(["Drivers", 'He said "hi"']), k.sent[-1])

    def test_missing_path_is_an_error_not_a_traceback_in_the_body(self):
        use(FakeKernel(lambda c: [result(repr(m._NOT_FOUND + "Drivers/Nope"))]))
        with self.assertRaises(ValueError) as cm:
            m.tree("Drivers/Nope")
        self.assertIn("Drivers/Nope", str(cm.exception))

    def test_empty_node_is_said_not_blank(self):
        use(FakeKernel(lambda c: [result(repr(""))]))
        self.assertEqual(m.tree("Drivers"), "(Drivers has no child nodes)")

    def test_dot_segments_are_refused(self):
        with self.assertRaises(ValueError):
            m.tree("../x")


class JobIds(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(m._check_job_id("_mcp_1712345678901"), "_mcp_1712345678901")
        for bad in ("../../etc/passwd", "'_mcp_1'", "_mcp_", "x_mcp_1", "", "_mcp_1/../x"):
            with self.assertRaises(ValueError, msg=bad):
                m._check_job_id(bad)

    def test_files_never_leave_the_job_dir(self):
        with self.assertRaises(ValueError):
            m._job_file("../canary")
        with self.assertRaises(ValueError):
            m._job_stopfile("../canary")
        self.assertEqual(m._job_file("_mcp_5").parent, m._JOB_DIR)

    def test_new_ids_are_distinct_even_within_a_millisecond(self):
        ids = {m._new_job_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)
        self.assertTrue(all(m._JOB_ID_RE.match(i) for i in ids))

    def test_get_result_rejects_a_bad_id_before_touching_anything(self):
        k = use(FakeKernel(lambda c: [result("'x'")]))
        with self.assertRaises(ValueError):
            m.get_result("../canary")
        self.assertEqual(k.sent, [])


class StopJob(unittest.TestCase):
    def setUp(self):
        for f in m._JOB_DIR.glob("*") if m._JOB_DIR.exists() else []:
            f.unlink()

    def test_running_job_is_flagged_and_marker_written(self):
        use(FakeKernel(lambda c: [result(repr(json.dumps(
            {"status": "running", "stop_requested": True})))]))
        r = json.loads(m.stop_job("_mcp_1"))
        self.assertTrue(r["stop_requested"])
        self.assertTrue(m._job_stopfile("_mcp_1").exists())

    def test_unknown_job_is_not_a_success(self):
        use(FakeKernel(lambda c: [result(repr(json.dumps({"status": "unknown"})))]))
        r = json.loads(m.stop_job("_mcp_2"))
        self.assertFalse(r["stop_requested"])
        self.assertEqual(r["status"], "unknown")
        self.assertFalse(m._job_stopfile("_mcp_2").exists(), "no marker for a job that does not exist")

    def test_finished_job_is_reported_finished(self):
        use(FakeKernel(lambda c: [result(repr(json.dumps(
            {"status": "done", "stop_requested": False, "note": "already finished; nothing to stop"})))]))
        r = json.loads(m.stop_job("_mcp_3"))
        self.assertFalse(r["stop_requested"])
        self.assertEqual(r["status"], "done")
        self.assertFalse(m._job_stopfile("_mcp_3").exists())

    def test_silent_kernel_with_marker_is_a_request_via_marker(self):
        def dead(code):
            raise RuntimeError("kernel wedged")
        use(FakeKernel(dead))
        r = json.loads(m.stop_job("_mcp_4"))
        self.assertTrue(r["stop_requested"])
        self.assertEqual(r["via"], "stop marker on disk")
        self.assertTrue(m._job_stopfile("_mcp_4").exists())

    def test_silent_kernel_and_no_marker_is_an_error(self):
        def dead(code):
            raise RuntimeError("kernel wedged")
        use(FakeKernel(dead))
        saved = m._JOB_DIR
        blocker = Path(_TMP) / "not-a-dir"
        blocker.write_text("x")
        m._JOB_DIR = blocker                       # mkdir/write must fail here
        try:
            with self.assertRaises(RuntimeError) as cm:
                m.stop_job("_mcp_5")
            self.assertIn("could not be written", str(cm.exception))
        finally:
            m._JOB_DIR = saved

    def test_unknown_in_kernel_but_state_on_disk_returns_that_state(self):
        m._JOB_DIR.mkdir(parents=True, exist_ok=True)
        m._job_file("_mcp_6").write_text(json.dumps({"status": "done", "progress": "60/60"}))
        use(FakeKernel(lambda c: [result(repr(json.dumps({"status": "unknown"})))]))
        r = json.loads(m.stop_job("_mcp_6"))
        self.assertEqual((r["status"], r["stop_requested"], r["progress"]), ("done", False, "60/60"))


class NotebookStatus(unittest.TestCase):
    def test_no_sessions_is_a_sentence_with_candidates(self):
        saved = (m._nb_api, m._start_activity_watcher)
        m._start_activity_watcher = lambda: None
        m._nb_api = lambda path, method="GET", body=None: (
            [] if path in ("/api/sessions", "/api/kernels")
            else {"content": [{"type": "notebook", "path": "KAME_notebook.ipynb"},
                              {"type": "file", "path": "notes.txt"}]})
        try:
            out = m.notebook_status()
        finally:
            m._nb_api, m._start_activity_watcher = saved
        self.assertIn("No notebook sessions", out)
        self.assertIn("KAME_notebook.ipynb", out)
        self.assertNotIn("notes.txt", out)
        self.assertNotIn("state above", out)


if __name__ == "__main__":
    unittest.main()
