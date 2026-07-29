"""Provoke and measure STM starvation of a revocable priority, from inside KAME.

Run in the KAME Jupyter console / IPython kernel:

    exec(open('kame/script/starvation_probe.py').read())
    starvation_probe(target=Root()["Drivers"]["MyDriver"]["SomeSetting"])

What is being tested
--------------------
A priority whose privilege can be revoked -- LOWEST / UI_DEFERRABLE / SCRIPTING
-- is given a bound (``KAME_STM_LOWPRIO_STARVE_MS``, 1000 ms) after which it stops
retrying and the host-installed handler runs.  KAME's handler throws
``XKameError``, which arrives here as ``kame.KAMEError``.  NORMAL and HIGHEST are
excluded: their privilege never expires, so they are never revoked and need no
failure path.

Without the bound those priorities retry forever and the only exit is the
negotiation HANG watchdog aborting KAME after 3 x 5 s.

Three facts shape the setup, all measured in kamestm/tests
----------------------------------------------------------
* **One low-priority thread does not starve; two or more do.**  transaction_-
  latency_bench at a 2 ms bound: ``-L 1`` never fired, ``-L 2`` / ``-L 4`` /
  ``-L 8`` did.  Low-priority threads starve each *other*, being excluded from
  the per-Linkage owner-skip lease and (for LOWEST) the jittered gate.  Hence
  three threads by default -- and ``threads=1`` is the control that should NOT
  starve.

* **Whole-tree scope is what makes an attempt expensive enough to accumulate
  retries.**  Bundle churn is O(subtree), so a transaction on ``Root()`` pays for
  every node under it.  A shallow tree did not starve in the C++ test where the
  bench's deeper arm did.

* **A fresh Python thread starts at NORMAL.**  The priority is per-thread and only
  the interpreter thread was set to UI_DEFERRABLE (xpythonsupport.cpp), so each
  worker must set its own -- without that this probe would silently measure
  NORMAL and never fire.  ``Priority.SCRIPTING`` is also a one-way trapdoor per
  thread: a worker that enters it cannot leave, which is fine for a throwaway
  thread and is why the interpreter itself is left alone.

This writes to a node
---------------------
Every attempt writes a value to ``target`` -- the same value it just read, but a
write is a write: ``onValueChanged`` listeners fire, and on a driver-owned node
that can mean traffic to an instrument.  **Pass ``target`` explicitly and choose
something inert.**  ``find_writable()`` exists for convenience but must be
enabled with ``allow_autotarget=True``, because it cannot tell a harmless
setting from a live one.
"""

import threading
import time
import traceback


def _children(node):
    try:
        return list(Snapshot(node).list())
    except Exception:
        return []


def read_scalar(node):
    """Current value of a scalar node, or None if it is not one."""
    try:
        payload = Snapshot(node)[node]
    except Exception:
        return None
    for conv in (float, int):
        try:
            return conv(payload)
        except Exception:
            pass
    return None


def find_writable(root=None, limit=6, verbose=True):
    """First scalar node under `root` that accepts a same-value write.

    Does the trial write inside a real transaction, so a node that refuses it
    (wrong type, validator, driver-owned and disabled) is skipped rather than
    assumed.  NOT safe by itself -- see the module docstring.
    """
    root = root or Root()
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        val = read_scalar(node)
        if val is not None:
            try:
                for tr in Transaction(node):
                    tr[node] = val
                if verbose:
                    print("auto target: %s = %r" % (node.getName(), val))
                return node
            except Exception:
                pass
        if depth < limit:
            for c in _children(node):
                stack.append((c, depth + 1))
    if verbose:
        print("no writable scalar node found under Root()")
    return None


def starvation_probe(target=None, threads=3, seconds=10.0, priority=None,
                     scope=None, allow_autotarget=False):
    """Contend at whole-tree scope from several revocable-priority threads.

    :param target:   scalar node written on every attempt.  Required unless
                     ``allow_autotarget=True``; see the module docstring on why.
    :param threads:  workers.  Run ``threads=1`` as the control -- one
                     revocable-priority thread is expected NOT to starve.
    :param priority: ``Priority.SCRIPTING`` by default;
                     ``Priority.UI_DEFERRABLE`` is what the GUI and this
                     interpreter already run at.
    :param scope:    node the transaction covers.  ``Root()`` by default, which
                     is what makes each attempt expensive.
    """
    if priority is None:
        priority = Priority.SCRIPTING
    if target is None:
        if not allow_autotarget:
            print("Refusing to pick a target automatically: every attempt "
                  "writes to it and listeners fire, which on a driver-owned "
                  "node can talk to an instrument.\n"
                  "Pass target=<an inert scalar node>, or "
                  "allow_autotarget=True if you accept that.")
            return None
        target = find_writable()
        if target is None:
            return None
    scope = scope or Root()

    base = read_scalar(target)
    if base is None:
        print("target is not a scalar node this probe can write")
        return None

    stop = threading.Event()
    stats = [dict(commits=0, starved=0, other=0, max_s=0.0)
             for _ in range(threads)]
    start_gate = threading.Barrier(threads + 1)

    def worker(i):
        setCurrentPriorityMode(priority)
        st = stats[i]
        start_gate.wait()
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                # Whole-tree scope: bundles everything under `scope`, so any peer
                # commit in the tree invalidates this attempt.  The retry loop is
                # the `for`, driven by Transaction.__next__ -> commitOrNext.
                for tr in Transaction(scope):
                    tr[target] = base
                st["commits"] += 1
            except KAMEError:
                # KAME's starvation handler throws XKameError, registered as
                # kame.KAMEError.  Other KAME-reported errors land here too,
                # which is why the first one is printed in full.
                st["starved"] += 1
                if st["starved"] == 1:
                    print("thread %d first KAMEError:" % i)
                    traceback.print_exc(limit=1)
            except Exception:
                st["other"] += 1
                if st["other"] == 1:
                    print("thread %d unexpected exception:" % i)
                    traceback.print_exc(limit=2)
            st["max_s"] = max(st["max_s"], time.perf_counter() - t0)

    ts = [threading.Thread(target=worker, args=(i,), daemon=True)
          for i in range(threads)]
    for t in ts:
        t.start()
    start_gate.wait()
    t0 = time.perf_counter()
    time.sleep(seconds)
    stop.set()
    for t in ts:
        t.join(timeout=30.0)
    elapsed = time.perf_counter() - t0

    total = dict(commits=0, starved=0, other=0, max_s=0.0)
    print("\n%-7s %10s %9s %8s %10s" %
          ("thread", "commits", "starved", "other", "max ms"))
    for i, st in enumerate(stats):
        print("%-7d %10d %9d %8d %10.1f" %
              (i, st["commits"], st["starved"], st["other"], st["max_s"] * 1e3))
        for k in ("commits", "starved", "other"):
            total[k] += st[k]
        total["max_s"] = max(total["max_s"], st["max_s"])
    print("%-7s %10d %9d %8d %10.1f   over %.1f s at %s" %
          ("total", total["commits"], total["starved"], total["other"],
           total["max_s"] * 1e3, elapsed, priority))

    if total["starved"]:
        print("\nStarvation fired -- the bound working. Without it these "
              "attempts would have retried indefinitely, and the only exit was "
              "the HANG watchdog aborting KAME after 3 x 5 s.")
    else:
        print("\nNo starvation. Any of:\n"
              "  * the tree is not contended enough -- are drivers running? try "
              "more threads, or a scope with more traffic under it;\n"
              "  * the bound is compiled out (KAME_STM_LOWPRIO_STARVE_MS=0);\n"
              "  * no handler is installed -- with none, the bound is reached "
              "and deliberately does nothing (that is what keeps enabling it "
              "from introducing an unhandled exception);\n"
              "  * threads=1, which is expected not to starve.")
    if threads > 1:
        print("Control: starvation_probe(target=..., threads=1) should report "
              "0 starved.")
    return total
