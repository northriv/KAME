"""Exercise the STM starvation bound from inside KAME, deterministically.

Run in the KAME Jupyter console / IPython kernel:

    exec(open('kame/script/starvation_probe.py').read())
    starvation_probe(Root()["..."]["SomeInertSetting"])

What is being tested
--------------------
A priority whose privilege can be revoked -- LOWEST / UI_DEFERRABLE / SCRIPTING
-- gets a bound (``KAME_STM_LOWPRIO_STARVE_MS``, 1000 ms) after which it stops
retrying and the host-installed handler runs.  KAME's handler throws
``XKameError``, which arrives here as ``kame.KAMEError``.  NORMAL and HIGHEST are
excluded: their privilege never expires, so they are never revoked.

Without the bound those priorities retry forever, and the only exit is the
negotiation HANG watchdog aborting KAME after 3 x 5 s.

Why a slow transaction is not enough on its own
-----------------------------------------------
The bound needs BOTH an age past the limit AND at least
``KAME_STM_LOWPRIO_STARVE_MIN_RETRIES`` (8) retries -- the retry gate is what
keeps the clock off the fast path.  So a transaction that merely takes a long
time does not fire: its retry count is 0.

And in Python the retry count only advances on a *failed* commit.
``Transaction.__next__`` calls ``commitOrNext()`` only when the transaction has
been modified, and ``commitOrNext()`` calls ``++(*this)`` -- the increment, and
the bound -- only when that commit fails.  A body that modifies nothing loops
without ever incrementing anything.

So each iteration here modifies the target, then commits a **nested** transaction
on the same node, which invalidates the outer one.  The outer commit then fails,
the retry count advances, and a short sleep ages it.  No contention, no other
threads, no drivers needed, and the same result every run.

This writes to a node
---------------------
Two same-value writes per iteration, about a dozen iterations -- but a write is a
write: ``onValueChanged`` listeners fire, and on a driver-owned node that can mean
traffic to an instrument.  **Pass an inert scalar node.**
"""

import time
import traceback


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


def starvation_probe(target, priority=None, retries=12, pause=0.12,
                     verbose=True):
    """Age one transaction past the bound and report what came out.

    :param target:   scalar node to write.  Same value each time; still fires
                     listeners, so choose something inert.
    :param priority: ``Priority.SCRIPTING`` by default.  Pass
                     ``Priority.NORMAL`` for the control -- NORMAL is not
                     revocable and must NOT throw however long it retries.
    :param retries:  outer iterations; needs to exceed
                     KAME_STM_LOWPRIO_STARVE_MIN_RETRIES (8).
    :param pause:    seconds per iteration; retries * pause should exceed the
                     bound (1 s by default), so 12 x 0.12 s = 1.44 s.
    :return: dict with what happened.
    """
    if priority is None:
        priority = Priority.SCRIPTING
    base = read_scalar(target)
    if base is None:
        print("target is not a scalar node this probe can read and write")
        return None

    # Per-thread and, for SCRIPTING, a one-way trapdoor -- so this changes the
    # calling thread's priority for good.  Run it from a throwaway thread if the
    # interpreter must stay at UI_DEFERRABLE.
    setCurrentPriorityMode(priority)
    if verbose:
        print("priority now %s; target %s = %r; aiming for %d retries over "
              "%.2f s" % (getCurrentPriorityMode(), target.getName(), base,
                          retries, retries * pause))

    result = dict(iterations=0, thrown=False, error=None,
                  elapsed=0.0, priority=str(priority))
    t0 = time.perf_counter()
    try:
        for tr in Transaction(target):
            result["iterations"] += 1
            # Modify, so __next__ actually attempts commitOrNext().
            tr[target] = base
            if result["iterations"] > retries:
                break               # let this one commit and end the loop
            # Invalidate ourselves: a nested transaction that DOES commit bumps
            # the node, so the outer commit below fails, ++(*this) runs, and the
            # retry count -- which the bound gates on -- advances.
            for inner in Transaction(target):
                inner[target] = base
            time.sleep(pause)
    except KAMEError as e:
        result["thrown"] = True
        result["error"] = str(e)
    except Exception as e:
        result["error"] = "unexpected %s: %s" % (type(e).__name__, e)
        if verbose:
            traceback.print_exc(limit=2)
    result["elapsed"] = time.perf_counter() - t0

    if verbose:
        print("iterations %d, elapsed %.2f s, threw %s"
              % (result["iterations"], result["elapsed"], result["thrown"]))
        if result["thrown"]:
            print("  " + (result["error"] or ""))
            print("\nThe bound fired. Without it this transaction would have "
                  "retried indefinitely; the only exit was the HANG watchdog "
                  "aborting KAME after 3 x 5 s.")
        elif result["error"]:
            print("  " + result["error"])
        else:
            print("\nNothing thrown. Expected for Priority.NORMAL. At a "
                  "revocable priority it means one of:\n"
                  "  * the bound is compiled out "
                  "(KAME_STM_LOWPRIO_STARVE_MS=0);\n"
                  "  * no handler is installed -- with none, the bound is "
                  "reached and deliberately does nothing, which is what keeps "
                  "enabling it from introducing an unhandled exception;\n"
                  "  * retries * pause did not exceed the bound, or retries did "
                  "not exceed KAME_STM_LOWPRIO_STARVE_MIN_RETRIES.")
    return result


def starvation_probe_control(target, **kw):
    """The control: NORMAL is not revocable and must not throw."""
    return starvation_probe(target, priority=Priority.NORMAL, **kw)
