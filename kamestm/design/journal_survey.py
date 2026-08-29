"""Survey the node tree for the provenance journal design.

Answers, on a REAL setup, the questions the design has so far only assumed:
  * how many nodes there are, and how they split by runtime / kind
  * which non-runtime nodes change while nobody is touching the UI
    (those are outputs whose `runtime` flag is missing -- the leakage the
    design expects but has never measured)
  * how fast settings actually change

Nothing is written to the instrument and no file of KAME's is touched; the
report goes to ~/kame_journal_survey.txt.

Run it from KAME:  Script -> Run..., or paste into the Python line shell.
Leave the measurement RUNNING and hands off the mouse for the sampling
period, or the "changed by itself" column means nothing.
"""

import os, time, collections

SAMPLE_SECONDS = 120.0
SAMPLE_INTERVAL = 0.25
REPORT = os.path.expanduser("~/kame_journal_survey.txt")


def walk(node, path, out, depth=0):
    """Collect (path, node, runtime) for every node under `node`.

    Alias lists are not descended into, the same rule the .kam writer follows
    (`XAliasListNode` children are navigated, never created).  They reference
    nodes another parent owns -- /Interfaces holds every driver's "Interface"
    -- so walking them counts hard-linked nodes once per parent and invents
    ambiguity that KAME itself never has: the address of that node is
    /Drivers/<name>/Interface, and nothing addresses it through the list.

    Even so, paths are not unique keys here: a calibration table holds
    hundreds of children with empty names, where order is what carries the
    meaning.  Each node therefore gets an index; the path is for reading only.
    """
    try:
        shot = Snapshot(node)
        children = shot.list(node)
    except Exception:
        children = None
    try:
        runtime = node.isRuntime()
    except Exception:
        runtime = None
    out.append((path, node, runtime, depth, bool(children)))
    if not children:
        return
    try:
        if node.getTypename().startswith("XAliasListNode"):
            return      #references, not ownership -- see the docstring
    except Exception:
        pass
    for child in children:
        try:
            name = child.getName()
        except Exception:
            continue
        walk(child, path + "/" + name, out, depth + 1)


def value_of(node):
    """String value, or None when the node holds no value."""
    try:
        return str(node)
    except Exception:
        return None


def main():
    t0 = time.time()
    nodes = []
    walk(Root(), "", nodes)

    total = len(nodes)
    runtime_yes = sum(1 for _, _, r, _, _ in nodes if r)
    runtime_no = sum(1 for _, _, r, _, _ in nodes if r is False)
    runtime_unknown = total - runtime_yes - runtime_no
    lists = sum(1 for _, _, _, _, has in nodes if has)
    maxdepth = max((d for _, _, _, d, _ in nodes), default=0)

    # Only non-runtime value nodes matter: those are what the journal would
    # subscribe to.
    watched = []
    for i, (path, node, runtime, _, _) in enumerate(nodes):
        if runtime:
            continue
        v = value_of(node)
        if v is None:
            continue
        watched.append((i, path, node, v))

    #How badly paths collide is itself a design input.
    seen_paths = collections.Counter(path for _, path, _, _ in watched)
    colliding = {p: n for p, n in seen_paths.items() if n > 1}

    lines = []
    lines.append("KAME journal survey  %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("")
    lines.append("nodes total          : %d" % total)
    lines.append("  runtime == True    : %d" % runtime_yes)
    lines.append("  runtime == False   : %d" % runtime_no)
    lines.append("  flag unreadable    : %d" % runtime_unknown)
    lines.append("  with children      : %d" % lists)
    lines.append("  max depth          : %d" % maxdepth)
    lines.append("value nodes the journal would subscribe to (non-runtime): %d"
                 % len(watched))
    lines.append("  distinct paths among them: %d  (%d paths cover %d nodes)"
                 % (len(seen_paths), len(colliding), sum(colliding.values())))
    lines.append("tree walk took       : %.2f s" % (time.time() - t0))
    lines.append("")
    lines.append("Sampling %.0f s at %.2f s -- keep the measurement running and "
                 "do NOT touch the UI." % (SAMPLE_SECONDS, SAMPLE_INTERVAL))
    print("\n".join(lines))

    # --- who changes by itself?
    last = {i: v for i, _, _, v in watched}
    counts = collections.Counter()
    samples = 0
    t_end = time.time() + SAMPLE_SECONDS
    while time.time() < t_end:
        sleep(SAMPLE_INTERVAL)
        samples += 1
        for i, _, node, _ in watched:
            v = value_of(node)
            if v is None:
                continue
            if v != last.get(i):
                counts[i] += 1
                last[i] = v

    duration = samples * SAMPLE_INTERVAL
    lines.append("")
    lines.append("samples              : %d over %.0f s" % (samples, duration))
    lines.append("non-runtime nodes that changed with nobody touching them: %d"
                 % len(counts))
    lines.append("(each is an output whose runtime flag is missing, or a "
                 "setting a driver writes back)")
    lines.append("A rate above %.2f /s would be impossible at this sampling "
                 "interval and means the key is wrong." % (1.0 / SAMPLE_INTERVAL))
    lines.append("")
    paths = {i: path for i, path, _, _ in watched}
    for i, n in counts.most_common():
        lines.append("  %6.2f /s  [#%d] %s" %
                     (n / duration if duration else 0, i, paths.get(i, "?")))
    if not counts:
        lines.append("  (none -- the runtime flag is doing its job here)")

    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-(len(counts) + 8):]))
    print("\nwritten to " + REPORT)


main()
