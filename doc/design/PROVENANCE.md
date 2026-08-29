# Measurement provenance: journal and replay

Status: the library core has landed (`kamestm/transaction_journal.h`,
`atomic_bounded_ring` in `kamestm/atomic_queue.h`, `transaction_journal_test`).
Everything below the core is design, not code.

## Why this, and why it is cheap here

A `Snapshot` in this framework is O(1).  That makes something normally
expensive nearly free: recording what the whole instrument looked like at an
instant.  Keep one such snapshot and the ordered stream of what changed after
it, and **any** intermediate state can be reconstructed — not merely the two
endpoints a saved `.kam` gives you.  In a framework that had to gather every
parameter by walking its objects, this would not be worth doing.

There is also a concrete hole to fill.  `XRawStreamRecorder` writes each
driver's raw record with its timestamp — and nothing else.  Replaying that
stream through `XRawStreamRecordReader` therefore re-analyses it **with
today's settings**, not the window function, fit range or calibration curve
that were in force when it was taken.  The journal supplies the missing half,
and the two join on time.

## What is recorded

**The class belongs to the write, not to the node.**  This is the conclusion
the design arrived at last, and it overturns the obvious approach.  A flag can
say one thing about a node; the distinction that matters here is what a
particular write *was*:

- a **request** — the user or a script asked for a value;
- a **report** — a driver wrote back what the instrument says, at open or as
  it goes.

The same node takes both.  `XThamwayPROT` creates `RXGain` and `RXPhase` as
`runtime == true` — settings the user changes, marked runtime because they are
read back from the instrument rather than saved.  `ODMR2D/Average` is
`runtime == false` and written by its driver while accumulating.  No flag can
be right about either, because both answers are right at different moments.
Attribution is per write, comes free in the serial's low bits, and says
exactly this.

A useful consequence: `runtime` is used inconsistently across 120 drivers and
5890 nodes, and **none of this depends on it being right**.  Subscription,
classification, capping and restore are all decided without consulting it, so
no audit of those flags is a prerequisite for any of this.  The order in fact
reverses — running the journal *produces* the audit, since it records which
nodes are written by whom and how often.  The flag keeps its existing meaning
for `.kam`, where its inconsistencies are already what users have.

So **subscription is not filtered by the flag**: doing that would have lost
every change to `RXGain`, which is to say a record of the user operating the
instrument.  Subscribe to value nodes broadly — 7103 nodes is an upper bound
and a listener each is well under a megabyte — and decide at *record* time.

| | Default | Rule |
|---|---|---|
| Requests (any node) | **always on** | the user operated something; never dropped |
| Reports, rarely changing | **always on** | falls under any cap; this is how the state a device reports at open is captured |
| Reports, at acquisition rate | capped | decimated by the same cap; "what was the temperature at 3:14" needs no kHz |
| Raw driver records | unchanged | user-chosen file, as today |

The rate cap is keyed on attribution rather than on the flag, and it does more
than guard against floods: it **classifies**.  A value a device reports once
when the interface opens passes any cap untouched; a measurement updating ten
times a second is thinned.  Nobody has to decide which is which.

Two things make that deliberate rather than incidental.  Interfaces already
announce themselves — `onOpen` and `onClose` — so the journal dumps a driver's
subtree when its interface opens, and "the instrument reported firmware 2.31
that day" is recorded on purpose rather than caught in passing.  And restore
replays **request entries only**: for `RXGain`, the gain the user asked for,
never the value the device happened to report.

Settings cost a few hundred KB a day, so there is no reason to make the user
decide.  Observations cost roughly 36 MB/hr at 20 values × 10 Hz — nothing
against 10 GB/hr of raw data, but no longer free.

**Observations are selected by a rule, not by a per-driver checkbox** (120
drivers' worth of checkboxes is not a user interface): record a driver's
observations only when its raw records are *not* being recorded, since raw
records plus settings can regenerate them.  A per-node rate cap keeps a driver
that commits at kHz from flooding the file; "what was the temperature at 3:14"
does not need kHz resolution.

The one case where recording both is right: regenerating an observation
depends on the analysis code, so a KAME upgrade can change the number.  If the
value as published must be preserved, capture both — it costs 0.4% of the raw
stream.  Hence: exclusive by default, both selectable.

**Node flags are not journaled.**  `setUIEnabled` is called from 452 places
and flips on every tuning cycle; it is derived UI state that drivers
re-derive during replay, so restoring it would fight live logic — the same
reason outputs are never restored.  Flag changes arrive on `onUIFlagsChanged`,
a different talker, so subscribing only to `onValueChanged` excludes them with
no special case.  The flags **as of a node's first appearance** do go in its
table entry: static, one line, and it makes the journal self-describing about
why a node was included.

## Architecture

**The library holds only what does not depend on meaning**: the bounded ring,
the accounting of what could not be kept, and the ordering rule.  Names,
paths, formatting, attribution and files are the application's.

**There is deliberately no hook in the commit path.**  Subscription is the
switch: with no listener attached, `Talker::createMessage()` returns nullptr
before allocating anything, so a journal that is off costs *exactly* nothing
and needs no global flag and no branch to say so.  KAME subscribes to
`onValueChanged`, `onTouch` and `onListChanged`, re-attaching as the tree
grows (drivers created, `.kam` loaded).  Runtime subtrees can be pruned
wholesale, since the flag propagates to descendants.

### Starting: subscribe first, read second

Beginning to journal means two things — writing down what is there now, and
subscribing to what happens next — and whichever comes first leaves a window.
Dump before subscribing and a change in the gap appears **nowhere**: the dump
holds the old value and no entry records the new one.  Subscribe before
dumping and it appears **twice**.  A loss cannot be repaired; a duplicate can
be dropped, and re-applying a value that is already set is idempotent anyway.
So: subscribe first, and give every dumped value the serial of the snapshot it
was read from, so a reader can discard the entries that predate it.

The window closes entirely at node granularity, without any global
transaction:

```
for each subtree:
    subscribe to the list's onListChanged        <- before enumerating
    for each child:
        subscribe to its onValueChanged / onTouch <- before reading
        read its value and write it to the dump, with the serial
        recurse
```

Anything committed after the subscription is caught; the dumped value is read
after the subscription, so the worst case is a change that appears both as an
entry and in the dump, which the serial identifies.  A child inserted between
subscribing to a list and enumerating it likewise shows up in both.

**No root transaction, and no root snapshot.**  Strictness by atomicity is the
wrong instrument here: a snapshot of an ancestor bundles the subtree, and with
a hard link present it forces unrelated transactions to fail their CAS and
retry — stopping the whole tree to start a journal contradicts what the
journal is for.  Correctness comes instead from "nothing is lost, and
duplicates are identifiable", which is the same reasoning that keeps restore
per driver rather than per tree.

### Node identity

A **session-local id assigned when the node is first subscribed**, plus a
table record carrying its path, its relationship to its parent, its
registered type name, its position and its flags.  Not a path hash:

- paths are not unique — this is a DAG, and a hard-linked node has several;
- a path hash conflates a node that was deleted with a different node later
  created at the same path, which is exactly the case provenance must
  distinguish.

And not the path itself.  KAME's own rule is that a node which cannot be told
from its siblings by name is **not addressed through that list at all**: the
interface listed as `/Interfaces/Interface` for every driver is reached at
`/Drivers/<name>/Interface`, where the name is unique, and `XInterfaceList` is
an `XAliasListNode` precisely to say so — the same class the `.kam` writer
refuses to emit `create()` for.  So the canonical path runs through the parent
that **owns** a node, never through a list that merely references it.

Position cannot stand in either: it moves as the user reorders things in the
UI and depends on registration order.  It is meaningful only in the lists
where order *is* the meaning — a calibration table's rows, whose children have
empty names for that very reason.

Not topology indices either: sibling insertion and release shift them, so the
same node changes identity mid-session.  Cross-session stability is not
needed — each journal carries its own id → path table, and runs are compared
on paths.  The path is computed once, at subscribe time, so no DAG walk ever
happens on the capture path.

### Enough to rebuild the tree, not merely to re-set it

Values alone cannot reconstruct anything: `.kam` writes
`create(typename, name)` because dynamically created children — drivers,
entries, graphs, math tools — have to be brought into existence before
anything can be set on them.  A dump that stands in for `.kam` has to carry
the same three-way distinction its writer makes:

- a child of an `XListNodeBase` — **created**, with its registered type name
  (omitted for `XListNode`, whose element type is already fixed);
- a child of an `XAliasListNode` — **navigated** by name, never created;
- any other child — exists as soon as its parent does; navigated.

Plus its position, where order carries meaning (a graph's axes).

The same applies to structure that appears mid-session: a driver added at
14:22 is an `onListChanged` event, and unless the type name is recorded **at
that moment** the addition cannot be replayed.  Type names travel with
structural events, not only with the opening dump.

One trap comes along with this.  `getTypename()` defaults to
`typeid(*this).name()`, which for a template instantiation alias
(`using XFoo = XFooX<Functor>`) is a mangled name that matches no key in
`XTypeHolder` — the same defect that breaks `.kam` round-trips.  A dump
should resolve every type name it writes against the registry and **say so
loudly** when one does not resolve, rather than writing a line that will
quietly fail to recreate anything.  (`.kam` writes it regardless today, and
the Python loader's `_KamFakeNode` swallows the failure on the way back in.)

### Attribution

The serial already carries the committing thread in its low 16 bits, so
"who" costs nothing.  Scripts add context through TLS (the IPython cell
label already exists).  Thread identity classifies nodes empirically:

- written by UI or script threads → a setting (intent);
- written only by driver threads → an output (observation).

This is worth more than a flag — though not for the reason first supposed.
Mis-flagged outputs were expected and have not been found; what exists instead
is nodes written by both, where no flag could decide because both answers are
correct at different moments (see the measurements below).  Attribution also
implements "my edits win" during replay: a node the user has touched since
opening a recording is theirs, and the journal does not overwrite it.  A
`NODE_NOT_JOURNALED` bit costs nothing in memory (`m_flags` has spare bits)
and can be added later — but only together with a `tools/audit` checker, or
it will leak exactly as `runtime` did.

### Values

Recorded in the node's **own textual form** (`to_str()`), the one `.kam`
uses.  That is not a shortcut but the point: `.kam` stores an `XComboNode` as
its item label rather than its index precisely because the list of choices is
built at run time, so an index means something different tomorrow.  The rule
generalises — record a value in the form that survives changes to the state
around it — and it has the side benefit that restoring goes through exactly
the setter `.kam` uses.

Numbers need a second field.  `to_str()` on an `XDoubleNode` goes through
`formatDouble()`, which is `%.12g` by default and, where a display format is
set, that format — so `.kam` does not round-trip a double today, and a node
displayed to three digits is *saved* to three digits.  Tolerable for a
settings file; not for provenance, where it produces both a spurious `diff`
between identical settings and a "reproduction" that used a different number
than the original.

So a floating-point entry carries the displayed string **and** the exact
value as **base64 of its eight bytes, binary64, little-endian** — stated in
the format rather than inherited from whichever machines happen to run KAME.
The readable half is already `to_str()`; the exact half only has to be exact,
and as bytes it is exact by construction:

- nothing depends on `to_chars` being available for floating point, which has
  real version floors across libstdc++, libc++ and MinGW;
- there is no locale to get wrong — `snprintf("%.17g")` writes `3,14` under a
  German or French locale, which is both a wrong number and invalid JSON;
- `inf`, `nan`, `-0` and subnormals need no special case, where decimal would
  have to escape them into strings because JSON admits no such literals, and
  that branch is where such writers go wrong.

Only floating-point nodes get it.  Strings, combos, booleans and integers
round-trip through `to_str()` already, so the presence of the exact field is
itself a marker that the readable one is rounded.

(That `.kam` itself rounds is a separate, pre-existing matter.  Changing its
precision would make old and new files differ for reasons that have nothing
to do with the settings, so it wants its own decision.)

Captured as a pool-allocated blob whose ownership passes through the ring
(freed by the drain, or by the producer when the ring is full).  **No
truncation**: KAME's allocator is lock-free, so allocating is acceptable even
on a real-time path, and a provenance record that silently shortens a string
is not one.  The record itself stays trivially copyable — it holds a pointer,
so nothing constructs or destructs inside the ring.

Losing records under pressure is allowed; losing them silently is not.  What
the ring refused is counted, and the drain writes a gap where it belongs.

## Files

One file, self-contained: **the dump is the head of the journal**.  That
removes the pairing problem entirely, makes every rotated segment
independently readable, and leaves the state at session start on disk even if
the session dies a second later.

| File | Extension | Contents |
|---|---|---|
| Settings snapshot (unchanged) | `.kam` | as today |
| Journal | **`.kamj`** | header record, dump, then one JSON object per line |
| Journal with observations | **`.kamj.gz`** | same format, gzip *members* per block |
| Raw stream | unchanged | as today |

JSON Lines because the two operations that matter — `diff run1.kamj
run2.kamj` and `grep` — then work with no tool at all, and because a
provenance record's value is in ten years, when the reader may be gone but
text with an obvious schema is not.  **Doubles are written `%.17g`**: JSON
does not distinguish int from float, and a value that fails to round-trip
turns "identical settings" into a spurious diff, which is the worst possible
failure for this file.

The extension is a hint; the file declares itself in its first line:

```json
{"format":"kame-journal","version":1,"session":"<uuid>","started":"...","kame":"9.0.0"}
```

Block compression uses concatenated gzip members (as BGZF does), so the file
stays seekable *and* `zcat`/`zgrep` still read it.

### Where they live

`QStandardPaths::AppLocalDataLocation` — `~/Library/Application Support/kame`
(macOS), `AppData\Local\kame` (Windows), `~/.local/share/kame` (Linux) — with
`journal/` and `log/` under it.  **Local**, not `AppDataLocation`: on Windows
that would be `Roaming`, and journals and logs are machine-local and not
small.  KAME already sets `applicationName("kame")` and no organization, so
the path has no vendor component.

The always-on session journal lives there, so provenance exists even when the
user has started no recording.  **When a recording is started, the journal is
also written beside the user's chosen file with a matching basename** —
`run042.gz` / `run042.kamj` — because users manage measurements as files and a
journal left behind in a hidden directory is a journal lost the first time the
data is copied to another machine.  Both carry the session UUID, and each
names the other, so either half leads to its counterpart.

`log/` is where `kame.log` should end up too — `$TMPDIR` is cleared on
reboot, which is the wrong property for the file you read after a crash, and
on Windows the executable's own folder may not be writable at all.  That move
is two-stage: the log stream is opened during static initialisation and
therefore cannot touch Qt (`QStandardPaths` needs `QCoreApplication`), so it
starts where it does today and is redirected once `main()` has Qt up.

### Separate from the raw stream, on purpose

Combining them would give one artifact and inherent ordering.  It would also
cost, and each of these is on its own decisive:

- **seek** — the raw stream is whole-file gzip read through `gzseek`, which
  decompresses forward from the start; at 10 GB/hr, reaching hour three means
  30 GB.  A journal inside it inherits that, and "jump to T with the settings
  of that moment" stops being tractable.  Scanned as its own small file, the
  journal is always cheap to seek.
- **existence** — the always-on settings journal must exist when no recording
  was started.  A combined file only exists while recording.
- **lifetime** — 240 GB/day gets deleted; a few hundred KB/day should be kept
  for years.  One file, one fate.

The residual risk, that the two are separated by hand, is mitigated by the
matching basename and the embedded cross-references, and can be reduced
further by mirroring the (tiny) settings changes into the raw stream as well,
so a lone raw file remains interpretable.

## Replay

Two operations, and they want opposite things:

- **Seek/prime** — collapse every entry from the head up to T into one state
  and apply it **per driver, in a single transaction, writing only what
  differs**.  Applying node by node makes analysis run on half-restored state
  (node A new, node B still old), which is both wasteful and wrong.  Per
  driver rather than per tree: a transaction on a common ancestor bundles the
  subtree and its CAS would keep losing.
- **Sequential replay** — apply entries at their original times, interleaved
  with the raw records.  Here analysis *should* re-run on each change: that is
  what happened.

Three policies, not two:

1. **off** — today's behaviour; re-analyse with current settings.  This is a
   real use, not a fallback: changing one parameter and re-running is the
   normal scientific move.
2. **strict** — exactly as recorded, for verification.
3. **user edits win** — the journal is the default, but any node the user
   touched since opening the recording stays theirs.  Attribution already
   distinguishes the two.

**Observations are never restored, in any policy.**  They are outputs; the
analysis recomputes them from the restored inputs, and writing them back would
conflict with the driver that owns them.  The record/restore asymmetry is
deliberate: record generously, restore conservatively.

Replay lives in kame.app, because `XRawStreamRecordReader` does.  Reading
provenance for its own sake — diffing two runs, answering "what was it at
3:14" — is an offline tool, and deliberately not in the live application,
where a reconstructed past state could be confused with the present one.

## `.kam` is not deprecated

A diff stream without a baseline means nothing, and every event-sourced design
keeps snapshots for the same reason.  Users also carry `.kam` files between
rigs and colleagues, which a session journal cannot do.

What changes is its status: from "save it often or lose your setup" to a
checkpoint KAME writes at session start and end by itself.  Its executable,
Ruby-flavoured form also stops being load-bearing for *reading*, since the
dump at the head of the journal gives tools a neutral path → value view.

## Measured, on an ODMR setup (2026-08-29)

7103 nodes, of which **1213 are non-runtime** — the journal's subscription
list.  A listener each is tens of kilobytes, and the whole tree walks in
0.02 s, so per-node subscription is affordable and the design's central
assumption holds.  83% of the tree is runtime and gets pruned.

One node changed with nobody touching anything while being
`runtime == false`: `/Drivers/ODMR2D/Average`, about 0.43 times a second.
That is **not** a missing flag — it is deliberately a setting, and the driver
writes it back while accumulating in incremental mode.  So the leakage this
design assumed exists has **no confirmed instance**; every other candidate in
that run was an artefact of the survey.

What the run did find is a category the design did not have: **a setting that
drivers also write**.  The user decides "average 100"; the driver reports 37
on its way there, through the same node.  Attribution is still needed, then,
but for this rather than for catching mis-flagged outputs:

- **Restore takes the last *user-attributed* value** — 100, the request, not
  37, the progress.  A driver's write-back is recorded and never restored,
  which is the same asymmetry that governs outputs.
- **The rate cap is keyed on attribution, not on the flag.**  `Average` is
  non-runtime and written at the acquisition rate, so capping only
  "observations" would have missed it.  What needs capping is what a *driver*
  writes, whatever the node is.

(The same run reported nodes changing hundreds of times a second, impossible
at four samples a second.  Two faults in the survey, both instructive: it
keyed by path, and it descended into alias lists.  The second inflates the
node count — a hard-linked node is counted once per parent — and manufactures
the very ambiguity KAME avoids by never addressing those nodes that way.  The
7103 above is therefore an upper bound; the shape of the answer is not in
doubt, but the number will come down when it is measured again.)

## Deferred

- Block framing plus an index for the raw stream, so `gzseek` stops being
  linear.  Independent of this work, and increasingly needed at 10 GB/hr.
- `NODE_NOT_JOURNALED`, only with an audit checker.
- Migrating the scattered `~/.kame_*` dotfiles under the managed directory.
