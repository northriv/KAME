# Measurement provenance: journal and replay

Status: the library core has landed (`kamestm/transaction_journal.h`,
`atomic_bounded_ring` in `kamestm/atomic_queue.h`, `transaction_journal_test`),
and so has the capture stage above it (`kame/xjournal.{h,cpp}` — subscription,
attribution and a survey report, no file format and no replay yet; see
"Stage 1: capture only" below).  Everything else is design, not code.

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
| Reports, rarely changing | **always on** | this is how the state a device reports at open is captured |
| Reports, at acquisition rate | **kept by a run, not by the session journal** | a Logbook is what the user asked for; the always-on file is not the place for an acquisition stream |
| Raw driver records | opt-in | the one thing the run mode chooses, being the one that costs 10 GB/hr |

The rule that separates them is keyed on attribution rather than on the flag,
and it does more than keep the always-on file small: it **classifies**.  A
value a device reports once when the interface opens is kept; a measurement
updating ten times a second is not, in the session journal.  Nobody has to
decide which is which.

It is **silence, not rate** — the first report after a quiet stretch, ten
seconds in the implementation.  A rate cap was tried and removed (user,
2026-08-29): see below.

Two things make that deliberate rather than incidental.  Interfaces already
announce themselves — `onOpen` and `onClose` — so the journal dumps a driver's
subtree when its interface opens, and "the instrument reported firmware 2.31
that day" is recorded on purpose rather than caught in passing.  And restore
replays **request entries only**: for `RXGain`, the gain the user asked for,
never the value the device happened to report.

Settings cost a few hundred KB a day, so there is no reason to make the user
decide.  Observations cost roughly 36 MB/hr at 20 values × 10 Hz — nothing
against 10 GB/hr of raw data, but no longer free.

**A run keeps every observation** — no per-driver checkbox (120 drivers' worth
of them is not a user interface), no rule about the raw stream, and no rate
cap either.  Two things were removed on the way here, both for the same
reason:

- the draft recorded a driver's observations only when its raw records were
  *not* being recorded, since raw plus settings regenerates them.  But
  regeneration depends on the analysis code, so a KAME upgrade can change a
  number that was published, and keeping them costs 0.4% of the raw stream.
- a per-node rate cap then survived as a flood guard.  It cost the one thing
  a provenance file must not lose — **records dropped silently** — to save a
  fraction of a percent, and it is the tier the user explicitly asked for.
  It also produced exactly one bug in its short life.

What bounds the damage instead is the mechanism that was already there and is
already honest: **the ring**.  If a driver floods faster than the drain, the
ring refuses and *counts* what it refused, and the count goes in the file.  A
loss that says where it is beats a decimation that does not.

The always-on session journal is a different question and keeps its own rule
(silence, above) — it is not a tier anybody chose.

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

**No root transaction — but the dump does take a root snapshot.**  An earlier
draft of this section forbade both, and that was wrong about the snapshot
(user, 2026-08-29).  A root Snapshot is ordinary in KAME: `XNodeBrowser`
takes one every time the pointed node changes, on a 500 ms timer, and
`XRubyWriter` takes one for every `.kam` save.  It bundles, so other threads
may lose a CAS once and retry — a cost the application already pays several
times a second whenever the node browser is open.  What is genuinely
forbidden is different and stays forbidden: a **root transaction**, which
would serialise every writer in the tree behind the journal, and a snapshot
taken from *inside* a transaction on a descendant, where the bundling changes
the packet the CAS compares against and the transaction can never commit.

And the dump *needs* it.  Reading values node by node has no consistency cut
at all: each value carries its own serial and the collection as a whole never
existed.  One root snapshot gives one serial, one instant, and an exact
de-duplication rule — an entry whose node and serial match the dumped
payload's serial IS the write that produced the dumped value, so no ordering
comparison is required to drop it.

The walk that *subscribes* is a different operation and stays single-nodal:
it re-runs on every structural change, and enumerating children needs no
cross-node consistency.  So: subscribe with per-node single-nodal snapshots,
then take one root snapshot and dump from that.

### Leaving is a membership question

An earlier version of the capture stage decided a node had gone by testing
whether its `weak_ptr` had expired, and reported nothing at all when a driver
was released — correctly, as it turned out: the script that had created it
still held its nodes, so nothing had been destroyed.  **Detached and
destroyed are two different states**, and the first is the interesting one.
Membership is what changed, membership is what a list announces, and only a
list can announce it.  The record and its statistics survive both.

### The journal must not touch the shape of a teardown

Two rules, both bought with a crash (2026-08-30, closing an NMR setup):

- **Stop before `terminate()`.**  A thread that walks the tree and commits on
  it has no business doing so while the tree is being destroyed.  The journal
  stops and restarts on the empty tree.
- **Never hold a node the tree has released.**  `pushPending` kept a
  `shared_ptr` to released nodes, which let one outlive its neighbours by up
  to a drain interval and changed the order `Node::releaseAll` destroyed
  things in.  Identity is all a release needs: a `weak_ptr` and the address,
  the address checked against the record before it is believed, since a freed
  one can be reused.

The crash itself was a null `m_link` — set in `Node`'s constructor and never
reset, so a null one is freed memory — reached through
`XPointerItemNode::onItemReleased`, which binds its listener by **raw
reference** and commits on `this` from inside it.  That shape is safe only
while the node is guaranteed alive whenever its list can emit a release, and
a teardown is exactly where that guarantee is in question.  It stopped
reproducing once the journal stopped perturbing the order; **which of the two
rules above did it, and how `m_link` came to read as null, are not
established** — the latent shape is still there for the next subsystem that
holds a node for a moment.

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

### The mangled name goes no further than this file

`getTypename()` is `typeid(*this).name()` with everything up to the first
`X` cut off.  For a plain `XLakeShore` that yields `LakeShore`, which is also
what `REGISTER_TYPE(XDriverList, LakeShore, …)` registers — the two agree by
coincidence of spelling, and that coincidence is the only reason `.kam` works
today.  It fails wherever the spelling stops coinciding:

- **template instantiations** — `XListNode<XInterface>` becomes
  `ListNodeI10XInterfaceE`, which matches no registry key;
- **compilers** — MSVC's `typeid().name()` is `class XListNode<class
  XInterface>`, so the same node writes a different string on Windows.  A
  settings file that does not cross platforms is not a settings file;
- **refactoring** — renaming a template parameter changes the identifier of
  every node instantiated from it, though nothing about the node changed.

So **a mangled name is never written as an instruction**.  What identifies a
type, for the purpose of bringing a node back, is *the string it was created
with* — the registry key — and the reliable way to have that string is to
record it at creation rather than derive it afterwards from the C++ type.
The mechanism already exists for exactly this reason:
`XGraphMathTool::setStoredTypename()` was added because template-alias tools
could not otherwise round-trip.  Generalising it — `createByTypename()`
stamping every node it creates — makes the round trip true by construction
instead of true by coincidence.

Where no key is needed at all, none is written: an alias-list child is
navigated, a fixed child exists as soon as its parent does, and an
`XListNode<NT>` child has its element type fixed by the list.  That leaves
the mangled name with no job in the file except documentation ("this was an
`XDoubleNode`"), which a reader may print and must never act on.

A dump therefore resolves every key it writes against the parent list's
registry and **says so loudly** when one does not resolve, rather than
writing a line that will quietly fail to recreate anything.  (`.kam` writes
it regardless today, and the Python loader's `_KamFakeNode` swallows the
failure on the way back in.)

**Implemented** (2026-08-29): `XListNodeBase::createByTypename()` is now a
non-virtual wrapper that stamps the key on whatever
`createByTypename_()` returns, and `XNode` carries it.  Three classes had
each grown their own copy of exactly this — `XGraphMathTool::m_storedTypename`,
`XPythonDriver::m_creation_key`, and the Python math-tool wrapper's — which is
as clear a statement as a codebase makes that the mechanism belonged in the
base.  All three are gone.

`.kam` is unaffected, which is what made it safe to standardise.  Statically:
123 of the 154 registered names are plain classes, where the typeid-derived
string already equals the key by construction (`REGISTER_TYPE(list, Foo, …)`
names the class `XFoo`); the other 31 are template aliases, every one of them
in a math-tool list that was already stamping; and the Python-registered types
already returned their creation key from an override.  So no `.kam` line
changes.

**Does stamping at `createByTypename` catch everything?**  Audited on the
current tree (2026-08-29), and yes:

- 154 classes are registered with `REGISTER_TYPE`, and **not one of them is
  instantiated anywhere by `create<>`, `createOrphan<>` or `new`** — every
  instance comes through a list's `createByTypename`.
- `creator(type)` has exactly four call sites, all of them inside such an
  implementation (`XDriverList`, `XCalibrationCurveList`,
  `XGraph1DMathToolList`, `XGraph2DMathToolList`), and the last two already
  stamp the key.
- Python cannot construct a node at all — no `XNode` subclass has a
  `py::init` — and `exportClass` only inserts a creator into the same
  registry, so a Python driver type also arrives through `createByTypename`.
- `XCalibratedEntryList::createByTypename` ignores the type string and always
  builds an `XCalibratedEntry`, so it behaves as a fixed-element list and
  needs no key either.

That is a fact about today's tree, not a guarantee about tomorrow's, so it
should be **enforced rather than re-audited**: a node sitting under a
registry list with no key recorded is a dump-time complaint.  Which is the
rule above, doing double duty — the day someone creates such a node directly,
the dump says so instead of writing a line that would not come back.

### Attribution

The serial already carries the committing thread in its low 16 bits, so
"who" costs nothing.  **The id alone is not enough, though** — measured on a
live session (2026-08-29): the driver committed as thread 6 and the IPython
kernel as thread 4, and nothing in either number says which is which.  So
each thread declares what it is, once, at its own start
(`XJournal::declareThisThread`): UI, script, or — by never saying anything —
a driver.  Scripts add further context through TLS (the IPython cell label
already exists).  Writes then classify:

- committed by the UI or a scripting thread → a request (intent);
- committed by any other thread → a report (observation).

Two things that run counter to intuition, both observed rather than
reasoned:

- **A script creating a driver is attributed to the UI thread**, because
  `createByTypename` on a list that is not thread-safe during creation is
  dispatched to the main thread by `kame_mainthread()`.  That is the right
  answer for the wrong-looking reason: both are requests.
- **A Python driver commits from the scripting thread**, so its reports
  count as requests.  Thread class cannot fix this; what would is a
  per-callback marker ("this write is inside a driver's record/analyse"),
  which is where this should go if the mis-classification ever matters.

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

## Stage 1: capture only

`kame/xjournal.{h,cpp}`.  It subscribes to every node, records what changes,
who changed it and when, and writes a survey — no file format, no dump, no
replay.  It exists to be measured on a real instrument, because the design
turns on a question no amount of reading the source answers: **which nodes are
written by the UI, by a script, and by a driver.**

Off unless `KAME_JOURNAL` is set in the environment
(`KAME_JOURNAL_REPORT_SEC` sets the report interval, default 60 s).  Not a
flag the capture path tests — subscription IS the switch, so an unset variable
means no listener is ever attached and `Talker::createMessage()` returns
nullptr before allocating anything.  The report lands in
`<AppLocalDataLocation>/journal/capture-<stamp>.txt` and is rewritten in place;
the last one is written as the session ends.

Three things in it are load-bearing beyond the measurement:

- **The walk never bundles.**  Both the per-node snapshot and the transaction
  that attaches the listeners are single-nodal (`Snapshot(node, false)`, the
  same `false` the `trans()` macro passes).  Not because a root snapshot is
  forbidden — the node browser takes one every time the pointed node changes
  — but because this walk **re-runs on every structural change** and needs no
  consistency: it only enumerates children and reads one flag.  Paying for a
  bundle of the whole tree at every driver creation, and again through a
  `.kam` load, is a cost with nothing to show for it.  The dump, which runs
  once and does need a consistency cut, is the opposite case and takes one.
- **Identity is a session-local id baked into the object the talker holds**, so
  the capture path does no lookup at all: it reads the write's serial off the
  committed payload and pushes one record into the ring.  A node reached again
  through a hard link is already subscribed and is skipped, so the count is of
  distinct nodes rather than of paths.
- **Structure is an event, not something to be rediscovered.**  Whether a
  node is in the tree is decided by one thing only: whether a list holds it
  (user, 2026-08-29).  Every other child is made by its parent's constructor
  and lives exactly as long as the parent, so the question never arises for
  it.  So the journal listens to `onCatch` / `onRelease` / `onMove` rather
  than `onListChanged` — they name the node, the list and the index at the
  moment it happens, where `onListChanged` is coalesced to one per
  transaction and says only that *something* changed.  A caught node's
  subtree is subscribed, and a released node's subtree marked off, on the
  journal's own thread; nothing but a ring push happens inside the
  committing thread's commit.  That hand-over **wakes the thread**, rather
  than waiting for its next pass: the gap between a node joining the tree
  and being subscribed is a gap in the record, so it is a condition signal
  and not a poll interval.
- **Entries are timestamped where the write happens**, not where they are
  drained.  Stamping at the drain would make the journal's time resolution
  the drain period — an answer to "what was it at 3:14" no better than the
  reader's own laziness.  The drain interval is then only how long records
  may sit in a ring that holds 8192 of them.
- **A full walk remains, but as a measurement rather than the mechanism.**
  Every 30 s it counts what the events failed to announce: arrivals the
  sweep found first, and departures it noticed first.  Both should be zero,
  and if they are on a real instrument then the rule above is proven and the
  sweep can go.  (`Node::insert` / `Node::release` are framework operations
  on *any* node; only `XListNodeBase` turns them into signals.  That is the
  gap the counter measures.)

### The alias guard that never fired

`XNode::getTypename()` strips everything up to and including the first `X` of
the mangled name, so `XAliasListNode<XInterface>` arrives as
`AliasListNodeI10XInterfaceE`.  Both existing tests for it — in
`xrubywriter.cpp` and in the survey script this stage replaces — compare
against `"XAliasListNode"` and therefore **never match anything**; and the
subclasses (`XInterfaceList`, `XScalarEntryList`, `XChartList`,
`XScriptingThreadList`, ...) do not carry the template's name at all, so no
string test could have worked.  Hence `XListNodeBase::isAliasList()`, a
predicate the classes answer themselves.

That is also why the node count below is an over-count: the survey walked the
alias lists after all, and counted every hard-linked node once per parent.
(The same dead check makes `.kam` emit `create()` for alias children, which
the loader silently ignores and the owning parent then restores correctly —
so it round-trips by luck.  Fixing that changes `.kam` output and wants its
own decision.)

## Files

One file, self-contained: **the dump is the head of the journal**.  That
removes the pairing problem entirely, makes every rotated segment
independently readable, and leaves the state at session start on disk even if
the session dies a second later.

| File | Extension | Contents |
|---|---|---|
| Settings snapshot (legacy) | `.kam` | as today; readable for ever, written until the dump replaces it |
| Journal | **`.kamj`** | header record, dump, then one JSON object per line — gzip inside, as `.docx` is a zip |
| Raw stream | **`.kamb`** (was `.bin`) | unchanged format; `.bin` keeps loading for ever |

JSON Lines because the two operations that matter — `diff run1.kamj run2.kamj` and `grep` — then work with no tool at all, and because a
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

### Saving is not appending — three things, none overloading another

A journal that keeps being written to the file the user "saved" is the most
natural thing to *use* and the wrong thing to call **Save**.  Ordinary Save
semantics say: the file is what you saw at that instant, closing without
saving discards, and copying the file gives you the whole of it.  An
appending file breaks all three — there is no moment at which it is what the
user saw when they pressed Save, and a copy taken mid-run is a truncation
rather than a document.

KAME already has the right idiom for a file that keeps being written, three
times over: a path field, a browse button and a **Write** toggle — the text
writer, the logger, and `XRawStreamRecorder`.  A continuously written journal
is one of those, not a File-menu item.  So:

| | What it is | Where it is said |
|---|---|---|
| Session journal | always on, never chosen | managed directory; no UI beyond an indicator and "open folder" |
| Run journal | the copy beside the user's data | destination + Write toggle, next to the raw-stream recorder, basename following it |
| **File → Save** | a checkpoint: dump only, no entries | ordinary Save semantics, unchanged |

The third is what replaces `.kam`: a journal whose head is a dump and whose
body is empty *is* a settings file, finished and portable, read by the same
reader as any other journal.

And the wish behind the question — "what I saved should stay up to date" — is
already granted by the first row rather than by changing the third.  The
state is not lost if the user forgets to save; that is the whole point of an
always-on journal, and it is why `.kam`'s status changes from "save often or
lose your setup" to a checkpoint.

### What a run records: three tiers, one combo

It is tempting to call this "include runtime nodes", and that is the one name
to avoid.  The flag does not decide what goes in — attribution does, and the
two disagree on real nodes (`RXGain` is `runtime == true` and is a setting;
`ODMR2D/Average` is `runtime == false` and is written by its driver).  A
runtime node the *user* sets is a request and is journaled either way.

What is left to choose, once observations are always kept, is one thing: **is
the raw stream recorded too** — the only part that costs 10 GB/hr.  That
makes the choice a ladder of magnitude rather than a matrix of switches:

| `m_journalMode` | files | holds | order of cost |
|---|---|---|---|
| **`Logbook`** | `run042.kamj` | the dump, and everything the instruments reported | ~11 MB/hr, measured |
| **`Logbook + raw`** | `+ run042.kamb` | and the raw records behind them | ~10 GB/hr |

There is no "settings only" tier, and the one that existed was removed
(user): **writing the settings once is `File → Save`.**  A `Write` switch
that wrote a file the instant it was pressed and then sat there doing nothing
is not something a user can be expected to make sense of — a switch that
records is one thing, an action that saves is another, and the tier list is
for the first only.  `Logbook` is the default, the cheaper of the two, so a
KAME nobody has configured does not imply that pressing Write means 10 GB/hr;
the mode is non-runtime and therefore saved, so a rig that records raw data
says so once.

`File → Save` accordingly offers `.kamj` beside `.kam`, and writes what it
has always written: **one file, one instant, finished when it returns**.  In
journal form that is a head with no body — which is what a settings file is.

A *logbook* is exactly what the middle tier is — what was set, and what the
instruments said, written down as it happened — and the word survives being
read in ten years by someone who never used KAME, which a label stored in a
file has to do.

**One combo rather than checkboxes** (`m_journalMode`, an `XComboNode`).  The
tiers are cumulative, so they are one choice, not two independent switches;
and a combo is stored by its **label**, which is the same reason `.kam`
stores one that way — a journal then says in words what the run was set to
record, rather than in booleans whose meaning depends on the version of the
code that wrote them.  The labels above are therefore load-bearing and fixed
now.

The dropped fourth state is worth naming so it is not reinvented: "raw
records, but drop the reports".  It saves 0.4% of the run and can lose the
number that was published (user).  Not an option; a trap.

**The pane says two things, so it has two parts** — an earlier version put
the session journal's path into the run's filename field, which made "always
on" visible at the price of a field that meant something different depending
on what was in it (user: the semantics were hard to follow).  Now:

    [x] Session journal          ~/Library/…/journal/session-…kamj
    Run  [ run042            ] [...]
    [ Logbook + raw ▾ ]   2.4 kB/s  51.4 kB      [ ] Write

The first row is the always-on file: a switch, and where it is going, shown
because a user who wants to find or copy it should not have to know where
KAME keeps its own files.  The rest is **this run** — its name, how much of
it is kept, what it is costing, and the switch that marks where it begins and
ends.  `Mode` and `Recording` stay disabled until a run is named, and `Mode`
also while one is open, since its tier is latched in the file's header.

Separating them also deleted a bug: "is there a run to configure" used to be
"is the field something other than the session path", a comparison that an
extension mismatch could get wrong.  Now it is "has one been named".

**Why `Recording` is not redundant.**  It looks like it could go — the mode
could gain an "Off", or naming a file could be the intent to record — and
both lose something.  A tier is a *preference* that outlives a run
(`Mode` is non-runtime and is saved; `Filename` and `Recording` are not), so
folding "off" into it means saving "off" and forgetting next session that
this rig records `Logbook + raw`.  And a name that starts recording cannot be
prepared in advance, nor stopped without destroying it.

The real answer is that the switch is not a mechanism at all: the journal is
always capturing, and the session journal is always being written.  What
`Recording` marks is **where a run begins and ends** — the only thing that
turns a continuous session into "this file is run042".  A design with no such
switch has no runs, only a session.

Once the field shows the session path, one might go further and say that
*changing the name* is what starts a run, leaving no switch at all.  It ends
at the same place: a run needs an END as well as a start, and a filename
cannot express "stopped, but still called run042" — reverting the name to
stop would destroy the name, and a name typed in advance would start a run
before its time.  So the switch stays, and what changes is when it means
anything.

The combo is where a *human* chooses; the mechanism stays where it is.
`XRawStreamRecorder`'s `Recording` node keeps being what turns the raw stream
on, driven by the combo, so scripts that set it directly go on working.

**Does `XRawStreamRecorder` still need to exist?**  The class does: somebody
has to subscribe to every driver's `onRecord`, hold the file mutex and write
the gz, and its base `XRawStream` is shared with the *reader*, which is
staying.  What it no longer is, is a place where anything is decided — its
`Filename` and `Recording` are now derived from the journal's.

**Done** (2026-08-29).  It was cheap after all: those two nodes are
`runtime == true`, so no `.kam` file references them; nothing in `kame/` or
`modules/` used `XMeasure::rawStreamRecorder()` beyond constructing it; and
the user knows of no script that reaches for `Root()["RawStreamRecorder"]`.
So it is now `/Journal/RawStream`, created and owned by the journal, and the
duplicate control surface is gone.

**Demote it; do not dissolve it.**  Folding the raw writing into
`XJournalRecorder` or `XJournal` looks like the tidier end state and is worse
on three counts:

- **Two disciplines that must not be confused.**  The journal's capture path
  runs inside every commit and is lock-free and allocation-light by
  construction; the raw writer takes a file mutex and does I/O from inside a
  listener.  In one class, nothing stops a later edit from taking the
  writer's mutex on the capture path — the exact deadlock the "never hold a
  plain mutex across a Snapshot/Transaction" rule exists for.  The separation
  is a safety property, not tidiness.
- **The friendship would widen.**  `XPrimaryDriver` makes both the recorder
  and the reader `friend`s, for the raw data.  Moving the writing into the
  journal means making *the thing that watches every node* a friend of every
  primary driver, where today it is the thing that writes the raw stream.
  Privileged access should stay narrow.
- **The base survives either way.**  `XRawStream` — the gz handle, the driver
  list, the mutex, the filename node — is shared with
  `XRawStreamRecordReader`, which is staying.  Dissolving the writer leaves
  that base with a single user, which is worse than what exists now.

What the demotion buys is exactly what is wrong today and nothing more: one
control surface instead of two, both files opened and closed at one moment,
byte accounting in one place, and `/RawStreamRecorder` out of the tree.

A script can still reach `/Journal/RawStream/Recording` and get a `.kamb`
with no `.kamj` beside it.  It is one node deeper and no longer beside the
journal's own switch, which is as much discouragement as it needs; if it ever
matters, the journal should notice and open a run of its own rather than the
two disagreeing.

Two things stay out of the user's hands.  Requests are never dropped, whatever
the setting, and neither are reports that change rarely — that is how "the
instrument said firmware 2.31 that day" gets recorded.  And the always-on
session journal always caps: it has to stay a few hundred KB a day, so the
mode is a property of a run, not of the session.

There is no cap to configure.  The `peak/s` column of the stage-1 survey
still earns its place — it says what a Logbook will cost per hour, and
whether the ring can keep up — but it is a sizing number now, not a tuning
one.

**What a Logbook line costs, measured** (a TestDriver run, 2313 entries):

| | per line | share |
|---|---|---|
| `"ts"` (readable time) | 39 B | 33% |
| `"x"` (exact double, base64) | 19 B | 16% |
| `"s"` (serial) | 15 B | 13% |
| `"v"` (value as text) | 15 B | 13% |
| `"c"` (request/report) | 13 B | 11% |
| `"t"`, `"id"` | 16 B | 14% |
| **total** | **118 B** | |

At 144 readings a second that is 17 kB/s *produced* — and **3 kB/s on disk**,
because the JSON is repetitive and gzip removes nearly all of it.  Which is
the useful lesson: **trimming the format buys almost nothing.**  Dropping the
readable timestamp for an epoch number and removing `"c"` takes 27% off the
produced bytes and **3% off the file**.

What does cost is the one field that is not repetitive: **the exact double is
half the compressed file** (removing it: 50% smaller on disk), since base64
of a measured value is pure entropy.  It stays anyway — `to_str()` on a node
with a display format renders `-0.1258[K]`, four digits and a unit, so
without the exact bytes an observation is not recoverable at all, which is a
worse failure than a file twice the size.

### One extension, and the compression inside it

`.kamj` is gzip, and its name does not say so — the same choice `.docx`,
`.jar`, `.epub` and `.nb` make.  The compression is part of the **format**,
not something done to the file afterwards, and `foo.kamj.gz` says the
opposite: a text file that someone has compressed.

An earlier draft kept the `.gz` on the grounds that `zcat` refuses a file not
named that way.  That argument does not survive contact with who would type
it (user): anyone reaching for `zcat` will rename the file or reach for
`gzip -dc`, and `zgrep` works on it either way.  Office formats do not append
`.gz`, and neither does this.

The reader **sniffs** rather than trusting the name — two bytes, `1f 8b` —
so a journal someone has unpacked to edit by hand still opens.  Editing one
by hand is a thing the format is meant to allow, so it should not depend on
whether the file was re-packed afterwards.

### Where they live

`QStandardPaths::AppLocalDataLocation` — `~/Library/Application Support/kame`
(macOS), `AppData\Local\kame` (Windows), `~/.local/share/kame` (Linux) — with
`journal/` and `log/` under it.  **Local**, not `AppDataLocation`: on Windows
that would be `Roaming`, and journals and logs are machine-local and not
small.  KAME already sets `applicationName("kame")` and no organization, so
the path has no vendor component.

The always-on session journal lives there, so provenance exists even when the
user has started no recording — **it is not optional, and that is the point**
(user, 2026-08-29): a `.kam` you must remember to save is exactly what this
replaces.  `session-<stamp>.kamj` is opened as KAME starts, headed by the same
dump a run gets, and it names each run as it begins and ends, so either half
leads to the other.

What separates it from a run's Logbook is not the format but **how much of
the acquisition stream reaches it**.  A request is never dropped.  A report
is kept only when it follows a silence — ten seconds, in the implementation
— which is precisely the state a device announces at `open` or when
something actually changed, and precisely not the stream a driver produces
while measuring.  That is what keeps a day's session in the hundreds of KB
the design assumed, without anyone choosing anything.

**Measured, and it decided the format** (2026-08-29): a session journal of an
ODMR rig was 608 KB after a few seconds, of which the dump was 592 KB and the
entries 30 — the silence rule was doing its job, and the *dump* is the cost.
3051 nodes at ~190 bytes each.  Gzipped it is 62 KB, so the file is gzip
throughout rather than plain for settings and compressed only for
observations: even the smallest tier is a full dump, and no rig's tree is
small.  `zcat` and `zgrep` keep it as readable as a provenance file has to be
in ten years.

The user's conclusion from the same measurement, and it is right: **it must
be possible to switch off**.  `Journal/SessionJournal` (saved, default on)
does that.  A dump is not free, and a background writer nobody can refuse is
impolite whatever its size.

**The switch is on probation, not permanent** (user): it stays while there is
any doubt, and goes when there is none.  What would settle it is measurable,
so it is worth writing down rather than deciding by feel:

- a session costs tens of KB, not hundreds, across real sessions rather than
  one;
- `dropped (ring full)` stays at zero on a rig that is actually acquiring;
- the opening dump does not show up in startup time (the survey reports the
  walk);
- and **a retention policy exists** — this is the real gate.  Always-on with
  no policy means files for ever, and a KAME restarted twenty times in an
  afternoon writes twenty near-identical dumps.  Keeping the last N sessions,
  or coalescing a dump that is identical to the previous one, is the missing
  piece.

When those hold, the switch leaves the pane; the node can stay for the rig
with no disk to spare, and for scripts.

(Retention is still not handled: one file per session, kept for ever.  They
are small now, but "for ever" wants a policy eventually.)  **When a recording is started, the journal is
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

### The user names the run, not the binary

The natural inversion (user, 2026-08-29): the file setting that exists today
belongs to `XRawStreamRecorder`, and it should belong to the **journal**, with
the raw stream taking the same basename — `run042.kamj` beside
`run042.kamb`.

The reason is not tidiness.  The journal always exists and the raw stream is
optional, so naming the optional one is backwards; and of the two, only the
journal is interpretable alone.  A lone `.kamb` cannot even say what wrote it
(see below), while a lone `.kamj` is a complete record of the session that
also names its data file.

One thing the inversion must not lose: **a separate path for the raw stream,
auto-filled from the journal's**.  10 GB/hr belongs on a scratch disk while
the few hundred KB belong next to the notebook, and that is a real way people
work.  So: one name for the run, one checkbox for "record raw records too",
and a raw path that is derived until someone overrides it.

### The raw stream does not say what it is

Worth recording while renaming it: the raw stream has **no header at all**.
It is a gzip of `[allsize u32][sec i32][usec i32][name\0\0][data][allsize
u32]` records, with no magic, no version and no writer identification — the
extension is the only clue there has ever been, which is an argument for a
distinctive one and a stronger argument for a header.

Adding one is compatible, and this is the moment: the first field of an old
file is a record length — a few hundred to a few million — so a magic word
(`KAMB`, 0x424D414B ≈ 1.1e9 as a little-endian `uint32`) cannot be confused
with one.  A reader that peeks four bytes accepts both, and new files gain a
version and a session UUID to pair them with their journal.  Cosmetic rename
alone, without this, leaves the file as anonymous as it is today.

### Copying a journal that is still being written

Windows makes this a real question, since sharing there is decided by the
*writer* at open time and refused for everyone else otherwise.  The answer is
that it works as long as nothing takes it away, and KAME already relies on
that: `XRawStreamRecorder` uses `gzopen`, the text writer and logger plain
`std::ofstream`, and both reach the CRT's default share mode, which denies
nothing.  A journal opened the same way can be copied by Explorer or
`robocopy` mid-run.  (Stated from the CRT's documented default rather than
from a test — worth confirming once on the Windows machine, with the journal
running.)

Three consequences that are ours to get right, not the OS's:

- **A copy is only as current as the last flush.**  So flush at a boundary —
  the end of a gzip member, or a whole line for the plain file — and never
  leave a half-written line in the buffer.
- **The reader must tolerate a truncated final line.**  A copy taken
  mid-write ends wherever it ends, and so does a file whose session was
  killed: the same shape, so one rule covers both.  With gzip *members* per
  block, everything up to the last complete member stays readable, which is
  the property whole-file gzip does not have.
- **On Windows the file cannot be deleted or renamed while KAME holds it
  open** — the CRT does not ask for `FILE_SHARE_DELETE`, and POSIX
  intuitions do not transfer.  Rotation, "save as" during a run, and any
  cleanup on exit have to close first.  Anything that ever opens the journal
  through `CreateFile` directly must pass all three share flags.

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

## What can read these files today

Worth stating plainly, since the writer has run ahead of the reader:

| File | Read by KAME | Read by anything |
|---|---|---|
| `.kamb` | **yes** — `XRawStreamRecordReader`, unchanged: the format is byte-identical to the `.bin` it renames, so old files and new ones are the same file with two names.  Its dialog offers both, and always will | — |
| `.kamj` | **the dump, yes** — `File → Open Measurement` applies it exactly as it applies a `.kam`; the entries after it are not read yet | `zcat`, `zgrep`, `zdiff` — which is not a placeholder but half of why the format is JSON Lines |
| `.kam` | yes, as ever | a Ruby interpreter |

**Two doors, and the file decides which half you get** (user, 2026-08-29):

- **`File → Open Measurement`** on a journal applies its **dump and stops**.
  The dump is what the tree was at one instant, so applying it is the same
  act as loading a `.kam` — which is why it belongs on the same menu item
  rather than on a new one, and why it is written in `xpythonsupport.py`
  beside `loadKam`, reusing the main-thread dispatch for lists that cannot
  be created off it and the `_KamFakeNode` tolerance for a tree that has
  moved on.  Values on device-reported (runtime) nodes are skipped, exactly
  as `.kam` comments them out: they are outputs, and writing them back would
  fight the drivers that produce them.
- **The record reader** is where the entries belong: opened beside the raw
  stream, it can restore the settings of the moment before re-analysing the
  records of that moment.  That is the hole this whole design exists to
  fill, and it is the next stage.

Measured against a real run (`test2.kamj.gz`, 50 nodes): 12 values applied,
26 runtime nodes skipped, nothing unresolved — and on a KAME where the driver
does not exist yet, one node created from its recorded type key, after which
its whole subtree resolves.

The `.kamb` magic header discussed above is also still unwritten, so a raw
file remains anonymous: the extension is the only clue to what wrote it.

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

### What overrides the journal, and how it is decided

Nothing needs to reason about authority.  Replay remembers, per node, the
serial its own write produced, and applies the next entry for that node only
while the node still carries it.  If the serial has moved, someone else wrote
— and whoever that was, they are more current than a recording.  It is a
compare-and-set, the same optimistic control the framework already runs on.

Everything intended falls out of that one test.  A user who edits a node keeps
it, because the edit moves the serial.  A driver writing back its own progress
keeps it, for the same reason, so the internal state it maintains is never
contradicted by a value read out of a file.  A node nobody has touched still
carries what replay last put there, and the next entry applies.

Note that serials cannot be compared *between* sessions — the counter is
thread-local and advances past whatever state it observes, so a recorded
serial means nothing here — and wall clocks are no help either, a recording
being always in the past.  The question is not "which is newer" but "has
anyone written since I did", and that is answerable exactly.

Attribution is then needed only to decide **what** to restore (a request, not
a report), never to decide whether an override happened.

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

## `.kam` is on its way out — kept for now

The decision (user, 2026-08-29): **`.kam` is to be retired eventually, and
maintained until then.**  What retires is the **writer**, not the reader.
Files that exist must keep loading essentially for ever — the requirement
stated earlier for the journal applies here just as much: backward
compatibility is needed, forward compatibility is not.

The baseline argument does not save the format, only the *idea* of a
baseline: a diff stream without one means nothing, which is why the dump sits
at the head of every journal.  Once the dump can do everything a saved `.kam`
is used for, the format has no remaining job.

What the dump has to absorb first — this is the retirement checklist:

- **Portability.**  Users carry settings between rigs and to colleagues.  A
  session journal as such cannot do that, but a journal whose head is a dump
  and whose body is empty *is* exactly a settings file.  "Save" becomes
  "write a journal with no entries after the head", and the same reader opens
  both.
- **Readable and editable by hand.**  `.kam` is text, and people edit it.
  JSON Lines is text too, `diff`s and `grep`s better, and the dump's neutral
  path → value view is easier to edit correctly than executable Ruby whose
  statements have to be run in order.
- **Version skew.**  `.kam` loading tolerates a tree that has changed since
  the file was written — the Python loader's `_KamFakeNode` silently absorbs
  nodes that are absent or of an unexpected type.  The dump reader needs the
  same tolerance, and should be able to *report* what it could not place
  rather than only swallowing it.
- **Structure, not just values.**  Everything `x.last.create(type, name)`
  does today: create for a list child, navigate for an alias-list child or a
  fixed child, and position where order is the meaning.

Until all four hold, `.kam` stays as it is: no format changes, no behaviour
changes, no new dependencies on it.

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
keyed by path, and it descended into alias lists — and its guard against the
latter could not have worked, for the reason given above.  Walking them
inflates the node count, since a hard-linked node is counted once per parent,
and manufactures the very ambiguity KAME avoids by never addressing those
nodes that way.  The 7103 is therefore an upper bound.  Stage 1 measures the
same things from inside the framework, where identity is the node itself
rather than its path; the survey script it replaces has been removed rather
than left around to be trusted again.)

## Deferred

- Block framing plus an index for the raw stream, so `gzseek` stops being
  linear.  Independent of this work, and increasingly needed at 10 GB/hr.
- `NODE_NOT_JOURNALED`, only with an audit checker.
- Migrating the scattered `~/.kame_*` dotfiles under the managed directory.
- `xrubywriter.cpp`'s two dead name tests (alias lists, and `XListNode`'s
  "no typename wanted"), which now have a predicate to use.  It changes what
  `.kam` files look like, so it is not a side effect of this work.
