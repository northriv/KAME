# TLA+ ↔ C++ Fidelity Dossier (Pre-Submission)

> **Status line.** Five linters produced **~264 findings** across 3 prose docs (VERIFICATION.md,
> proof_semantics.md, parameterized_cutoff.md), 7 EN slide decks (+ identical JA mirrors), 11 TLA+
> specs, the C11 GenMC test family, and 2 C++ source files (`kamepoolalloc/atomic_smart_ptr.h`,
> `kamestm/transaction_impl.h`). Headline counts: **~38 stale references** (line-number drift in
> the docs + both slide decks), **2 ghost/missing C++ symbols** (`m_priority_tidstamp`,
> `Transaction::priority_tidstamp()` — verified ABSENT from all C++ source) plus 1 informal-alias
> symbol (`negotiate()`/`m_link->negotiate()`) and 1 non-existent spec-cited symbol
> (`snapshotSupernode`), **0 memory_order mismatches** (every doc-claimed RC11 order confirmed
> against current source), and **~14 ghost spec-actions + ~78 orphan spec-actions** in the
> action-completeness sweep. The protocol-level correspondence (every TLA+ action → a real, current
> C++ function → a real C11 GenMC function) is **sound**; what is broken is almost entirely
> **citation hygiene** (line numbers drifted when `atomic_smart_ptr.h` was relocated/refactored to
> 2154 lines and `transaction_impl.h` grew) plus a **single substantive symbol error** (the
> priority-tag mechanism is misnamed throughout the docs/slides).

---

## §0 — How to use this dossier (READ FIRST)

This dossier was assembled by **AI linters**. Their job, and the limit of their authority, is:

- **AI did (and may assert):** *locate* the cited C++/C11/TLA+ symbols, *check for drift* (does
  `file:line` still point at the named code? does the named symbol exist verbatim? does the
  memory_order in the source equal the order claimed in the doc? is every spec `Next` disjunct
  documented, and is every documented action a real spec action?). These are mechanical,
  grep-checkable facts and are reported below with verbatim `file:line` citations.

- **AI did NOT (and must NOT) assert: semantic fidelity.** Whether each TLA+ action is a *faithful
  abstraction* of the C++ it cites — i.e. whether the spec models the right thing, whether the
  abstraction gaps are sound over-approximations, whether the RC11 reasoning is correct, and
  whether the headline verification numbers mean what the paper says they mean — is a **judgment**,
  not a lookup. **The author must personally certify it and sign §6.**

> **Paper-credibility note.** For a submission, an AI asserting "the spec faithfully models the
> implementation" is not a credible warrant — the AI can confirm a symbol *exists* and a line
> *matches*, but it cannot vouch that the abstraction is *sound*. If the abstraction soundness is
> attributed to the linters, a reviewer is right to discount it. The fidelity claim in the paper
> must rest on the **named author's** certification (§6). Use §1–§5 to make the mechanical layer
> airtight; then the author certifies §6 in their own name.

**Workflow:** (1) clear every box in §1 — stale/missing refs first; (2) verify §2–§4 are clean
after the §1 fixes regenerate the cited lines; (3) walk the §5 table and fill the author column;
(4) sign §6.

---

## §1 — ⚠️ Stale / missing references — FIX FIRST

Every `[STALE]` / `[MISSING]` / `[MISMATCH]` from the drift-docs and drift-slides linters, as a
checklist. Each item: where it is cited → what is wrong → the corrected target. The `[OK]` items
are collapsed into the counts at the end of this section.

### 1A. Ghost / missing C++ symbols — HIGHEST PRIORITY (substantive, not just line drift)

- [ ] **`m_priority_tidstamp` does NOT exist in C++.** Cited at `VERIFICATION.md:315`,
  `proof_semantics.md:44`, `proof_semantics.md:97`, and in `slides_layer2_LLfree.html`
  Slide 2/3 (as `Linkage::m_priority_tidstamp`). **Verified absent** from all of `kamestm/` and
  `kame/` C++ source (appears only in `.md`, slides, and one `.tla` comment). The real per-Tx
  priority mechanism is: `s_privileged_tidstamp` (inline-static slot, `transaction.h:593`),
  `try_register_privileged_tidstamp` (`transaction.h:611`), `release_privileged_tidstamp`
  (`transaction.h:614`, called `transaction.h:1835`), `i_am_privileged_now`
  (`transaction.h:646`), `fair_mode_blocks_me` (`transaction.h:634`). **Fix every doc/slide that
  attributes the TLA+ `priorityTag` to `m_priority_tidstamp`.**
- [ ] **`Transaction::priority_tidstamp()` does NOT exist.** Cited at `proof_semantics.md:44,97`
  and `slides_layer2_LLfree.html` Slide 2/3 as the C++ counterpart of TLA+ `MyTag`. No such
  method. The TLA+ `MyTag` corresponds to the registered privileged tidstamp via
  `try_register_privileged_tidstamp` / `i_am_privileged_now`; the per-node tag claim is
  `Snapshot::tag_as_contender` (`transaction.h:1630`). The `CanProceed` disjuncts
  (`m_priority_tidstamp.load()==0` / `==priority_tidstamp()`) must be re-expressed against
  `i_am_privileged_now` + `fair_mode_blocks_me`.
- [ ] **`snapshotSupernode()` does NOT exist as a symbol.** Cited at `slides_layer2_en.html`
  Slide 7 (and JA mirror) as `transaction_impl.h:696-755` for the `UnbundleWalk` action.
  **Verified absent** from `transaction.h` / `transaction_impl.h` / `transaction_definitions.h`.
  The `bundledBy`-chain walk is `reverseLookupWithHint()` (`transaction_impl.h:1490`),
  `forwardLookup()` (`transaction_impl.h:1544`), `reverseLookup()` (`transaction_impl.h:1585+`).
  Rename the slide's C++ citation.
- [ ] **`negotiate()` / `m_link->negotiate()` is an informal alias, not a verbatim symbol.**
  Referenced at `proof_semantics.md:44,97` (and §4 prose). There is no `Linkage::negotiate()`
  method; the real entry point is `ScopedNegotiateLinkage<XN>::_negotiate()`
  (`transaction_neg_impl.h:608`); `negotiate` appears only in comments and in the
  `_negotiate`/`negotiate_internal` names. Either rename to `_negotiate` or mark it explicitly as
  an informal alias in the text.

### 1B. `atomic_smart_ptr.h` line-range drift — VERIFICATION.md §2 table (all 9 rows stale)

The file was **relocated** `kame/atomic_smart_ptr.h` → `kamepoolalloc/atomic_smart_ptr.h` (per
CLAUDE.md) and **refactored/expanded to 2154 lines**; every out-of-line def now lives ~500–1100
lines below the cited range. Corrected line numbers **verified this run**.

- [ ] `VERIFICATION.md:215` `acquire_tag_ref_()` cited `1058-1108` → decl `1067`, **def `1632`**
  (1058-1108 is unrelated TaggedPtr/state-comment code).
- [ ] `VERIFICATION.md:216` `load_shared_()` (bulk) cited `1116-1128` → decl `1064`, **def `1684`**
  (1116-1128 is a comment block).
- [ ] `VERIFICATION.md:217` `release_tag_ref_(pref, T)` cited `1158-1206` → decl `1074`,
  **def `1730`**.
- [ ] `VERIFICATION.md:218` `compareAndSwap_()` (legacy 6-phase) cited `550-650` → **symbol no
  longer exists** as a function def; that range is `local_shared_ptr` ctors / `_use_count_`. Only a
  comment mention remains (e.g. `atomic_smart_ptr.h:1711`). See §2 for the replacement
  (`compareAndSet_impl_`).
- [ ] `VERIFICATION.md:219` `local_shared_ptr::swap(asp&)` cited `628-649` → that range is
  `_use_count_()`; real `swap(atomic_shared_ptr<T>&)` decl `717`, **def `2085`**.
- [ ] `VERIFICATION.md:220` `compareAndSet_impl_<SCOPED>` cited `1240-1450` → decl `1096`,
  **def `1811`** (cited range is mid-`scoped_atomic_view` body).
- [ ] `VERIFICATION.md:221` `scoped_atomic_view` ctor cited `598-700` → class decl `1139`,
  **explicit ctor `1169`/`1200`** (cited range is `local_shared_ptr` territory).
- [ ] `VERIFICATION.md:222` `scoped_atomic_view` dtor cited `730-845` → **`~scoped_atomic_view()
  noexcept { release_(); }` at `1271`**.
- [ ] `VERIFICATION.md:223` `local_shared_ptr::reset()` cited `433-444` → that range is
  `ref_traits_auto` trait code; real out-of-line **`reset()` def `1597`** (decl `720`).

### 1C. `transaction_impl.h` line-range drift — VERIFICATION.md & proof_semantics.md

- [ ] `VERIFICATION.md:683` `Packet::checkConsistensy` cited `transaction_impl.h:870-871` → **def
  `1001`** (870-871 is a `negotiate_internal` livelock-probe comment).
- [ ] `VERIFICATION.md:503` `reverseLookup` self-return cited "line 1440" → actual self-return at
  **`transaction_impl.h:1593`** (`if(&superpacket->node()==this) foundpacket=&superpacket;`);
  `reverseLookup` defs at 1590/1665/1675; nothing relevant at 1440.
- [ ] `proof_semantics.md:218,237,299` / §9 cite `tags_successful_cas` at
  `transaction_impl.h:2526-2528` → that range is now a DISTURBED-return/comment block; **real call
  site drifted to `2688`** (child) / `3019` (unbundle). (Symbol exists; line drifted.)
- [ ] `proof_semantics.md:231,232,233` / §9 audit cite `transaction_impl.h:2389` (Phase1 pre-CAS),
  `2397` (`tag_as_contender` on fail), `2407`, `2431` (ScopedNegotiateLinkage eager) → at HEAD
  2389 is a `ScopedNegotiateLinkage scope(...)` ctor; 2397/2407/2431 are unrelated comment /
  `if(!scope) continue;` lines. The Phase-1 superfine logic exists but the line numbers drifted.
- [ ] `proof_semantics.md:241` / §9 cite the Phase-3 child-CAS loop at
  `transaction_impl.h:2480-2506` → at HEAD that range is a
  `switch(status){case SUCCESS/DISTURBED}` / `child_scope.commit()` block, not the documented
  `for(i) compareAndSet(subwrappers_org[i], bundled_ref)` loop. Drifted.

### 1D. Path note (not a line, but stale)

- [ ] `VERIFICATION.md:29,195` cite the protocol source as `kame/atomic_smart_ptr.h`. The file is
  actually `kamepoolalloc/atomic_smart_ptr.h` (relocated per CLAUDE.md). Symbols are real; only the
  path prefix is stale. **Fix the `kame/` → `kamepoolalloc/` path everywhere it appears as a source
  prefix.**

### 1E. `slides_layer1_en.html` (+ JA `slides_layer1.html`) — atomic_smart_ptr.h ranges all stale

Every numeric `atomic_smart_ptr.h` range in the Layer-1 deck is stale (file grew). Corrected
targets are the same as §1B.

- [ ] Slide 2 `acquire_tag_ref_` cited `462-488` → **def `1632-1681`** (462-488 is `ref_traits_auto`
  boilerplate).
- [ ] Slide 3 `scan_: load_shared_()` cited `494-503` → **def `1684-1699`** (494-503 is
  `ref_traits` comment/Refcnt-trait code).
- [ ] Slide 4 `compareAndSwap_: 6 Phases` cited `556-603` → **no `compareAndSwap_` def exists**;
  556-603 is `ref_traits<T,1>/<T,2>` specializations. Relabel to `compareAndSet_impl_`
  (`def 1811`) / public `compareAndSwap()` wrapper (`2038`). Residual comment at `1711`.
- [ ] Slide 7 `acquire_tag_ref_()` cited `1058-1108` → points at decls/comments; **def
  `1632-1681`** (imprecise).
- [ ] Slide 7 `load_shared_()` bulk-transfer cited `1116-1128` → comment text only; impl in
  `scoped_atomic_view::promote/release_` ~`1373-1436` and `load_shared_()` `1684-1699`.
- [ ] Slide 7 `release_tag_ref_(pref, T)` cited `1158-1206` → inside `scoped_atomic_view` body;
  **def `1730-1776`**.
- [ ] Slide 7 `compareAndSwap_() 6 phases (legacy)` cited `550-650` → **removed**; region is
  `ref_traits` specializations.
- [ ] Slide 7 `compareAndSet_impl_<...>` cited `1240-1450` → inside `scoped_atomic_view` body;
  **def `1811-2034`** (decl `1096`).
- [ ] Slide 7 `scoped_atomic_view ctor/CAS/dtor` cited `598-845` → those lines are
  `local_shared_ptr` ctors / `atomic_shared_ptr` member decls; **class `1139-1532`** (fwd-decl
  `363`).
- [ ] Slide 7 `local_shared_ptr::reset()` cited `433-444` → `ref_traits_auto` trait code; **def
  `1597-1614`** (uses `unique()`/`decAndTest()` at 1606-1611, **not** a literal `fetch_sub(1)`).

### 1F. `slides_layer2_en.html` (+ JA `slides_layer2.html`) — transaction_impl.h ranges all stale

- [ ] Slide 7 `snapshot()` cited `transaction_impl.h:842-870` → **defs `2038`/`2053`** (842-870 is
  TLS diagnostic-state globals + `LivelockProbe::state()`).
- [ ] Slide 7 `bundle()` cited `1077-1171` → **def `2354`** (1077-1171 is PacketWrapper ctors +
  `insert()`).
- [ ] Slide 7 inner-bundle of `bundle()` cited `1100-1171` → inner/recursive bundle is
  `bundle_subpacket()` at **`2226`** (1100-1171 is inside `insert()`).
- [ ] Slide 7 parent/grand-scope `commit()` cited `1245-1270` → **def `2757`** (1245-1270 is inside
  `insert()`; `checkConsistensy` at 1265).
- [ ] Slide 7 `CommitRead` reload-before-CAS cited `1245-1260` → `commit()` `2757`; cited range is
  `insert()`.
- [ ] Slide 7 `CommitTryCAS` cited `1245-1270` → `commit()` `2757`; cited range is `insert()`.
- [ ] Slide 7 `CommitDone` finalization tail cited `1265-1275` → `commit()` `2757`; 1265-1275 is the
  tail of `insert()` + start of `release()`.
- [ ] Slide 7 `UnbundleWalk` cited `snapshotSupernode() transaction_impl.h:696-755` → **symbol does
  not exist** (see §1A); use `reverseLookup`/`reverseLookupWithHint`.
- [ ] Slide 7 `UnbundleCASAncestors/UnbundleCASLoop` cited `1367-1379` → **`unbundle()` def `2904`**;
  ancestor CAS ~`2545-2640`; 1367-1379 is inside `release(Transaction&,...)` (body `1325-1480`).
- [ ] Slide 7 `UnbundleCASChild` cited `1383-1389` → inside `release()` (`m_missing=false`
  subpacket-reset block); **`unbundle()` `2904`**.

### 1G. `slides_layer2_LLfree.html` — stale Lamport-helper cite + priority MISMATCH

- [ ] Slide 2 "Lamport serial helpers `transaction.h:547-576`" → 547-576 is
  `with_kind()`/`strip_kind()`/`is_active_stamp()` stamp-field helpers, **not** Lamport serial
  counter/TID helpers. The actual serial machinery is `SerialGenerator` at **`transaction.h:816`**,
  `gen()` at **`836`**.
- [ ] Slide 2/3 maps TLA+ `priorityTag[n]` → C++ `Linkage::m_priority_tidstamp` → **no such member**
  (see §1A). Re-map to the global `NegotiationCounter` static via
  `try_register_privileged_tidstamp` / `release_privileged_tidstamp` / `i_am_privileged_now`.

### 1H. `slides_hardlink_en.html` — one stale line cite (symbol & code correct)

- [ ] Slide 1 "SnapshotConsistency ... mirrors `Packet::checkConsistensy` at
  `transaction_impl.h:870-871`" → **`checkConsistensy` is at `1001`**; 870-871 is a `LivelockProbe`
  comment. (The symbol name and the quoted Phase-4 code are correct.)

> **JA mirrors:** `doc_ja/slides_layer1.html`, `slides_layer2.html`, `slides_layer2_LLfree.html`,
> `slides_hardlink.html`, etc. carry the **identical** numeric citations. **Every §1E–§1H fix must
> be applied in both `doc/` and `doc_ja/`.**

**[OK] collapsed counts for §1:** drift-docs linter: 19 `[OK]` (the two `.tla` SnapshotConsistency
cites `2level_LLfree_dynamic.tla:1093-1096` and `3level_LLfree_dynamic.tla:1831-1835` are **exact**;
`allSubReachable`/`checkConsistensy` `globalroot` params confirmed `transaction_impl.h:1001-1002,
1050-1051`; the Phase-4 reachability-gate order confirmed `2644-2653`; all named TLA+ operators
exist; config-default knobs confirmed `transaction_definitions.h:108,206,208`). drift-slides
linter: ~12 `[OK]` (all 12/12 commit hashes verified present with matching subjects; all on-slide
C++ symbol *names* — `hasPriority`/`bundledBy`/`payload`/`subpackets`/`isOlderThan`/`SerialGenerator`
/`finalizeCommitment`/`drop_tags_n_privilege`/`tag_as_contender`/`release_privileged_tidstamp` —
exist; quoted Phase-4 code faithful to `2644-2652`).

---

## §2 — Symbol existence

Named C++ identifiers, whether they exist verbatim, and the real location/spelling. **2 ghost
symbols + 1 informal alias + 1 non-existent spec-cite are flagged** (✗); all others confirmed (✓).

> **Author's correction (recorded this run).** The DEFAULT priority mechanism is the **per-linkage**
> path, gated by `KAME_PER_LINKAGE_PRIVILEGE` which **defaults to 1** (`transaction_definitions.h:230`,
> "Default: ON"). The TLA+ `priorityTag` therefore corresponds to the per-linkage slot
> `Linkage::m_transaction_started_time` (`transaction.h:905`) and the `Snapshot` tag machinery
> (`m_started_time` / `tag_as_contender` / `drop_tags_n_privilege`), **not** to
> `s_privileged_tidstamp` / `try_register_privileged_tidstamp` / `release_privileged_tidstamp`.
> Those `*_privileged_tidstamp` symbols are the **`KAME_PER_LINKAGE_PRIVILEGE=0` GLOBAL fallback only**
> (defined in `transaction_neg_impl.h`); they are NOT the default mechanism. The verified per-linkage
> correspondence is enumerated in §5 and in the table rows below.

| Symbol (as cited) | Exists verbatim? | Real location / spelling |
|---|---|---|
| `m_priority_tidstamp` | ✗ **GHOST** | does not exist; the per-node priority slot (DEFAULT, `KAME_PER_LINKAGE_PRIVILEGE=1`) is `Linkage::m_transaction_started_time` `transaction.h:905` |
| `Transaction::priority_tidstamp()` | ✗ **GHOST** | does not exist; a Tx's own tag (TLA+ `MyTag`) is `Snapshot::m_started_time` `transaction.h:1515` (tid-packed µs from `now_us_tagged()`); the per-node tag claim is `Snapshot::tag_as_contender` `:1630` |
| `snapshotSupernode()` | ✗ **NOT FOUND** | does not exist; real: `reverseLookupWithHint` `transaction_impl.h:1490`, `reverseLookup` `:1585` |
| `negotiate()` / `m_link->negotiate()` | ✗ **ALIAS** | no such method; real: `ScopedNegotiateLinkage<XN>::_negotiate()` `transaction_neg_impl.h:608` |
| `compareAndSwap_()` (legacy 6-phase) | ✗ **REMOVED** | subsumed into `compareAndSet_impl_` `atomic_smart_ptr.h:1811`; public `compareAndSwap()` `:2038`; comment-only residue `:1711` |
| `Linkage::m_transaction_started_time` (TLA+ `priorityTag[n]`, DEFAULT) | ✓ | `transaction.h:905` (per-linkage priority slot, `atomic`) |
| `Snapshot::m_started_time` (TLA+ `MyTag`, DEFAULT) | ✓ | `transaction.h:1515` set from `now_us_tagged()`, kinded via `with_kind(m_started_time,…)` `:1662`; `iter(t)`/TagOlder compared by `signed_diff_us_packed` `:1664` |
| `tag_as_contender` (TLA+ `TagAfterFail`/`PreemptTag`, DEFAULT) | ✓ | `Snapshot::tag_as_contender` `transaction.h:1630` (CAS: slot empty OR current tagger younger → overwrite; symmetric preempt-window `:1669`; pushes onto `m_tagged_linkages`) |
| `i_am_privileged_now` / `fair_mode_blocks_me` (TLA+ `CanProceed`, DEFAULT) | ✓ | `transaction.h:646` / `transaction.h:634` |
| `drop_tags_n_privilege` (TLA+ `ClearMyTags`, DEFAULT) | ✓ | `transaction.h:1802` (walks `m_tagged_linkages`, zeroes matching slots), called `1518/2206/2370` |
| `m_registered_privileged → StampKind::Reserved` (escalation, DEFAULT) | ✓ | `transaction.h:1659` |
| `s_privileged_tidstamp` (=0 GLOBAL fallback only) | ✓ | decl `transaction.h:593` (inline static); used only under `KAME_PER_LINKAGE_PRIVILEGE=0` (impl `transaction_neg_impl.h:139`) |
| `try_register_privileged_tidstamp` (=0 GLOBAL fallback only) | ✓ | decl `transaction.h:611`; impl `transaction_neg_impl.h:120` (non-default) |
| `release_privileged_tidstamp` (=0 GLOBAL fallback only) | ✓ | decl `transaction.h:614`, called `:1835`; impl `transaction_neg_impl.h:231` (non-default) |
| `ScopedNegotiateLinkage` | ✓ | `transaction.h:120/1331/1357`; impl `transaction_neg_impl.h:608` |
| `tags_successful_cas` | ✓ | called `transaction_impl.h:2688` (child), `3019` (unbundle) |
| `strip_kind` / `stamp_tid` | ✓ | `transaction.h:562` / `transaction.h:439` |
| `isOlderThan` | ✓ | `transaction.h:1615` (unsigned-sub-reinterpreted-signed) |
| `SerialGenerator` (`gen()`) | ✓ | `transaction.h:816` (`gen()` `:836`) |
| `finalizeCommitment` | ✓ | `transaction.h:2362` |
| `checkConsistensy` / `allSubReachable` | ✓ | `transaction_impl.h:1001` / `:1050` (both carry optional `globalroot`) |
| `reverseLookup` (self-return) | ✓ | `transaction_impl.h:1593` (defs `1590/1665/1675`) |
| `bundle` / `unbundle` / `commit` / `snapshot` | ✓ | `transaction_impl.h:2355` / `2904` / `2757` / `2038`,`2053` |
| `bundle_subpacket` | ✓ | `transaction_impl.h:2226` |
| `acquire_tag_ref_` | ✓ | decl `atomic_smart_ptr.h:1067`, def `:1632` |
| `load_shared_` | ✓ | decl `:1064`, def `:1684` |
| `release_tag_ref_(pref, added_global_rcnt, single_attempt)` | ✓ | decl `:1074`, def `:1730` (`added_global_rcnt` param confirmed) |
| `compareAndSet_impl_<OldrT,NewrT,SCOPED,RETAIN_NEWR>` | ✓ | decl `:1096`, def `:1811` |
| `scoped_atomic_view<T>` | ✓ | fwd-decl `:363`, class `:1139`, ctor `:1169/1200`, dtor `:1271` |
| `local_shared_ptr::swap(atomic_shared_ptr<T>&)` | ✓ | decl `:717`, def `:2085` |
| `local_shared_ptr::reset()` | ✓ | def `:1597` (decl `:720`) |
| `PacketWrapper::hasPriority()`/`bundledBy()` | ✓ | `transaction.h:857` / `:862` |
| `Packet::payload()`/`subpackets()` | ✓ | `transaction.h:253` / `:256` (accessor returns `local_shared_ptr<PacketList>`; index via `->at(i)`) |
| `KAME_STM_OPTIONAL_OPTIMIZATION` (default 1) | ✓ | `transaction_definitions.h:108` |
| `KAME_ENABLE_SPIN_BAND_GATE` (default 0) | ✓ | `transaction_definitions.h:206/208` |
| TLA+ ops `GenSerial`/`EncodeSerial`/`MyTag`/`TagOlder`/`CanProceed`/`TagAfterFail`/`TagAfterSuccess`/`ClearMyTags`/`PreemptTag` | ✓ | `BundleUnbundle_2level_LLfree.tla:171/169/194/197/210/221/236/242/282` |

> **Note (verified this run):** `grep -rn "priority_tidstamp" kamestm/ kame/ --include=*.{h,cpp,c}`
> (the ghost spelling, distinct from the `privileged_tidstamp` global-fallback symbols) returns
> **zero** non-`.md` hits, confirming both ghost symbols. `snapshotSupernode` likewise returns zero
> hits in `transaction*.h`. The DEFAULT priority path is per-linkage
> (`KAME_PER_LINKAGE_PRIVILEGE` defaults to 1, `transaction_definitions.h:230`), so the verified
> correspondence is `Linkage::m_transaction_started_time` / `Snapshot::m_started_time` /
> `tag_as_contender` / `drop_tags_n_privilege` (rows above); `s_privileged_tidstamp` &
> `*_register/release_privileged_tidstamp` exist but are the `=0` GLOBAL fallback only.

---

## §3 — memory_order audit

Doc §1 ("Ordering in original") was audited op-by-op against current source. **Result: ZERO
mismatches.** The doc's primitive names map through the thin wrapper in `kamepoolalloc/atomic.h`:
`compare_set_weak` = `compare_exchange_weak(acq_rel, relaxed)` (`atomic.h:84-88`), `compare_set_strong`
= `compare_exchange_strong(acq_rel, acquire)` (`atomic.h:81`), `decAndTest` = `fetch_sub(1, acq_rel)`
(`atomic.h:90`).

| Operation | C++ order (file:line) | Doc-claimed | Match |
|---|---|---|---|
| `compare_set_weak` success | `acq_rel` — `atomic.h:87` | `acq_rel` | ✓ |
| `compare_set_weak` failure | `relaxed` — `atomic.h:87` | `relaxed` | ✓ |
| `load_shared_` global refcnt `fetch_add(rcnt)` | `relaxed` — `atomic_smart_ptr.h:1696` | `relaxed` | ✓ |
| `compareAndSet_impl_` step-4 transfer `fetch_add(step4_amount)` | `relaxed` — `atomic_smart_ptr.h:1944` | `relaxed` | ✓ |
| CAS-failure rollback `fetch_sub(step4_amount)` | `relaxed` — `atomic_smart_ptr.h:2017` | `relaxed` (doc loosely says "fetch_add(negative)") | ✓ (order holds; phrasing loose) |
| step-6 final `fetch_sub(sub)` + delete check | `acq_rel` — `atomic_smart_ptr.h:1974` | `acq_rel` | ✓ |
| `release_tag_ref_` fallback `fetch_sub(sub_amount)` | `acq_rel` — `atomic_smart_ptr.h:1770` | `acq_rel` | ✓ |
| `local_shared_ptr::reset` `decAndTest()` (`= fetch_sub(1,acq_rel)`) | `acq_rel` — `atomic_smart_ptr.h:1610` (`atomic.h:90`) | `acq_rel` | ✓ |

**Off-table but verified sound (no §1 row — flag for the paper):** the §1 table covers the
**strong-refcount core only** and is now **incomplete** relative to source. The following carry
sound orders but have no doc row — consider adding rows if the paper claims §1 enumerates the full
set: `compare_set_strong` (`acq_rel`/`acquire`, `atomic.h:81`); `load_tagged_`/`ref_ptr_` relaxed
pointer loads (`atomic_smart_ptr.h:1039,1056`); `swap()` CAS+transfer (`:2097` acq_rel/relaxed +
`:2094` relaxed `fetch_add` + `:2105` `release_tag_ref_`); the **weak_refcnt** path
(`weak_refcnt.fetch_sub(1,acq_rel)` `:203,348,854`; `fetch_add(1,acq_rel)` `:813,823`; acquire
loads `:200,345,901`); `local_shared_ptr` copy-ctor `fetch_add(1,relaxed)` (`:1575,1586`);
`scoped_atomic_view` promote/owned RMWs (`fetch_add` relaxed `:1295,1392,1433`; `fetch_sub` acq_rel
+ delete `:1439,1462,1517`); the **biased-refcount** path (all relaxed + release-on-publish `:144`,
**gated OFF — no type opts in**, `:122,129-148`); and the `transaction_impl.h` runtime atomics
(`s_count` CAS `:955`; `RunnerCounterEntry` reclaim chain head/CAS/claim `:154,255-258,210-225`,
acquire/release-paired and self-consistent per `:284-299`). All `transaction_impl.h` `g_*`
instrumentation `fetch_add(relaxed)` is inside `#if defined(KAME_ADAPT_INSTRUMENT)` (`:344-487,
577-837`) and **compiled out by default**. `AcquireOneCount`/`ReleaseOneCount` RMWs
(`fetch_add/fetch_sub(1,release)`) live in `transaction.h:769,786,803,808` — outside the two
audited files — and pair with `v.load(acquire)` at `transaction_impl.h:304`.

---

## §4 — Spec-action completeness

Per spec: **ORPHAN** = in the spec's `Next`/`NextStep` disjuncts but undocumented; **GHOST** =
named in docs/slides but **absent from the spec**. Counts (sweep over 11 current specs ×
VERIFICATION.md + all EN/JA slides): ~150 actions checked; ~70 OK (documented), **~78 ORPHAN**,
**14 GHOST**.

### 4A. GHOSTS — documented action names that DO NOT exist in the spec (must fix before slides ship)

**Layer-1 slides (`slides_layer1_en.html` + JA `slides_layer1.html`) — spec was renamed; deck is
stale (slide lines 210-266, 370-401, 487-493):**

| GHOST name in slide | Real spec action (`atomic_shared_ptr.tla`) |
|---|---|
| `ReserveScanRead` (`:210`) | `AcquireTagRefRead` (`Next:1092`) |
| `ReserveScanCAS` (`:124/217`) | `AcquireTagRefCAS` (`Next:1093`) |
| `ScanIncGlobal` (`:242`) | `LoadSharedIncGlobal` (`Next:1095`) |
| `LeaveScanCAS` (`:252`) | `ReleaseTagRefCAS` (`Next:1098`) |
| `LeaveScanGlobal` (`:266`) | `ReleaseTagRefGlobal` (`Next:1099`) |
| `LoadSharedBulkAdd` (`:488`, "(new)") | realized by `LoadSharedIncGlobal` |
| `ReleaseTagRefExcess` (`:401/489`) | folded into `ReleaseTagRefCAS` (`atomic_shared_ptr.tla:481-486`) |
| `ScopeAcquire` (`:370/493`) | `ScopeStartAcquire` (+`ScopeSetState`) (`Next:1111/1112`) |
| `ScopeCASConsume` (`:376/493`) | `ScopeCASCleanup` (`Next:1119`) |
| `ScopeDestruct` (`:493`) | `ScopeDtor` (`Next:1121`) |

**Hardlink nonatomic docs (`VERIFICATION.md:902-905` + `slides_hardlink_en.html`) — conceptual
labels presented as per-thread steps; absent from the spec's actual `Next`
(`DoStep1..4Enter/Exit`, `LoopIterEnd`, `Finalize{C,A}{Walk,Cas}`):** `NonTxInsertAC`,
`TxInsertHardlink`, `NonTxReleaseAC`, `TxReleaseHardlink`. Fine as prose; **wrong if cited as spec
actions.**

### 4B. ORPHANS — in `Next`, undocumented (benign, but breaks any "1:1 action↔doc" claim)

Documentation is at the **phase level** (BundlePhase1-4, Commit*, Unbundle*, the migration
pipeline); most fine-grained sub-steps are unnamed in any doc. By spec:

- **`atomic_shared_ptr.tla`** (`Next:1089-1122`, 32 disjuncts): `StartLoadShared` `:1091`,
  `AcquireTagRefNull` `:1094`, `LoadSharedIncGlobal` `:1095`, `LoadSharedStartRelease` `:1096`,
  `ReleaseTagRefRead` `:1097`, `ReleaseTagRefGlobal` `:1099`, `StartCAS` `:1101`, `CASFailDone`
  `:1105`, `CASUndo` `:1108`, `ScopeStartAcquire` `:1111`, `ScopeSetState` `:1112`,
  `ScopeCAS{Start,PreInc,Load,Check,Transfer,Swap,Cleanup,Undo}` `:1113-1120`, `ScopeDtor` `:1121`,
  `ReturnToIdle` `:1122`.
- **`BundleUnbundle_2level_LLfree.tla`**: `BeginChildIteration` `:861`, `CommitStart(t,c)` `:862`.
- **`BundleUnbundle_3level_LLfree.tla`**: `BeginChildIteration` `:1642`, `CommitStart(t,n)` `:1643`.
- **`2level_LLfree_dynamic.tla`**: `InsertStart` `:1054`, `ReadParent` `:1055`, `InsertCASParent`
  `:1060`, `InsertReadChild` `:1061`, `InsertCASChild` `:1062`, `InsertFinal` `:1063`,
  `BeginChildIteration` `:1066`, `CommitStart` `:1067`, `CommitSkip` `:1068`, `ReleaseStart`
  `:1075`, `ReleaseCASParent` `:1076`, `ReleaseReadChild` `:1077`, `ReleaseCASChild` `:1078`,
  `SkipIteration` `:1079`.
- **`3level_LLfree_dynamic.tla`**: same Insert*/Release*/CommitStart/CommitSkip/SkipIteration/
  BeginChildIteration family (`:1788-1817`).
- **`hardlink_4node.tla`**: `BundleStart` `:409`, `BundlePhase1B` `:411`.
- **`hardlink_dynamic.tla`**: `MigrateClearOther` `:1121`, `MigrateClearOtherFail` `:1122`, plus the
  Insert*/Release*/CommitStart/CommitSkip family (`:1117-1145`).
- **`hardlink_external.tla`**: `ExternalMigration` `:318`, `BundleStart` `:319`.
- **`hardlink_external_migration.tla`**: `BundleStart` `:300`, `BundlePullP1Fail` `:303`,
  `BundleCASP2Fail` `:305`, `BundleUpdateGN1Fail` `:307`, `BundlePhase4Fail` `:309`, `P1RaceBundle`
  `:311`.
- **`hardlink_nonatomic.tla`**: `DoStep1` `:205`, `DoStep2Enter` `:206`, `DoStep2Exit` `:207`,
  `DoStep3` `:208`, `DoStep4Enter` `:209`, `DoStep4Exit` `:210`, `LoopIterEnd` `:211`,
  `StartFinalizeC` `:212`, `FinalizeCWalk` `:213`, `FinalizeCCas` `:214`, `FinalizeAWalk` `:215`,
  `FinalizeACas` `:216`.
- **`hardlink_self_collision.tla`**: `InsertHardLink` `:319`, `BundleStart` `:320`.

### 4C. NON-FINDINGS (checked, confirmed present — NOT ghosts)

`SnapshotForUnbundle` is a **recursive operator** present in `BundleUnbundle_3level_LLfree.tla:1147-1148`
(and 3level_dynamic) — an operator, not a `Next` action. `CommitChild` appears in spec **comments**
(`BundleUnbundle_2level_LLfree.tla:51-52`) as narrative shorthand for the per-child commit realized
by `CommitStart/CommitRead/CommitTryCAS/CommitDone` — documentation shorthand, not a clean ghost.
`ClearMyTags`/`MyTag`/`TagOlder`/`CanProceed` are operators/helpers present in the LLfree specs
(out of `Next`-enumeration scope).

---

## §5 — Consolidated correspondence checklist

The big table: TLA+ action → C++ fn@loc (current, corrected) → C11 GenMC fn → current code
snippet/note → **author column (blank for the author to mark ✅ / ❓)**. All C++ locations here are
the **corrected** post-drift line numbers. The protocol-level mapping was verified **sound** by the
correspondence linter; the author certifies the *semantic* soundness per row.

### Layer 0 — `atomic_shared_ptr` (`kamepoolalloc/atomic_smart_ptr.h`)

| TLA+ action | C++ fn @ loc | C11 fn | Snippet / note | Author |
|---|---|---|---|---|
| `AcquireTagRefRead`/`AcquireTagRefCAS` | `acquire_tag_ref_` @`1632` | `acquire_tag_ref` `test_atomic_shared_ptr.c:66` / `test_scoped_atomic_view.c:95` | single `load_tagged_()` + `compare_set_weak` to +1 local tag (`:1636,1663`) | ☐ |
| `LoadSharedIncGlobal` (+ `StartLoadShared`) | `load_shared_` @`1684` | `load_shared` `test_atomic_shared_ptr.c:132` | `pref->refcnt.fetch_add(rcnt, relaxed); release_tag_ref_(pref, rcnt);` (`:1696-1697`) | ☐ |
| `ReleaseTagRefCAS`/`ReleaseTagRefGlobal` | `release_tag_ref_` @`1730` | `release_tag_ref` `test_atomic_shared_ptr.c:92` / `test_scoped_atomic_view.c:120` | drains `min(rcnt_old, added_global_rcnt)` tags + `fetch_sub` excess `acq_rel` + delete check (`:1730+`) | ☐ |
| `CASPreInc/Reserve/Check/Transfer/Swap/Cleanup` | `compareAndSet_impl_<…>` @`1811` | `compare_and_swap` `test_atomic_shared_ptr.c:166` / `compare_and_set_scoped` `test_scoped_atomic_view.c:170` | `SCOPED`/`ACQUIRE` constexpr dispatch, 6 common steps; SCOPED step4=`rcnt_old`, `NEWR_ADD=2` when `RETAIN_NEWR` (`:1842,~1925`) | ☐ |
| `CASSwap`/`StartSwap` | `local_shared_ptr::swap(asp&)` @`2085` | `swap` `test_atomic_shared_ptr.c:241` | unconditional CAS-loop hold-transfer: `acquire_tag_ref_` → `fetch_add(rcnt_old-1)` → `compare_set_weak` → `m_ref=pref` (`:2092-2100`) | ☐ |
| `ScopeStartAcquire`/`ScopeSetState` | `scoped_atomic_view` ctor @`1169` | acquire path `thread_scope_cycle` `test_scoped_atomic_view.c:239` | `acquire_tag_ref_(&rcnt,weakly)`; if `rcnt<threshold` → `m_pref=p; m_tag_held=true` (`:1175,1184-1185`) | ☐ |
| `ScopeDtor` | `~scoped_atomic_view` @`1271` → `release_()` @`1511` | `ScopeDtor` `test_scoped_atomic_view.c:254` | **minor wording drift:** dtor calls `release_()`; TagHeld branch is `release_tagheld_zeroreset_(false)` @`1514` (NOT literal `release_tag_ref_(pref,1)` as §2 claims); semantically the 1-tag release | ☐ |
| `Reset` | `local_shared_ptr::reset()` @`1597` | `reset_hold` `test_atomic_shared_ptr.c:155` | drops global ref + delete check; `unique()` path `store(0)+deleter` @`1606-1608`; non-unique fallback `decAndTest()=fetch_sub(1,acq_rel)` @`1610` | ☐ |

### Layer 1/2 — bundle/unbundle + priority (`kamestm/transaction_impl.h`, `transaction_neg_impl.h`, `transaction.h`)

| TLA+ action | C++ fn @ loc | C11 fn | Snippet / note | Author |
|---|---|---|---|---|
| `BundlePhase1-4` | `Node<XN>::bundle` @`2355` | `try_outer_bundle`/`try_bundle` `test_bundle_3level_LLfree.c:568` (P2@636/P3@642/P4@662) | explicit `//--- Phase 1` `:2451`; Phase2 CAS parent `scope.compareAndSetRetain` `:2525`; Phase3 CAS each child `:2531`; Phase4 finalize `:2626` | ☐ |
| Bundle Phase-3 skip-Null (hard-link fix) | `bundle` @`2538` | (in `try_*_bundle`) | `if((*subpackets)[i]) bundled_ref=make_local_shared<PacketWrapper>(…) else …subwrappers_org[i]` — mirrors TLA+ `if(local.subpackets[c]==nullptr) continue` | ☐ |
| Bundle Phase-4 `allSubReachable` gate | `bundle` @`2644-2653` | (in `try_*_bundle`) | `newpacket->m_missing=false; if(allSubReachable(...)) [[likely]] {…checkConsistensy…} else { m_missing=true; return DISTURBED; }` — clear-before-gate / restore-on-fail | ☐ |
| `is_bundle_root` missing override | `bundle` @`2508-2511` | — | `if(is_bundle_root){ assert(&supernode==this); missing=false; }` | ☐ |
| `InnerPhase2/3/4` (recursive inner bundle) | `bundle_subpacket` @`2226` | `try_inner_bundle` `test_bundle_3level_LLfree.c:453` | inner/recursive bundle invoked from `snapshot()` outer retry | ☐ |
| `SnapRead`/`SnapCheck` | `Node<XN>::snapshot` @`2053` (Tx overload @`2038`) | `snapshot_grand` `test_bundle_3level_LLfree.c:680` | outer retry loop drives bundle on missing parent | ☐ |
| `CommitGrand`/`CommitParent`/`CommitRead`/`CommitTryCAS`/`CommitDone` | `Node<XN>::commit` @`2757` | `commit_grand` `:709` / `commit_child` `:763` / `commit_parent` (2level) | `hasPriority` → direct `scope.compareAndSetWithHint(newwrapper, tr.m_started_time)` @`2820`; else `unbundle()` @`2829` | ☐ |
| `UnbundleWalk`/`UnbundleCASLoop`/`UnbundleCASChild` | `Node<XN>::unbundle` @`2904` | `commit_child` (bundledBy-chain walk) `test_bundle_3level_LLfree.c:763` | multi-level walk; ancestor CAS ~`2545-2640`. **(slide cites stale `1367-1389` / non-existent `snapshotSupernode` — see §1F)** | ☐ |
| `CanProceed` | `i_am_privileged_now` @`transaction_neg_impl.h:189` + `fair_mode_blocks_me` @`:248` | `can_proceed_with_preempt` `test_bundle_3level_LLfree.c:297` | gate which thread may CAS; C11 fuses CanProceed+PreemptTag (comment `:294-296`). **NOT `m_priority_tidstamp` — see §1A** | ☐ |
| `TagAfterFail`/`PreemptTag` | `Snapshot::tag_as_contender` @`transaction.h:1630` (preempt `transaction_neg_impl.h:1106,1174`) | `tag_after_fail` `:318` + preempt fused in `can_proceed_with_preempt` `:297` | older-overwrites-younger age-ordered preempt | ☐ |
| `ClearMyTags` | `Snapshot::drop_tags_n_privilege` @`transaction.h:1802` | `clear_my_tags` `:340` | zeroes `m_transaction_started_time` on matching `(us,tid)` linkages at Tx-scope end (`:1802-1813`); C11 comment "released ONLY on commit success" `:288` | ☐ |
| Priority gate disable knob | `KAME_STM_OPTIONAL_OPTIMIZATION`=1 `transaction_definitions.h:108` (guards `transaction_impl.h:2161,2393`) | `LLFREE_PRIVILEGE==0` `test_bundle_3level_LLfree.c:78-83` (gate no-ops `:362-367`) | `LLFREE_PRIVILEGE==0` == TLA+ `Privilege=FALSE` | ☐ |
| `SnapshotConsistency` (invariant) | `Packet::checkConsistensy` @`transaction_impl.h:1001` | (assertion in tests) | mirror; **`.tla` cites exact: `2level_LLfree_dynamic.tla:1093-1096`, `3level_LLfree_dynamic.tla:1831-1835`**. **(hardlink slide cites stale `870-871` — §1H)** | ☐ |

> **C11 inventory (verified present):** all 15 `test_*.c` named in VERIFICATION.md §1 exist in
> `kamestm/tests/tlaplus/`: `test_atomic_shared_ptr.c`, `test_scoped_atomic_view.c`,
> `test_stm_commit.c`, `test_bundle_{2,3}level{,_LLfree,_LLfree_dynamic}.c`,
> `test_bundle_hardlink_{4node,dynamic,external,external_migration,nonatomic,self_collision}.c`.

---

## §6 — Author's irreducible sign-off

§1–§5 establish the **mechanical** layer (refs resolve, symbols exist, orders match, action sets
accounted). The following are the **semantic judgments the linters cannot make** — they require the
author's understanding of intent and abstraction. **Only the named author may certify these; the AI
linters explicitly do not and cannot.** Sign each line.

**(a) Each spec action is a sound abstraction of the cited C++.** For every row of §5, confirm the
TLA+ action models the *behaviour* of the cited function — not merely that the symbol exists — at
the right granularity (e.g. that the `compareAndSet_impl_` SCOPED step-4 prepay/rollback genuinely
captures the C++ tagged-pointer transfer; that `bundle` Phase-3 skip-Null and the Phase-4
`allSubReachable` gate are modelled with the correct effect-on-hardlink / no-op-on-single-parent
branch).
- ☐ Certified by: ______________________  Date: __________

**(b) The disclosed abstraction gaps are sound over-approximations AND are stated in the paper:**
- ☐ **ABA is out of scope** — stated, and justified (tagged-pointer + GenMC scope).
- ☐ **WEAK-CAS is not modeled** (specs use atomic CAS; C++ uses `compare_exchange_weak`) — stated;
  the spurious-failure case is argued benign.
- ☐ **§7.5 liveness bounded-disturbance obligation** — the bounded-disturbance assumption under
  which liveness holds is stated as an explicit obligation, not silently assumed.
- ☐ **Hard-link Phase-3/Phase-4** — the hard-link self-collision / migration cases (the
  `BundleUnbundle_hardlink_*` family) and the Phase-3 skip-null fix are disclosed as the boundary of
  the static-single-parent-tree scope.
- ☐ **Thread-axis ∀T-as-conjecture** — the all-threads generalization is presented as a
  *conjecture* (model-checked at bounded T), not a proven ∀T theorem.
- ☐ Certified by: ______________________  Date: __________

**(c) The RC11 memory-order reasoning is correct.** §3 confirms each source order *equals* the
doc-claimed order, but only the author can certify that those orders are *sufficient* for the
claimed happens-before / publication safety (the acquire/release pairing, the relaxed-add +
acq_rel-drain refcount discipline, the delete-check synchronization), **and** that the §3 "off-table
but sound" paths (weak_refcnt, scoped_atomic_view, biased-OFF) do not need to appear in the paper's
ordering table — or, if the paper claims §1 is exhaustive, that the table is extended.
- ☐ Certified by: ______________________  Date: __________

**(d) The headline numbers are correct and current.** State counts, bound sizes (N, T, depth),
GenMC/TLC configurations run, and any "verified" claim against the *current* specs — and confirm the
git commit-hash references in the docs (`VERIFICATION.md` §3 `87892b35`/`92b15f62`/`404fa137`/
`b12e1895`/`1ffd8dce`; §nonatomic `b23fa954`/`ead762be`/`b7a4d882`; §6 `472d193d`/`9a0f9848`;
proof_semantics `2d141d5`/`0141ac11`) resolve to commits with the subjects the docs claim. *(The
slide-deck commit hashes — 12/12 — were verified present with matching subjects by the linter; the
prose-doc hashes above were NOT verified this run and are the author's to confirm.)*
- ☐ Certified by: ______________________  Date: __________

---

### Appendix — items the linters explicitly left UNVERIFIED (author's to close or scope out)

- `parameterized_cutoff.md:74-81` / §8.1 footprint action names for the **3level** spec
  specifically (`SnapshotForUnbundle`, `UnbundleWalk`, `UnbundleCASLoop/Child`, `BundlePhase1-4`,
  `Commit*`) were not exhaustively grep-confirmed against `BundleUnbundle_3level_LLfree.tla` this
  run (the 2level/3level family `BundlePhase1-4`/`Inner*` were confirmed via cross-refs).
- The `cds_atomic_shared_ptr/*.c` GenMC memory-order annotations were **out of scope** this run
  (task targeted `tlaplus/test_*.c`); the `compareAndSwap_` rows in VERIFICATION.md §1's GenMC
  ordering table name a **removed** C++ symbol and should be relabeled to `compareAndSet_impl_`.
- All prose-doc git commit-hash references (listed in §6(d)) are not `file:line` citations and were
  not checked against git history.

---

## §7 — Author focus: action correspondence ranked by Claude's confidence (LOW first)

§6 lists *what* the author must certify; this section says *where to spend the time*.
The ranking below is **Claude's own self-assessed confidence** that each TLA+ action is a
*sound abstraction* of the cited C++ — it is a model-introspection aid, **not** a verification
result. Certify 🔴 first; 🟢 can be skimmed. (All `file:line` are in `kamestm/transaction.h`
unless noted; bundle/commit code is `transaction_impl.h`.)

### 🔴 Lowest confidence — certify first

1. **`TagAfterFail` (3-stage ladder) + independent `PreemptTag`  ↔  `Snapshot::tag_as_contender()`**
   (transaction.h:1630, preempt-window :1669)
   *Concern:* the spec **splits** the mechanism into "CAS-fail ladder" + a *separate* preempt
   action, whereas C++ does it in one CAS-loop and uses a **µs `signed_diff` preempt-window** that
   has no analogue in the spec's pure lexicographic order. This is the area where the docs had
   named a *phantom* C++ symbol — i.e. demonstrably under-examined.
   *Certify:* (a) `tag_as_contender`'s real overwrite condition (slot empty ∨ current younger,
   within the preempt-window) is faithfully covered by `TagAfterFail` + `PreemptTag`; (b) the
   two-action decomposition introduces **no interleaving absent in C++** and loses none present.

2. **`iter(t)=MaxCommits−iterBudget` / `TagOlder` lex  ↔  `m_started_time` / `signed_diff_us_packed`**
   (transaction.h:1515, :1664)
   *Concern:* `iter` (a commit counter that **increases** as a Tx progresses) and `started_time`
   (a wall-clock µs stamp **fixed at construction**) are different quantities; the spec assumes
   "older = smaller iter", the C++ "older = earlier µs". Whether these are order-isomorphic for
   the oldest-wins arbitration the liveness proof needs is the subtlest abstraction in the model.
   *Certify:* the **direction** of "older" matches in both, and the iter↔started_time mapping
   preserves the total order used by the ranking argument.

3. **Liveness §7.5 — bounded structural disturbance (acknowledged OPEN gap)**
   (`parameterized_cutoff.md` §7.5)
   *Concern:* the `(progress)` lemma assumes the privileged `t★` completes without unbounded
   retry, but a peer on a *different* linkage can raise `DISTURBED`/`COLLIDED` through the bundle
   chain. The "each disturbance removes a younger element from M ⇒ bounded" step is **not
   mechanized**. This is a real proof gap, not doc drift.
   *Certify:* whether the bounded-disturbance claim actually holds in the C++ (or scope it as
   conjecture in the paper).

### 🟠 Medium-low

4. **superfine inner-bundle recursion + Phase-3 `DISTURBED`** — `InnerPhase2/3/4`, `BundlePhase3`
   ↔ `bundle()` recursive inner bundle (transaction_impl.h:2355). Several `InnerPhase` restart
   bugs surfaced *during modelling* (see verification_log.md) ⇒ intricate. *Certify:* the spec's
   restart points match the C++ recursion's actual `DISTURBED` returns.
5. **`ClearMyTags` (clears MY tag at *every* node)  ↔  `drop_tags_n_privilege()` walking
   `m_tagged_linkages`** (transaction.h:1802). *Concern:* "all nodes" vs "only the linkages I
   tagged", and "commit-success only" vs the dtor/abort path. *Certify:* cleared-set + timing.
6. **Hard-link Phase-4 reachability gate** — `allSubReachable` / `checkConsistensy(globalroot)`
   (transaction_impl.h:1050 / :1001, gate call :2645) ↔ the `_hardlink_*` models. *Certify:* the
   `globalroot` threading and the DISTURBED-on-unreachable behaviour for cross-tree migration.

### 🟡 Medium (local accounting — bounded)

7. **`scoped_atomic_view` consume `fetch_sub(2)`** (atomic_smart_ptr.h:1811-2034) — the "tag +
   m_ref release absorbed together" refcount accounting.
8. **drain `release_tag_ref_` excess-undo** (atomic_smart_ptr.h:1730-1775) — the
   `cas_rcnt`/`added_global_rcnt`/`drained` balance.

### 🟢 High confidence — skim

`acquire_tag_ref_` CAS-loop (`AcquireTagRef*`), `load_shared_` global `fetch_add`
(`LoadSharedIncGlobal`), bundle 4-phase **coarse** structure (`BundlePhase1-4`), direct commit CAS
(`CommitTryCAS`), `local_shared_ptr::reset` decAndTest (`Reset`), and the **memory_order** table
(independently re-checked this run: 8/8 match — §3).

### Thread-role / bundle-by-which-thread coverage — RESOLVED

*Concern (anticipated reviewer question):* is `bundle` pinned to a fixed thread (e.g. thread 1),
so the model never explores "thread 2 bundles while thread 1 commits"?

*Resolution:* No pin exists. The bundle-triggering action is `SnapRead(t)` guarded by
`t \in RootThreads` (spec :298) and all bundle machinery is `t`-indexed — there is no literal tid
anywhere. Which threads bundle is purely the cfg's `RootThreads` set, and coverage is two-pronged:
- **Structural / "every thread bundles":** the **symmetric** cfgs run it directly — `micro` /
  `Is_bothroles` (`RootThreads = {1,2}`, `LeafThreads = {1,2}`) and `confC` / `Is_allroot`
  (`RootThreads = {1,2,3}`). All threads share the root role, so every "who-bundles-when"
  permutation is explicitly in the state graph (no symmetry argument needed); structural saturation
  (σ = 6, `T=2 ≡ T=3`) is measured on these.
- **Priority directions / asymmetric contention:** `confA` (`RootThreads = {2}`,
  `LeafThreads = {1,3}`) places the lone bundler at the **middle** tid. Since `TagOlder` orders by
  `(iter, tid)` (smaller tid = older ⇒ `t1 ≺ t2 ≺ t3`), the bundler `t2` contends with **both** an
  older peer (`t1`) and a younger peer (`t3`). `tag_as_contender` compares **pairwise** (current
  slot vs my stamp, one peer at a time), so "bundler older → preempt" **and** "bundler younger →
  yield" are both exercised in this single run — the relabeling that would put a different tid as
  the bundler adds no new pairwise priority relationship.

Hence the asymmetric cfgs need **no** TLC `SYMMETRY` reduction (which is unavailable anyway, §"i"):
`confC`/`micro` cover all-threads-bundle structurally, and `confA`'s middle-tid bundler covers both
priority directions. The `iter`-axis relabelings (rank changing as a Tx advances) are unrolled by
TLC within `MaxCommits`. (This supersedes the earlier "asymmetric cfgs check only one labelling"
caveat — confA's placement closes it for the priority axis.)

### Inner-bundle restart-point correspondence (🟠4) — EXAMINED (AI cross-check), accepted

*AI cross-check of `InnerPhase2/3/4` (3level spec :579–696) ↔ the recursive C++ `bundle()`.*
Confirmed: the C++ "inner bundle" is `bundle_subpacket()` (transaction_impl.h:2226) recursively
calling `subnode->bundle(... , false)` (~:2300) when a child is `missing`; so `InnerPhaseN` is the
recursive `bundle()`'s own Phase 2/3/4, flattened into separate TLA+ actions.

**Phase structure matches** (Inner2 = CAS child `missing=TRUE`; Inner3 = CAS each grandchild to
`BundledRefWrapper`, one per step; Inner4 = finalize `missing=FALSE`; success → outer Phase 1).

**One real difference — restart *granularity* (not a soundness bug):**
- **TLA+:** any `InnerPhaseN` failure → unconditional `snap_check`, **clearing all** outer
  `wrapper`/`subwrappers`/`subpackets` and re-collecting **every** child (:606-614, :662-670, :693-).
- **C++** (:2480-2490): on inner `DISTURBED`, `if(supscope == *supernode.m_link) continue;` retries
  **only the disturbed child** (keeping siblings' collected `subwrappers_org[i]`); it bails to the
  caller (`return DISTURBED`) **only if the outer parent changed**. I.e. the C++ does a finer,
  per-child restart that the TLA+ collapses into a coarse full re-collect.

**Author decision: accepted as a sound abstraction — no additional model-checking run.**
Rationale (a): the C++ re-validates everything before committing — Phase 2 parent CAS
(`compareAndSetRetain`, :2525) and Phase 3 per-child CAS (:2530) fail → `DISTURBED` if any kept
sibling view went stale, so no inconsistent bundle can *commit*; the reachable **committed** states
therefore coincide with the TLA+'s coarse-restart model. The only thing the coarse model does not
represent is the C++ *intermediate* "stale sibling carried into Phase 3" — and although the
σ-saturation that would directly exercise such intermediates was measured on 2-level all-root and
only *extrapolated* to 3-level (§5.1), the Phase-2/3 re-validation argument is deemed sufficient.
Option (b) — a targeted 3-level inner-bundle Phase-3-with-stale-sibling run — was **considered and
declined** as unnecessary.

*Minor (record only):* the `InnerPhase2/3/4` `\*`-comments in `BundleUnbundle_3level_LLfree.tla`
still cite `bundle()` at lines `1249-1258 / 1260-1282 / 1286-1299 / 2487-2511`; current anchors are
`bundle()` :2355, parent CAS :2525, Phase 3 :2530, `bundle_subpacket` :2226. (Spec-comment drift,
not checked code — fix opportunistically.)
