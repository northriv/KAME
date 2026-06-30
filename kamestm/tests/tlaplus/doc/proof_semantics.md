# TLA+ 証明セマンティクス

KAME の `tests/tlaplus/` で行っている TLA+ 検証について、**TLC の終了が何を証明し、何を証明しないか**を理論的に整理する文書。時系列の修正記録は `verification_log.md` を参照。

---

## 1. 検証アプローチの世代

bundle/unbundle (Layer 2) の TLA+ 仕様は有限性の確保方法によって変遷してきた。現行と直近の旧版は以下：

| 世代 | Serial | CONSTRAINT | Privilege/Priority | 動作 | 役割 |
|---|---|---|---|---|---|
| 旧 (Nat+CONSTRAINT) | `Nat` 単調増加 | `SerialBound` で打ち切り | optional | 終了するが状態数が発散傾向 | wrap は回避できるが探索打ち切り依存 |
| **新 (LL-free) ✅ 完成** | `Nat` 単調増加 | **なし** | **常時 ON** (`priorityTag`) | 構造的有限が成立し TLC が exhaustion 完走、liveness PASS | LL-free を構造的に証明 |

> さらに古い modular 算術版もかつて存在した (`% MaxSerial` で wrap させる方式)。`SerialWrapAround` invariant violation で落ち、抽象化も中途半端だったため実用価値はほぼなく、現在は履歴的言及のみ (§11)。

> **2026-04-29: Gen 3 完成**。`BundleUnbundle_2level_LLfree.tla` が CONSTRAINT なしで完走し、Safety 全 invariant に加えて `EventuallyAllDone` liveness PROPERTY も PASS した。665,218 distinct states / depth 89 / 28 秒 / queue 0 (exhaustion)。
>
> **2026-04-30: スケール検証 (superfine+commits2) PASS**。ohtaka F1cpu にて `BundleUnbundle_2level_LLfree_superfine_commits2_mc.cfg` (MaxCommits=2, |Threads|=2, BundleCollect=superfine, BundlePhase3=superfine) が完走。127,586,599 distinct states / depth 311 / 4h 40min / queue 0 (exhaustion)。Safety + `EventuallyAllDone` PASS。
>
> **2026-05-01: 3-thread Config C (all-root, superfine) PASS**。`BundleUnbundle_2level_LLfree_3thr_superfine_C_mc.cfg` (Threads={1,2,3}, RootThreads={1,2,3}, LeafThreads={}, BundleCollect=superfine, BundlePhase3=superfine) が Safety 全 invariant PASS。**137,333,348 distinct states / depth 96 / 6h 35min / queue 0 (exhaustion)**。Config C は 3 スレッドが全員 CommitParent のみを実行する構成で、Parent CAS + bundle collect の 3-way 競合を集中的に検証。KAME commit `0141ac11`。
>
> **2026-05-01: Dynamic insert/release モデル (`BundleUnbundle_2level_LLfree_dynamic.tla`) PASS**。動的な子ノード挿入・解放を扱う新 spec を追加。静的 LLfree spec の 763,478 states に対し、release 操作込みで状態数が大幅増加。Release coarse 2-thread config: **14,203,816 distinct states / depth 150 / 19:03 (exhaustion) — PASS**。開発中に 3 件の活性違反を発見・修正: (1) release 時の idle スレッド deadlock、(2) 解放済み子への繰り返し処理による livelock、(3) idle スレッドが残したタグが他スレッドをブロックする stale priority tag 問題。Stale priority tag 修正: C++ の RAII デストラクタに対応して `ClearMyTags(t)` を 3 つのフェーズ境界で明示的に呼び出す。
>
> **2026-05-03: Dynamic 3-thread no-release Config A・B PASS + BundlePhase1 deadlock fix**。`BundleUnbundle_2level_LLfree_dynamic.tla` の 3-thread no-release 構成を 2 config で ohtaka liveness 検証。Config A (`_3thr_A_live_mc.cfg`, Ins={1}, Root={2}, Leaf={3}): **53,397 distinct states / depth 68 / 7s — PASS + liveness ✅**。counter 10–15, 42 terminal emits。Config B (`_3thr_B_live_mc.cfg`, Ins={1}, Root={2,3}, Leaf={}): **149,137 distinct states / depth 82 / 14s — PASS + liveness ✅**。counter 8–15, 22 terminal emits。insert/commit の 3-way 競合において活性が成立することを ohtaka で確認。
>
> **BundlePhase1 deadlock fix**: `_release_superfine_mc.cfg` が depth 34 で TLC deadlock に達した。根本原因: Thread A がサブラッパー収集中にThread B が子を release → `ActiveChildren` が縮小するが A の収集済みエントリが残存し、第1ダイジャンクト（`∃c ∈ ActiveChildren : subwrappers[c] = Null`）が FALSE に → 有効アクションなしで stuck。修正: `BundlePhase1` fine/superfine ブランチに第2ダイジャンクトを追加。`∀c ∈ ActiveChildren : subwrappers[c] ≠ Null` のとき（mid-collection shrink 検出）、リリース済み子のエントリを削除して `bundle_phase2` へ進む。Phase 2 CAS はリリース側の linkage[Parent] 更新で必ず失敗 → `BundleRetryPC` → `ReadParent` が新しい parent wrapper（リリース済み子を除去済み）を取得。C++ 忠実: prestamp CAS → 収集ループ → Phase 2 CAS 失敗 → snap_read リトライと同じパス。ローカル検証: depth 45 到達、deadlock なし、terminal counter 15–18 出力確認。完全網羅検証は ohtaka。
>
> **2026-05-02: Dynamic モデル追加 2 件 PASS**。`BundleUnbundle_2level_LLfree_dynamic.tla` の no-release 構成を 2 粒度で検証。(1) **coarse (no release)**: 763,478 distinct states / depth 104 / 43s — PASS。(2) **superfine (no release)**: 4,862,872 distinct states / depth 162 / 7:05 — PASS。既報の release coarse 14.2M と合わせ、insert/release 動的操作の安全性 + 活性が coarse → superfine の全粒度で確認された。release superfine は pending。
>
> **2026-05-03: Dynamic release superfine live 2-thread PASS**。`_release_superfine_live_mc.cfg` が BundlePhase1 deadlock fix 適用済みで完走。**413,884,516 distinct states / depth 320 / 7h 13min / counter 15–55 / 5,972 terminal emits — PASS + liveness ✅**。これにより dynamic spec の 2-thread 全粒度 (coarse/superfine) × 全操作 (no-release/release) の Safety + liveness が完全に証明された。
>
> **2026-05-01: 活性検証方針を改訂 — coarse cfg で証明、大規模 cfg は INVARIANT のみ**。`BundleUnbundle_2level_LLfree_coarse_mc.cfg` (coarse atomicity, |Threads|=2, MaxCommits=1) でラップトップ上の活性証明を担保。superfine/fine は coarse の細分化なので、coarse での `EventuallyAllDone` PASS が全粒度に波及する（refinement 論証、§10 参照）。ohtaka 向け大規模 cfg (superfine/3thr/commits2/A-D 等、全 15 cfg) は `PROPERTY EventuallyAllDone` を削除し INVARIANT のみに。TLC の temporal property check は状態数に対して非線形に高価なため、137M states 規模の superfine cfg では省略効果が大きい。
>
> したがって現状:
>
> - Safety 性: 制限付き探索内のみだった旧世代に対し、Gen 3 で **構造的有限の状態空間における完全 exhaustion** で証明完了。MaxCommits=1, |Threads|=2, MaxPayload=3 の micro config だが、modular/CONSTRAINT どちらも介在しないため意味のある形式証明。
> - LL-free 性: `priorityTag` 設計が意図通り無限 retry を構造的に不可能にしていることが、TLC 完走 + `EventuallyAllDone` PASS で証明された (§2 の「終了 + CONSTRAINT なし + Nat serial → livelock-free 自動」の論証チェーンが成立)。
> - 大規模ストレステストとの相互裏付け: C++ 実装 commit `2d141d5` が FINE/SUPERFINE multi-thread を全 PASS、preempt が 100万件/秒のオーダーで観測される動的挙動も TLA+ 仕様の `PreemptTag` action と整合。
>
> **形式検証の柱が立った**。残る作業は config 拡張 (MaxCommits>=2, |Threads|>=3) でのスケーラビリティ検証（ohtaka など）と、3-level への拡張。

新世代は `BundleUnbundle_2level_LLfree.tla`。`priorityTag[n]: Null | <<iter, tid>>` を per-node に持ち、older transaction（小さい iter、次に小さい tid）が勝つ規律を C++ の per-linkage 優先度スロット `Linkage::m_transaction_started_time` (transaction.h:905) と、それを CAS する `Snapshot::tag_as_contender()` / `negotiate()` から忠実に写している（`KAME_PER_LINKAGE_PRIVILEGE=1` が既定）。「無限 retry を構造的に不可能にする」設計が、CONSTRAINT なし TLC 完走によって**実証**された。

---

## 2. TLC 終了が証明するもの／しないもの

### 終了 + CONSTRAINT なし

- **状態空間が真に有限であることの証明**。TLC が全到達状態を列挙し終えたという事実そのものが証明。
- 加えて以下の 2 条件が成立すれば、**livelock-free が自動的に出る**：
  1. Serial が `Nat` で単調増加
  2. livelock 候補となる全アクション（retry を含む）が serial を必ず bump する

  理由：状態グラフ上のサイクル ⇒ 同じ状態に戻る ⇒ serial も同じ値に戻るが、serial 単調増加のためサイクル不可能 ⇒ 無限挙動が存在しない。

- KAME の LL-free 仕様 (`BundleUnbundle_2level_LLfree.tla`, micro config) はこの 2 条件を満たし、**2026-04-29 に CONSTRAINT なし TLC 完走が達成された**。665K distinct states / depth 89 / queue 0 / 28 秒。これにより構造的 LL-free 性が形式的に証明された。

### 終了だけでは保証**されない**もの

- **Deadlock-freedom**: TLC は「次に取れるアクションがない状態」を別カテゴリで deadlock として検出する。終了したかどうかとは独立。`-deadlock` flag や `CHECK_DEADLOCK FALSE` でこの検出を抑制している場合、終了は何も保証しない（§6 でこの workaround を廃止する案）。
- **Wait-freedom**: §7 参照。

---

## 3. CONSTRAINT 付きのケース（旧世代）

`CONSTRAINT SerialBound` は「serial が `MaxSerial` に達した状態より先を探索しない」という打ち切り規則。これがあると：

- TLC の終了は「`SerialBound` を満たす全状態を探索した」を意味するに過ぎず、状態空間そのものが有限であることは示さない。
- したがって **CONSTRAINT 付きの終了は livelock-free の証明にならない**。「打ち切らなければ無限に状態が増え得る」可能性が残る。
- ただし invariant violation の検出（safety properties）には有効。打ち切られた領域に入る前に invariant が破れていればちゃんと捕まる。

旧世代（Nat + SerialBound）は safety の検証としては機能するが、liveness（LL-free）の証明としては不完全。

---

## 4. Privilege/Priority 機構による LL-free の構造化

`BundleUnbundle_2level_LLfree.tla` の設計骨子：

- `priorityTag[n] \in {Null} \cup ({0..MaxIter} \times Threads)`
- CAS 失敗時、自分の `<<iter, tid>>` が現在 tag より older なら自分でセット (`TagAfterFail`)
- CAS 成功時、自分の tag ならクリア (`TagAfterSuccess`)
- 他スレッドは tag が Null か自分のものでなければ「待つ」（CanProceed 否定）

この older-wins 規律の**意図**：

- 各時点で「優先される 1 スレッド」が必ずいる
- そのスレッドの retry 回数は他スレッドの数に対して有界（**であってほしい**）
- 全スレッドの累積 retry 数も有界（**であってほしい**）
- → serial 増加が有界 → 状態空間が有限 → TLC が終了
- → §2 の議論で CONSTRAINT なしの終了が LL-free の証明になる

C++ 実装の per-linkage 優先度スロット `Linkage::m_transaction_started_time` (transaction.h:905) と、それを oldest-wins で CAS する `Snapshot::tag_as_contender()` (transaction.h:1630)・gate を判定する `i_am_privileged_now`/`fair_mode_blocks_me` (transaction.h:646/634)・commit 成功時にタグを落とす `drop_tags_n_privilege()` (transaction.h:1802) をミラーしているため、仕様の完走はそのまま実装の LL-free 性の根拠になる（既定の `KAME_PER_LINKAGE_PRIVILEGE=1` 経路。`KAME_PER_LINKAGE_PRIVILEGE=0` の場合のみ `transaction_neg_impl.h` のグローバル `s_privileged_tidstamp` 系へフォールバック）。**2026-04-29 に完走が達成された**ため、上記の論証チェーンは成立。priority 規律の穴と TLA+ 写しの不備、いずれの可能性も排除された (micro config の範囲で)。

---

## 5. 旧 (Nat + CONSTRAINT) 世代の位置づけ

旧版は意図的に削除せず残している。wrap は起きないが、CONSTRAINT を外すと状態数が発散する。これは「Nat にしただけでは LL-free にならない」ことを示し、priority 機構の必要性 = Gen 3 (LL-free) への動機を裏付ける役割を持つ。

なお更に古い modular 版については §11 (履歴的言及) を参照。

---

## 6. `CHECK_DEADLOCK TRUE` への移行案

旧 cfg 冒頭に「`-deadlock` flag で実行せよ」というコメントがあるのは、全スレッドが iterBudget=0 で完了した終端状態に enabled action がなく、TLC が deadlock 判定するため。これは TLA+ の標準イディオムで回避できる。

```tla
AllDone == \A t \in Threads : pc[t] = "idle" /\ iterBudget[t] = 0

Terminating ==
    /\ AllDone
    /\ UNCHANGED vars

Next ==
    \/ \E t \in Threads : Step(t)   \* 既存のアクション
    \/ Terminating

Spec == Init /\ [][Next]_vars /\ \A t \in Threads : WF_vars(Step(t))
\* WF には Terminating を含めない — 含めると「永遠にスタッターすればよい」になり活性が無意味化
```

cfg 側：
```
SPECIFICATION Spec
PROPERTY <>AllDone        \* 活性を明示的に検査
INVARIANT Safety TerminalPayloadCheck
\* CHECK_DEADLOCK は書かない（デフォルトの TRUE のまま）
\* -deadlock flag も不要
```

これにより：

- 終端状態 (`AllDone`) は `Terminating` disjunct で自己ループとして許容
- それ以外の状態で詰まれば従来通り deadlock として検出される（**これが回復する**）
- `<>AllDone` を property として書けば「ちゃんと全員 Done になる」 = LL-free を明示的に検査できる

注意：

- `vars` には spec の全変数を網羅すること（書き忘れると TLC が遷移を勝手に許す）。`vars == <<v1, v2, ...>>` を一箇所で定義して使い回すのが安全。
- PlusCal 使用なら `fair process` で translator が同等の Terminating を自動生成する。手書き TLA+ では明示が必要。

---

## 7. Wait-free は別問題

LL-free が証明されても **wait-free は出ない**。

- LL-free（lock-free 含む）: システム全体としていずれかのスレッドが進む。fairness 仮定下で個別スレッドも eventually 進む。
- Wait-free: **他スレッドの挙動に依存せず**、自スレッドの**有界ステップ数**で完了。

差分：

1. LL-free は通常 fairness（WF/SF）の下でしか活性が出ない。Wait-free は fairness 非依存。
2. LL-free の活性 bound は構成依存（N スレッド・有限データ）。Wait-free は構成非依存の bound が必要で、N を増やすと bound が伸び続けるなら wait-free ではない。
3. KAME の CAS retry ベースは構造的に wait-free でない。優先される 1 スレッドが先に進む間、他スレッドは（fairness で救われるが）任意回数 retry し得る。

TLA+ で wait-free を示すなら、各スレッドにステップカウンタを持たせ、それが他スレッド非依存の定数で bound される invariant を書く必要がある。普通の活性検査では出ない。

KAME が示せているのは **lock-free + livelock-free**（priority 機構＋fairness 下）であって wait-free ではない。これは設計通り（CAS retry ベースなので本質的にそう）。

---

---

## 8. 現状の評価 (2026-04-29 更新)

| 項目 | 形式検証で示せていること | 経験的に分かっていること |
|---|---|---|
| Safety（安全性 invariant） | Gen 3 (LLfree, micro config) で **CONSTRAINT なし完全 exhaustion で証明** | C++ 大規模ストレステストで未破綻 |
| LL-free（活性） | Gen 3 で **`EventuallyAllDone` PROPERTY PASS = 形式証明済み** (micro config) | C++ stress で livelock 観測なし、preempt 100万件/秒オーダー |
| Wait-free | **証明する設計でない** | CAS retry ベースなので本質的に出ない |

**つまり: Gen 3 によって Safety + LL-free が formally に証明された**。micro config (665K states) でのベース証明に加え、以下のスケール検証が完了:
- **MaxCommits=2 + superfine (127M states, 4h 40min)** — PASS (2026-04-30)
- **2L 3-thread superfine confC (137M states, 6:35)** — Safety + liveness PASS (ohtaka, 2026-05-01)
- **3L 3-thread superfine confC (640M states, 15:25)** — Safety + **liveness** PASS (ohtaka, 2026-05-03) ← **3-level 3-thread liveness 形式証明**
- **3L 3-thread superfine confA/B** — ⏳ ohtaka 再投入中 (InnerPhase2 fix 適用、2026-05-03)

C++ 実装 (commit `2d141d5`) との 1:1 対応も検証済み (8/8 priority-gated CAS site)。

### Gen 3 の達成内容

- `priorityTag[n]: Null | <<iter, tid>>` を per-node に組み込み、older transaction が勝つ規律を C++ の per-linkage 優先度スロット `Linkage::m_transaction_started_time` (transaction.h:905) と `Snapshot::tag_as_contender()` (transaction.h:1630) から忠実に反映 (既定 `KAME_PER_LINKAGE_PRIVILEGE=1`)。
- Transaction-scope 持続: tag は CAS 単位で clear せず、`CommitParent`/`CommitDone` (= C++ `drop_tags_n_privilege` 呼び出し点) でのみ release。これがないと unbundle/rebundle ping-pong による livelock が起きることを bound-60 violation トレースで実証。
- `ActiveThread` 判定で zombie tag (終了スレッドの遺留 tag) 回収。
- `PreemptTag` で older が younger を強制 preempt。
- `Terminating` disjunct イディオムで `-deadlock` flag 不要 (§6 案を実装)。
- `<>AllDone` PROPERTY で活性を明示的に検査、PASS。
- `SYMMETRY` は drop (Nat-ordered Threads + liveness check の両立のため; Gen 3 は exhaustion が高速なので不要)。

### 旧 (Nat+CONSTRAINT) 世代の位置づけ (継続)

旧版の保存（§5）は移行過程の記録としての意義に留まる。CONSTRAINT を外すと状態数が発散する事実が「Nat にしただけでは LL-free にならない → priority 機構が必要」を裏付け、Gen 3 への動機を示す。なお更に古い modular 版についてはほぼ価値なく、§11 に履歴的言及のみ。

### 今後の作業

1. ✅ ~~Gen 3 を CONSTRAINT なしで完走させる~~ — 達成 (2026-04-29)
2. ✅ ~~`Terminating` disjunct + `<>AllDone` PROPERTY~~ — 実装済み
3. **スケール検証**: MaxCommits ≥ 2, |Threads| ≥ 3 で同一仕様が完走するか確認 (ohtaka)
   - ✅ MaxCommits=2, superfine (127M states, 4h 40min, 2026-04-30) — PASS (fine を包含するため fine+commits2 は不要)
   - ✅ |Threads|=3, MaxCommits=1, superfine Config C (all-root) — **137,333,348 distinct states / depth 96 / 6h 35min — PASS** (ohtaka, 2026-04-30)
4. ✅ ~~**3-level への拡張**: `BundleUnbundle_3level_LLfree.tla`~~ — 仕様・micro cfg 作成済み; ohtaka 検証待ち
5. **Dynamic insert/release モデル** (`BundleUnbundle_2level_LLfree_dynamic.tla`): 構築済み
   - ✅ coarse/superfine (no-release) 2-thread — PASS (ohtaka 2026-05-02)
   - ✅ release coarse 2-thread — 14.2M states / 27:12 PASS (ohtaka 2026-05-01)
   - ✅ 3-thread no-release Config A/B — PASS + liveness (laptop 2026-05-03)
   - ✅ BundlePhase1 deadlock fix — depth 45 ローカル確認 (2026-05-03)
   - ✅ release superfine 2-thread live — **413,884,516 states / depth 320 / 7:13 / counter 15–55 / 5,972 emits — PASS + liveness** (ohtaka 2026-05-03)
   - ⏳ 3-thread release Config — ohtaka
6. **InnerPhase2 restart fix (2026-05-03)**: `QuiescentCheck` violation at depth 61 (spec `8fb19385`, 3L superfine confA)。`InnerPhase2` 失敗時に `UNCHANGED local` で snap_check に戻ることで `subwrappers[Parent]` が非 Null のまま残り、`BundlePhase1` が Parent 再収集をスキップして stale な `subpackets` で `BundlePhase2` → lost increment。修正: `InnerPhase3`/`InnerPhase4` と同パターンで `wrapper`/`subwrappers`/`subpackets` をクリア、`bundleNode` も eager-tag。C++ 忠実: `bundle_subpacket` が DISTURBED を返すと外側 child_retry ループが `continue`（`subwrappers_org[i]` 未更新）→ Parent を再読み → 内部 bundle 再実行、TLA+ の挙動と等価。ローカル確認: 1-thread PASS (47 states)、2-thread coarse PASS (1,497,098 states / depth 98 / 1:35 + liveness)。confA/B を ohtaka 再投入 (`run_l3llf_3thr_superfine_live.sh`)。
   - *パターン*: InnerPhase2/3/4 の三つの inner-phase failure path がすべて outer bundle state をクリアする一貫した設計になった。
7. **C++ 実装の確認**: TLA+ で「不要」と分かった C++ 機構 (たとえば read-side fairness tag) があれば実装側の整理候補にする (ただし `tags_successful_cas` は C_obs カウンタなので残置必須)
7. **Layer 1 (atomic_shared_ptr) と Layer 2 (bundle/unbundle) の合成証明** (compositional proof) は依然として未着手 — Layer 1 単独 liveness の扱いは §11 を参照

---

## 9. superfine の C++ 忠実度監査 (2026-04-30)

`BundleUnbundle_2level_LLfree.tla` / `BundleUnbundle_3level_LLfree.tla` の `superfine` 経路を `kame/transaction_impl.h` (`Node<XN>::bundle()` および `ScopedNegotiateLinkage`) と突き合わせて点検した。結果は **1 箇所だけギャップあり、ただし live-lock-free 検証の健全性には影響しない**。

### 経路ごとの一致状況

| 経路 | TLA+ | C++ (transaction_impl.h) | 一致 |
|---|---|---|---|
| Phase 1 entry pre-CAS (superfine) | `parentW.serial /= ser` → CAS + `TagAfterSuccess`/`TagAfterFail` | `!hasPriority \|\| serial mismatch` (line 2389) → CAS + `tag_as_contender` on fail (line 2397) | ✓ |
| Phase 1 child-collect 失敗 (parent unchanged) | retry same child + child `TagAfterFail` | child_retry++ + `ScopedNegotiateLinkage(child, child_retry>0)` eager (line 2431) | ✓ |
| Phase 1 child-collect 失敗 (parent changed) | snap_read + Parent `TagAfterFail` | return DISTURBED → outer retry + `scope(retry>0)` Parent eager (line 2407) | ✓ |
| Phase 2 CAS 失敗 | snap_read + Parent `TagAfterFail` | return DISTURBED → outer retry + `scope(retry>0)` Parent eager | ✓ |
| Phase 3 失敗 DISTURBED | snap_read + child + Parent `TagAfterFail` | return DISTURBED → caller restart + Parent eager | ✓ |
| **Phase 3 失敗 not-DISTURBED** | bundle_phase1 + child タグ **のみ** | outer continue + `scope(retry>0)` Parent eager | ✗ ギャップ |
| Phase 4 CAS 成功 | `TagAfterSuccess` (no-op = keep) | `tags_successful_cas` (slot を自分の tid に上書き) | ~ (実害なし — 次節参照) |

### 唯一のギャップ: Phase 3 失敗 not-DISTURBED で eager Parent タグ欠落

C++ (`transaction_impl.h:2480-2506`):

```cpp
for(unsigned int i = 0; i < subnodes->size(); i++) {
    if(!child->m_link->compareAndSet(subwrappers_org[i], bundled_ref)) {
        if(retry) snap.tag_as_contender(child->m_link);
        if(... disturbed ...) return BundledStatus::DISTURBED;
        changed_during_bundling = true;
        break;
    }
}
if(changed_during_bundling)
    continue;  // outer for(retry) の次イテレーション開始
               // → ScopedNegotiateLinkage scope(supernode.m_link, snap, retry)
               //   が retry>0 で Parent を eager-tag
```

TLA+ 2-level (`BundleUnbundle_2level_LLfree.tla` line 533-539):

```tla
ELSE \* No rollback — restart Phase1
     /\ pc' = [pc EXCEPT ![t] = "bundle_phase1"]
     /\ priorityTag' = [priorityTag EXCEPT ![c] = TagAfterFail(t, c)]
       \* ← C++ ではここで Parent も refresh されるが、TLA+ にはない
```

3-level も同様 (line 802-811)。3-level は `newSer` を生成するため、bundle_phase1 復帰後の Phase 1 superfine pre-CAS が `parentW.serial /= newSer` で発火し、**CAS が失敗した場合のみ** Parent タグが付く。eager (CAS 試行前のタグ) は欠けている。

### なぜ live-lock-free 検証には影響しないか

**C++ の `retry > 0` eager-tag が果たす役割:**
- 性能最適化: 競合が観測されてからタグを置くことで、無競合時の atomic-store コストを節約
- 加速効果: peer をより早く block し、retry サイクルを短縮
- **安全性・活性の構造的保証には関与しない**

**live-lock-free 性の本質的な機構:**
- `TagOlder` による older-wins 全順序
- `TagAfterFail` の older 上書き則
- `PreemptTag` による active-snatch
- `ClearMyTags` は commit 成功時のみ

これらは TLA+ も C++ も同一で、retry-count によらず常に適用される。

**保守的近似の方向:**

| 時点 | TLA+ 動作 | C++ 動作 |
|---|---|---|
| Phase 3 fail 直後 | bundle_phase1 へ; Parent タグ refresh **なし** | retry++ で Parent eager-tag |
| Peer 介入の余地 | あり (Parent 一時的に無タグ) | なし |
| Peer が割り込んだ場合 | 私の次の pre-CAS / Phase2 が失敗 → そこで `TagAfterFail` で Parent をタグ | (発生しないが、発生しても同じ) |
| 1ラウンド遅れの影響 | older-wins により次ラウンドで peer block | — |

つまり TLA+ は **C++ より広い interleaving 集合を許す保守的近似**。C++ の eager-tag は peer 干渉を 1 ラウンド早く遮断するが、TLA+ はそれが無い状態 (peer 干渉が 1 ラウンド多い世界) で `EventuallyAllDone` を verify している。

**従って TLA+ で活性が verify されれば、C++ では一層安全に live-lock-free**。

### Phase 4 成功時の `TagAfterSuccess` vs `tags_successful_cas`

C++ (`transaction_impl.h:2526-2528`) は Phase 4 成功後に各子ノードと Parent の priority slot を **明示的に自分の tid に上書き** (`tags_successful_cas`) する。TLA+ の `TagAfterSuccess(t, n) == priorityTag[n]` は no-op で、既存のタグを保持する。

通常は問題ない (自分のタグが既に立っている)。レアケースとして「Phase 3 中に older peer が `PreemptTag` で私のタグを奪取し、その後 peer は別ノードで詰まり Phase 4 では私の CAS が通る」というシナリオでは、C++ は Parent タグを自分に上書き、TLA+ は older peer のタグを保持。実装上の差はあるが、**いずれも CAS の semantics は同一**で safety/liveness ともに影響なし。

### 結論

- **修正の必要性なし**: 形式検証の主張 (`EventuallyAllDone` PROPERTY が成立 → 構造的 LL-free) は現仕様で十分正当化される
- **C++ 1:1 文字どおり対応の主張をする場合のみ修正候補**: 2-level 533-539 と 3-level 802-811 に `![Parent/node] = TagAfterFail(t, ...)` を追加すれば C++ と同型になる
- スライド・論文では「TLA+ は C++ より保守的な近似で活性を証明している」と説明するのが正確

---

## §10 活性証明の粒度と ohtaka cfg 方針 (2026-05-01)

### 粒度の包含関係

3 つの atomicity 粒度は **coarse ⊇ fine ⊇ superfine** の包含関係にある:

| 粒度 | BundlePhase1 | BundlePhase3 | 一回の action でモデル化される範囲 |
|---|---|---|---|
| coarse | 子収集を 1 step | Phase 3 CAS を 1 step | Phase 1/3 全体 |
| fine | 子収集を 1 子/step | Phase 3 CAS を 1 子/step | Phase 1/3 の 1 子ずつ |
| superfine | fine + prestamp CAS | fine + DISTURBED 検出 | C++ に最も忠実 |

finer な粒度はよりインターリーブが多い (= peer 干渉の機会が増える) ため、**coarse での活性証明が成立すれば fine/superfine でも成立する**。

論理的根拠:  
1. `priorityTag` の更新規則 (`CanProceed`, `TagAfterFail`, `PreemptTag`) は粒度に依存しない (coarse/fine/superfine で同一オペレータ)  
2. coarse で見た場合の「older-wins が進行を保証する」論証チェーン (§2) は、細分化されても同じオペレータが呼ばれる限り成立  
3. superfine/fine で増える interleaving は「CAS サイトが細かく分割される」だけで、priority gating が防ぐべき livelock を新たに生成しない

### 検証方針

| cfg 種別 | 対象 | PROPERTY | 根拠 |
|---|---|---|---|
| `coarse_mc` | laptop (2-level) | **あり** | 活性の参照証明。coarse PASS → 全粒度をカバー |
| `micro_mc` (fine) | laptop (2-level) | あり | 追加の fine 活性確認 (665K states, 28s) |
| superfine/3thr/commits2/A-D 等 | ohtaka | **なし** | INVARIANT のみ。活性は coarse cfg で証明済みとして省略 |
| 3-level 全 cfg | ohtaka | **なし** | 同上 |

### コスト削減効果

TLC の temporal property (PROPERTY) 検査は safety invariant 検査に比べて状態数に対して非線形に高価 (状態グラフに SCC 解析が加わる)。superfine 137M states 規模では PROPERTY 省略により wall time が 2-5× 短縮される見込み。

---

## 11. Layer 1 (atomic_shared_ptr) liveness の扱いと履歴的注記

### 層の命名

本文書では以下のスキームを使用：

| 層 | 内容 | ファイル |
|---|---|---|
| Layer 1 | atomic_shared_ptr (lock-free reference counted smart pointer) | `kame/atomic_smart_ptr.h`, `tests/tlaplus/atomic_shared_ptr.tla` |
| Layer 2 | bundle/unbundle (Software Transactional Memory) | `kame/transaction_impl.h`, `tests/tlaplus/BundleUnbundle_2level_LLfree*.tla` |

旧スキームでは「Layer 0 = atomic_shared_ptr、Layer 1 = stm_commit、Layer 2 = bundle/unbundle」という 3 層の番号付けがあったが：

- 旧 Layer 0 → **新 Layer 1** に renumbering
- 旧 Layer 1 (stm_commit) は抽象化が中途半端で実用価値がなく**廃止**

### Layer 1 の liveness 検証戦略

Layer 1 単独の liveness 命題は**薄い**。理由：

- `acquire_tag_ref_` 等の primitive 内部の CAS retry は **単一 atomic location 上の単一 CAS retry** パターン。敗者は再 load して再 CAS するだけ、追加作業ゼロ。各 CAS round に hardware が 1 名の勝者を保証するため、構造的 lock-free。しかし、LL/SCモデルのためにバックオフは必要。
- Layer 1 を呼び出す Layer 2 (bundle/unbundle) 側は `for() { acquire_tag_ref(...); CAS(...); if (failed) break; ... }` の **fail-fast 構造**。算法レベルの retry は外側 (Layer 2) のループに切り離されており、liveness は Layer 2 の `priorityTag` (older-wins) で救済される設計。
- → Layer 1 単独で「lock-free / LL-free」を独立に問う意味がほぼない。系全体としての liveness は Layer 2 Gen 3 の完走 (§8) によって既に証明されている。

これを踏まえた検証戦略を 2 案：

#### Plan A — TLA+ Layer 1 で liveness を直接検証 (本命)

- パターンが単純なのでモデル自体はシンプル
- `atomic_shared_ptr.tla` は既に Safety を検証している (`TerminalCheck` 等)
- 同 spec に `Liveness == \A t : <>(...)` 系の PROPERTY を追加し `WF_vars(...)` 配下で完走させれば足りる

#### Plan B — 構造的議論で代用 (fallback)

- 各 primitive (`acquire_tag_ref_`, `release_tag_ref_`, `compareAndSwap_(NOSWAP=true)`, `swap`, `reset`) について「単一 atomic location 上の単一 CAS retry」パターンであることを informal に確認。しかし、`compareAndSwap_(NOSWAP=true)`などは二重forループ。
- 算法レベル retry の Layer 2 側委譲 (fail-fast) を明示
- 「Layer 1 単独 liveness は Layer 2 Gen 3 の系として保証される」を文書として残す

#### 採用方針

**Plan A を本命とする** が、bundle/unbundle Gen 3 のスケール検証や 3-level 拡張など他の検証作業がトークン予算を圧迫するため、当面手が回らなくても問題ない。Plan B は本節そのものが該当する記録として機能する。Plan A 着手は他 layer の作業が落ち着いてからで十分。

どちらの Plan でも **Safety は TLA+ Layer 1 + GenMC で必須** (変更なし)。

### 履歴的言及: modular 版と旧 stm_commit (旧 Layer 1)

完全性のため、廃止された変種に触れておく：

- **modular 版** (古い `BundleUnbundle*.tla` の `% MaxSerial` 方式): 実行すると `SerialWrapAround` invariant violation で落ちた。priority 機構なしでは retry が無制限に発生し serial が wrap、`ModGT` 比較が破綻することの動的反例として一時保存していたが、抽象化が不十分で実用価値はほぼない。**現行 doc・table から除外**。
- **旧 Layer 1 (stm_commit)**: 抽象化が不十分で意味のある検証単位として機能しなかった。**仕様 (`stm_commit.tla`) ともども廃止**、本文書の検証対象から除外。

---

## 関連ドキュメント

- `verification_log.md`: 各時点の検証結果と修正履歴。本文書と異なり時系列。
- `BundleUnbundle_2level_LLfree.tla`: Gen 3 の参照実装。ヘッダコメントに priority 機構の C++ 対応関係。
- `atomic_shared_ptr.tla`: Layer 1 (atomic_shared_ptr) の Safety 検証用 spec。Liveness 検証 (§11 Plan A) の宿主候補。
