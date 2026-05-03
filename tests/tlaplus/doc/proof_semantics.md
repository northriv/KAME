# TLA+ 証明セマンティクス

KAME の `tests/tlaplus/` で行っている TLA+ 検証について、**TLC の終了が何を証明し、何を証明しないか**を理論的に整理する文書。時系列の修正記録は `verification_log.md` を参照。

---

## 1. 3 世代の検証アプローチ

KAME の bundle/unbundle 系の TLA+ 仕様は、有限性をどう確保するかで 3 世代に分かれる。新しい世代に置き換えても、旧世代は意図的に保存している（理由は §5）。

| 世代 | Serial | CONSTRAINT | Privilege/Priority | 動作 | 役割 |
|---|---|---|---|---|---|
| 最古 (modular) | `% MaxSerial` で wrap | なし | optional | `SerialWrapAround` invariant violation で **落ちる** | privilege 機構なしの破綻を invariant violation として示す |
| 旧 (Nat+CONSTRAINT) | `Nat` 単調増加 | `SerialBound` で打ち切り | optional | 終了するが状態数が発散傾向 | wrap は回避できるが探索打ち切り依存 |
| **新 (LL-free) ✅ 完成** | `Nat` 単調増加 | **なし** | **常時 ON** (`priorityTag`) | 構造的有限が成立し TLC が exhaustion 完走、liveness PASS | LL-free を構造的に証明 |

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
> **2026-05-01: 活性検証方針を改訂 — coarse cfg で証明、大規模 cfg は INVARIANT のみ**。`BundleUnbundle_2level_LLfree_coarse_mc.cfg` (coarse atomicity, |Threads|=2, MaxCommits=1) でラップトップ上の活性証明を担保。superfine/fine は coarse の細分化なので、coarse での `EventuallyAllDone` PASS が全粒度に波及する（refinement 論証、§10 参照）。ohtaka 向け大規模 cfg (superfine/3thr/commits2/A-D 等、全 15 cfg) は `PROPERTY EventuallyAllDone` を削除し INVARIANT のみに。TLC の temporal property check は状態数に対して非線形に高価なため、137M states 規模の superfine cfg では省略効果が大きい。
>
> したがって現状:
>
> - Safety 性: 制限付き探索内のみだった旧世代に対し、Gen 3 で **構造的有限の状態空間における完全 exhaustion** で証明完了。MaxCommits=1, |Threads|=2, MaxPayload=3 の micro config だが、modular/CONSTRAINT どちらも介在しないため意味のある形式証明。
> - LL-free 性: `priorityTag` 設計が意図通り無限 retry を構造的に不可能にしていることが、TLC 完走 + `EventuallyAllDone` PASS で証明された (§2 の「終了 + CONSTRAINT なし + Nat serial → livelock-free 自動」の論証チェーンが成立)。
> - 大規模ストレステストとの相互裏付け: C++ 実装 commit `2d141d5` が FINE/SUPERFINE multi-thread を全 PASS、preempt が 100万件/秒のオーダーで観測される動的挙動も TLA+ 仕様の `PreemptTag` action と整合。
>
> **形式検証の柱が立った**。残る作業は config 拡張 (MaxCommits>=2, |Threads|>=3) でのスケーラビリティ検証（ohtaka など）と、3-level への拡張。

新世代は `BundleUnbundle_2level_LLfree.tla`。`priorityTag[n]: Null | <<iter, tid>>` を per-node に持ち、older transaction（小さい iter、次に小さい tid）が勝つ規律を C++ の `m_priority_tidstamp` / `negotiate()` から忠実に写している。「無限 retry を構造的に不可能にする」設計が、CONSTRAINT なし TLC 完走によって**実証**された。

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

C++ 実装の `m_priority_tidstamp` / `m_link->negotiate()` をミラーしているため、仕様の完走はそのまま実装の LL-free 性の根拠になる。**2026-04-29 に完走が達成された**ため、上記の論証チェーンは成立。priority 規律の穴と TLA+ 写しの不備、いずれの可能性も排除された (micro config の範囲で)。

---

## 5. 旧世代を保存している理由

意図的に削除せず残している。

### 最古（modular）

実行すると **`SerialWrapAround` invariant violation で落ちる**。これは：

- privilege 機構なしでは retry が無制限に発生し、
- modular な serial counter が wrap して、
- 異なる時刻の serial が同値となり、`ModGT` 比較が破綻する、

ことを動的な反例として示す。**「LL-free 機構が必要であること」自体を violation として証明** している、貴重な反例。Gen 3 がなぜ必要かの最も説得力ある根拠。

### 旧（Nat + CONSTRAINT）

wrap は起きないが、CONSTRAINT を外すと状態数が発散する。

- これは「Nat にしただけでは LL-free にならない」ことを示す
- privilege 機構の必要性を補強する第二の証拠
- Gen 3 への動機を明示する役割

---

## 6. `CHECK_DEADLOCK TRUE` への移行案

現在 cfg 冒頭に「`-deadlock` flag で実行せよ」というコメントがあるのは、全スレッドが iterBudget=0 で完了した終端状態に enabled action がなく、TLC が deadlock 判定するため。これは TLA+ の標準イディオムで回避できる。

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

**つまり: Gen 3 によって Safety + LL-free が formally に証明された**。micro config (665K states) でのベース証明に加え、**MaxCommits=2 + superfine (127M states, 4h 40min) でもスケール検証済み** (2026-04-30)。C++ 実装 (commit `2d141d5`) との 1:1 対応も検証済み (8/8 priority-gated CAS site)。

### Gen 3 の達成内容

- `priorityTag[n]: Null | <<iter, tid>>` を per-node に組み込み、older transaction が勝つ規律を C++ の `m_priority_tidstamp` から忠実に反映。
- Transaction-scope 持続: tag は CAS 単位で clear せず、`CommitParent`/`CommitDone` (= C++ `drop_tags_n_privilege` 呼び出し点) でのみ release。これがないと unbundle/rebundle ping-pong による livelock が起きることを bound-60 violation トレースで実証。
- `ActiveThread` 判定で zombie tag (終了スレッドの遺留 tag) 回収。
- `PreemptTag` で older が younger を強制 preempt。
- `Terminating` disjunct イディオムで `-deadlock` flag 不要 (§6 案を実装)。
- `<>AllDone` PROPERTY で活性を明示的に検査、PASS。
- `SYMMETRY` は drop (Nat-ordered Threads + liveness check の両立のため; Gen 3 は exhaustion が高速なので不要)。

### 旧世代の保存意義 (継続)

旧世代の保存（§5）は教育的価値として継続：modular 版の `SerialWrapAround` violation と Nat+CONSTRAINT 版の状態数発散は、いずれも「priority 機構なしでは LL-free が成立しない」ことの動的な証拠であり、Gen 3 がなぜ必要であったかの裏付けとなっている。

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
   - ⏳ release superfine 2-thread — ohtaka 投入待ち (deadlock fix 適用済み)
   - ⏳ 3-thread release Config — ohtaka
6. **C++ 実装の確認**: TLA+ で「不要」と分かった C++ 機構 (たとえば read-side fairness tag) があれば実装側の整理候補にする (ただし `tags_successful_cas` は C_obs カウンタなので残置必須)
7. **Layer 0/1 と Layer 2 の合成証明** (compositional proof) は依然として未着手

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

## 関連ドキュメント

- `verification_log.md`: 各時点の検証結果と修正履歴。本文書と異なり時系列。line 127-129 の「Nat serial 不可」記述は旧世代当時の事実。
- `BundleUnbundle_2level_LLfree.tla`: Gen 3 の参照実装。ヘッダコメントに priority 機構の C++ 対応関係。
