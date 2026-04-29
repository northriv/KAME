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

**つまり: Gen 3 によって micro config 範囲で Safety + LL-free が初めて formally に証明された**。C++ 実装 (commit `2d141d5`) との 1:1 対応も検証済み (8/8 priority-gated CAS site)。

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
3. **スケール検証**: MaxCommits ≥ 2, |Threads| ≥ 3, MaxPayload ≥ 4 で同一仕様が完走するか確認 (ohtaka など)
4. **3-level への拡張**: `BundleUnbundle_3level_LLfree.tla` を作成、同様の検証
5. **C++ 実装の確認**: TLA+ で「不要」と分かった C++ 機構 (たとえば read-side fairness tag) があれば実装側の整理候補にする (ただし `tags_successful_cas` は C_obs カウンタなので残置必須)
6. **Layer 0/1 と Layer 2 の合成証明** (compositional proof) は依然として未着手

---

## 関連ドキュメント

- `verification_log.md`: 各時点の検証結果と修正履歴。本文書と異なり時系列。line 127-129 の「Nat serial 不可」記述は旧世代当時の事実。
- `BundleUnbundle_2level_LLfree.tla`: Gen 3 の参照実装。ヘッダコメントに priority 機構の C++ 対応関係。
