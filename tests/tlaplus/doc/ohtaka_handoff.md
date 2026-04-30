# Ohtaka スパコン TLC 検証 引き継ぎ

## 前提

- ラップトップ検証済み cfg は `verification_log.md` 参照（全 PASS）
- spec: `BundleUnbundle_3level_LLfree.tla`, `BundleUnbundle_2level_LLfree.tla`
- TLC: `tla2tools.jar` (2026.04.06.150550)

## 実行コマンド

```bash
java -XX:+UseParallelGC -Xmx<MEM>g -cp tla2tools.jar tlc2.TLC \
  -workers auto \
  -config <CFG>.cfg \
  <SPEC>.tla > <OUTPUT>.log 2>&1
```

- `-deadlock` は不要（`Terminating` disjunct で吸収、proof_semantics.md §6）
- `-Xmx`: i8cpu ノードなら `-Xmx180g` 程度
- 出力は必ずファイルにリダイレクト（grep パイプは行欠落のリスクあり）

## Lamport counter 出力

### `PrintTerminalMaxCounter`（スパコン用・最小バイト数）
AllDone 状態ごとに最大 counter のみ出力:
```
22
```

cfg で `INVARIANT PrintTerminalMaxCounter` を使用。
counter の意味: `counter = serial ÷ SerialBase`, `SerialBase = 1 + |Threads|`

### `PrintTerminalSerial`（ラップトップ用・詳細）
AllDone 状態ごとにスレッド別 serial を出力:
```
<<"Terminal serial[t]:", <<52, 62>>>>
```

## スパコン対象 cfg（優先順）

### Tier 1a: 2L 3thr fine（実行中 — 先に完了させる）

3-way priority resolution の基本正しさを確認。fine は簡略抽象だが、
FAIL すれば superfine に進む意味がないため、スクリーニングとして先行。

| cfg | spec | Threads | MaxCommits | 実測 distinct | メモ |
|---|---|---|---|---|---|
| `2level_LLfree_3thr_mc` | 2L | {1,2,3} | 1 | **650M+ (24h f1fat, queue 114M)** | 実行中、`PrintTerminalMaxCounter` 追加要 |

f1fat 24h 時点 (2026-04-30): 3.9B generated, 650M distinct, 114M queue, depth 49。

### Tier 1b: 2L 3thr superfine（fine PASS 後）

C++ 忠実モードでの最終証明。Phase 0 prestamp CAS と Phase 3 DISTURBED 検出を含む。
2thr スケーリング比: fine 804K → superfine 2.5M (3.1x)。

| cfg | spec | Threads | MaxCommits | 推定 distinct | メモ |
|---|---|---|---|---|---|
| (新規) `2level_LLfree_3thr_superfine_mc` | 2L | {1,2,3} | 1 | **2B+** (fine 650M × 3x) | i8cpu 推奨 |

### Tier 2: 3L 3thr（2L superfine 完了後）

| cfg | spec | Threads | MaxCommits | 推定 distinct | メモ |
|---|---|---|---|---|---|
| (新規) `3level_LLfree_3thr_mc` | 3L | {1,2,3} | 1 | **数十B** | 2L の数倍以上 |

### Tier 3: MaxCommits=2（低優先）

2thr MaxCommits=1 で全 CAS パスを網羅済み。旧 spec (非LLfree) で MaxCommits=2
PASS 済みのため追加検証価値は薄い。

| cfg | spec | Threads | MaxCommits | 推定 distinct | メモ |
|---|---|---|---|---|---|
| `2level_LLfree_commits2_mc` | 2L | {1,2} | 2 | 30–50M | ラップトップで 33.6M まで確認 |

## cfg 作成テンプレート

### 2L 3thr に PrintTerminalMaxCounter 追加

`BundleUnbundle_2level_LLfree_3thr_mc.cfg` 末尾に追加:
```
INVARIANT PrintTerminalMaxCounter
```

### 3L 3thr micro（新規: `BundleUnbundle_3level_LLfree_3thr_mc.cfg`）

```
\* 3-thread 3-level LL-free micro. Ohtaka target.
\* Walk/CAS = superfine (root-first), Collect/Phase3 = fine.

SPECIFICATION Spec

CONSTANTS
    Threads = {1, 2, 3}
    Grand = Grand
    Parent = Parent
    Child1 = Child1
    Child2 = Child2
    Null = Null
    MaxCommits = 1
    UnbundleWalkAtomic  = "superfine"
    UnbundleCASAtomic   = "superfine"
    BundleCollectAtomic = "fine"
    BundlePhase3Atomic  = "fine"
    Privilege = TRUE

INVARIANT SnapshotConsistency
INVARIANT NoPriorityLoss
INVARIANT BundleChainValid
INVARIANT BundledByCorrect
INVARIANT MissingPropagation
INVARIANT TerminalPayloadCheck
INVARIANT QuiescentCheck
INVARIANT DebugSerialBound
INVARIANT PrintTerminalMaxCounter

PROPERTY EventuallyAllDone
```

## 結果の読み方

```bash
# PASS 確認
grep "Model checking completed" output.log

# 統計
grep "states generated.*distinct" output.log | tail -1
grep "depth of" output.log
grep "^Finished" output.log

# Lamport max counter 一覧 (PrintTerminalMaxCounter 使用時)
grep -E "^[0-9]+$" output.log | sort -n | uniq
# → min = 先頭, max = 末尾

# terminal state 数 (重複あり — unique counter 数ではない)
grep -E "^[0-9]+$" output.log | wc -l
```

## 注意事項

- `Privilege = FALSE` は意図的に発散（proof_semantics.md §4–§5）
- `SYMMETRY` は使用不可（`TagOlder` が Nat 順序を要求、liveness も mask される）
- `MaxSerial` は LLfree spec の CONSTANTS から削除済み（cfg からも除去済み）
- TLC `PrintT` は `System.out.println()` 経由でスレッドセーフ（文字混在なし）
