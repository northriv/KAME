# Ohtaka TLC 検証 引き継ぎガイド

検証結果の全詳細は `verification_log.md` を参照。本文書は ohtaka 投入の操作手順と現在の pending ジョブのみを記載する。

---

## 実行コマンド

```bash
java -XX:+UseParallelGC -Xmx<MEM>g -cp tla2tools.jar tlc2.TLC \
  -workers auto \
  -config <CFG>.cfg \
  <SPEC>.tla > <OUTPUT>.log 2>&1
```

- `-deadlock` 不要（`Terminating` disjunct が AllDone 終端状態を吸収。`proof_semantics.md §6`）
- `-Xmx`: i8cpu ノード (384 GB) なら `-Xmx180g` 程度
- 出力は必ずファイルにリダイレクト（grep パイプは行欠落のリスクあり）
- sbatch スクリプト: cfg を `CFG=<cfg filename>` 環境変数で渡す

### sbatch 例

```bash
# 3L superfine confA/B (safety のみ)
CFG=BundleUnbundle_3level_LLfree_3thr_superfine_A_mc.cfg \
  sbatch ohtaka/run_l3llf_3thr_superfine_live.sh

CFG=BundleUnbundle_3level_LLfree_3thr_superfine_B_mc.cfg \
  sbatch ohtaka/run_l3llf_3thr_superfine_live.sh

# safety PASS 後に live 版
CFG=BundleUnbundle_3level_LLfree_3thr_superfine_A_live_mc.cfg \
  sbatch ohtaka/run_l3llf_3thr_superfine_live.sh

# 2L dynamic 3thr release
CFG=BundleUnbundle_2level_LLfree_dynamic_3thr_release_mc.cfg \
  sbatch ohtaka/run_l2llf_dynamic_live.sh
```

---

## 現在の pending ジョブ

| spec | cfg | 状況 | 備考 |
|---|---|---|---|
| **L1** | `atomic_shared_ptr_3thr_cas_mc.cfg` | **要投入** | 3-thread + load+CAS+scope, MaxOps=2, SYMMETRY. Local 41% / 1h44m で queue 増加 → ohtaka 必須。`sbatch run_atomic_shared_ptr.sh` |
| **L1** | `atomic_shared_ptr_3thr_full_mc.cfg` | **要投入** (cas PASS 後) | + EnableSwap=TRUE |
| 3L | `_3thr_superfine_A_mc.cfg` | **要投入** | casOldWrappers fix 適用済み |
| 3L | `_3thr_superfine_A_live_mc.cfg` | safety PASS 後 | |
| 3L | `_3thr_superfine_B_mc.cfg` | **要投入** | casOldWrappers fix 適用済み |
| 3L | `_3thr_superfine_B_live_mc.cfg` | safety PASS 後 | |
| 2L-dyn | `_3thr_release_mc.cfg` | **要投入** | 3-thread + release |
| **3L-dyn** | `_3thr_release_mc.cfg` | **要投入** | casOldWrappers fix 適用済み, local 2t PASS ✅ |

### 完了済み (最新)
- **2L superfine 3t confC live — 137M states, depth 96, 2h53m, PASS + liveness (ohtaka, slurm-2906729, 2026-05-13)**
  - PHASE=1 (safety, slurm-2905644) は 16:23 で終了 → checkpoint 未作成のまま Exit
  - PHASE=2 は `No checkpoint found — starting fresh.` で再走 (fresh 開始)、結果 2h53m で safety+liveness 完了
- **3L-dyn release superfine 2t — 921M states, PASS (ohtaka, slurm-2898329, 2026-05-05)**
- 3L-dyn 3thr-A live — 122K states, PASS + liveness (2026-05-04)
- 3L-dyn 3thr-B live — 120K states, PASS + liveness (2026-05-04)
- 3L superfine confC live — 640M states, PASS + liveness (2026-05-03)
- 2L-dyn release superfine live — 413M states, PASS + liveness (2026-05-03)
- 2L-dyn 3thr-A/B live — 53K / 149K states, PASS + liveness (2026-05-03)

---

## 出力の読み方

```bash
# PASS 確認
grep "Model checking completed" output.log

# 統計
grep "states generated.*distinct" output.log | tail -1
grep "depth of" output.log
grep "^Finished" output.log

# Lamport max counter 分布 (PrintTerminalMaxCounter 使用時)
grep -E "^[0-9]+$" output.log | sort -n | uniq -c
#  → 左列: terminal state 数, 右列: counter 値

# min/max counter
grep -E "^[0-9]+$" output.log | sort -n | awk 'NR==1{min=$1} END{print "min="min, "max="$1}'

# terminal state 総数
grep -E "^[0-9]+$" output.log | wc -l
```

counter の意味: `counter = serial ÷ SerialBase`, `SerialBase = 1 + |Threads|`

---

## cfg ガイド

### PrintTerminalMaxCounter vs PrintTerminalSerial

| invariant | 出力 | 用途 |
|---|---|---|
| `PrintTerminalMaxCounter` | 各 AllDone 状態で最大 counter 1 整数 | ohtaka（バイト数最小） |
| `PrintTerminalSerial` | 各 AllDone 状態でスレッド別 serial タプル | ラップトップ（詳細） |

ohtaka 向け大規模 cfg は `INVARIANT PrintTerminalMaxCounter` を使用。

### PROPERTY の方針

- liveness (`PROPERTY EventuallyAllDone`) は状態数に対して非線形に高価（SCC 解析）
- 2-thread coarse で liveness 証明 → 全粒度に波及（`proof_semantics.md §10`）
- 100M 超の ohtaka cfg は原則 INVARIANT のみ（live 版は `_live_mc.cfg` を別途）

### 注意事項

- `Privilege = FALSE` は意図的に発散（proof_semantics.md §4–§5）
- `SYMMETRY` 使用不可（`TagOlder` が Nat 順序を要求、liveness も mask される）
- `DebugSerialBound` は定数レベル TRUE（Lamport serial は非有界のため）— TLC 警告は無視してよい
- TLC `PrintT` はスレッドセーフ（`System.out.println()` 経由）

---

## spec の現在の状態 (2026-05-04)

| spec | ファイル | 主な修正履歴 |
|---|---|---|
| 2L LLfree | `BundleUnbundle_2level_LLfree.tla` | GenSerial Lamport, UnbundleWalk root-first, Terminating disjunct |
| 2L dynamic | `BundleUnbundle_2level_LLfree_dynamic.tla` | 上記 + BundlePhase1 second disjunct (deadlock fix) |
| 3L LLfree | `BundleUnbundle_3level_LLfree.tla` | 2L 修正 + InnerPhase2/3/4 outer-bundle-clear on DISTURBED + **casOldWrappers fix (2026-05-05)** |
| **3L dynamic** | **`BundleUnbundle_3level_LLfree_dynamic.tla`** | **3L static + 2L dynamic 統合. BundlePhase1 dispatch on bundleNode, SubDomainOf = AllChildren, InnerPhase3 fixed grandchild set, CommitGrand dynamic discovery. casOldWrappers fix (2026-05-05)** |

詳細な修正経緯は `verification_log.md` の Notes 節を参照。

### 3L dynamic sbatch コマンド (local sanity PASS 後に投入)

```bash
# 3L dynamic release superfine 2t (all-roles, PHASE=1 safety only) — ✅ DONE (slurm-2898329)
PHASE=1 CFG=BundleUnbundle_3level_LLfree_dynamic_release_superfine_mc.cfg \
  sbatch run_l3llf_dynamic_live.sh

# 3L dynamic 3thr release (insert/root/leaf + all release) — casOldWrappers fix 適用済み
CFG=BundleUnbundle_3level_LLfree_dynamic_3thr_release_mc.cfg \
  sbatch ohtaka/run_l3llf_3thr_superfine_live.sh
```

### 3L dynamic ローカル sanity チェック手順

```bash
cd tests/tlaplus

# 1-thread (< 1s)
java -Xmx4g -cp tla2tools.jar tlc2.TLC \
  -config BundleUnbundle_3level_LLfree_dynamic_1thr_mc.cfg \
  BundleUnbundle_3level_LLfree_dynamic.tla

# 2-thread coarse (目安: 数分)
java -Xmx8g -workers auto -cp tla2tools.jar tlc2.TLC \
  -config BundleUnbundle_3level_LLfree_dynamic_coarse_mc.cfg \
  BundleUnbundle_3level_LLfree_dynamic.tla

# 2-thread coarse with release
java -Xmx8g -workers auto -cp tla2tools.jar tlc2.TLC \
  -config BundleUnbundle_3level_LLfree_dynamic_release_coarse_mc.cfg \
  BundleUnbundle_3level_LLfree_dynamic.tla
```
