# kamepoolalloc §75 realtime work — Linux-side handoff (**CLOSED**)

Both items are done. The results live in `design/RT_READINESS.md` — §G9 for
item 1, §G6(a) for item 2 — which is the permanent home; this file is kept
for the *method*, and in particular for the environment traps below, which
cost real time to find and which anyone re-running these measurements on
Linux will hit again.

Host used: 4 vCPU Intel Xeon @ 2.80 GHz (Firecracker microVM, 16 GiB),
Ubuntu 24.04, glibc 2.39, kernel 6.18.5, GCC 13.3, Release. `ctest` 18/18
green on **both x86-64 and `-m32`** (see the ILP32 note below).

**Not a realtime kernel.** `PREEMPT_DYNAMIC`, not `PREEMPT_RT`
(`/sys/kernel/realtime` absent); no `isolcpus` / `nohz_full` / `rcu_nocbs`;
`sched_rt_runtime_us` at the default 950 ms/1 s, so the `SCHED_FIFO prio 80`
the harness reports getting is still throttled and preemptible; 4-vCPU guest,
so steal time is in every sample.  The G6(a) mechanism results (does
`MADV_NOHUGEPAGE` actually stop 2 MiB zeroing) are unaffected by that — they
are page-fault-path facts — but the `MAX` / `p99.99` cells are not WCET
numbers and §G6(a) says so explicitly.  Anyone re-running this to establish a
bound needs a `PREEMPT_RT` host with the measured thread on an isolated,
`nohz_full` core.  **A recipe for standing one up is at the end of this file**
("Standing up the `PREEMPT_RT` host").

---

## Item 1 — G9 negative control: does the regression test have teeth? — **YES**

The question was narrow: with the `3145e139a` guard reverted, does
`alloc_thread_exit_unarmed_test` fail on Linux? If not, the test is toothless
and needed strengthening rather than closing.

It fails, with exactly the predicted signature.

| | `units_live` | `chunks_live` |
|---|---|---|
| guard present (both linkages) | 6 → 10, plateau from cycle 40 | 5 → 6, plateau |
| guard reverted (both linkages) | 26 → 129 | **25 → 125 over 100 cycles = +1/cycle** |

The revert was minimal — `|| kame_thread_torn_down()` dropped from the
`l1_push` guard and nothing else, restoring exactly the pre-`3145e139a` gap.
`l1_pop_fit`'s guard (which is about `g_lrc_l1_threads` counter drift, not
stranding) was left alone. Guard restored afterwards; nothing reverted was
committed.

Two things confirmed rather than assumed, both recorded in §G9:

* **glibc ordering.** A probe registering both a C++ `thread_local` destructor
  and a `pthread_key` destructor on one thread shows the `thread_local` one
  running first on glibc 2.39 — the window is real, and it is the window macOS
  does not open.
* **Which term is load-bearing.** Instrumenting `l1_push` at the moment the
  consumer's `pthread_key` destructor frees the foreign block shows
  `s_l1_drained == 0` but `kame_thread_torn_down() == 1`. The consumer never
  armed its L1, so the pre-existing `s_l1_drained` check cannot fire; its
  *bucket-tier* TLS was armed, so `AllocThreadExitCleanup`'s `thread_local`
  destructor had already set `s_alloc_tls_off`. The added term is the only
  thing closing the window, which is what makes the test a real regression
  test for `3145e139a` and not a coincidence.

---

## Item 2 — G6(a) `MADV_NOHUGEPAGE` — **implemented, opt-in, measured**

Shipped as `kame_pool_set_thp_policy()` / `kame_pool_get_thp_policy()` with
`KAME_THP_SYSTEM` / `ALWAYS` / `NEVER`. Full rationale, numbers and the
"should realtime mode imply it?" decision (**no**, and the measurement backs
the instinct: up to +58 % on a TLB-bound working set is too surprising for a
knob documented as silencing background maintenance) are in §G6(a).

Three things the original plan did not anticipate, all found by measuring:

1. **The large-VA tier needed the advice more than the regions did.** The plan
   named only `mmap_new_region()`. But blocks above `LRC_HI` are a fresh
   32 MiB-aligned mmap per allocation and are the coldest, largest memory the
   pool hands out — advising only regions left every one of their 2 MiB spans
   faulting as a hugepage. `large_va_raw_map()` now carries the same policy.
2. **`MADV_NOHUGEPAGE` does not split hugepages that already exist.** It stops
   future hugepage faults and future khugepaged collapses, which is what
   matters for latency, but a range already backed by THP stays backed. Hence
   the documented call order: policy **before** prewarm.
3. **`KAME_THP_SYSTEM` cannot be re-applied.** Linux has no "clear" advice;
   `MADV_HUGEPAGE` and `MADV_NOHUGEPAGE` each clear the other's flag and
   neither restores neutral. Policy 0 returns 0 from the re-advise walk and
   affects new regions only.

### Traps to know before re-running any of this on Linux

* **Containers commonly set `PR_SET_THP_DISABLE` on the whole process tree.**
  This one invalidates everything silently: every VMA reports
  `THPeligible: 0`, `AnonHugePages` stays 0 no matter what you advise, and
  both the "policy 2 holds it at 0" check and every latency arm pass for the
  wrong reason. `/sys/kernel/mm/transparent_hugepage/enabled` says `[always]`
  the whole time and tells you nothing.
  Check `THP_enabled:` in `/proc/self/status` (0 = disabled). Clear it with a
  tiny wrapper that calls `prctl(PR_SET_THP_DISABLE, 0)` and then `execvp`s
  the real binary — the flag is in `MMF_INIT_MASK`, so the cleared state
  survives the `exec`. **Always run a control first**: a plain
  `mmap` + `MADV_HUGEPAGE` + `memset` should show `AnonHugePages` equal to the
  touched size. If the control is 0, nothing downstream means anything.
  (Sub-test (2e) in `alloc_rt_thread_test` encodes this: it measures a
  baseline and *skips* rather than asserting when the host backs no THP.)
* **`bench_rt_wcet` could not see a hugepage fault at all**, for two reasons
  worth knowing: its first touch was outside the timed window (the
  `now_ns()` pair bracketed only `kame_pool_malloc`), and even once timed, the
  steady-state loop is prewarmed and the large tier hands back a pointer whose
  first page the allocator itself has already written a header into. Fixed by
  timing the touch (`1st-touch` histograms) and adding `--faults`, which times
  one write per 4 KiB page across freshly-mapped `> LRC_HI` memory. `--thp
  system|always|never` selects the arm.
* **The arms cannot be interleaved inside one process** — see (2) above, the
  policy cannot be un-applied to faulted memory. A/B across *processes*,
  alternating order, median of ≥ 7. On a 4-vCPU shared VM the max is dominated
  by preemption (single-run maxima ranged 0.98–32.8 ms for the same arm), but
  p50/p99.9/p99.99 medians were stable to within a bucket across independent
  sessions.
* **`bench_loop` cannot measure the THP cost** — it keeps one block live, so
  its working set is TLB-trivial. `tests/bench/bench_tlb.c` was added for
  this: a dependent random pointer chase over a pool-allocated working set,
  which is deliberately the *worst* case for TLB reach.
* **Do not use an *unadvised* range as the "THP is on" baseline.** Under the
  common `defrag = madvise` setting the kernel will not compact to find a
  2 MiB block for a range nobody asked about, so a `KAME_THP_SYSTEM` baseline
  reads 0 kB as soon as memory is mildly fragmented — and a check written as
  "baseline > 0, then assert NEVER == 0" then skips itself for no reason.
  With a `SYSTEM` baseline, sub-test (2e) passed 64-bit and skipped 32-bit on
  the same host and kernel. Use `MADV_HUGEPAGE` as the baseline arm; it earns
  the compaction effort and makes the A/B deterministic.

### ILP32

Worth doing, since the standalone library claims Linux 32-bit support and
this work touches page-level code. `apt-get install gcc-multilib
g++-multilib`, then configure with `-DCMAKE_C_FLAGS=-m32
-DCMAKE_CXX_FLAGS=-m32 -DCMAKE_EXE_LINKER_FLAGS=-m32
-DCMAKE_SHARED_LINKER_FLAGS=-m32`. Result: 18/18, warning-free, and (2e)'s
behavioural half passes with the same 10,240 kB vs 0 kB figures as 64-bit —
THP is a kernel page-table property, not a function of pointer width, so an
ILP32 process on an x86-64 kernel gets hugepages normally.

The two defects that build found were both in the new test/bench code, not
the library: a `%lx`-into-`uintptr_t` scan in (2e)'s smaps parser (ILP32
`uintptr_t` is `unsigned int`), and a `size_t` overflow in `bench_tlb` for a
multi-GiB working set that surfaced as a misleading "working set too small".
Both fixed. The library itself needed nothing — its ILP32 handling
(`ALLOC_MAX_REGIONS = 96` = 3 GiB, `RADIX_VA_LIMIT = ~0`) predates this work.

### Reproducing

```bash
cd kamepoolalloc && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build -j && (cd build && ctest --output-on-failure)

# THP must be available AND not prctl-disabled — see the traps above.
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

./build/tests/bench_rt_wcet --faults 24 --thp system   # tail, THP arm
./build/tests/bench_rt_wcet --faults 24 --thp never    # tail, anti-THP arm
./build/tests/bench_tlb 512 1024 6000000               # TLB cost, default
KAME_POOL_NOHUGEPAGE=1 ./build/tests/bench_tlb 512 1024 6000000
```

---

## Standing up the `PREEMPT_RT` host

This converts the one caveat the two items above could not remove — "the `MAX`
and `p99.99` cells are not WCET numbers" — into numbers that are.  Nothing
here changes a result already recorded; it is the missing *instrument*.

The target written up here is a **spare Intel iMac 27" (Retina 5K, 2017)**
running **Ubuntu Server 26.04 on its own internal SSD**, because that is the
machine this project has and because it turned out to be good enough — see the
gate result below.  Everything except the Apple-specific parts applies to any
x86-64 box with ≥ 4 cores.  Speed is irrelevant — WCET work measures
determinism, not throughput.

Total spend: nothing.  Read the gate section first; it is designed to tell you
whether a candidate machine is usable *before* you install anything on it or
buy anything for it.

Why x86-64 rather than an ARM host, given ARM has no SMM/SMI and is the more
likely long-term realtime target: the entire G6(a) analysis is built on 4 KiB
base pages with a 2 MiB PMD hugepage, and distro aarch64 kernels are split
between 4 K and 64 K base pages (with a 512 MiB PMD).  The existing numbers
are x86 and `timeStamp()`'s rdtsc path is x86.  Measure on the architecture
the corpus is in; port the corpus afterwards if the target moves.

### The gate — run this from a live USB, before installing or buying anything

**`hwlatdetect` does not need `PREEMPT_RT`.**  It drives ftrace's `hwlat`
tracer, which busy-polls the TSC with interrupts disabled and reports the gaps
— i.e. the intervals where firmware (SMI/SMM) took the CPU away from the
kernel entirely.  That is a property of the *machine*, not of the kernel, which
is why the number it gives from an ordinary live session is final.  Boot an
Ubuntu Desktop live USB (⌥ Option at the chime → `EFI Boot`; the live user is
`ubuntu` with an empty password, and `sudo` needs none):

```bash
grep HWLAT /boot/config-$(uname -r)     # CONFIG_HWLAT_TRACER must be =y
sudo apt install -y rt-tests lm-sensors
sudo hwlatdetect --duration=30m --threshold=10
```

If the tracer is not compiled in, the free gate is not available on that ISO
and you have to install the RT kernel first — so spend the 30 seconds on that
`grep` before anything else.  (A counter-example exists: the Firecracker
kernel this project's cloud sessions run on has `# CONFIG_HWLAT_TRACER is not
set`.)

Whatever it reports is a **floor no kernel setting can lower**, and on a Mac
there is no BIOS knob to attack it with.  If it is 200 µs then no allocator
claim below 200 µs is meaningful on that box, and saying so is the honest
outcome rather than publishing a number the platform manufactured.

**Result on this iMac** — Ubuntu 26.04 live session, `7.0.0-14-generic`:

```
hwlatdetect:  test duration 1800 seconds
	detector: tracer
	parameters:
		Latency threshold: 10us
		Sample window:     1000000us
		Sample width:      500000us
	     Non-sampling period:  500000us
Max Latency: 13us
Samples recorded: 1
Samples exceeding threshold: 1
ts: 1785918598.514060378, inner:0, outer:13, cpu:0
```

One 13 µs excursion in a full 1800 s run; everything else below 10 µs.
`inner:0, outer:13` places it *between* iterations of the sampling loop rather
than inside one — the ordinary shape of an SMI.  Better than a consumer x86
box has any right to be; Apple's EFI/SMC is not doing anything pathological.

Note the default duty cycle is 50 % (1 s window, 0.5 s width), so ~900 s was
actually observed and the true event *rate* is around twice what is seen.
Raise it with `--window=1000000 --width=900000` if the rate matters.  It does
not change the amplitude, which is the part that does.

**13 µs is this project's measurement floor on this host, and it belongs next
to every number the campaign produces.**  For scale, the phenomena being
chased are an order of magnitude above it — a 2 MiB huge-page zeroing fault is
~100–200 µs, and the deferred-unmap / RT-gate effects are milliseconds.

#### The untuned `cyclictest` baseline, and why it is worth keeping

Taken in the same live session — so **generic kernel, no `isolcpus`, no
affinity, desktop running**.  It is not an RT result and must not be recorded
as one; it is the *before* picture.

`cyclictest -m -p99 -t1 -i200 -d0 -D10m -h400 --quiet`, 3,000,000 samples:

| min | avg | p99.99 | p99.999 | max |
|---|---|---|---|---|
| 1 µs | **2 µs** | ~11 µs | ~21 µs | **97 µs** |

with 2,987,492 samples (99.58 %) landing in the 2 µs bucket.

Keep it because the pair decomposes the tail: firmware can account for at most
13 µs of that 97 µs max, so **~84 µs is software — scheduling and preemption
— which is exactly what the RT kernel, `isolcpus` and pinning attack.**
Neither number alone tells you how much of the tail is reachable.

One detail from the output, `# /dev/cpu_dma_latency set to 0us`: cyclictest
holds a PM-QoS request that forbids deep C-states for its own duration.  So
C-state exit latency is *already* excluded from the 97 µs above — but
`bench_rt_wcet` does not do this, which is where the `intel_idle.max_cstate=1`
family in the tuning section earns its place (after the fan check, not before).

### Boot medium — the internal blade, and what to do about the Fusion Drive

The machine has a **Fusion Drive**, which Linux does not understand: Apple's
CoreStorage / APFS Fusion is a macOS logical volume, so Linux sees two
unrelated devices (a small NVMe blade — 32 GB on 1 TB Fusion, 128 GB on
2/3 TB — and a 3.5" HDD).

Since macOS on this machine is stuck at Ventura and therefore out of security
support, the resolution is to stop keeping it: **wipe, and install to the
blade.**  An external SSD keeps macOS bootable and is the alternative if you
want that, but then the unpatched OS is a liability that depends on nobody ever
booting it, and a Thunderbolt enclosure plus a drive costs about what a used
small-form-factor PC does — at which point buying the PC dominates.

* **Update macOS fully *before* wiping.** Mac EFI/SMC firmware ships only
  inside macOS updates, so whatever is installed at wipe time is frozen
  forever.  Since the firmware is exactly what `hwlatdetect` measures, take the
  last one available.
* Check the blade's wear first — it is an 8-year-old drive that has been the
  SSD half of a Fusion pair:
  `sudo apt install nvme-cli && sudo nvme smart-log /dev/nvme0 | grep -E 'percentage_used|data_units_written'`.
  On this machine it reads **`percentage_used: 1%`** — effectively unworn, so
  the blade is fine as the system disk.  (Worth knowing *why* the fear was
  misplaced: Apple's Fusion is a **tiering** scheme where blocks migrate by
  access frequency, with only a small write buffer — not a write-through cache
  that funnels every write through the SSD.  Expect wear closer to an ordinary
  boot drive's than to a cache device's.)
  Note the blade may enumerate as AHCI rather than NVMe on some models, in
  which case it is `/dev/sda` and `smartctl -a` is the tool; `lsblk -o
  NAME,SIZE,MODEL,TRAN,ROTA` settles it. If the `nvme` command itself is
  missing, that is just `nvme-cli` not being installed in the live session.
* 32 GB is enough.  **Install Ubuntu Server, not Desktop** — no GUI is needed
  (everything here is CLI), it is a third of the size, and it removes the
  compositor from a thermal budget that already worries us.  Measured
  footprint: the repo is 52 MB and the whole `kamepoolalloc` CMake build is
  **7.6 MB**; the disk goes to the OS and toolchain, ~8–9 GB in total.
* **No swap.**  A page fault that reaches swap is unbounded, which is the
  opposite of the property being measured — and it saves the couple of GB the
  installer would otherwise take.
* **Leave the 3.5" HDD out of `/etc/fstab` entirely.**  Nothing needs it, and a
  spinning disk is interrupts and heat.
* Boot Camp Assistant is a *Windows* tool and must not be used: it can leave a
  hybrid MBR, a GPT/MBR inconsistency Linux tooling then has to fight.
* If you do go the external route after all, **use manual partitioning and
  point the bootloader at the external disk.**  Left to itself the installer
  writes GRUB into the *internal* ESP — the one way this procedure can damage
  a macOS install you meant to keep.

### Ubuntu + the realtime kernel — use 26.04 LTS, not 24.04

Now that `PREEMPT_RT` is fully upstream, **Ubuntu 26.04 LTS ships the realtime
kernel (7.0) in the main archive** — no Ubuntu Pro, no token, no `pro attach`:

```bash
sudo apt update && sudo apt install ubuntu-realtime
sudo reboot
# Confirm you actually got RT — do not skip this, a non-RT kernel still boots
# happily and every number below would then be meaningless:
cat /sys/kernel/realtime        # must print 1
uname -v | grep -o PREEMPT_RT
```

Pick 26.04 specifically.  On **24.04 and earlier the realtime kernel is behind
Ubuntu Pro** (free for personal use on ≤ 5 machines, but it is an account, a
token and an attach step).  That subscription gate used to be the one good
argument for Debian's `linux-image-rt-amd64` here; on 26.04 it is gone, so
there is no longer a reason to split the distro from whatever else you run.

Kernel 7.0 is new but buys nothing to fear on 2017 hardware: `rt-tests` is
ftrace plus userspace, and Polaris/`amdgpu` has been settled for a decade.

What *does* still matter more than the distro: the kernel must not change
under you mid-campaign.  Do not use a rolling release for a machine whose
whole purpose is reproducible numbers.

Refs: <https://ubuntu.com/real-time>,
<https://documentation.ubuntu.com/real-time/latest/reference/releases/>,
<https://documentation.ubuntu.com/real-time/latest/how-to/enable-real-time-ubuntu/>

**This host does not need to build KAME.** Only the CMake test/bench tree is
required — no Qt, no Ruby, no pybind11:

```bash
sudo apt install build-essential cmake git rt-tests
cd kamepoolalloc && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

### Tuning — all of it from the kernel command line, because it is a Mac

A Mac has **no BIOS setup screen**: Turbo, C-states and SMT cannot be disabled
in firmware.  Everything below is `GRUB_CMDLINE_LINUX_DEFAULT` in
`/etc/default/grub`, then `sudo update-grub && sudo reboot`.

```
isolcpus=nohz,domain,2-3 nohz_full=2-3 rcu_nocbs=2-3 irqaffinity=0-1
intel_pstate=disable tsc=reliable nmi_watchdog=0
```

Then, per boot (or via a unit):

```bash
# performance governor on every CPU
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# SCHED_FIFO must not be throttled — the default is 950 ms out of every 1 s
echo -1 | sudo tee /proc/sys/kernel/sched_rt_runtime_us
# mlock: the default 8 MB cap silently defeats page pinning (KAME now warns)
printf '* soft memlock unlimited\n* hard memlock unlimited\n' \
    | sudo tee /etc/security/limits.d/99-kame.conf
```

Deliberately **not** in the list above, and why:

* `nosmt` — only relevant if it is the i7-7700K (4C/8T).  The i5-7500/7600 are
  4C/4T and there is nothing to disable.
* `intel_idle.max_cstate=1 processor.max_cstate=1 idle=poll` — these are the
  usual next step when `cyclictest` shows C-state exit latency, but on this
  machine they make two cores spin at 100 % *before* you have confirmed the
  fans respond under Linux.  Add them after the fan check below, not before.

**Fans.** macOS drives the fans from the SMC; Linux may not ramp them, and a
30-minute WCET run that thermally throttles produces numbers that are about
the cooling, not the allocator.  Check `sensors` (the `applesmc` module), and
either install `macfanctld` or raise the floor by hand via
`/sys/devices/platform/applesmc.*/fan1_min`.  Run measurements from a text
console with the GUI stopped — the 5K panel and the Radeon Pro are a
meaningful share of the thermal budget.

### The scheduling floor — `cyclictest`, once the tuning above is in place

The firmware floor was already established from the live USB (the gate section
near the top of this chapter).  What the installed and tuned system adds is the
*scheduling* component on top of it:

```bash
sudo cyclictest -m -p99 -t1 -a2 -i200 -d0 -D30m -h400 --quiet
```

on the isolated core (`-a2`).  Only this run counts: a `cyclictest` taken from
the live session is a non-RT kernel with no `isolcpus` and a desktop running,
so it says nothing about the tuned machine.  The `hwlatdetect` number, by
contrast, is kernel-independent and does not need repeating.

Record both floors.  They belong in any §G6(a) revision alongside the
allocator numbers, exactly as the Ohtaka rules in `CLAUDE.md` require the
partition, node ID, governor and turbo state.

### Running the measurement

```bash
# pin to an isolated core and take the RT priority the harness asks for
sudo chrt -f 80 taskset -c 2 ./build/tests/bench_rt_wcet --faults 24 --thp system
sudo chrt -f 80 taskset -c 2 ./build/tests/bench_rt_wcet --faults 24 --thp never
sudo chrt -f 80 taskset -c 2 ./build/tests/bench_rt_wcet --faults 24 --thp always
```

Same protocol as everywhere else in this project: **interleave the arms inside
one session**, median of ≥ 5, report min/max beside it, and ≥ 10⁶ samples
before reading anything at p99.9 or deeper (see the two methodology traps at
the end of this file).  THP state is runtime-settable, so the three arms need
no reboot — but re-read the `PR_SET_THP_DISABLE` trap above before trusting an
`AnonHugePages: 0`.

What this campaign is expected to produce: §G6(a)'s "mechanism trustworthy /
absolute WCET not trustworthy" split collapses into a single set of numbers
carrying the `hwlatdetect` floor as their stated resolution.

## Context you may want

- `design/RT_READINESS.md` — the whole programme, G1–G10, with what is claimed
  and what explicitly is not. G6(a) and G9 carry these results.
- README §"The realtime contract" — preconditions → guarantees → exclusions.
- `tests/alloc_rt_thread_test.cpp` — asserts each RT claim by observing a
  counter move rather than trusting that a call was made; (2d) is the G6(b)
  `mlock` one and (2e) the G6(a) THP one. Both are written to tolerate a
  hostile CI environment (low `RLIMIT_MEMLOCK`; no THP) by skipping visibly
  rather than passing vacuously. This host: `ulimit -l` = 8192 kB, so `mlock`
  returns a short count — documented behaviour, not a failure.

## Two methodology traps already paid for (from the macOS side)

- **Deep-tail comparisons need ~10⁶ samples.** At 120 k the cross-thread arms
  ordered *backwards* at p99.9; at 4 M they were equal and the RT arm won the
  deep tail. Below ~10⁶ the deep-tail buckets hold single digits and the
  ordering is noise.
- **`p50 = 0 ns` is the clock floor, not a sub-nanosecond free.** Apple
  Silicon's `steady_clock` ticks at ~41.7 ns. On this x86 host the floor is
  far finer (clock-overhead mean 22 ns), which is why the 27 ns vs 2,048 ns
  p50 split in the fault measurements is readable at all.
