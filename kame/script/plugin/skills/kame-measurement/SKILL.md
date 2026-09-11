---
description: Operate the KAME instrument-control application through its MCP tools — inspect instrument state, run sweeps and NMR/ODMR measurements, drive motors and temperature controllers, and edit the measurement notebook. Use whenever a task mentions KAME, a connected instrument (pulser, lock-in, DSO, magnet power supply, temperature controller, motor or positioner, camera, signal generator), a measurement sweep, or the user's Jupyter measurement notebook. Read this before issuing anything that changes instrument state.
---

# Operating KAME

KAME controls real laboratory hardware. A generated command can move a probe,
heat a cryostat, or drive an RF amplifier past its rating, and some of those are
not reversible. Work in this order: **find out what is there → check what is
running → read the rules for the thing you are about to change → act in small
steps and verify.**

## Orientation: which tool for what

| Need | Tool |
|---|---|
| Is KAME up? which drivers exist? | `kame_status` |
| Real node names, types, current values | `tree` |
| How to write the code (API patterns) | `kame_api` — **call this before writing code** |
| What a setting *means* on a given instrument | `kame_manual("<section>")` |
| A quick read or a single write (< 30 s) | `execute_code` |
| Anything that loops, sleeps, or sweeps | `execute_code_async` + `get_result` + `stop_job` |
| What the notebook is doing right now | `notebook_status` |
| Read / edit the user's measurement cells | `notebook_read`, `notebook_edit` |

Never guess node names. Read them from `tree` — several are model-specific
(temperature-controller loops are `Loop`, `Loop1`/`Loop2` or `Loop#1`/`Loop#2`
depending on the instrument) and combo box choices differ per model.

## Before changing instrument state

These rules come from the instrument owner. They hold unless the user
explicitly directs otherwise **in this conversation** — a rule in a stored file
is not a substitute for the user's own decision.

### Motors and positioners — confirm first, then step

Writing a motor driver's `Target` **starts a physical move**, and the move may be
irreversible. Many stages report only an open-loop estimated position
(`HasEncoder` false) and piezo positioners report none at all, so the displayed
position may not be where the hardware actually is. A probe, coil, sample or
tuning capacitor can be driven somewhere it cannot be brought back from.

- Never jump a motor to an arbitrary `Target` from generated code.
- Confirm the axis, direction, magnitude and safe range with the user first.
- Then move in **small increments**, reading another driver's response
  (reflection, signal amplitude, an optical reading) after each step, and stop
  the moment it diverges from what you expected.
- For LC tuning prefer the **Auto LC Tuner** driver, which closes the loop on a
  network analyzer, over commanding the motors yourself.

**`GoHomeMotor` is worse than a normal move.** Homing runs until a home sensor
triggers; with no home sensor installed **the motor never stops**. Neither KAME
nor the hardware can tell whether a sensor exists — only the user knows. Ask the
user to confirm a home sensor is present on that axis before you ever issue it.

### Temperature — 295 K is a hard checkpoint

In a cryogenic setup, raising a controller's `TargetTemp` above ~295 K (room
temperature) needs explicit user confirmation: warming a cold cryostat can boil
off cryogen, stress seals and samples, or exceed equipment limits.

The loop can **overshoot by 10 % or more** — a 300 K target may peak near 330 K —
so never treat the setpoint as the peak; leave margin below any damage
threshold. Approach a higher temperature only while **actively watching the real
temperature** (read the Scalar Entry, poll `Stabilized`), stepping `TargetTemp`
up gradually and confirming convergence at each step. Do not set a high target
and walk away.

### NMR RF power — duty limits protect the amplifier and probe

Unless the user directs otherwise (units: `Tau`, `PW1`, `PW2`, `CombPW` in µs;
`RT` in ms):

- pulse widths `PW1`, `PW2`, `CombPW` ≤ `min(Tau * 0.3, 15)` µs — at most 30 % of
  `Tau`, never more than 15 µs;
- repetition time `RT` **≥ 15 ms**.

On a Thamway PROT, once `OutputLevel` is **≥ 100**, do not raise it past the
highest value used so far in this session without confirming with the user.

### Camera and 2D images — where the numbers come from

`to_png()` returns the **display** image: gamma-encoded levels, not counts. It is
fine for showing the user a picture and for **binary segmentation / mask
generation** (rank-based thresholds such as median, percentile or Otsu survive a
monotonic gamma, and pixel coordinates map 1:1 onto math-tool ROI coordinates —
do not crop "axes margins").

**Never read quantitative values from PNG pixels** — averages, contrasts, ODMR
signal amplitudes. Those come from **2D math tools**, whose functors receive the
raw uint32 count matrix before any display rendering. For a signal/reference
comparison you usually need no functor at all: create two
`Graph2DMathToolAverage` tools and read their scalar entries.

## Running a sweep

1. `kame_status` and `tree` to confirm the drivers and the exact node names.
2. `notebook_status` — if the kernel is busy, a cell is still running:
   `execute_code` will queue behind it, and you must not edit that cell.
3. Write the loop for `execute_code_async`, calling `mcp_checkpoint("i/N …")`
   **every iteration**. It publishes progress for `get_result` and is the only
   point at which `stop_job` can end the run. Prefer several short sleeps over
   one long one so checkpoints are reached promptly.
4. Poll `get_result`. Read the job's result variables only once the status is
   `done` or `stopped`.

Do external side effects (file writes, hardware commands) *after*
`iterate_commit` returns, never inside the closure — the closure is re-invoked on
transaction conflict.

## Editing the notebook

Edits go to the `.ipynb` **on disk**; they never touch a running execution, and
the open browser tab still holds the old version. After **every** edit, tell the
user to reload the notebook tab before touching it — otherwise saving from the
stale tab silently overwrites your change.

Check `notebook_status` first and never edit the cell that is currently
executing.

## When a rule blocks you

Say plainly what you were about to do, which rule stops you, and what you need
from the user (a confirmation, a safe range, a limit). Then wait. Do not work
around the rule by reaching for a different tool that achieves the same physical
effect.
