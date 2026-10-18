# stt/: the ear, ideas, hypotheses, what to test

> Numbers and sources: `RESEARCH.md` (Part 0 is the current spine, Part 1 is the frozen baseline).
> Pole level (the law, the ear/mouth split, the shared hypotheses H1..H10, the slices A..G):
> `../IDEAS.md`. Vision: `../../IDEAS.md` section 0.
> Live test on the microphone and the acquisition list: `TODO.md`.
> Status: **the first measurements exist** (2026-10-18, real microphone, `π-ear` v0, everything
> below the shadow line). The hypotheses stay hypotheses until each one gets its own number; the
> measurements section at the bottom of this file says which numbers already exist.

---

## What this file owns

`../IDEAS.md` owns the pole: the ear/mouth split, the law as it applies to both halves, and the
hypotheses shared with the mouth. **This file owns the ear**: its state, its record, its three
decisive questions, and the hypotheses the restart produced. `../tts/IDEAS.md` owns the mouth the
same way. Nothing here repeats the pole file.

---

## The shift this file is built on

The first pass studied the ear as a **frozen** pipeline: an utterance went into a frozen encoder, came
out as a vector, was compared by cosine to an enrolled voiceprint, and a constant threshold decided.
That answered "which model", never "which paradigm". The restart (`RESEARCH.md` Part 0) moves the
centre of gravity:

| | frozen ear (first pass) | living ear (this file) |
|---|---|---|
| where the learning lands | nowhere, after training | the state, continuously |
| the voiceprint | enrolled in advance | grown out of what is heard |
| the criterion | a constant, chosen by hand | measured from the sensor's own margin noise, never counted from the inventory |
| cold start | useless until enrolled | useful from second zero, with an empty state |
| the output | `[timestamp] text` | events: text + speaker + voice physics, all numbers |
| the state | implicit, inside the weights | an artifact on disk: readable, inspectable, resettable |

**The encoder stays.** It stops being the system and becomes a sensor. That single sentence is the
difference between the two passes.

---

## The second law, and the row of my own table it kills (KπX, 2026-10-19)

**No constant, no threshold, no hand-set value in any decision path. Ever.** The row of the table
above that says *"the criterion: measured from the sensor's own margin noise, never counted from
the inventory"* is **still a heuristic**: it is my formula, `margin >= z_margin * sigma`, with
`z_margin` chosen by me. Measured consequence: on the real sessions it admits everything and holds
one entry for four voices, because the margin it compares against is 0.0005 of noise. The row is
corrected here and the constant dies with the rest of the decision layer.

**The direction is not data science.** Classic practice cleans a dataset, trains, deploys, then
finds out in the field whether it works, and a correction costs a new dataset and a new training
run. π does the opposite: **a model with good bases, alive and human**, imperfect by construction,
corrected **one case at a time**, in place, from use.

| | Classic pipeline | π |
|---|---|---|
| where the truth comes from | a curated dataset, beforehand | the stream, in place |
| the imperfections | hidden until deployment | assumed, corrected as they appear |
| a correction | new dataset, new run, new deploy | one labelled case, obeyed now, absorbed into the weights |
| afterwards | unknown | **the error does not come back** |

**The paradigm, in his words (KπX, 2026-10-19): human models.** We aim at no score. We preach by
demonstration, meaning the ability to show a model that is not perfect but fallible, and that learns.
This holds for every sub-model: each one must at least learn. The stance is counter-current on
purpose: refusing the current binary world's habit of spending its time on training first, then
inference.

**What the ear becomes, concretely:** one sensor that embeds (trained), one head that judges a pair
of embeddings as same or different with a third output for "neither" (trained, small, ours), one
learned speech/non-speech judgement in front. `E2` and `E3` above stop being about a threshold and
become about whether the head's verdict reproduces what KπX says it is. Nothing else in this file
changes: the state, the record, the corpus and the correction channel are what make the model
replaceable without losing anything.

**Open, and deliberately not decided: the form of a correction.** How KπX says "from this second to
that second there were two voices", and how that case is written down, is a question to think through
before anything is fixed. This file carries no format until it is.

---

## Hypotheses (each one is falsifiable)

| # | Hypothesis | How to falsify it |
|---|---|---|
| E1 | An explicit state (a JSON plus a hash-keyed embedding store) makes the ear useful at t=0 from an empty file, so no enrollment ritual is ever needed. | Delete the state, speak 30 s, and check whether a usable record exists. If nothing usable appears before a manual enrollment, E1 is dead. |
| E2 | Evidence-gated registration (a speaker exists only after K mutually compatible embeddings) prevents phantom speakers. | Play a cough, a door and a chair 20 times each. Count the speakers the system created. |
| E3 | **FALSIFIED, 2026-10-19, on real audio.** The adaptive threshold `tau_t = min(tau_base + lambda*N_t, tau_max)` does not beat a constant: it is worse than any constant. Measured on 379 seconds of real speech and film: the counter fed itself, so every registration moved the boundary, the boundary admitted the next voice, and 56 identities were created for two real voices with tau pinned at its ceiling. Replaced by a criterion measured from the sensor's own margin noise, which holds one entry on the same log. | Run the same log twice, once adaptive and once with the best fixed constant found by hand. If the fixed one wins, E3 is unnecessary complexity. |
| E4 | A speaker's identity can be a short hash of its committed centroid, stable across process restarts. | Restart on the same day's audio and compare keys. |
| E5 | The event record is cheap: under 400 bytes per excerpt, ten hours under 10 MB. | Measure it. Do not estimate it. |
| E6 | `f0_pctl`, relative to the speaker's own observed distribution, separates speakers better than any absolute frequency. | Two speakers, same sentence. Does the percentile separate them, and does raw Hz alone? |
| E7 | Test-time adaptation in bias-only mode improves the embedding under a shifted condition (whisper, another microphone) without degrading the normal one. | Measure both conditions before and after adaptation. |
| E8 | The record is sufficient: the agentic pole can decide from it without re-deriving anything the ear already knew. | Hand the log to a decision loop and count what it has to recompute itself. |
| E9 | The whisper regime is a family-B problem (adapt online) rather than a fine-tuning problem. | TTT-online against fine-tuning, same 30-phrase corpus, same budget. |

---

## The record contract (the ear's whole output)

One line of JSONL per emitted excerpt, every field a number, nothing prose. The full example and its
rationale are in `RESEARCH.md` R0.7. Three rules that are not negotiable:

1. **`decided_at` exists from the first line.** The record is written immediately, the attribution may
   be revised when the evidence lands. A revision after the fact is normal, not an error.
2. **`kind` is `known`, `new`, `candidate`, `unknown` or `silent`.** An undecided excerpt is a
   legitimate record. The ear never guesses to look complete.
   **Corrected 2026-10-18 against the running implementation:** the original line named three
   kinds. The POC distinguishes five, and the two extra ones are not decoration. `candidate` is an
   observation that clusters with something but has not crossed the evidence bar yet: without it,
   the evidence mechanism is invisible while it works. `silent` is a window with no speech at all:
   without it, "there is no speaker" is indistinguishable from "I am not sure which speaker".
3. **No verdict labels.** Numbers only: no "grave", no "calm", no "stressed", no intent. Any
   interpretation belongs downstream. A label written by the ear is a decision the ear had no right
   to make.

And the storage split that makes it cheap: events in an append-only JSONL, embeddings in a
hash-keyed store with **one row per speaker or candidate, never one per excerpt**.

---

## The three questions, and nothing else, for now

Stated fully in `RESEARCH.md` R0.11 and turned into a running plan in `TODO.md`:

1. Can the ear produce a log it is worth reading, while deciding nothing at all?
2. Does the state build itself, single-pass, with a threshold that moves on its own?
3. Does it break? A cough, a door, a television, a second person, another microphone.

Until those three have numbers attached, no threshold value means anything.

---

## Anti-goals for the ear

- Any label written as a verdict (regime, mood, intent). Numbers only.
- Any setup ritual: enrollment, calibration, a first-run wizard.
- Any threshold set before it has been measured.
- Raw audio leaving the machine, and raw audio kept longer than the test for which it was recorded.
- A second VAD. The one we have is neural and sufficient.
- A decision the ear has no right to make: whether to act, whether to answer, whether to speak.
- Optimising the encoder before the state exists. The encoder is the sensor, the state is the system.

---

## Measurements (real, 2026-10-18, this machine's microphone)

The POC `π-ear` (v0: numpy only, no model, no download) was run on real audio the same day it was
built. These are the first numbers inside π, and they show what an honest cold start looks like.

**Environment.** `plughw:0,6` (the DMIC array) accepts 16 kHz mono s16 natively, so nothing
resamples. `arecord --use-strftime --max-file-time 1` produces one chunk per second, 32 000 bytes
each, and because the names come from `%Y%m%d-%H%M%S`, lexicographic order is chronological order.
That directory is the entire transport between the two servers.

**The mechanism, on four seconds of room silence, with the first gate (voiced_floor 0.05, which
turned out to be too permissive):**

| window | kind | cos to best | tau | f0 | voiced ratio | level |
|---|---|---|---|---|---|---|
| 1 | unknown | none | 0.985 | 275.9 Hz | 0.08 | -69.9 dB |
| 2 | candidate | 0.975 | 0.985 | 179.8 Hz | 0.13 | -65.3 dB |
| 3 | new, key `11cb4660f7210774` | 0.996 | 0.985 | 186.0 Hz | 0.15 | -62.4 dB |
| 4 | known, same key | 0.999 | 0.987 | 181.8 Hz | 0.12 | -63.4 dB |

Two things are proven and one is exposed.

- **Proven: the threshold moved on its own.** `0.985 -> 0.987`, exactly `tau_base + lambda*N` for
  `N = 1`. Nothing was calibrated and a criterion still existed, which is the whole claim of E3.
- **Proven: the path works from an empty state.** unknown, then candidate, then new, then known, in
  three seconds, with a key that is stable because it was computed once (E4, first half: stability
  within a run; across a restart is still unmeasured).
- **Exposed (E2, and section 0 of the pole law):** a silent room was enough to register a speaker.
  The gate was `voiced_ratio > 0.05` while room noise yields 0.08 to 0.15, so the ear invented a
  person out of nothing. That is exactly what the law forbids: it must say unknown, not invent.

**The fix, and its measurement.** The gate is now `voiced_floor`, a configuration value (environment
variable `PI_EAR_VOICED_FLOOR`, CLI flag `--voiced-floor`), default `0.25`, placed between the
measured noise band (0.08 to 0.15) and the expected normal voice (0.4 to 0.7). Re-run on five
seconds of the same room with a fresh state:

| windows | silent | observations | registered | candidates |
|---|---|---|---|---|
| 5 | 5 | 0 | 0 | 0 |

The ear now says nothing rather than something wrong. **E2 is half answered:** the evidence bar plus
the voiced gate produced zero phantoms on silence; the other half (20 coughs, a door, a chair) is
the next measurement, and it is now a one minute test with `make rec` and `make run` in two panes.

**Correction, later the same evening.** The five window result above was lucky, and it is recorded
here as what it was: a five second sample of a quiet room. Running the real thing, with `make rec`
live and a keyboard in the room, the phantom speaker came back. The cause was deeper than the gate,
and it had nothing to do with the gate:

| What was seen | What it was |
|---|---|
| `f0 = 457.1 Hz` on every window, silence included | 16000/35, the FIRST lag of the pitch search range |
| `pct = 100.000` on every window | every window had the same pitch, so the percentile meant nothing |
| `cos = 1.000` between unrelated noisy windows | identical artefacts are identical vectors |
| a speaker registered from room noise | the consequence, not the cause |

When nothing is periodic the autocorrelation decays, so its maximum lands on the shortest lag the
search allows, and the detector announced the highest frequency that range can express. Because
that artefact is identical in every noisy window, every noisy window looked like the SAME voice,
with a cosine of 1.000, and the evidence bar was satisfied in three windows. A pitch gate cannot
repair a broken pitch.

**The fix, at the physics level.** A maximum on the first lag of the search range is not a pitch, it
is the absence of one: a genuine period always peaks strictly inside the range. The detector now
returns "no pitch" on that boundary, so the artefact cannot enter a window at all. Four tests hold
it down (`tests/test_physics.py`), including the end of the chain: transient bursts register nobody.

**The measurement that decides it.** With `make rec` running live on the real microphone, and the
ear ingesting the stream from the beginning:

| windows | silent | observations | registered | candidates |
|---|---|---|---|---|
| 95 | 95 | 0 | 0 | 0 |

Ninety five seconds of real room, real keyboard and real microphone produced zero phantoms, against
five seconds before the fix. **E2 is now answered for the silence and transient half.** The other
half (a cough, a door, music, a second voice) is a one minute test: `make rec` and `make run` in two
panes, then `make snapshot` before and after.

**Measured 2026-10-19.** Separation between two real voices is now measured, and it is zero: the
two real voices of the long session scored 0.9997 against each other, with a median
best-minus-runner-up margin of 0.0005, and the inventory that once created 56 identities for them
now creates one. Everything else above is still open.

---

## The measured failure, and what replaced the counter (2026-10-19, real audio)

Two sessions on the real microphone, archived as event logs in `tests/artefacts/`: the first run
(103 windows) and the long one, a hummed note then a film with a single voice (414 windows). Both
were replayed through the inventory. This is what came out of them.

**Before.** 56 identities for two real voices. One voice was present in eight separate entries.
Every entry sat at a cosine of 0.9999 from the others: the sensor could not tell its own entries
apart, and nothing in the state said so. Four causes, measured rather than guessed:

| Cause | The number | Effect |
|---|---|---|
| the threshold counted the inventory | `tau_base + lam*N`, pinned at `tau_max` | every registration moved the boundary, so the boundary admitted the next voice |
| no merge invariant existed | 8 entries for one voice | one voice became many keys, and the record could not say which was the person |
| a growing chunk was read while it grew | 4179 events for 379 s of audio | the `k_evidence = 3` bar was diluted tenfold: three observations meant 0.3 s of voice |
| `frames` carried a sample count | 48 944 "frames" for a 3.06 s window | a record field that meant two things at once |

**After.** The same 414 windows, the same audio, through the new inventory: **1 entry**,
`registered_new = 1`, `merged_into_existing = 0`, and every window admitted to that one entry. The
counter is gone, so nothing can feed itself. The invariant is in place, so a candidate that passes
the test a window passes is absorbed instead of registered, and the absorbed key is mapped in
`absorbed` rather than lost. A chunk still being written is no longer read at all, so one second of
voice is one observation, and the evidence bar means what it says again.

**What the numbers say about the vector, and it is not flattering.** The criterion is now two
conditions: `cos_best >= tau_floor` AND `margin >= z_margin * sigma`, where `sigma` is the spread of
the margins this sensor has itself produced. On the real session the margins sit at a median of
0.0005: the best match and the runner-up are 0.0005 apart, which is nothing. Two synthetic voices an
octave apart (110 Hz and 215 Hz) sit at a cosine of 0.962, closer to each other than two windows of
one real voice are to themselves. **So v0 holds one entry for two voices, and that is the honest
answer, not a regression.** At a margin of 0.0005 no threshold could separate them; the only choice
available was which error to make, inventing 56 people or admitting that this sensor cannot tell
them apart. The second is what the pole law asks for, and it is now the behaviour.

**Consequence for v1: the criterion needs no change.** A 512-d CAM++ embedding puts different
speakers at a cosine of 0.1 to 0.3 and the same speaker above 0.5, so `tau_floor = 0.5` separates,
`sigma` becomes a real number instead of 0.0, and the invariant finally has something to bite on.
What is left to build is the sensor, not the state.

**And the audio is no longer lost.** Measured the same evening: six minutes of real test survived
only as numbers, so the exact session could not be replayed at all. The ear now copies every chunk
it ingests into a durable, bounded corpus under the home (`--corpus-dir`, one hour by default),
outside the RAM directory. That is what turns the sentence above into a plan instead of a hope.
