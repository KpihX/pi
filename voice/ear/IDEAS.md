# ear/: the ear, ideas, hypotheses, what to test

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

---

## The rebuild: π-ear v1 (2026-10-18, night, on KπX's order "refais entièrement l'oreille")

The order of `TODO.md` ("the POC is to be rebuilt, ultra compact") is executed. v0's decision layer
(`physics.py`, `inventory.py`, 582 lines, seven deciding constants) is gone; the transport, the
atomic state, the append-only record, the file controls and the corpus survived, as planned. The
package is now 1928 lines, 1407 of them code once docstrings and comments are out: 1116 for the
system itself and 291 for the downloads and the simulated rooms (`fetch.py`, `public.py`). The
decision module, `model.py`, is 374 lines of code, and `tests/test_law.py` fails if a hand-set
number comes back into it. Honest note against the target of "a few hundred lines": the system is
smaller than v0 (1543 lines of source) but not yet a few hundred; the docstrings are long on
purpose, they carry the measurements.

### The answer to "the form of a correction" (left open above, now proposed and built)

A correction is **a constraint between rows of the record**: `same`, `different`, `not_voice`,
`name`. One line appended to `corrections.jsonl`, written by `π-ear correct`. It is obeyed at the
next tick (the clusters change at once) and absorbed into the weights at the refresh that follows
(`same` adds every cross pair to the statistics `kappa` is estimated from; `different` adds a
cannot-link that no merge may cross; `not_voice` moves the windows into a noise cluster, which then
competes for every future window). Still open for KπX: whether a correction should also be able to
name a *time span* ("from this second to that second there were two voices"); rows were chosen
because they point at exact evidence.

### Sensor choice: measured, not taken from the literature

EER on real public voices, 2-second windows unless stated (lab script, 2026-10-18):

| encoder | LibriSpeech (3 spk) | FSDD (6 spk) | Piper TTS (12 voices) |
|---|---|---|---|
| **CAM++ zh-cn common (27 MB, 192-d)**, 1.0 s | 11.8 % | 12.7 % | 26.8 % |
| **CAM++ zh-cn common**, 1.5 s | 4.7 % | 9.3 % | 21.0 % |
| **CAM++ zh-cn common**, 2.0 s, **chosen** | **0.0 %** | **8.9 %** | 13.3 % |
| CAM++ en VoxCeleb, 2.0 s | 0.0 % | 17.4 % | 33.3 % |
| WeSpeaker ResNet34, 2.0 s | 3.1 % | 19.8 % | 11.7 % |
| ERes2Net base, 2.0 s | 0.6 % | 9.3 % | 6.7 % |

Synthetic voices are the hardest case: one VITS model makes 904 voices that sit closer to each
other than real people do. ERes2Net is the upgrade path if the twins case matters (38 MB).

### Three measured failures of v1 during its own build, and what they changed

| What was tried | What was measured | What replaced it |
|---|---|---|
| `kappa` estimated from the model's own clusters | the v0 loop again in a new form: tighter clusters raise kappa, which cuts tighter clusters | `kappa` learns only from pairs *known* to share a voice: the labelled base and the `same` corrections |
| `kappa` from two consecutive windows of one run (self-supervision from time) | kappa 283 -> 527 in five simulated minutes: adjacent windows are closer than one voice across recordings | dropped |
| an "effective dimension" vMF (d_eff ~ 11 measured from pair moments) | more fragmentation, not less: a broad prior on an 11-sphere makes "someone new" too probable | reverted to the true 192-d sphere |
| no noise model at all | a sustained pouring-water clip, which Silero calls speech, became a "speaker" in 1 room of 6 | the base hears ESC-50 folds 1-3 (5 noise clusters); tests only use fold 5 |

### Measurements of v1 (simulated rooms from public recordings, 1 s chunks, the real path)

- **12 random rooms** (6 real voices, indoor sounds between turns, gains -12..0 dB, floor -70..-50
  dBFS): purity **1.000 in 12/12**, phantoms **0 in 12/12**, coverage 0.85-0.91. One voice split
  in two in **4/12** rooms (FSDD lucas across takes, LibriSpeech 5703 across gains). The split is
  the safe error, and one `same` fixes it for the next sessions (`test_a_split_voice_said_once_to_be_one_stays_one`).
- **160 indoor sounds** never heard by the base: no speaker, except at most one long laugh. That is
  not a phantom by the pole's own definition: CAM++ hears the person who laughs.
- **two minutes of room floor**: zero observations (v0 registered a person in four seconds).
- **fast conversation** (pyannote sample): 2 speakers, purity >= 0.7 with 2 s windows over turns
  of 0.4-4 s. This is the window's limit, not the model's.
- **four Mandarin speakers**: 4.
- **restart**: same keys for the same voices the next day, rows and revision continue (E4, now
  answered across restarts).
- **cost**: ~50x real time on 2 CPU threads; a record line is ~270 bytes (E5).

### What the hypotheses E1..E9 now say

| # | status after v1 |
|---|---|
| E1 explicit state, useful at t=0 | **confirmed**: unknown, then new, then known, from an empty memory, no ritual |
| E2 evidence-gated registration | **replaced and confirmed**: the evidence bar is a posterior, 0 phantoms in 12 rooms + 160 sounds |
| E3 adaptive threshold | falsified (v0), and there is no threshold left to adapt |
| E4 stable key across restarts | **confirmed** (`test_a_restart_keeps_every_name_and_every_row`) |
| E5 cheap record | **confirmed**: ~270 B per window, ten hours of speech ~5 MB |
| E6 `f0_pctl` separates speakers | **dropped**: the pitch estimator was hand-written physics; CAM++ separates, and the record carries `rms_dbfs` only |
| E7 TTT bias-only adaptation | open, next |
| E8 the record is sufficient for the agentic pole | open: needs the arbiter |
| E9 whisper regime | open: needs KπX's own voice, on the real microphone |

### Next, in order

1. **The real microphone.** Everything above is simulated from public recordings. `make rec` +
   `make run` on `plughw:0,6`, KπX's voice, a film, a second person: the numbers that matter.
2. **Fragmentation across conditions.** The model treats windows as independent given a voice,
   which is over-confident when one voice has modes (two rooms, two days). The principled next
   step is a third level (voice -> condition -> window), with the condition spread learned from
   `same` corrections. Measured cost of not having it: one split voice in 4 rooms out of 12.
3. **Turn changes inside a 2 s window** (the conversation case): a sub-window re-cut when the
   posterior flips, or a speaker-change detector in front of the encoder.

---

## v1.1 (2026-10-19): the returns of the first live session, answered

`../../RETURNS.md` (the first live session on the real microphone, 21 minutes) made ten proposals,
R1 to R10; Claude's reading added five, P1 to P5. All fifteen are in. What each became:

| # | What it asked | What v1.1 does |
|---|---|---|
| R1 | continuity between windows | a **continuity prior** P(the next window of a run keeps its voice), learned **only** from labelled transitions (a `same` or `different` on two adjacent windows). Off until both kinds exist. Learned from the model's own decisions it would feed itself, which is the v0 loop |
| R2 | the doubt is invisible (`p = 1.0` 104 times in 112) | **`margin`** in nats on every decision. Measured on 12 rooms, 1041 decisions: p = 1.0 in 91 %, while the margin spans 8.9 (5th percentile) to 110 nats (95th), median 56; 5 windows sit below 1 nat. Those are the ones worth a look |
| R3 | negative dBFS is hard to read | **`rel_db`**: the level above this session's non-speech floor, the floor learned from the chunks where the VAD heard no speech. 2 min of room floor at -60 dBFS reads -60.0 +/- 1.5 (`test_two_minutes_of_room_floor_observe_nothing`) |
| R4 | Whisper on the loop: 144 % of a core | a **scribe thread**: the loop queues a finished run and goes on. Tested with a transcriber that takes 1 s per run: the loop finishes in less than one run's time |
| R5 | voices inside music | not built: separation and target-speaker extraction come after the arbiter. Gating (below) removes the music runs from Whisper |
| R6 | the record mixes v0 and v1 lines | every line carries **`port: 1`** and its `session`; a `session` line opens each session (with its first chunk); a record that does not start on the port is archived whole at startup |
| R7 | no correction was ever written | owed by KπX, live. Everything it needs is tested: obey, log, absorb, undo |
| R8 | candidates are invisible | `candidates` in the state: rows, evidence, waiting time, current weight |
| R9 | Whisper tiny is poor | **measured** (below), default is now `small`, int8, off the loop; `PI_EAR_WHISPER` and `PI_EAR_LANGUAGE` choose otherwise |
| R10 | a contract for consumers | **`port.py`**, version 1: typed `Session`, `Decision`, `Text`, `Correction`, `read()` and `follow()` |
| P1 | the 81-second hole, unexplained | **`π-ear probe`**: Silero's own probability per 32 ms frame, level and decisions, second by second, on the corpus |
| P2 | a wrong correction would be forever | **`π-ear undo N`**: the model is rebuilt from the base by replaying every row and every other correction at the moment it was first applied |
| P3 | rows are not how a person thinks | **`--from/--to`**: stream seconds of a session, resolved to rows; the span is kept in the line |
| P4 | why did it decide that | **`π-ear why ROW`**: then (p, margin, level) and now (the three best explanations, in nats behind) |
| P5 | candidates never die | **forgetting, learned**: see below |

### Transcription, measured (whisper int8, 2 threads, this container)

| model | WER English (LibriSpeech, 3 files) | WER French (Piper fr_FR, 5 sentences) | seconds per second of audio, French given / detected | words invented on 10 noise clips |
|---|---|---|---|---|
| tiny | 6 % | 16 % | 0.05 / 0.12 | 7-9 |
| base | 5 % | 9 % | 0.06 / 0.17 | 9-10 |
| **small** | **2 %** | **3 %** | 0.22 / 0.54 | 6-9 |

Every size invents words on coughs, keyboards and laughter: a transcriber cannot be the judge of
whether there was speech. Hence the **gate, which is not a threshold**: a finished run goes to
Whisper only if the model recognised a voice in it (`known` or `new`). Music runs, which wrote
`[MUSIC]` and invented Japanese in the live session, never reach it now (`runs_without_voice`).
The French test set is synthetic; the real French number is owed by the live microphone.

### Forgetting: the design that failed first, and the one that holds

First design: a candidate's weight is `exp(-age * rate)`, with the rate estimated by survival
analysis (confirmations over total waiting time, censored candidates included). **It cannot
forget**: a candidate that keeps waiting adds its own age to the waiting time, so the rate falls as
it ages, and its weight never drops below `exp(-confirmations)`. Caught by
`test_a_candidate_fades_at_the_rate_this_stream_confirms_voices` before it shipped.

What holds: a **cure-rate model**. A candidate is a real voice with probability `pi`, and a real
voice is confirmed after an exponential delay of rate `lam`; otherwise (a cough, a door) it never
is. Both are fitted by EM on this stream's own candidates (`sphere.cure_rate`). A candidate's
weight is the posterior that it is a voice still to come, `pi e^(-lam a) / (pi e^(-lam a) + 1 - pi)`,
which goes to zero as it waits; when it is exactly zero in float64, the candidate is dropped and
counted. Before the first confirmation nothing fades. No constant: the delays are this room's.

### Exactness of the replay, and why it needed one line

`undo` is only honest if a replay gives the very same model. It did not at first: the live path
decided on the float64 embedding, the replay on the float32 row read back from disk. The live
path now decides on the stored float32 row too, and `test_the_memory_rebuilds_exactly_by_replay`
compares the two models' JSON, bit for bit.

### A bug the margin revealed

Run tails of 6 to 22 ms (less than one fbank frame) came out of CAM++ as NaN, and v1.0 wrote
`"p": NaN` into the record (invalid JSON, silently: 6 decisions across the 12 rooms). A sensor that cannot see now
produces no observation (`unseeable` counter), and every simulated room asserts that no NaN
reaches the record.

### Unchanged, on purpose

12 random rooms: purity 1.000 and 0 phantoms in 12 of 12, one voice split in 4 of 12: identical to
v1.0 to the last decimal. Expected: continuity is off without labels, and four minutes is too
short for forgetting to touch anything. v1.1 changes what the ear can say about itself and how it
can be taught, not what it hears.

---

## The third law, in this pole: a model that knows when it does not know (KπX, 2026-10-20)

The guiding line of every model brick of π (`../../IDEAS.md`, section 0): not a model that always
runs toward a minimum, but one that says, with each output, whether it knows, hesitates, or does
not know, and which kind of not-knowing it is in.

What the ear already does, and what it still lacks:

| | already | still owed |
|---|---|---|
| who | `unknown` as an answer, `margin` in nats, `p`, candidates that fade | calibration of the margin against corrections (is a small margin often wrong?) |
| what | nothing: Whisper writes text with no doubt attached | a per-word confidence, "I did not understand" as an output, a revision in the silences (proposed 2026-10-20, not decided yet) |
| the sensor | a NaN tail is not an observation | saying *which* doubt: unseen voice, true ambiguity, or a sensor out of its domain (music, whisper, overlap) |

---

## v2 (2026-10-20): who and what, one ear (KπX: "W1 à W6, Q1 à Q6, la totale")

The third law, built in both halves. Every number below is measured on public recordings
written as 1 s chunks and replayed through the real path; the tests hold them.

### What, the plan and what it became

| | asked | built |
|---|---|---|
| W1 | the text while it is said | `partial` lines: the fast recogniser (Parakeet TDT 0.6b v3) re-reads the run in progress whenever it is free, latest audio wins, nothing queues |
| W2 | revise in the silences | `text` r0 from the fast reading at the run's end, r1 from Whisper turbo in the next silence; corrections are absorbed there too (the scribe's thread never runs ahead of the decision loop) |
| W3 | say what it does not know | a named doubt per text, a trust and `agreed`/`new` marks per word, the other readings kept, and a `question` when the doubt is `ambiguous` or `not_understood` |
| W4 | corrections of text | `text` corrections and answers: a lexicon per voice, substitutions learned from the alignment, recall of corrected runs; candidates, never silent rewrites |
| W5 | learned weights | the reader is a Bayesian conditional logit, evidence maximised (`sphere.fit_choice`), refitted on the slow clock on every example ever seen |
| W6 | calibrated | `p` of the chosen reading is the logit's own, trained on outcomes; words likewise |

The recognisers, measured (WER en on LibriSpeech, fr on Piper UPMC, int8, 2 threads):
parakeet-v3 3 % / 6 % at 0.16 s per s; whisper-turbo 2 % / 9 % at 1.14; whisper-small 2 % /
19 %; canary-180m 1 % / 15 % but 2 to 9 words invented on 10 noise clips. Two different models on
purpose: where they disagree is where the ear does not know.

**A correction is trusted the first time.** First design: the base weights learned on plain runs
only; the weights of `edits` and `recall` started at zero and a corrected error came back once
before the reader learned to trust what it was taught. Fixed by *correction episodes* in the
base: every public run the recognisers got wrong is corrected once and heard again, with and
without the recall of that very run.

**The first sentence of a new voice was never read.** v1.1 sent a run to Whisper only if a voice
was recognised in it; a voice's first run is still a candidate (`unknown`), so its first sentence
was lost. Now every run that is not entirely a known sound is read, and whether anything was said
is the reader's call. To make that call honest the reader gained `none_unspoken`, one minus
Silero's mean speech probability over the run (measured: 0.82 to 0.94 on speech, under 0.78 on 27
of 28 indoor sounds that pass the VAD), and the base now hears the ESC-50 sounds as the VAD lets
them through. On the 160 sounds of fold 5, 14 runs reach the reader: 8 invented a word before, 4
now (`"Yeah."`, marked `unseen`), 10 are `not_understood`.

**A word confirmed by its own reading.** The two recognisers agreeing put a word in the lexicon
*before* the decision, so a word heard for the first time was already "known" and the doubt read
`clear`. The lexicon now learns after the decision: the first time is `unseen`, then known.

**A doubt the memory resolves is not a doubt.** After one answer, the recognisers still disagree on
"KπX", and v2's first build asked again every time. New doubt kind `taught`: they disagree, and
what KπX taught is the most probable reading. Not asked.

### Who, v2

**Q2, conditions and voices.** Base kappa is now measured *within* one recording in one place
(465), kappa_c across places for one voice (449). 12 random rooms: no voice split (v1: 4 of 12).

**The place prior, found by a failure.** With the hierarchy alone the pyannote two-person sample
became *one* voice. Not kappa (forcing it from 476 to 600 changed nothing) and not the code: the
generative model itself preferred one voice by 26 nats, because the two voices of that room are
closer to each other (cos 0.49) than the population says two voices are (0.35): they share a
channel. Centring the prior of a new voice on the room reversed it (+42 nats for two). Built as a
level of the model: places, with `kappa_in` (voices around their place, 212) and `kappa_p`
(places around the population, 359), measured on four simulated places in which every base voice
was heard, and the place's mean estimated from the voices already heard there. The pyannote
sample: 2 voices.

**Q4, turns.** Shadow windows offset by 1 s are decided (never learned: they would count the same
audio twice), and the `turns` line gives each half window the voice of the more confident window
covering it: 0.625 of the learned windows right on the pyannote sample, 0.75 of the time by the
turns; 0.99 or more in the rooms. **Failed design**: detecting a turn inside a window by
comparing its two 1 s halves. Measured on that sample: same-speaker halves at cos 0.40 to 0.72,
two-speaker halves at 0.28 to 0.51. Too much overlap for a sensor; not built.

**Another place, another day.** The six voices of day 1 heard on day 2 through another simulated
place: most come back under new keys (the place moves every voice further than the base's places
did). One `same` per voice, and day 3, in that second place, gives each voice its day-1 key
again. Two things this needed: noise heard through places too in the base (a door and a toilet
flush heard through a reverberant room became two phantoms), and the memory's clock never running
backwards (a replay faster than real time started its session before the last one ended, and
"the oldest voice keeps its name" kept the wrong one).

**Q3, person or device.** The mean spectral shape of a voice's windows (24 mel bands, level
removed) against two diagonal Gaussians learned on public recordings heard live and through a
simulated loudspeaker. 3 pairs (a person in the room, another voice from a loudspeaker): 5 of 6
right, the miss a device heard as a person at p 0.77. `device` or `person` said once is final for
that voice and teaches the two classes.

**Q1, trust.** `trust` = P(right | margin), a logistic on the margin fitted on the base's own
replay (182 public decisions, 93 % right: 0.50 at 0 nats, 0.71 at 10, 0.94 at 30), then on what
every correction says about the decisions it touches. None before it has seen itself both right
and wrong.

**Q5, asking a name.** A voice of an earlier session, back and still nameless: one `question`,
answered by `inbox.answer(id, text=...)`; the answer is replayed exactly by `undo`.

**Q6, music: separation measured and refused.** Spleeter (2 stems, int8, sherpa-onnx 1.13.8) made
reading worse at every level of music: WER 2 % -> 5 % at 6 dB, 3 % -> 18 % at 0 dB, 5 % -> 36 %
at -6 dB (Parakeet; Whisper turbo worse still). Its stems came out about 130 times louder than the
input and uncorrelated with the voice; only their sum gave the mix back. The recognisers read
through the music on their own. The separator slot stays (`PI_EAR_SEPARATOR`), off. For who:
the room with music all along at 10 dB under the speech keeps purity 1.000 and no split; at 5 dB
two voices split (the safe error); the music alone is never a person.

### The plug points

`port.py` is version 2: `Turns`, `Partial`, `Text`, `Question` join `Session`, `Decision` and
`Correction`. `port.Inbox` is the one way in, for a terminal, the arbiter or a voice: `correct`,
`answer`, `undo`. Recognisers by registry name or `module:factory`. Nothing in the ear knows who
is on the other side of the inbox, which is the point: when the mouth exists, a spoken answer
is the same line as a typed one.

**Who is asked, and when.** First build: a `question` for every `ambiguous` or `not_understood`
text, so every cough that reached the reader asked "what was said?". Now: `ambiguous` asks; a
`not_understood` asks only in a run of a known voice (someone spoke and the ear missed it); a
sound nobody owns, not understood, asks nothing. Measured on fold 5: no question from 160 sounds.
