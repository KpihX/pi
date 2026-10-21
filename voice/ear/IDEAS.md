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

---

## v2.1 (2026-10-21): the first live test, PSG - Barça, answered

KπX ran the three panes for 2 h 11 min in front of a football broadcast (a laptop
microphone, the television across the room, KπX in it), and sent the record, the memory
and the last hour of corpus. That hour is kept on this machine as a **live recording**
(`π_ear.live`, never committed: broadcast audio and the voices of a home); its manifest,
`tests/live/2026-10-20-psg-barca.json`, names ten segments and what makes each hard, and
`make test-live` replays them through the ear from the memory it had. CAM++ run again on
the corpus reproduces the stored rows with cos 1.00000: the replay is exact.

No reading is taken as the truth. Every number below is a word recall against two other
systems' readings of the same hour (Whisper turbo told French on 30 s blocks; Parakeet on
20 s blocks), next to what v2.0 wrote live, and next to what the ear's own two readings
alone would have given: a measurement of where the loss is, not a target.

### What the live test showed

| | measured live (v2.0) |
|---|---|
| who | the commentator (1145 windows) and the consultant (685) stable for 100 min; studio host, consultants, pitch-side, a sponsor's voice found again 65 min apart; KπX's own voice (`b54d13`, learned on 2026-10-19) in the room all along; 40 voices for 12 to 15 people (fragments); an independent judge (ERes2Net) agrees on 75 % of windows |
| person or device | **a bug**: a memory older than v2 had 1449 rows and no spectra; spectra were appended from index 0, so every decision read another window's spectrum (KπX: device) and late voices had none (`NaN`). Aligned again: every television voice `device`, KπX `person` at 1.000 |
| what | 37 % of the words written, where Whisper alone held 53 %; 318 of 662 texts "not understood" (106 real sentences), 516 questions, 2 texts `clear` in two hours |
| why | Parakeet answering in English on studio talk (more the longer the piece); Whisper's language detection writing Russian, Portuguese, Turkish, Japanese or Chinese on short phrases, mostly KπX's; Whisper reading only the first 30 s of a run (a 70.9 s run lost half); Silero's mean on real speech 0.3-0.7 and Parakeet's token log-probability -0.8 where the base had learned 0.9 and -0.1; `ambiguous` named whenever two readings differed by one word; the slow pass 16 min behind on studio talk that never paused |

### What changed, each measured on the replay

**Pieces** (`scribe.py`). The unit of text is a piece of a run, cut where the voice changes
(the turns of the model of who) and where Whisper's window would overflow, at Silero's
deepest dip, never shorter than the VAD's own shortest speech. A piece is final while its
run goes on. An interview's question and answer, one text live, are two pieces, two voices.

**Reading with context.** Whisper reads better the longer its input (0.84 on 30 s of match
commentary, 0.76 on 15 s, 0.64 on 8 s) and wrote "..." on short pieces read alone. The slow
recogniser now reads as many consecutive pieces as its window holds; its sentences, timed
(sherpa-onnx's segment timestamps), go back to the pieces they were said in.

**Language, learned per voice** (`language.py`). A Dirichlet per voice over what Whisper
*detected* it speaking, its prior the room, itself a Dirichlet with one pseudo-reading over
the known languages and one for a language never written here; the detector's reliability
as a function of length, measured on the base then on the stream; the voice's language
given to Whisper when it is the better bet than detection at this length. In the reader it
is not a learned feature but a probability, added with Bayes' coefficient one:
`log P(the reading's language | its voice)` against the voice's best.

The designs that failed on the way, each on the replay of KπX's segment:

1. counting every reading, told ones included: one misdetected phrase of KπX ("I'm so happy
   to have you") locked his voice, then the room, into English, each forced reading
   confirming it: recall 0.002;
2. the room as a plain average: one detection made it 100 % English;
3. a learned weight on `1 - in_language`: trained on translation episodes the base made
   separable, it grew to -18, then -53, and a right French reading paid for the room's
   still uncertain language;
4. measured against the other readings only: a lone Chinese reading of KπX paid nothing;
5. the expectation over the text's language: the text detector's own doubt (a word model:
   "J'imagine des mains." was "another language", two words unseen) made right readings pay.
   Now: the text's most probable language, by character trigrams with an open class.

**Every sensor level relative to the regime.** Silero's mean and Parakeet's token
log-probability are read against their means over the known voices of this place (the
base's own as one pseudo-piece), not as levels: a television through a laptop is quiet to
both, and that is not "nothing was said". The base reader now also hears its Piper
sentences through a loudspeaker in a room over indoor sounds (three regimes).

**The doubt is the posterior.** `ambiguous`: the reader cannot choose (the chosen reading
more probably not the best). `partial` (new): it can, but the best reading there is holds
more words probably wrong than right, by the word model's own trust ("I understood bits of
it"): a reading's `p` is "the best of the readings", not "a right reading", and over 300
synthetic pieces of every garbling, no piece was ever `ambiguous` without it. The word model
learns from every word of every reading against the truth (it once learned from the
reader's choice with unfitted weights, and agreed words came out untrusted). Two readings
apart are a doubt about speech only in one language (a translation is not).

**Questions worth someone's attention.** Only about a person (a commentator is nobody to
ask), one open per voice, and about the words: a sure sentence around one unsure name asks
about that name, and a word KπX confirmed is never asked again.

**A slow pass that keeps pace.** Partials and revisions take turns (a first build always
read the partial first, and revised nothing while the television spoke); people first,
then the most doubtful; nothing slow while a person speaks; its own speed measured.

**Bugs the live test found**, each with a test that fails without its fix: the spectra of
an old memory (aligned, missing rows put back in front); a saved reader that froze the
weights of the base it was saved under (they are refitted on the current base now); the
run in progress past 60 s sliced from a negative index (the partials of studio talk read
the wrong audio); a scribe job that failed killed its thread and the ear waited forever
(errors are written to the record now, and the scribe goes on); a late `hear` re-opening
a finished run.

**Loops are not speech.** Whisper's hallucination loop ("la Céline de la Céline de la
Céline...") was accepted on KπX's segment. A probability again, not a weight: the term added
to a reading's score is log P(it is speech, not a loop | it repeats k of its m word
trigrams), speech repeating Poisson(q m), a loop any share uniformly, q and the share of
loops learned without one label by EM on everything the recognisers wrote (the base's public
readings, then the stream's), the base's truths held as speech. 90 % of 2405 live readings
repeat nothing, the loops 30-80 %.

Three designs failed on the replay first, each measured. The pace (words per second): the
loops ran at 2.5-5.6, inside speech's own 0.6-4.4. A cost per repeated trigram: a 100-word
studio reading with three natural repeats paid 20 nats and lost to a short translation
(studio 0.78 -> 0.55). Poisson(q m) against its most probable count, q learned on accepted
readings: the base's short sentences made q 0.0014, and a player's 67-trigram answer
repeating "c'est vrai que" and "sur le terrain" paid 5.4 nats and lost to "I did not
understand" (player 0.75 -> 0.57); learning q from what the reader accepted could also teach
it that loops are speech, the way the first language model locked itself. Against a loop,
that answer's two repeats cost a few hundredths of a nat, a loop of thirteen in twenty
forty.

**Language, a second time: the voice's regime.** The replay of KπX's segment never told
Whisper "fr" for KπX (59 pieces of 60 left to detect): detection at 2 s was measured right 6
times in 7 on the base (clean speech), and KπX's detections, 6 French in 18 and the rest
Portuguese, English, Chinese, Russian, Polish, Turkish, were counted as languages spoken, so
the voice was 35 % French. Told French, the same pieces read "C'est toujours comme ça, j'ai
toujours jamais vu..." where detection read "Eee" in Turkish: recall 0.318 -> 0.377 against
Whisper-fr, 0.475 -> 0.480 against Parakeet (the neutral one). Now each detection is judged
against its voice without it (a voice heard once was always "right": 0.98 where the truth
was 0.02), each voice learns how much worse or better its own detections are than their
length predicts (KπX over the television: one in three at 2 s), its detections count by
that, and a voice speaks like the room as strongly as the voices of the room do (a Dirichlet
concentration estimated on all of them at once, 21 on this record). On the replay's own
detections: KπX French at 0.71 against a detection right at 0.30, so French is given; the
commentators keep detecting (0.99).

Three more things the replay showed before that held. The language model was refitted only
when the scribe was idle, and a television never lets it be: the regime of KπX's voice was
never learned in time (it is refitted after every detection now; a vectorised logit and a
cached room keep it at 0.06 s for 2000 detections, where the first version took 1.6 s). A
detection was judged without itself in its voice but not in the room: at the start of a
session, KπX alone in it, the room was KπX, and the first misdetection (Chinese) confirmed
itself; three pieces were told Chinese, until 352 s. And the base had measured the detector
at 2-4 s only: the slope of reliability in length was free, and three detections of one
voice turned it over (longer came out less reliable). The base now measures it up to the
window, one voice's sentences then every voice's of a language one after another. Replayed,
KπX's voice is told French from its first French detection on.

Questions are asked only about a voice the model knew before the piece: on the goal and the
crowd, KπX's shouts made voices of one window each (16 dB over the floor, a person's
spectrum: rightly people), and each was asked what it had said. A voice of one window may be
nobody; a real one comes back.

And then the live test (`make test-live`) replayed KπX's segment once more and Whisper was
told Chinese four times: the order of the scribe's jobs varies with the clock, people's
pieces are revised first, and for minutes the room was KπX's voice alone, its misdetections
confirming each other while the television's French waited in the queue. A second detector
was there all along: the language of what the fast recogniser wrote, at once, on every
piece, with its own reliability learned like Whisper's (`"fast"`). It detected nothing, and
that showed the deepest bug of the pass: the text's language (character trigrams with an
open class) took the trigrams already seen as the open class's universe, so a French trigram
seen once was likelier "in another language" than in French, and on the base's few hundred
words "le match est fini ce soir" was 99.9999 % another language. The reader's language term
had paid it on every French reading until the stream's own words outgrew it. The universe is
every trigram that could be written now ((characters + 1)^3). A detector never seen wrong
also came out "unknown" instead of "reliable" (Laplace's succession now: one right and one
wrong pseudo-detection at the data's mean length). Replayed three times, KπX's voice is told
French 11 to 13 times and never another language.

Fixing the text's language moved the match: the reader prefers the fast reading more often
(52 pieces of 94 against 43), says "not understood" less (17 against 23) and "ambiguous"
more (6 against 1); against Whisper-fr 0.438 -> 0.396, against Parakeet 0.582 -> 0.598. Each
reference favours its own recogniser, and nothing unsupervised says which recogniser is
right in this regime: the reader's preference is still the base's, until KπX corrects a
piece. Not tuned to move a number.

**Zero trust, the review.** Every module read again, line by line, each finding with a test
that fails without its fix: a voice change was gated on `trust or 1` (a calibrated trust of
0 read as 1); the audio of every piece was kept for the whole session (460 MB on two hours);
undoing a text correction rebuilt the reader from the base and forgot every voice's language
and every word both recognisers had confirmed; an answer with neither a text nor a choice
crashed the scribe two calls later (refused now, the question stays open; "" is an answer:
nothing was said); a refit that raised left the reader dirty and the scribe refitted again
at once, forever, and the ear waited on it (an error is written once now); the replay starts
from a memory that already heard the segment (said in `live.py`: who is easier on the replay
than it was live).

**One model per process.** The full suite ran this container out of memory: every ear a
test built loaded its own Whisper turbo (a gigabyte). Recognisers are loaded once per
process and shared, behind a lock (their language is set per call).

### The replay, segment by segment (word recall, vs Whisper-fr / vs Parakeet)

Final build, from the memory of the session's end (`make test-live`; lab runs 6-11). "Only
fast/slow" is what one recogniser alone would have given; "oracle" the better reading of
each piece against Whisper-fr. The reader beats either recogniser alone where they disagree
by regime, and sits at the oracle where one of them is simply right.

| segment | v2.0 live | v2.1 | only fast | only slow | oracle | told French |
|---|---|---|---|---|---|---|
| match | .426 / .593 | .396 / .598 | .430 / .588 | .433 / .549 | .420 | 23 of 94 |
| KπX | .235 / .436 | .312 / .480 | .256 / .468 | .348 / .486 | .377 | 12 of 18 of KπX's (0 another language, 3 replays) |
| studio | .392 / .506 | .775 / .673 | .593 / .651 | .808 / .637 | .842 | 0 of 94 (detection right) |
| player | .013 / .016 | .752 / .144 | .285 / .171 | .718 / .134 | .752 | 0 of 17 |
| crowd | | .217 / .377 | .210 / .441 | .252 / .364 | .280 | 15 of 53 |
| whistle, ads | | .167 / .587 | .167 / .659 | .190 / .641 | .167 | 0 of 21 |
| another programme | | .333 / .342 | | | | |
| 17 min in one stretch | | .784 / .613 | | | | slow pass 0.74 s/s, 0 behind |

What the numbers do not say: the Whisper-fr reference was itself told French, so telling
French flatters it (the Parakeet column is the neutral one); on segments shorter than its
30 s blocks (the interview, 48 s; the goal, 30 s) it is not a measurement at all, a block
straddling the segment counting whole or not at all. KπX's own segment is bounded by the
recognisers (oracle .38), not by the reader. Cold start is the remaining weakness of the
language model: alone at the start of a memory and misdetected the same way by both
detectors, a voice is told that language (the goal replayed from nothing: English, once)
until another voice or a longer detection says otherwise; a memory keeps its room's
languages from one session to the next.


**Correction 2026-10-21 (KπX): a misreading of the live test, corrected here, append-only.** The
lines above say the match was "a football broadcast (a laptop microphone, the television across
the room)" and speak of "a television heard by a laptop". That is wrong. KπX watched the match
in the browser of this very laptop: the "television" was **the machine's own sound**, played by
its own speakers and picked up by its own microphone. Two consequences the analysis above missed:

- what the machine plays is not an unknown source to be guessed through the microphone: it is a
  signal the machine already has, exactly, before it reaches the air (the output's monitor);
- the acoustic path is the laptop's speakers to the laptop's microphone, the standard case of
  echo cancellation, not a television across a living room.

Every "device" voice of that session was the machine itself. Nothing measured above changes;
what it means does (see the proposals that follow this correction).

---

## v2.2 (2026-10-21): an ambient ear (KπX: "un système humain, ambiant, jamais confiné")

KπX's correction, kept here because it is the brief of everything below: the ear is not for one
voice. KπX alone, with friends or classmates, in a meeting with a trusted group, with voices
coming out of the phone or the PC, with a video playing: it must understand the scene the way
a person who is there does, and assume no scenario. And two answers: the machine's own output
may be captured, by configuration (`PI_EAR_OUTPUT`, on by default; off, the microphone alone is
the ear); what it played is kept, optimised (text and metadata, no audio past the corpus bound).

### Step 1: two streams, one ear

**Why.** What the machine plays exists, exactly, before it reaches the air: its output's
monitor. A person in the room does not have that; the ear does, so it should not guess
through the microphone what it can read directly.

**What it is.** `make rec-out` records the monitor beside the microphone (same chunks, same
names: one clock) and `π-ear apps` writes which applications play, each second
(`apps.jsonl`, from the sound server). The ear has two channels (`ear.Channel`): the
microphone and the machine. Each has its own sensor, runs and floor; both feed **one** model
of who, the machine's windows in a place of their own. Every decision, turn and text says its
`channel`; the machine's say which `apps` played them (port 4).

**Taking the machine out of the microphone (`echo.py`), measured before it was built**
(lab `aec_domain`, `aec_compare`: a simulated laptop, its small speakers and the room, French
sentences said over LibriSpeech it plays):

| canceller | echo removed | the person's words read (Parakeet), person as loud as the machine |
|---|---|---|
| none | 0 dB | almost none |
| WebRTC AEC3 | 20-25 dB | **none**: its residual suppressor gates the near end in double talk |
| Speex MDF, linear only | about 16 dB | **three quarters** |

AEC3 is built for telephony, where silencing the near end while the far end speaks is a
feature; for an ear that must hear people while the machine plays, it is the failure itself.
MDF estimates its own step from the stream (Valin 2007), so nothing is set. Its limit is
physical and measured: a small driver pushed into its nonlinearity leaves what no linear
filter models (a scene drawn with a harsher driver: 9-11 dB removed, the person still unread).

**The offset between the two recorders is learned**, never assumed: they start at different
moments, so the streams are apart by up to a chunk plus the output's latency, more than any
filter tail. GCC-PHAT on each stretch of played sound finds it (to the sample on four lags
from 0 to 900 ms, plus the speaker's and room's own few milliseconds); the most frequent
estimate since the last hole in the stream is used; until there is one, nothing is subtracted.

**Two bugs found by building it, each with its test.** The two streams arrive one chunk at a
time live but all at once in a replay: the machine's chunks were pushed first and the bounded
reference had dropped the start when the microphone needed it. The streams are now merged by
their seconds (a machine chunk goes first while it is not further ahead than an offset can
reach). And a microphone chunk waiting for the machine's could see its file pruned before the
corpus copied it: the corpus copies at the read now.

**The canceller's doubt, two numbers** (the third law): `echo` on every microphone decision,
the share of its energy that went with the machine; `left` per chunk, how much of what remains
the machine still explains (the coherence of the cleaned microphone with the reference). Where
the canceller failed, `left` is high, and the state says so.

**What it does to a scene** (`tests/test_output.py`: a podcast in English played by the
laptop, a person speaking French in the room, partly over it):

| | the person over the machine | the person after it |
|---|---|---|
| output off | the sentence is lost: the microphone writes the podcast in its place | read |
| output on | "Je voudrais qu'on reprenne le chapitre 3 demain matin." | read, unchanged |

And the podcast itself is read from the machine, clean, attributed to the browser. Until the
offset is learned (the first stretch of played sound) the microphone still writes the podcast
heard through the air: the machine, heard twice. That is step 2's first fact to use.

**Degrading visibly.** Output on but its recorder not running: the microphone is never held
back (a microphone chunk waits at most for the next one, and not at all when no machine chunk
ever came), and `ctl status` says `DEGRADED: no chunk of the machine's sound yet`.

### Step 2: where a voice comes from, the scene (`scene.py`)

**Why.** "Person or device" was the wrong question for an ambient ear. A person who is there
tells four kinds of voice apart, because they answer them differently: someone **in the room**
(listen, answer), someone **reached through a device** (a meeting, a call: a person too),
something **recorded** (a video, a podcast: context, nobody to ask), and **π itself** (never
someone else). Each voice gets P(room, remote, media, self), on every decision (`scene`,
`scene_p`), every text, and in `ctl status`.

**What is a fact, and what is learned.** Nothing is written by hand; three things are facts:

* *through this machine*: a voice on the machine's channel was played by it. Its microphone
  windows count for the room only by the share of their energy the canceller did not explain
  (`1 - echo`). A Beta with one pseudo-window each way: P(machine) = (M + 1) / (M + room + 2).
* *through another loudspeaker* (a phone, a television): the voice's long-term spectrum, the
  existing `source.py` model, as evidence.
* *π itself*: the machine windows π's mouth played (`contract.MOUTH_APP`): the sound server
  names the application.

**Remote or media: the turns.** Among the voices that come out of a device, the one thing that
separates a call from a video, whatever the application, is that **a conversation waits for
the room, a recording does not**. So the ear counts, over the seconds it heard: k1 of the n1
seconds the voice spoke had the room speaking too; k2 of the n2 seconds without it. Two
accounts of these four counts, with no parameter:

* *media*: the room speaks at one rate whatever the voice does, p1 = p2 ~ U(0, 1);
* *conversation*: the room speaks less over the voice than in its silences, p1 ~ U(0, 1),
  p2 ~ U(p1, 1) (so p1 has the same prior in both, and only the comparison counts).

The log of their ratio (the Bayes factor, one integral, `conversation_evidence`) is added to
the application's prior log-odds. Its intuition, on the timelines of `tests/test_scene.py`:

| what the counts show | evidence (nats) | then P(remote), P(media) |
|---|---|---|
| a meeting: 4 s each, the room answers in the voice's silences only | +41.8 | 0.99, 0.00 |
| a video with pauses, the room talks whenever it wants (40 % of seconds) | -1.4 | 0.19, 0.80 |
| a video that never pauses: no silence of it to compare with | **0, exactly** | 0.50, 0.50 |
| a voice that pauses, and nobody ever takes the turn | -4.1 | 0.02, 0.98 |

The third row is the third law at work: with nothing to compare, the evidence is not small,
it is none, and the two stay as likely. The fourth: a conversation would have the room speak
in the voice's silences; a voice nobody ever answers is better told as a recording.

**The first form, refused by its own test.** p1 was compared with a plug-in rate for the room,
½ before the voice had ever been silent: a video that never paused, talked over by the room
40 % of the time, came out a conversation (40 % < 50 %). The plug-in was a number set by hand
in disguise; the two-hypothesis form has none.

**Who spoke in the room, for the turns.** Not one window's echo share: a person talking over a
loud video is half the machine's energy, and still the room speaking (the end-to-end scene
lost the person's seconds that way). A voice of the microphone speaks in the room unless it is
shown to come out of a device, by its windows (this machine) or its spectrum (another
loudspeaker: the video heard through the air, which first counted as the room answering).
And a window marks every second it covers, not only its middle (half the seconds were lost).

**What an application plays** is learned from KπX's corrections (`room`, `remote`, `media`,
`self` on the port): told once that a voice of `Meet` is a person, the next voice out of `Meet`
starts as one, before any turn; another application learns nothing from it. Before any
correction, an application is as likely a call as a video: no list of names.

**End to end** (`tests/test_output.py`, synthetic scenes, the whole ear, one run each):

| scene | the device's voice | the room's person |
|---|---|---|
| the browser plays a podcast, the person speaks partly over it | media 0.69, remote 0.11 | room |
| a call: a voice out of `Meet` asks, the person answers, five times | remote 0.70 | room 0.81 |

The remaining mass of the device's voice (room 0.2-0.29) is the microphone's copies of it from
before the canceller learned the offset: real evidence the cluster mixes, honestly kept.

**The machine heard twice, now said.** A microphone piece whose words are found, in order and
more than not, in what the machine's channel wrote at that time is marked `echo_of` those
pieces, never asked about; and since the microphone's recogniser can be first, a machine
reading that arrives later revises the microphone pieces it explains. In the podcast scene,
every microphone text from before the canceller converged is now marked.

**What it changes.** Questions and the slow pass's priority go by "a person" (room or remote),
not by "not a loudspeaker": a colleague in a call is asked about, a video is not. A tie (the
video that never pauses) is a coin, off ½ only by the one pseudo-window of the prior: whether
to ask then is a matter of the value of the answer, step 3.

### Step 3: second thoughts (`revise.py`)

**Why.** KπX: a human "lives in the present, has a first idea, then a second, and adjusts
with time". The ear's model of who already changes its mind on its own (two clusters fuse
when one voice explains them better, a candidate becomes a speaker, a correction moves
windows), but what it had written stayed written: the first window of every voice said
`unknown` forever, though a second later the model knew whose it was.

**What it is.** Every window the session published is kept with what was said of it. At
each silence (a run closes), after a correction, after an undo, the ear asks its model again,
for each, what it would say now:

* a learned window: where the model has put it now (its cluster's voice);
* a window never learned from (a shadow, a glimpse, a forgotten candidate): its best
  explanation under the model as it is, scored **without learning**. Only the clusters that
  changed since the last time (and "a new voice") are scored again against the one it was
  given: the others are where they were. Measured on a synthetic stream of six voices, half
  of the windows unlearned: 3000 windows thought over in about 60 ms, linear in the windows;
  a two-hour session (some 7200) in a fraction of a second per silence;
* a window heard in another place than the model is in (the machine's channel while the room
  speaks) is not scored again: a new voice's prior is centred on the place, and changing
  place resets what the place knows.

Where the answer changed, a `revision` is published (port 5), with why: `merged` (its voice
is now part of another), `corrected` (a correction named it), `learned` (more evidence). The
run's turns are written again (`revision` counts), the pieces it is in are given their voice
now and their texts decided again under it (that voice's language, its vocabulary), and the
scene follows the fusions (one voice, one body of evidence). A revision tells the model
nothing: it is the model's own state, read back, so it cannot feed itself. The invariant,
tested end to end on the podcast scene: **what the ear says last of every window is what its
model says now.**

**The question worth asking.** "Are these two voices one?" is worth KπX's attention only if
the answer is expected to change something. If B (n windows) were said to be A, all of B's
windows would change hands; the model's own P that B's main condition belongs to A, against
staying where it is, makes the expected change min(P, 1 - P) n. It is asked when that is at
least one window's worth (the less likely answer at least 1/n probable), the most valuable
pair first, one question open at a time, never while a person speaks, never twice for a pair
(the pairs asked are kept with the memory). The answer is `same` or `different`, which is the
correction it asked for; anything else is refused, visibly.

**What it measured about itself, honestly.** On voices drawn with CAM++'s measured
geometry, the model is almost never in that doubt: between "one voice" and "two", its link
evidence jumps from a fusion (the clusters merge) to tens of nats apart (27 nats with two
windows of the second voice, 60 with three, `tests/test_revise.py`). So it asks nothing there,
and should not. Where it will ask is where its evidence really is thin: a voice of few windows
heard through two very different channels. Whether its nats are as sure as they say (the per
window margins are calibrated by corrections, Q1; the voice links are not yet) is the open
question: each answer to a `same` question is exactly the label that would calibrate them.

**Not yet.** A piece whose voice changed is decided again from its readings; it is not
re-read by the recognisers in its voice's language (its audio is gone once read by both):
re-reading from the corpus is step 5's. The windows of earlier sessions are not thought over
again (their record is closed; their voices are, through the model).
