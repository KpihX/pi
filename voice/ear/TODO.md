# ear/TODO.md: the live test on the microphone

> The paradigm is not tested by a download (`../RESEARCH.md` R0.11). It is tested by **one capture
> session** and **three questions**. Everything below runs on this machine, locally, offline.
> Machine state verified 2026-10-18: phase 1 needs **zero install and zero download**.
> Hypotheses being tested: `IDEAS.md`, E1..E9.

---

> **FLOW CORRECTED 2026-10-18 (KπX).** Everything below was written as a batch take: one 20 minute
> recording, then processing. That is exactly the shape `../../IDEAS.md` section 0 forbids, and the
> sequence it implies (train, then calibrate, then infer) is what this POC exists to refuse. The
> live flow is now the reference. The phases below survive as the **replay path**, meaning the same
> questions asked of the recorded backlog instead of a one-shot take.
>
> ### The live flow (current)
>
> Two independent servers, coupled by a directory of one-second raw chunks. Setup in order, then
> three panes:
>
> | Step | Command | Role |
> |---|---|---|
> | 1 | `make sync` | base env: numpy + sherpa-onnx + typer, resolved by uv |
> | 2 | `make models` | the CAM++ ONNX, 27.4 MB, needed from v1, not by v0 |
> | 3 | `make rec` | pane 1: the recorder, endless one-second chunks, **never stopped** |
> | 4 | `make run` | pane 2: the ear, slides a 3 s window, keeps `state.json` live |
> | 5 | `make watch` | pane 3: `jq . /tmp/π/state.json` in a loop: watch it move |
>
> While it runs: `make status`, `make events`, `make snapshot`, `make diff` (the visible proof that
> something moved), `make pause`, `make resume`, `make reset`, `make clean`.
> Positional flags go through `RUN_FLAGS`, for example `make run RUN_FLAGS=--keep-audio`.
>
> What this changes for the protocol below:
>
> - **the take is no longer a take.** Play the film and talk over it live, watch the state, and use
>   `make snapshot` before and after each test, then `make diff`, to see what the system did with it;
> - **the delegation order of the phases is gone.** The inventory of phase 2 runs from second zero,
>   with no download, and the record of phase 1 is produced continuously by the ear instead of once;
> - **phase 0 stays true** (nothing to install, nothing to clone) and the per-phase definitions of
>   done stay true, they are simply measured continuously instead of once;
> - **the corpus is automatic now, and it is not in `/tmp`.** Every chunk the ear ingests is copied
>   to `--corpus-dir` (default `$HOME/.cache/π/corpus`), bounded to the last `--corpus-max-chunks`
>   chunks. `make corpus` shows it, `make replay-corpus` re-analyses a whole recorded session with
>   other parameters. `--keep-audio` still exists, but it keeps chunks in the RAM directory, so it is
>   the wrong tool for anything longer than one test. The falsification table of phase 3 runs against
>   the durable corpus, not against a hand-built take;
> - **the guardrails below all stay in force**, in particular: raw audio never leaves the machine,
>   the log carries numbers and never audio, and only one variable changes at a time.

---

## The POC is to be rebuilt, ultra compact (KπX, 2026-10-19)

**Graved: the current ear POC is not the shape we keep.** Measured today: **2361 lines** across
`src/π_ear/` and `tests/` (1543 of source, 818 of tests), of which the decision layer alone is **582
lines** of hand-made rules (`physics.py` 184, `inventory.py` 398), plus **seven constants that
decide** (`window_s`, `hop_s`, `level_floor_db`, `voiced_floor`, `k_evidence`, `tau_floor`,
`z_margin`) on top of three that only pace the loop (`idle_sleep_s`, `persist_every_s`,
`gap_factor`). Thousands of lines built around decisions a model should take, which is what the
second law (`IDEAS.md`) forbids.

**What dies with the rebuild:** the descriptor, the pitch estimator, the voicing gate, the identity
boundary, and the tests that hold those rules in place.

**What survives, because it is not a decision:** the transport (one-second chunks, one directory),
the completeness gate on a chunk still being written, the file-as-control (`PAUSE`, `RESET`), the
live loop, the state written atomically, the append-only record, the durable corpus, the replay.

**The target shape: a real POC, ultra compact.** A few hundred lines, no constant anywhere in a
decision path, the parameters in the weights on disk. The infrastructure above is what makes the
replacement measurable instead of a gamble.

---

## Measured before any new test (2026-10-19): the fix, replayed

Nothing to launch for this one. It is already done and it is in the repo:
`tests/test_corpus.py` replays the two real sessions recorded on 2026-10-18 (their event logs are
kept in `tests/artefacts/`) through the inventory. Before the fix: 56 identities for two real
voices. After: 1. The four causes and their numbers are in `IDEAS.md`, section "The measured
failure". Re-run it with `make test`: 30 tests, no microphone, no model, no download.

Still owed here, both measured on 2026-10-19 rather than guessed:

- **the revision restarts at 1 while the log is append-only.** After a restart the log
  carries two lines numbered 1, 2, 3, and no field says they belong to different sessions.
  A reader keyed on `revision` cannot tell them apart. The fix is a design choice, not a
  patch: a `session` id on every line, or a revision read back from the last line at
  startup;
- **the state is written to disk and never read back at launch.** Observed: the same film's
  voice, re-ingested after a restart, was registered under a new key (`9421dc615a099d15`,
  then `db24a423840c6add`), because the inventory lives in RAM only. The pole asks for the
  opposite: an agent that knows you must keep the same name for you across restarts.

Then the falsification pass below, against the durable corpus that now exists by default,
and CAM++ on the same corpus, to finally separate two voices.

---

## Phase 0: what is already here (verified facts, not assumptions)

| Need | Present | Detail |
|---|---|---|
| capture | `arecord`, `ffmpeg`, `sox` | cards: `card 0: sofhdadsp`, device 0 `HDA Analog`, device 6 `DMIC Raw` (the laptop mic array) |
| VAD | `~/.local/share/whisper-cpp/ggml-silero-v5.1.2.bin` | 885 KB, neural, already downloaded |
| STT | 6 ggml models | incl. `ggml-large-v3-turbo-q5_0.bin` (574 MB); `whisper-cli --vad` wires the VAD above |
| runtime | `uv 0.10.3`, `python3 3.14.4` | nothing is installed in the system |
| model hub | `hf` CLI, token already stored | no login needed |
| GPU | `/dev/dri/renderD128` | Intel Arc iGPU, for later phases only |
| shell tools | `jq`, `sox`, `ffmpeg` | enough to inspect and convert without writing much Python |

**Nothing to clone for phases 1 to 3. Nothing to install. Nothing to pull yet.**

---

## Phase 1: capture and log. Decide nothing.

**One session, ~20 minutes, containing everything at once.** This is deliberate: the corpus must
contain both the normal case and the destructive cases, otherwise the inventory will look perfect and
teach us nothing.

### 1a. Capture

```bash
# most robust first: plughw lets ALSA do the rate/channel conversion
arecord -D plughw:0,6 -f S16_LE -r 16000 -c 1 -d 1200 /tmp/π-live-raw.wav
# fallbacks if that fails: -D plughw:0,0 (HDA Analog), or -D default
# then normalise to the contract every model below expects: 16 kHz, mono, s16
ffmpeg -y -i /tmp/π-live-raw.wav -ar 16000 -ac 1 -c:a pcm_s16le /tmp/π-live.wav
```

Sanity check before the long take: `arecord -D plughw:0,6 -f S16_LE -r 16000 -c 1 -d 3 /tmp/t.wav`
then play it back. If capture fails entirely, the fix is a system audio package and therefore a
`sudo` action, which goes through tmux, not through this shell.

**What to put in the take, in this order** (it answers E1, E2 and E3 at once):

| Segment | Duration | Why |
|---|---|---|
| silence, empty room | 30 s | the floor: what is "nothing" |
| your own voice, normal | 3 min | the baseline speaker |
| your own voice, soft then whispered | 1 min | the low-volume regime (E9, and `../RESEARCH.md` Part 1 section 5) |
| a second person speaking | 2 min | the new voice, never named (E1, E2) |
| coughs, throat clearing, 20 of them | 1 min | phantom speakers (E2) |
| door, chair, keyboard, objects | 1 min | phantom sounds |
| a film or a YouTube video with speech | 3 min | does a stranger merge into a known speaker |
| music only | 1 min | is the VAD honest about non-speech |
| a film with music under speech | 2 min | the realistic mixed case |
| your own voice over the phone, other end of the room | 1 min | cross-microphone identity (E7) |

### 1b. Transcribe with the VAD, JSON out

```bash
whisper-cli -m ~/.local/share/whisper-cpp/ggml-large-v3-turbo-q5_0.bin \
  --vad --vad-model ~/.local/share/whisper-cpp/ggml-silero-v5.1.2.bin --vad-threshold 0.5 \
  -oj -of /tmp/π-live /tmp/π-live.wav
```

Verify the flag names once with `whisper-cli --help` before relying on them; the intent is fixed
(JSON output with segment timestamps), the exact spelling may drift between builds.

### 1c. Emit the record, with no decisions

A single small script (`uv run --with numpy`) that, for each segment from the JSON above, writes one
line of JSONL:

- `t0`, `t1` from the segment timestamps;
- `text`;
- `speaker: {"kind": "unknown"}` for now, and nothing else;
- `voice`: `rms_dbfs`, `f0_hz`, `f0_pctl`, `formants_hz`, `hnr_db`, `voiced_ratio` computed with
  numpy only (autocorrelation for f0 and HNR, LPC for formants, RMS for level). No model, no download.

**Definition of done for phase 1:** a JSONL file that reads well to a human, and the measured
bytes-per-excerpt number. That is E5 answered, and E1 half answered.

---

## Phase 2: the inventory, single-pass over the log. The only download.

```bash
mkdir -p ~/.cache/π/models
hf download csukuangfj/speaker-embedding-models \
  3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx --local-dir ~/.cache/π/models
hf download csukuangfj/speaker-embedding-models \
  3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx    --local-dir ~/.cache/π/models
```

Then, over the log, in order, with `uv run --with sherpa-onnx --with numpy`:

1. segment, embed (CAM++), and **hold** the embedding in a candidate pool;
2. register a speaker only once K mutually compatible embeddings have accumulated (start with K=3);
3. apply the measured criterion: `cos_best >= tau_floor` AND `margin >= z_margin * sigma`, where
   `sigma` is the spread of the margins the sensor has itself produced (**the counter form
   `min(tau_base + lambda*N_t, tau_max)` is falsified: measured 2026-10-19 it created 56 identities
   for two real voices**; the numbers are in `IDEAS.md`);
4. print the final state: how many speakers, which excerpts, and what stayed undecided.

The two CAM++ variants are the only real choice here, and it is empirical: run both on your own
voice, keep the one that separates you from the second person better. 72 MB total, five minutes.

**Definition of done:** the state file, and the answer to "how long did a new voice need to become a
speaker". That is E2 and E3.

---

## Phase 3: try to break it (the falsification pass)

Re-run phase 2 and look only at the segments below. Each one targets one failure mode, and each one
has a number attached in `IDEAS.md`.

| Segment from the take | Question | Hypothesis at risk |
|---|---|---|
| 20 coughs | how many speakers were created | E2 |
| door, chair, keyboard | what does the system think these are | E2 |
| film speech with a stranger | did the stranger merge into a known speaker | E2, E4 |
| music only | is it labelled as speech | E2 |
| your voice over the phone | does the key stay stable across microphones | E4, E7 |
| whispered speech | where does it break | E9 |

---

## Phase 4: the two heavier bets (only if phases 1 to 3 pass)

| Bet | What | Cost | Via |
|---|---|---|---|
| non-speech classes | YAMNet, 521 AudioSet classes, 16 kHz mono, 0.96 s patch / 0.48 s hop, own 1024-d embedding | ~16 MB ONNX | `hf download` (repo id to verify at download time) |
| plasticity | TTT-online-bias: adapt bias parameters only, keep the updated weights for the next sample | 2 to 3 GB in the uv cache | `uv run --with torch` (no system change) |

Neither is needed to answer the three questions. Both should wait for a number that justifies them.

---

## Acquisition summary: what, via what, and what is deliberately skipped

| What | Size | Via | Needed for |
|---|---|---|---|
| CAM++ ONNX, zh-cn-common | 27.4 MB | `hf` (installed, authed) | phase 2 |
| CAM++ ONNX, en_voxceleb | 44.8 MB | `hf` | phase 2, comparison |
| YAMNet ONNX | ~16 MB | `hf`, repo to verify | phase 4 |
| torch | 2 to 3 GB, uv cache only | `uv run --with torch` | phase 4 |
| VoxCeleb1 test set (EER benchmark) | ~1.3 GB | official site, then mirrors | **not needed** for the live test |
| Kaggle | nothing | **skipped**: it costs an account and a token for no gain here | |
| git clones | none | phases 1 to 3 need no repository | |

Two repositories are worth **reading** later, not cloning yet: `state-spaces/mamba` (the reference
implementation of the state-as-memory family) and `IDSIA/modern-srwm` (self-referential weight
matrices). Both are Phase 4 material at the earliest.

---

## Guardrails

- **16 kHz mono everywhere.** It is the contract of CAM++ and of YAMNet. Convert once, at capture.
- **Raw audio never leaves the machine.** The event log never contains raw audio, only numbers.
- **The take contains other people's voices**, from films and from a second person. That is biometric
  data. Rule: keep the log, delete the raw audio when the session is done, unless a named reason to
  keep it exists.
- **One variable at a time.** Never change the microphone and the VAD threshold in the same session.
- **Logs before models.** Every phase produces a file that can be replayed. Nothing is measured once
  and thrown away.

## Definition of done for this file

The three questions of `IDEAS.md` answered with numbers, written back into `IDEAS.md`, and every
hypothesis marked confirmed, falsified, or still open.

---

## Done 2026-10-18 (night): the ear is rebuilt (KπX: "refais entièrement l'oreille, end to end")

The section "The POC is to be rebuilt, ultra compact" above is executed. Summary, details in
`IDEAS.md` ("The rebuild: π-ear v1") and `README.md`:

- **dead**: the descriptor, the pitch estimator, the voicing gate, the identity boundary, the seven
  deciding constants, and the tests that held them;
- **kept**: the transport, the completeness gate, PAUSE/RESET as files, the live loop, the atomic
  state, the append-only record, the durable corpus, the replay;
- **new**: Silero + CAM++ as frozen sensors, one Bayesian model (vMF + CRP) whose parameters are
  statistics, base weights measured on public data, corrections as a channel, a checkpoint read
  back at launch (the two items "still owed" above are both fixed: every line carries a `session`,
  and the state is read back, so a voice keeps its key across restarts), and a law test.

**Still owed, and only KπX can do it: the live test on the real microphone.** Every number above
comes from public recordings replayed through the real path. In three panes: `make rec`,
`make run`, `make watch`; then a film, a second person, coughs, KπX's own whisper. After it,
correct what is wrong with `π-ear correct`, and run it again: the paradigm says the same errors do
not come back, and that is now a measurement to make, not a promise.

---

## Done 2026-10-19: v1.1, from RETURNS.md (KπX: "carte blanche pour la v1.1")

All ten proposals of `../../RETURNS.md` section 8, plus five of Claude's, are in, measured, and
tested (91 tests). The list, the numbers and the two designs that failed on the way are in
`IDEAS.md`, section "v1.1".

**Owed by KπX on the real microphone, in this order:**

1. `make probe FROM=84 TO=163` on the corpus of session `20261019-071257`, if it is still in the
   corpus (one hour rolling): it says whether the 81-second hole was speech or not.
2. One correction, live: `make correct C="name ROW --label KπX"`, then the same situation again.
3. One `undo`, to see the model come back exactly.
4. A conversation with a second person in the room, then `make why ROW=...` on a doubtful window
   (the smallest `margin` in the record).
