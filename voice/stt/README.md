# π-ear: the streaming ear of π

Two servers and one directory. The recorder never stops, the ear follows it, and
the state of what has been heard is on disk at every instant. There is no
"process the recording afterwards" step anywhere in this package, and that
absence is the design.

This directory is two things at once, deliberately:

| Path | What it is |
|---|---|
| `IDEAS.md`, `RESEARCH.md`, `TODO.md` | the ear's memory: the thinking, the sources, the plan |
| `pyproject.toml`, `src/`, `tests/`, `Makefile` | the system itself: module `π_ear`, command `π-ear`, distribution name `pi-ear` (ASCII, imposed by the packaging standard) |

## The architecture

```
        [microphone]
             |
     SERVER A: the recorder          arecord, one raw chunk per second,
     (never stopped)                 endless, into the audio directory
             |
      <the audio directory>          the interface between the two servers
             |
     SERVER B: the ear               follows the directory, slides a 3 s window,
     (pausable, restartable)         decides, rewrites the state
             |
     /tmp/π/state.json              atomic rewrite, ~1 per second: what is
                                     known right now, and who is in the room
     /tmp/π/events.jsonl            append-only, one line per window: the record
     $HOME/.cache/π/corpus/*.raw    the audio of the session, bounded, outside RAM
             ^
     /tmp/π/PAUSE  /tmp/π/RESET     control is a file, not a signal
```

Why a directory of one-second chunks and not a pipe: a pipe dies with either
side, cannot be paused without losing audio, and cannot be replayed. With chunks
the recorder survives everything, a pause becomes a backlog that is drained on
resume, and the same audio can be re-analysed later with other parameters.

## Setup, in this order

```bash
make sync     # 1. base env: numpy + sherpa-onnx + typer (uv resolves it)
make models   # 2. the only download: CAM++ ONNX, 27.4 MB (needed from v1)
```

Then three panes:

```bash
make rec      # pane 1: the recorder. Start once, never stop.
make run      # pane 2: the ear. Prints one line per decision.
make watch    # pane 3: the state moving on disk.
```

Then live your life, play a film, talk to someone. In another pane:

```bash
make status     # human readable summary
make events     # tail -f the append-only log
make pause      # freeze the ear; the recorder keeps recording
make resume     # drain the backlog and catch up
make snapshot   # freeze a comparable copy of the state
make diff       # unified diff of the two last snapshots
make reset      # archive the state, restart from the backlog
make corpus     # what the durable corpus holds, and what it costs in disk
make replay-corpus  # run a recorded session again, with other parameters
make clean      # wipe /tmp/π (the corpus is not in /tmp: it survives)
```

## Every option

| Flag (`π-ear run`) | Default | What it does |
|---|---|---|
| `--root` | `/tmp/π` | runtime root: audio, state, events, snapshots, archive |
| `--device` | `plughw:0,6` | ALSA capture device. Must match the recorder |
| `--rate` | `16000` | sample rate in Hz. The model contract |
| `--chunk-s` | `1` | chunk duration. Must match `arecord --max-file-time` |
| `--window-s` | `3.0` | analysis window: what the inventory is shown |
| `--k-evidence` | `3` | observations needed before a speaker is registered |
| `--tau-floor` | `0.5` | the cosine the closest voice must clear |
| `--z-margin` | `3.0` | how many sigma above the observed margin noise the winner must be |
| `--voiced-floor` | `0.25` | minimum voiced ratio before a window counts as speech |
| `--from-start` | off | ingest the backlog instead of skipping it |
| `--keep-audio` | off | keep consumed chunks in the RAM directory (needed by `replay`) |
| `--audio-dir` | none | read chunks from here instead of `<root>/audio` (a frozen corpus) |
| `--corpus` / `--no-corpus` | on | copy every ingested chunk into the durable corpus |
| `--corpus-dir` | `~/.cache/π/corpus` | where that corpus lives |
| `--corpus-max-chunks` | `3600` | how many chunks it keeps (one hour at one chunk per second) |
| `--log-silence` | off | log every window, silent ones included |
| `--max-seconds` | none | stop after N seconds (demos, tests) |

| Make variable | Default | Note |
|---|---|---|
| `ROOT` | `/tmp/π` | tmpfs on this machine: RAM, see the choices below |
| `DEVICE`, `RATE`, `CHANNELS`, `CHUNK_S` | `plughw:0,6`, `16000`, `1`, `1` | the capture contract |
| `WINDOW_S`, `K_EVIDENCE`, `TAU_FLOOR`, `Z_MARGIN`, `VOICED_FLOOR` | `3.0`, `3`, `0.5`, `3.0`, `0.25` | the inventory knobs |
| `CORPUS_DIR`, `CORPUS_MAX_CHUNKS` | `~/.cache/π/corpus`, `3600` | the durable corpus |
| `RUN_FLAGS` | empty | extra flags for `make run`, e.g. `make run RUN_FLAGS=--keep-audio` |
| `MODELS` | `~/.cache/π/models` | where `make models` puts the ONNX |

## The choices, and why

1. **16 kHz mono s16le.** Not a preference: CAM++, YAMNet and sherpa-onnx all
   expect it, and `plughw:0,6` on this laptop accepts it natively (verified), so
   nothing resamples and no quality is lost to a conversion.
2. **One-second chunks.** Cost: up to one second of latency in the state. Bought:
   pause, restart, catch-up and replay. Below one second `arecord --max-file-time`
   does not go, and the pipe alternative gives up all four properties.
3. **A three-second window, one decision per second.** Long enough for pitch to
   be measurable, short enough that "now" still means now.
4. **No VAD in v0.** The chunk boundary already segments the stream. A voice
   activity detector earns its place in v1, when excerpts must be cut inside a
   chunk rather than at its edge.
5. **Registration on evidence, not on a single observation.** One noisy second
   cannot create a phantom speaker. `k_evidence = 3` compatible observations
   are required.
6. **A criterion that is measured, not counted, plus an invariant that holds it.**
   A window belongs to a known voice when it is at least `tau_floor` close to the
   closest one AND closer to it than to the runner-up by `z_margin` times the
   spread of the margins this sensor has itself been producing. The spread
   calibrates the ear, so nothing is hand-tuned. This replaces
   `tau = min(tau_base + lam * N_registered, tau_max)`, kept here as the failure it
   turned out to be: measured on 379 seconds of real audio, that counter made every
   registration move the boundary, so the boundary admitted the next voice, and 56
   identities were created for two real voices with tau pinned at its ceiling.
   Alongside it sits the invariant that two entries may never describe a voice this
   sensor cannot tell apart: a candidate that passes the same test a window passes
   is absorbed rather than registered, and its old key is mapped in `absorbed`
   instead of being lost.
7. **The identity key is computed once, at commit, and versioned.** A centroid
   keeps moving as evidence accumulates; a key recomputed on every observation
   would silently change who someone is overnight. Hence `version`, and hence a
   key that is a function of the committed centroid only.
8. **No `regime` field, ever.** The record carries `f0_pct_in_speaker`, the rank
   of this pitch inside that speaker's own observed distribution. Writing the
   word "bass" would be a decision the ear has no right to take: 120 Hz is low
   for most voices and high for some.
9. **The provisional identity vector is a placeholder, and it is labelled as
   one.** Four numbers (pitch, brightness, level, noisiness), no model, no
   download. It is NOT biometric: a voice on a film will collide with a voice in
   the room. It exists so the mechanism (candidates, evidence, registration,
   moving threshold) can be watched working tonight. v1 replaces it with a CAM++
   embedding and changes nothing else.
10. **`/tmp` is tmpfs here, so audio is consumed and deleted, and the corpus lives
    in the home instead.** 14.5 GB free in RAM, and the ear gains nothing from
    keeping what it has already read, so the audio directory empties itself as it
    goes. Measured 2026-10-19: that also meant six minutes of real test (a hum, then
    a film) survived only as numbers, and the exact session could not be replayed
    against another identity vector. Hence the **durable corpus**: every chunk that
    is ingested is copied to `--corpus-dir`, bounded to the last
    `--corpus-max-chunks` chunks, outside RAM. It is on by default in `make run`,
    visible with `make corpus`, and reusable with `make replay-corpus`. A long pause
    is the one case that still grows in RAM: 1.9 MB per minute of backlog.
11. **The state is rewritten atomically.** A half-written JSON read by the
    agentic pole would be worse than a stale one.
12. **Pause is a backlog, not a discard.** Freezing the state while still reading
    the audio would silently throw the paused seconds away. The cursor is not
    advanced, the audio accumulates, and `backlog_chunks` in the state says how
    much.
13. **The voiced gate is a measured value, not a guess.** A silent room still
    yields a voiced ratio of 0.08 to 0.15 (noise slipping through the periodicity
    test) while a normal voice yields 0.4 to 0.7, so the default is 0.25 and it is
    a first-class option. Before that value was measured, four seconds of empty
    room were enough to register a phantom speaker, which is the exact failure the
    pole law names: the ear must say unknown, it must not invent. The measurement
    is written up in `IDEAS.md`.
    The gate alone was not enough, and that is worth knowing: with a live recorder
    and a keyboard in the room, the phantom came back, because the pitch detector
    was reporting an artefact (see 14) that made every noisy window look like the
    same voice. A level gate cannot repair a broken pitch.
14. **A peak on the boundary of the pitch search range is not a pitch.** When
    nothing is periodic, the autocorrelation is maximal at the shortest lag the
    range allows, which reads as the highest frequency it can express: 457.1 Hz
    here, on every noisy window, identical every time. That is how ninety seconds
    of room and keyboard noise looked like one speaker with a cosine of 1.000, and
    it registered a person out of nothing. The detector returns no pitch on that
   boundary, so the artefact never reaches the inventory. `tests/test_physics.py`
   holds the whole chain down, from white noise to the registration decision.
15. **A chunk the recorder is still writing is not a chunk yet.** It is read only
    once the recorder has moved on to the next one. Measured 2026-10-19, before this
    gate existed: reading a growing file made every poll analyse one more observation
    of the SAME second of voice, so 4179 events covered 379 seconds of audio, the
    evidence bar was diluted tenfold, the process sat at 99.9% of one core and the
    log would have cost 330 MB per day. After the gate: one observation per second,
    one analysis per chunk, and the evidence bar means what it says. The last chunk
    of a live session, legitimately partial, is documented as the one that does not
    make the cut; a replay of a frozen corpus reads it, because nothing there is
    still growing.
16. **The loop sleeps at the end of every iteration, and `PAUSE` really pauses.** Two
    controls that looked implemented and were not, both measured 2026-10-19. The sleep
    sat after the `break`, so it was dead code no reader could reach and a live ear spun:
    one core pinned at 99.9% for a loop whose only job is to notice that nothing is new,
    while the analysis itself costs 28 ms on real audio. And the paused branch waited
    without a `continue`, so it fell through to the consumer: the cursor advanced while
    paused and the state was rewritten as `paused=False` with the PAUSE file sitting right
    there. A control that lies about its own state is worse than a missing control. Both
    are held down by `tests/test_loop.py`, and the state now says `paused` when it is.

## What v0 proves, and what it does not

Proves: chunks are ingested exactly once, in order, and a chunk still being written
is not read at all; a window becomes numbers; candidates accumulate evidence; a
speaker is registered when evidence allows; **no two entries ever describe a voice
this sensor cannot tell apart**; a speaker is recognised again after a silence; the
state on disk is rewritten live and the log is append-only; the audio of a session
survives in a bounded corpus outside RAM.

Does not prove: that the identity vector separates real voices. It does not, and
that is now measured rather than suspected. Two synthetic voices an octave apart sit
at a cosine of 0.962 in this space; on the real two-voice session every window
scored 0.9997 against every entry, with a median best-minus-runner-up margin of
0.0005. So v0 holds ONE entry for two voices, and that is the honest answer from a
sensor that cannot tell them apart. Telling them apart is v1's job with CAM++, and
the corpus is what will let last night's exact session be replayed against it.

## Tests

```bash
make test    # 30 tests, no microphone, no model, no download
make check   # ruff: imports, format, lint
```

The synthetic session in `tests/synthetic.py` builds two voices, a gap, and the
first voice returning. `test_pipeline.py` runs it through the real pipeline and
checks the state, the log and the corpus; `test_inventory.py` checks the claims the
mechanism rests on; `test_corpus.py` **replays the real failures**, out of the event
logs kept in `tests/artefacts/`, and asserts that the session which once produced 56
identities produces one. Those artefacts are the reason the replay exists at all: a
fix proven on a synthetic mood is not a fix.
