# π-ear v1.1: the streaming ear of π

Two servers and one directory, as before. The recorder never stops, the ear follows it, and what
the ear has learned is on disk at every instant. What changed is everything that decides: v0 was
582 lines of hand-made rules and seven hand-set constants; v1 is **one probabilistic model** whose
decisions are posteriors, whose parameters are statistics learned from public data and from the
stream, and whose only teacher besides the stream is **a correction, said once**.

```
      [microphone]
           |
   SERVER A: arecord            one raw chunk per second, endless
           |
   <the audio directory>        the whole transport between the two servers
           |
   SERVER B: the ear
     Sensor   Silero VAD  ->  CAM++ (2 s windows, cut live)
     Model    who is it: argmax of a posterior, never a threshold
     Scribe   its own thread: Whisper small on each finished run in which a voice was recognised
           |
   /tmp/π/state.json            atomic, ~1/s: what is known now
   /tmp/π/events.jsonl          append-only record: one line per window, per text, per correction
   ~/.cache/π/ear/model.json    the checkpoint: clusters, links, learned statistics (read at launch)
   ~/.cache/π/ear/embeddings.f32  one row per window heard, append-only
   ~/.cache/π/ear/observations.jsonl  when and how each row was decided: the model replays from it
   ~/.cache/π/ear/corrections.jsonl  what KπX said was wrong (append-only, obeyed live, undoable)
   ~/.cache/π/corpus/*.raw      the audio of the session, bounded, outside RAM
           ^
   /tmp/π/PAUSE  /tmp/π/RESET   control is a file, not a signal
```

## The model, in one screen

```
  speaker means   mu_k ~ vMF(mean0, kappa0)     what voices look like, as a population
  one window      x    ~ vMF(mu_k,  kappa)      how much one voice varies
  who it is       z    ~ CRP(alpha)             how often a new voice shows up
```

Every window is scored against every cluster and against "someone new", and the most probable
explanation wins. Every v0 rule has a probabilistic replacement:

| v0 rule (dead) | v1 mechanism |
|---|---|
| `voiced_floor`, `level_floor_db`, a hand-written pitch estimator | Silero VAD at its published operating point, plus **noise clusters**: the base heard ESC-50 folds 1-3, and `not_voice` corrections add more |
| `tau_floor`, `z_margin` (the identity boundary) | the vMF likelihood, with `kappa` and `kappa0` **estimated**, never chosen |
| `k_evidence = 3` (the anti-phantom bar) | a cluster becomes a speaker when "one voice" is more probable than "all different voices". With one window the two are equal, so **one window never makes a person** |
| the merge invariant | two clusters fuse when one voice explains them better than two |
| a fixed window of 3 s | `ENCODER_WINDOW_S = 2.0`, the sensor's validity domain, **measured** (EER 11.8 % at 1 s, 0.0 % at 2 s on real LibriSpeech voices). Shorter leftovers are *glimpses*: decided, never learned from |

**The law is a test.** `tests/test_law.py` parses `model.py` and fails if it contains a numeric
literal other than 0 and 1, or a comparison against a constant. It also checks that the checker
itself catches the v0 shape (`cos_best >= 0.5 and margin >= 3.0 * sigma`).

**Where the numbers live.** `src/π_ear/weights/base-*.json`: sufficient statistics measured by
`π-ear base` on public voices and sounds that no test uses (FSDD jackson/nicolas/theo, two
LibriSpeech speakers, ESC-50 folds 1-3). The stream adds its own statistics on top, and `refresh`
re-estimates every parameter after each decision, never before it. `contract.py` holds what is
not a decision: the audio format, the sensors' published operating points, resource bounds.

**What `kappa` learns from, and what it refuses to learn from.** Only pairs *known* to share a
voice: the labelled base, then every `same` correction (all cross pairs of the two clusters, in one
dot product of their sums). Never from the clusters the model built itself, which is the
self-feeding loop that made v0 create 56 identities. Never from two consecutive windows of one run
either: measured, they sit closer than one voice does across recordings.

## Corrections: one case, obeyed now, absorbed into the weights, undoable

```bash
π-ear correct same 812 840                          # these windows are one voice (fuses, lowers kappa)
π-ear correct different 812 901                     # two voices (splits, never re-merged)
π-ear correct not_voice --speaker KEY               # that "speaker" was never a voice (becomes noise)
π-ear correct name 812 --label KπX                  # call this voice KπX
π-ear correct name --from 163 --to 288 --label vid  # the same, by stream seconds of the latest session
π-ear undo 3                                        # forget correction number 3, exactly
```

Rows are the `row` field of `events.jsonl`; `--from/--to` is the human shortcut, resolved to the
rows of that span (the span is kept in the correction line, the rows are the truth). A correction
is one appended line in `corrections.jsonl`; the running ear applies it at its next tick (fast
clock), logs it in the record with its `line` number, and re-estimates its parameters (slow
clock). `undo N` appends one more line: the ear rebuilds its whole model from the base by
replaying every row and every other correction at the moment it was first applied. The replay is
bit for bit (`test_the_memory_rebuilds_exactly_by_replay`), so an undo is exact, not approximate.

Two corrections also teach what the stream cannot: a `same` or `different` on two **adjacent**
windows of one run is a labelled transition, the only evidence the continuity prior learns from.

## Looking under a decision

```bash
π-ear why 812                      # what was decided then (p, margin, level), the 3 best explanations now
π-ear probe --from 84 --to 163     # the corpus, second by second: dBFS, Silero's own probability, decisions
π-ear probe --from 84 --to 163 --asr   # + Whisper on the whole span, ungated
```

`probe` answers "was it not speech, or speech the detector missed?" from the audio itself, the
question RETURNS.md left open for the 81-second hole.

## Setup, in this order

```bash
make sync     # 1. numpy + sherpa-onnx (+ sherpa-onnx-core: its onnxruntime) + typer
make fetch    # 2. models + public corpora: ~1.1 GB downloaded once, ~850 MB kept in ~/.cache/π
```

Then three panes:

```bash
make rec      # pane 1: the recorder. Start once, never stop.
make run      # pane 2: the ear. One line per decision.
make watch    # pane 3: the state moving on disk.
```

Then `make status`, `make events`, `make correct C="different 812 901"`, `make undo N=3`,
`make why ROW=812`, `make probe FROM=84 TO=163`, `make pause`, `make resume`, `make snapshot`,
`make diff`, `make reset`, `make corpus`, `make replay-corpus`.

`PI_EAR_WHISPER=base` (or `tiny`) trades transcription quality for CPU; `PI_EAR_LANGUAGE=fr`
skips language detection (2.5x faster, and no invented Japanese over music).

## The record, and the port (`port.py`, version 1; values below are illustrative)

```json
{"port":1,"session":"20261019-071257","kind":"session","weights":"…","rows":121,"first_chunk":"20261019-071259.raw"}
{"port":1,"session":"…","row":42,"t0":82.6,"t1":84.6,"decided_at":84.62,"kind":"known",
 "speaker":"1abeee72123db13d","label":"KπX","p":1.0,"margin":187.4,
 "runner_up":{"id":null,"p":0.0},"evidence":12,"learned":true,"run":80.6,"rms_dbfs":-28.7,"rel_db":24.1}
{"port":1,"session":"…","kind":"text","t0":80.6,"t1":88.1,"rows":[41,42,43],"text":"…"}
{"port":1,"session":"…","kind":"correction","what":"different","rows":[812,901],"line":3,"refused":null}
```

`kind` is `unknown` (heard once, or a glimpse nobody matches), `new` (this window made a speaker),
`known`, or `noise`. `p` is the posterior; it saturates at 1.0 on 192 dimensions, so **`margin`**
(nats between the decision and the runner up) carries the doubt. `rel_db` is the level above this
session's non-speech floor, learned from the stream (no calibration, no SPL guess). A record that
does not start on the port (v0, v1.0) is archived whole when the ear starts.

A consumer reads the port, never the internals: `port.read(path, session)` gives typed
`Session`, `Decision`, `Text` and `Correction` events, `port.follow(path)` yields them live.

## What is measured (2026-10-18, this repository's simulated rooms)

| Scene (real public recordings, written as 1 s chunks) | Result |
|---|---|
| 6 real voices taking turns + coughs, doors, keyboard, laughter, noise floor (`room`) | 6 speakers, 0 phantoms, purity 1.000, coverage 0.86-0.91 |
| the same room, 12 random seeds, random gains (-12..0 dB) and floors (-70..-50 dBFS) | purity 1.000 in 12/12, 0 phantoms in 12/12 (see `IDEAS.md` for fragmentation) |
| 160 indoor sounds of fold 5, never heard by the base, no voice | 0 speakers apart from, at most, one long laugh (laughter is a voice) |
| 2 minutes of room floor | 0 observations |
| pyannote two-person sample, turns of 0.4-4 s | 2 speakers, purity >= 0.7 |
| sherpa-onnx Mandarin file, 4 speakers | 4 speakers |
| synthetic twins (Piper), fused by the sensor, one `different` each | apart in the next session |
| a laugh that became a "person", one `not_voice` | goes to noise in the next session |
| speed | ~50x real time on 2 CPU threads; Whisper never on the decision loop (v1.1) |

## Tests

```bash
make test-fast   # 50 tests, no model, no download, < 1 s: law, maths, model, disk
make test        # + the real sensors and 20 simulated rooms (needs make fetch), ~6 min
make check       # ruff
```

`tests/artefacts/` keeps the two v0 event logs of 2026-10-18: the evidence of the 56-identity
failure. v1 cannot replay them (they hold numbers, not audio), which is why the corpus exists.
