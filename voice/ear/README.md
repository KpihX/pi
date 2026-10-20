# π-ear v2: the streaming ear of π, who and what

Two servers and one directory, as before. The recorder never stops, the ear follows it, and what
the ear has learned is on disk at every instant. v1 answered **who**; v2 answers **what** too, in
the same spirit: one learned model per question, every decision a posterior, every doubt named,
and one teacher besides the stream, **a correction said once** (typed, or spoken through the
mouth and heard back by the ear: the inbox does not care).

```
      [microphone]
           |
   SERVER A: arecord            one raw chunk per second, endless
           |
   <the audio directory>        the whole transport between the two servers
           |
   SERVER B: the ear
     Sensor   Silero VAD  ->  CAM++ (2 s windows, cut live, + shadow windows offset by 1 s)
     Model    who: windows -> conditions -> voices, in a place; person or device
     Scribe   what, on its own thread: Parakeet v3 while it is said, Whisper turbo in the silence,
              a learned reader choosing between them, the memory of what KπX taught
           |
   /tmp/π/state.json                  atomic, ~1/s: what is known now
   /tmp/π/events.jsonl                the port (out): one line per window, turn, text, question, correction
   ~/.cache/π/ear/corrections.jsonl   the inbox (in): corrections and answers, append-only, undoable
   ~/.cache/π/ear/model.json          the checkpoint (who + the reader)
   ~/.cache/π/ear/embeddings.f32      one row per window heard; spectra.f32 its spectral shape
   ~/.cache/π/ear/observations.jsonl  when, how and where each row was decided: the replay
   ~/.cache/π/ear/runs.jsonl          every reading of every run; taught.jsonl every text taught
   ~/.cache/π/corpus/*.raw            the audio of the session, bounded, outside RAM
```

## The three laws, as code

1. **100 % stream.** No setup phase, no calibration phase: the first window is decided like the
   thousandth.
2. **Nothing tuned, everything learned.** `tests/test_law.py` parses `model.py`, `words.py` and
   `source.py` and fails on any numeric literal but 0 and 1, and on any comparison with a
   constant. The numbers live in `weights/base-*.json`, measured by `π-ear base` on public
   recordings no test uses, then moved by the stream and by corrections.
3. **Every model knows when it does not know.** Who: `unknown`, `margin` (nats), `trust` (a
   calibrated probability). What: a named doubt per text (`clear`, `unseen`, `taught`,
   `ambiguous`, `not_understood`), a trust per word, and *questions* when it hesitates between
   readings, or when a known voice spoke and it did not understand (a cough asks nothing).

## Who (`model.py`), in one screen

```
  a place        p     ~ vMF(mean0, kappa_p)    rooms and microphones differ
  a voice        v     ~ vMF(p, kappa_in)       voices of one place share its channel
  a condition    mu_c  ~ vMF(v, kappa_c)        one voice drifts across rooms and days
  one window     x     ~ vMF(mu_c, kappa)       how much one voice varies in one place
  who            CRP(alpha) over conditions, CRP(alpha_v) over voices
```

Every concentration is measured on the base: FSDD and LibriSpeech voices heard in four simulated
places (`public.place`: a room, a microphone band, a gain, a noise floor), then refined by `same`
corrections. What each level fixed, measured:

| level | without it | with it |
|---|---|---|
| conditions -> voices (Q2) | 4 of 12 random rooms split a voice | 0 of 12 |
| the place prior | the pyannote sample's two voices heard as one (they share a channel: cos 0.49 where the population says 0.35) | two voices |
| shadow windows + `turns` (Q4) | who spoke when, by the 2 s window | by the half window: 0.625 -> 0.75 of the time on that sample, 0.99+ in the rooms |
| person / device (Q3) | a loudspeaker is a person | spectral shape, learned on simulated loudspeakers: 5 of 6 right, the miss unsure (p 0.77); `device` once fixes it for good |
| trust (Q1) | a margin in nats | P(right \| margin), from the base's own replay, then from every correction |

A voice heard again in a later session, still nameless, makes the ear **ask its name, once** (Q5).

## What (`asr.py`, `words.py`, `lexicon.py`, `scribe.py`)

International by default, French measured too, anything else plugs in:

| recogniser (int8, 2 threads) | WER en | WER fr | s per s of audio | words invented on 10 noise clips |
|---|---|---|---|---|
| **parakeet-v3** (fast, 25 languages) | 3 % | 6 % | 0.16 | 2 |
| **whisper-turbo** (slow, 99 languages) | 2 % | 9 % | 1.14 | 8 |
| whisper-small | 2 % | 19 % | 0.89 | 6 |
| canary-180m | 1 % | 15 % | 0.25 | 2-9 |

```
 speaking ... partial (fast, re-read while the run goes on: the text appears as it is said)
 run ends ... text r0 (fast, at once)
 silence  ... text r1 (slow)   the reader chooses among: fast, slow, the substitutions it
                               learned, the corrected runs it recalls, and "I did not understand"
 doubt    ... question         when it is ambiguous or not understood: answer it once
```

The **reader** is a Bayesian conditional logit (evidence maximisation, no hand-set prior) over 11
features of each reading: who wrote it, how much the other recogniser agrees, token confidence,
learned substitutions, recall of a corrected run, known words, and, for "I did not understand",
low confidence, two recognisers apart, and a run Silero itself found little like speech. Its base
weights come from 80 public runs (Piper voices in English and French, clean and degraded) and the
ESC-50 sounds as the VAD lets them through, plus *correction episodes*: every run the
recognisers got wrong, corrected once and heard again, so that a correction is trusted from the
first time. Each word carries its own trust and `agreed` / `new` marks.

## The port and the inbox: plug anything in

```json
{"port":2,"kind":"session","weights":"…","fast":"parakeet-v3","slow":"whisper-turbo",…}
{"port":2,"row":42,"t0":82.6,"t1":84.6,"kind":"known","speaker":"1abe…","condition":"9f0c…",
 "label":"KπX","source":"person","source_p":0.99,"p":1.0,"margin":187.4,"trust":0.998,"shadow":false,…}
{"port":2,"kind":"turns","run":"…:80.6","segments":[[80.6,81.6,"1abe…"],[81.6,84.6,"77d2…"]]}
{"port":2,"kind":"partial","run":"…:80.6","text":"send the notes to"}
{"port":2,"kind":"text","run":"…:80.6","revision":1,"text":"send the notes to KπX","doubt":"taught",
 "p":0.97,"words":[{"w":"send","trust":0.99,"agreed":true,"new":false},…],"readings":[…]}
{"port":2,"kind":"question","id":"…:80.6#text","about":"text","choices":["… pi x","… pie ex"],"doubt":"ambiguous"}
{"port":2,"kind":"correction","what":"text","answers":"…:80.6#text","line":7,"refused":null}
```

A consumer (the arbiter, the mouth) reads `port.read()` / `port.follow()` and gets typed events;
it never reads the internals. It speaks back through **`port.Inbox`**, the same three calls a
terminal uses:

```python
inbox = port.Inbox(memory)
inbox.correct("text", rows, text="send the notes to KπX")   # or same/different/not_voice/name/person/device
inbox.answer(question_id, text="Alice")                        # a name, or what was said
inbox.answer(question_id, choice=1)                            # or one of the question's choices
inbox.undo(line)                                               # forget one, exactly
```

So a vocal correction is: the mouth reads a `question` aloud, the ear hears the answer, the
arbiter calls `inbox.answer`. Recognisers plug in by name (`PI_EAR_FAST`, `PI_EAR_SLOW`: any
`asr.REGISTRY` name, `none`, or `package.module:factory`); a factory takes the models directory
and returns `samples -> Hypothesis`.

## Corrections

```bash
π-ear correct same 812 840                        # one voice (joins voices, teaches kappa_c)
π-ear correct different 812 901                   # two voices
π-ear correct not_voice --speaker KEY             # never a voice
π-ear correct name 812 --label KπX                # call it KπX
π-ear correct device --speaker KEY                # a loudspeaker (or `person`)
π-ear correct text --from 163 --to 170 --text "le cours de Galois"   # what was said
π-ear questions                                   # what the ear is asking
π-ear answer 'S:163.2#text' --choice 1            # answer it (or give the text)
π-ear undo 3                                      # forget correction 3, exactly
```

Every line is obeyed at the next tick and absorbed into the weights. `undo` rebuilds by replay:
the model of who from the base, every row and every other correction at the moment it was
applied (bit for bit, tested); the reader from its base and every other text taught. A text
correction finds its run even sessions later (`runs.jsonl`).

## Setup, in this order

```bash
make sync     # 1. numpy, scipy, onnxruntime, sherpa-onnx, typer
make fetch    # 2. models + public corpora into ~/.cache/π (~2.4 GB downloaded once)
make rec      # pane 1: the recorder. Start once, never stop.
make run      # pane 2: the ear. One line per decision and per text.
make watch    # pane 3: the state moving on disk.
```

Then `make status`, `make events`, `make questions`, `make answer Q=… A="…"`,
`make correct C="text 812 --text '…'"`, `make undo N=3`, `make why ROW=812`,
`make probe FROM=84 TO=163`, `make pause`, `make resume`, `make snapshot`, `make diff`.

`PI_EAR_LANGUAGE=fr` fixes Whisper's language; `PI_EAR_SLOW=none` keeps only the fast reader;
`PI_EAR_FAST=none` turns the *what* off.

## What is measured (2026-10-20, real public recordings written as 1 s chunks, the real path)

| scene | result |
|---|---|
| 6 voices + coughs, doors, keyboard, laughter (`room`) | 6 speakers, 0 phantoms, purity 1.000 |
| 12 random rooms (gains, floors, takes) | purity 1.000 in 12/12, no voice split (v1: 4/12), one laugh heard as the person who laughs |
| the same people in another place (day 2) | most voices come back under new keys; one `same` each, and on day 3 each is its day-1 voice |
| the room with music all along, 10 dB under speech | purity 1.000, no split, the music never a person |
| pyannote two-person sample | 2 speakers, turns right 75 % of the time |
| 4 Mandarin speakers | 4 speakers |
| 160 indoor sounds, no voice | nobody (but, at most, one laugh); with the recognisers on, 14 runs read: 10 `not_understood`, 4 one-word inventions marked `unseen`, no question |
| person vs loudspeaker, 3 pairs | 5/6 right, the miss unsure; corrected once, it stays |
| English + French, clean and under music at 0 dB, Parakeet + turbo | 3 errors in 49 words (6 %), two of them one homophone; every text revised in the silence |
| a name mis-heard, answered once | written right the next time, no question asked |

## Tests

```bash
make test-fast   # no model, no download: law, maths, model, reader, disk, CLI
make test        # + the real sensors, recognisers and simulated rooms (needs make fetch)
make check       # ruff
```
