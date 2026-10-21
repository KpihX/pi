# π-ear v2.1: the streaming ear of π, who and what

Two servers and one directory, as before. The recorder never stops, the ear follows it, and what
the ear has learned is on disk at every instant. v1 answered **who**; v2 answers **what** too, in
the same spirit: one learned model per question, every decision a posterior, every doubt named,
and one teacher besides the stream, **a correction said once** (typed, or spoken through the
mouth and heard back by the ear: the inbox does not care). v2.1 is v2 after its first live test
(two hours of a football broadcast heard by a laptop, KπX in the room): what it got wrong, and
how each fix was measured on the replay of that session, is in `IDEAS.md`, section "v2.1".

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
     Scribe   what, on its own thread: runs cut into pieces (a voice's phrase, under Whisper's
              window), Parakeet v3 at once, Whisper turbo on groups of pieces with context,
              a learned reader choosing between them, each voice's language, what KπX taught
           |
   /tmp/π/state.json                  atomic, ~1/s: what is known now
   /tmp/π/events.jsonl                the port (out): one line per window, turn, text, question, correction
   ~/.cache/π/ear/corrections.jsonl   the inbox (in): corrections and answers, append-only, undoable
   ~/.cache/π/ear/model.json          the checkpoint (who + the reader)
   ~/.cache/π/ear/embeddings.f32      one row per window heard; spectra.f32 its spectral shape
   ~/.cache/π/ear/observations.jsonl  when, how and where each row was decided: the replay
   ~/.cache/π/ear/pieces.jsonl        every reading of every piece; taught.jsonl every text taught
   ~/.cache/π/corpus/*.raw            the audio of the session, bounded, outside RAM
```

## The three laws, as code

1. **100 % stream.** No setup phase, no calibration phase: the first window is decided like the
   thousandth.
2. **Nothing tuned, everything learned.** `tests/test_law.py` parses `model.py`, `words.py`,
   `source.py` and `language.py` and fails on any numeric literal but 0 and 1, and on any comparison with a
   constant. The numbers live in `weights/base-*.json`, measured by `π-ear base` on public
   recordings no test uses, then moved by the stream and by corrections.
3. **Every model knows when it does not know.** Who: `unknown`, `margin` (nats), `trust` (a
   calibrated probability). What: a named doubt per text, from the reader's own posterior
   (`clear`, `unseen`, `taught`, `partial`: "I understood bits of it", `ambiguous`,
   `not_understood`), a trust per word, and *questions*, only to people, about the words it is
   unsure of (a commentator on television is nobody to ask; a cough asks nothing).

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
 speaking ... partial     the open part of the run, re-read when the scribe is free
 a piece  ... text r0     cut where the voice changes, or before Whisper's 30 s window
              overflows, at Silero's deepest dip: read by Parakeet at once
 later    ... text r1     Whisper reads as many pieces as its window holds (it reads better
              with context), in the language that voice speaks; its timed sentences go back
              to their pieces; the reader chooses among both readings, the substitutions it
              learned, the corrected pieces it recalls, and "I did not understand"
 doubt    ... question    to a person, about the words it is unsure of: answer it once
```

The **reader** is a Bayesian conditional logit (evidence maximisation, no hand-set prior) over
the features of each reading: who wrote it, how much the other recogniser agrees, token
confidence (against what it usually is in this place), learned substitutions, recall of a
corrected piece, known words, and, for "I did not understand", low confidence, two recognisers
apart in one language, and a piece Silero found less like speech than the speech of this place.
Every sensor level is relative to the place: a television through a laptop is quiet to Silero
and to Parakeet, and that is not "nothing was said". **Language** is not a learned feature but a
probability added with Bayes' coefficient: `log P(the reading's language | its voice)`, learned
per voice from what Whisper detected it speaking (`language.py`), so a Parakeet translation into
English of a French voice loses, and an English guest in a French room is still written in
English. The base weights come from Piper voices in English and French in three regimes (clean,
in a room, played by a loudspeaker over indoor sounds), the ESC-50 sounds as the VAD lets them
through, and correction episodes (a correction trusted the first time).

## The port and the inbox: plug anything in

```json
{"port":3,"kind":"session","weights":"…","fast":"parakeet-v3","slow":"whisper-turbo",…}
{"port":3,"row":42,"t0":82.6,"t1":84.6,"kind":"known","speaker":"1abe…","condition":"9f0c…",
 "label":"KπX","source":"person","source_p":0.99,"p":1.0,"margin":187.4,"trust":0.998,"shadow":false,…}
{"port":3,"kind":"turns","run":"…:80.6","segments":[[80.6,81.6,"1abe…"],[81.6,84.6,"77d2…"]]}
{"port":3,"kind":"partial","piece":"…:80.6#2","run":"…:80.6","t0":84.1,"text":"send the notes to"}
{"port":3,"kind":"text","piece":"…:80.6#2","run":"…:80.6","speakers":{"1abe…":0.92,"77d2…":0.08},
 "revision":1,"text":"send the notes to KπX","doubt":"taught","p":0.97,"language":"en","told":"en",
 "speech":0.61,"words":[{"w":"send","trust":0.99,"agreed":true,"new":false},…],"readings":[…]}
{"port":3,"kind":"question","id":"…:80.6#2#text","about":"text","piece":"…:80.6#2",
 "choices":["… pi x","… pie ex"],"words":["pi","x"],"doubt":"unseen"}
{"port":3,"kind":"correction","what":"text","answers":"…:80.6#2#text","line":7,"refused":null}
```

Since port 4 every decision, turn and text also says its `channel` (`mic`, `machine`) and its
`scene` (`room`, `remote`, `media`, `self`); since port 5 a window can be said again
(`revision`), and the ear may ask whether two voices are one:

```json
{"port":5,"kind":"revision","revises":17,"run":"…:16.4","was":{"kind":"unknown","speaker":null},
 "now":{"kind":"known","speaker":"05d9…","label":null,"scene":"room"},"why":"learned"}
{"port":5,"kind":"turns","run":"…:16.4","segments":[…],"revision":1}
{"port":5,"kind":"question","id":"1abe…=77d2…#same#…","about":"same","speakers":["1abe…","77d2…"],
 "rows":[40,61],"choices":["same","different"],"p":0.31,"value":4.3}
```

The examples above the `port` 5 lines are of port 3: their fields have kept their meaning.

A consumer (the arbiter, the mouth) reads `port.read()` / `port.follow()` and gets typed events;
it never reads the internals. It speaks back through **`port.Inbox`**, the same three calls a
terminal uses:

```python
inbox = port.Inbox(memory)
# what: text, same, different, not_voice, name, person, device, room, remote, media, self
inbox.correct("text", rows, text="send the notes to KπX")
inbox.answer(question_id, text="Alice")  # a name, or what was said
inbox.answer(question_id, choice=1)  # or one of the question's choices
inbox.undo(line)  # forget one, exactly
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
make rec-out  # pane 4: what this machine plays, and which application plays it.
```

**The machine's own sound** (`PI_EAR_OUTPUT=on`, the default). What this machine plays is
recorded beside the microphone from its output's monitor (`PI_EAR_OUTPUT_SOURCE`, default
`@DEFAULT_MONITOR@`, through `pipewire-pulse`), read as a channel of its own (its voices are the
machine's, with the application that played them), and taken out of the microphone (`echo.py`),
so the room is heard over a video, a call or music. `PI_EAR_OUTPUT=off`: the microphone alone,
as before. Without `make rec-out`, the ear behaves as off and `make status` says so.

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

### On a real session (PSG - Barça, 2026-10-20, replayed: `make test-live`)

Word recall against two other systems' readings of the same audio (Whisper turbo told
French on 30 s blocks / Parakeet on 20 s blocks): a measurement, not a target, and
neither reference is the truth. v2.0 is what the ear wrote live.

| segment | what makes it hard | v2.0 live | v2.1 replayed |
|---|---|---|---|
| match, 9 min | TV across the room at -40 dBFS, crowd under it | .426 / .593 | .396 / **.598** |
| KπX in the room, 8.5 min | short phrases over the TV, Whisper detecting 7 languages | .235 / .436 | **.312 / .480** |
| studio, 12 min | talk that never pauses; Parakeet translating to English | .392 / .506 | **.775 / .673** |
| a player interviewed, 2 min | question and answer in one run | .013 / .016 | **.752** / .144 |
| 17 min without a cut | whistle, ads, studio, interview | | .784 / .613, slow pass 0.74 s per s, never behind |
| almost silence, 3 min | -50 dBFS | | 4 pieces, 80 % of the words written heard by another system |

Correction 2026-10-21 (KπX), append-only: "TV" in this table is a misreading. The match played
in the browser of this laptop, through its own speakers: the machine's own sound, not a television
across the room (`IDEAS.md`, end of section v2.1).

Who on the same record: every television voice `device`, KπX `person`; questions only about
people already known (KπX's name, KπX's doubtful words), 0 to 4 per segment; 0 scribe errors.
Limits, measured: KπX over the television stays near the recognisers' own oracle (.38: what
the better of the two readings would give, piece by piece); a voice heard alone at the very
start of a memory, misdetected the same way by both detectors, is told that language until
another voice or a longer detection says otherwise (the goal, 30 s cold: English, once). On
the match the two references disagree (each favours its own recogniser): the reader's
preference between recognisers is the base's until KπX corrects a piece.

## Tests

```bash
make test-fast   # no model, no download: law, maths, model, reader, disk, CLI
make test        # + the real sensors, recognisers and simulated rooms (needs make fetch)
make test-live   # + a real session replayed (needs its recording on this machine)
make check       # ruff
```

`make test-live` replays segments of a real session (`tests/live/<name>.json`, its
manifest: where each segment is and what makes it hard) through the ear, from the memory
it had then, with the real recognisers. The recording itself stays on the operator's
machine (`PI_EAR_LIVE`, default `~/.cache/π/data/live/`): broadcast audio and the voices of
a home are never committed. No reading is the truth: each segment is measured against two
other systems' readings and against what the ear wrote live.
