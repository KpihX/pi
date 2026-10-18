# voice/: ideas, hypotheses, experiments

> Research and numbers: `stt/RESEARCH.md` (filled and sourced: Part 0 is the current spine, Part 1
> the frozen baseline) and `tts/RESEARCH.md` (blank, to be filled). Vision: `../IDEAS.md`.
> Each half now owns its own ideas file: **`stt/IDEAS.md`** (the ear) and **`tts/IDEAS.md`** (the
> mouth). This file stays the pole: the law, the shared hypotheses H1..H10, the slices A..G, the
> mouth's four-verb contract. Live ear test and acquisition list: **`stt/TODO.md`**.
> Status: **nothing measured yet inside π.** Everything below is a hypothesis to test.

---

## The goal, both directions

Build an **ear** that knows *who* is speaking and reacts only to the owner, at whisper
volume, on any device, with the heavy work only paid when it matters. And build the
**mouth** that answers: believable, in a voice that is deliberately chosen, interruptible
mid-phrase, and only open when the arbiter decides to speak.

The two halves share one thing and must never duplicate it: the **decision to speak**
belongs to the arbiter (`../agentic/`). The ear asks *may I listen now*, the mouth asks
*may I speak now*, and both take `no` for an answer.

---

## The law, applied to the vocal pole (KπX, 2026-10-18, `../IDEAS.md` section 0)

π is 100% stream, 100% live. So the gate is **never "enrolled, then used"**: enrollment is
not a phase, it is an event in the stream.

The clause of the law this pole cannot dodge:

> **The gate must be useful at t=0, with an empty state, and there is no calibration step.**

Consequences. Two of them **re-read** lines above instead of replacing them:

| Consequence | What it changes here |
|---|---|
| **Enrollment is a stream event** | H2 ("3 to 10 s of enrollment") survives as a *measurement* of how fast the state becomes usable, but those seconds can no longer be a setup ritual: the first observed speech **is** the initial state, and the user labels it if and when he wants to |
| **The boundary comes from the stream** | Slice B ("enroll once, then print a score every 100 ms") is the right probe but the wrong shape: the score must be produced with **nothing enrolled yet**, and the boundary derived from observed margin and cohort statistics rather than a constant chosen in advance |
| **Unknown is a first-class answer** | The state must hold *unlabelled recurring clusters*, not only the owner. A new voice is "new and recurring" long before it has a name, and naming happens *when it matters*, by asking, not up front |
| **Silence is already an answer** | The law adds one thing: the ear must also listen without requiring the owner to have spoken first |

**Cold start, precisely.** At t=0 there is no target embedding, so no target-conditioned
model can run yet. The open-set half still runs: segment, embed, cluster, count. The moment
a cluster gets a name, every target-conditioned tool downstream becomes available for that
entity. Semantics **emerge** from the stream instead of preceding it, which is also what
makes the wake phrase optional (see "Do we even need a wake phrase" below).

---

## The second law, in this pole: nothing is tuned, everything is learned (KπX, 2026-10-19)

**No constant, no threshold, no hand-set value in any decision path, in any pole of π. Ever.** A
number written by hand is a decision the model should have taken. Every threshold of the first ear
POC (the identity descriptor, the pitch estimator, the voicing gate, the identity boundary) is
therefore an architectural defect, not a tuning detail, and the list lives in `stt/TODO.md` with
the line counts.

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

**What this pole loses, named:** the boundary inside `stt/IDEAS.md` is still **my** formula
(`z_margin * sigma`), the voicing gate is two constants I chose between two measured bands, the pitch
estimator and the identity descriptor are hand-written physics, and H2's "3 to 10 s of enrollment" is
a duration guessed in advance. All four become learned judgements: when a sound counts as speech,
whether two voices are the same, and how long the state really needs to become useful.

**Open, and deliberately not decided: the form of a correction.** How KπX says "from this second to
that second there were two voices", and how that case is written down, is a question to think through
before anything is fixed. This file carries no format until it is.

---

## Hypotheses (each one is falsifiable)

| # | Hypothesis | How to falsify it |
|---|---|---|
| H1 | A speaker-embedding gate in front of the ASR removes the "answers someone else's turn" class of failure **upstream**, better than any prompt or router tweak. | Take atlas's three real fixtures (`Hey Pavel...`, `Avery, do you see...`, `No, Pavel, I'm talking to you`) and run them through the gate. If the gate says "me" for them, H1 is dead. |
| H2 | 3 to 10 s of enrollment is enough for a usable personal gate on *this* voice, in *this* room, on *this* mic. | Enroll once, then measure the false-accept and false-reject rates on a day of real audio. |
| H3 | Whisper needs an adaptation path for whispered speech; the raw model is not enough. | Record the same ~30 phrases at normal / soft / whisper volume. Measure CER. Compare against the published 18.93% (Whisper-v3, whisper speech). |
| H4 | A two-stage always-on cascade (cheap gate, then identity, then ASR) holds a laptop battery budget. | Measure CPU/energy of stage 1 and 2 alone over a full working day. |
| H5 | The mic already reaches "very soft voice" without new hardware; the gap to true silence is a *sensor* problem, not a model problem. | Compare RMS and CER between soft-voice and whisper on the same phrases. |
| H6 | Enrollment can improve itself from daily use (self-augmenting embedding), so the gate gets better with time instead of drifting. | Reproduce the ICASSP 2026 short-enrollment self-augmentation idea; measure after 5 updates. |
| H7 | The mouth already exists on this machine (`piper` is installed), so the open problem is **scheduling**, not synthesis. | Measure first-audible latency. If synthesis itself eats the budget, H7 is dead. |
| H8 | First audible sound under 300 ms is reachable by streaming the first clause while the rest is still being generated. | Measure from "decision to speak" to "first audible sound" on a real sentence. |
| H9 | Interruption must cut the audio device **locally, in one frame**, and must never wait for the generator to stop. | Speak a long sentence, cut in, check whether the tail is still heard. |
| H10 | π needs an explicit **voice identity** (which voice, which language, which speed) held as configuration, never as a constant. | Try to change pi's voice without touching any code. |

---

## Experiments, in order of return on effort

| Slice | What to actually do | Why first |
|---|---|---|
| **A** | Turn on the neural VAD that is already installed: `ggml-silero-v5.1.2.bin` in `~/.local/share/whisper-cpp/`. One config key in live-listen, zero code. | Kills the hallucination class (the `Radio-Canada x7` bug) upstream, in minutes. |
| **B** | Enroll the voiceprint once, then a small script that prints a `me / not-me` score every 100 ms. | This is the real heart. Everything else is downstream of it. |
| **C** | Build the tiny labelled corpus (normal / soft / whisper) and measure CER with the current local model. | Decides between "fine-tune" and "whisper-to-normal conversion" on numbers, not taste. |
| **D** | Same chain on the phone. | Validates the "everywhere" promise with the same models. |
| **E** | Wire slice B into the arbiter: the gate becomes a hard, non-semantic prior for "is this addressed to me". | Closes `PBS-001` for real. |
| **F** | **Mouth end-to-end**: speak a real sentence with the local TTS, measure first-audible latency and cut latency. | It is the mouth's slice A: cheap, immediate, and it produces the two numbers that constrain everything else. |
| **G** | **Voice identity study**: the local voices, then the cloning options (ZipVoice and Pocket TTS both ship inside sherpa-onnx). | Decides *what π sounds like* before any pipeline is built around it. |

---

## The mouth's own contract

The mouth is a port with four verbs, and the verbs are the specification:

| Verb | Meaning |
|---|---|
| `say(text)` | stream audio, never wait for the full generation |
| `stop()` | cut in one frame, locally, without a round trip |
| `vary(...)` | voice, language, speed, emphasis: configuration, not code |
| `state()` | is it speaking right now, and since when |

The decision (`speak` or `not`) is deliberately **not** one of these verbs. That belongs
to the arbiter (`../agentic/`), so that the mouth can only ever obey or refuse, never
choose.

---

## Open questions

- **Where does the enrollment live, and how is it protected?** It is a biometric. A
  recording of the owner's voice must not become a key (see `stt/RESEARCH.md` section 10).
- **One voiceprint or several?** Room, headset, phone, whisper, and loud speech are
  physically different signals. One vector, or a small set with a selection rule?
- **Does the gate ever open for a second person on purpose?** In a meeting, the system
  must know *who* spoke, not just *not the owner*. That is diarization, a different port.
- **Do we even need a wake phrase?** If the voiceprint is the trigger, the phrase becomes
  optional. That is the conceptual prize (see `stt/RESEARCH.md` section 9).
- **Latency floor.** What is the smallest honest end-to-end latency from "he stopped
  speaking" to "first audible sound"? Every stage we add must be charged against it.
- **Whose voice should π have?** A neutral synthetic voice, a deliberately chosen one, or
  the owner's own cloned voice? The last is the most personal and the most dangerous: it
  turns the assistant into a forgery of the person it serves (see `stt/RESEARCH.md`
  section 10).
- **Does the mouth have a veto on readiness?** If the arbiter says speak while the
  pipeline is not ready, does the mouth stay silent, delay, or announce?
- **One mouth or several?** A short reflex voice ("understood, looking") and a full reading
  voice do not share the same latency budget, and may not deserve the same engine.

---

## Anti-goals

- A wake word shouted loud enough to wake the room.
- Sending raw audio anywhere. Vectors and gated segments only.
- Another VAD. The one we have is neural and fine.
- Choosing a model before measuring it.
- A voice that cannot be interrupted.
- A cloned voice chosen before the identity question is settled.
