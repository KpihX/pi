# tts/: the mouth, ideas, hypotheses, what to test

> Numbers and sources: `RESEARCH.md` (blank: the mouth is the next research target).
> Pole level (the law, the ear/mouth split, the hypotheses H7..H10, the slices F and G, the
> four-verb contract): `../IDEAS.md`. The ear: `../ear/IDEAS.md`. Vision: `../../IDEAS.md` section 0.
> Status: **nothing measured yet inside π.** Everything below is a hypothesis, none of it is a decision.

---

## What this file owns

`../IDEAS.md` owns the pole and already carries the mouth's four-verb contract and its open questions
about voice identity. **This file owns the mouth's *state* and its *budget***: what the mouth must
remember, what it must write, what it must never decide. Nothing here repeats the pole file.

---

## The mouth under the law

π is 100% stream, 100% live. For the mouth that forbids one specific shape, and it is the shape every
TTS demo has:

> **generate the whole sentence, then play it.** That is a batch assumption with a speaker attached.

What the law requires instead:

| The law | What it means for the mouth |
|---|---|
| fast clock never waits for the slow one | the first phoneme is emitted before the sentence is finished generating |
| one act, no phases | there is no "prepare the reply" stage; the reply starts as it is decided |
| state is explicit and persistable | which voice, which speed, which language, and what was said, survive a restart |
| unknown is a legitimate value | a refusal to speak is an event, never a silence |
| a decision belongs to the arbiter | the mouth can only obey or refuse, never choose |

---

## The second law, in this pole: the mouth is learned and corrected (KπX, 2026-10-19)

**No constant, no threshold, no hand-set value in any decision path, in any pole of π. Ever.** For
the mouth that kills the shape every TTS pipeline ships: a fixed silence budget, a fixed chunk size,
a fixed speed, a fixed prosody, chosen by a developer. Those are decisions the model should take.

**The direction is not data science.** Classic practice cleans a dataset, trains a voice, deploys it,
and a correction costs a new training run. π does the opposite: **a model with good bases, alive and
human**, imperfect by construction, corrected **one case at a time**, in place, from use. A mouth is
exactly where this matters: a voice that reads a formula wrongly once must never read it wrongly
again.

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

**Open, and deliberately not decided: the form of a correction.** How KπX says "you read that badly",
or "you spoke when you should have stayed silent", and how that case is written down, is a question
to think through before anything is fixed. One thing only is graved for now: a mouth that spoke when
it should have stayed silent is a correction of the **arbiter** as much as of the mouth, which is why
the mouth writes its own record (`M3`).

---

## Hypotheses (each one is falsifiable)

| # | Hypothesis | How to falsify it |
|---|---|---|
| M1 | First audible sound under 300 ms is reachable by streaming the first clause while the rest is still being generated. | Measure from "decision to speak" to "first audible sound" on a real sentence. |
| M2 | Interruption must cut the audio device locally, in one frame, and must never wait for the generator to stop. | Speak a long sentence, cut in, and check whether the tail is still heard. |
| M3 | The mouth must write the same kind of event record as the ear (`t0`, `t1`, text, voice id, interrupted or not), so one log holds the whole conversation. | Try to reconstruct a dialogue from the ear's log alone. If you cannot, M3 was needed. |
| M4 | Voice identity is state, not configuration: which voice, which speed, which language must survive a restart as fields of the persistent state. | Restart the process and check that it speaks again with the same voice, with no code touched. |
| M5 | A refusal is an event: when the arbiter asks for speech and the mouth cannot, the refusal is recorded rather than silently dropped. | Ask for speech while the engine is unavailable; look for the record. |
| M6 | The synthesis engine is not the open problem on this machine; scheduling is. | Measure the raw synthesis time of one sentence and compare it to the first-audible latency observed. |
| M7 | The mouth's latency and the ear's latency share one budget, so they must be measured together, not separately. | Measure "he stopped speaking -> first audible sound" end to end, once. |
| M8 | The mouth must be interruptible even mid-phoneme, at the device level, not at the sentence level. | Interrupt 100 ms after the onset, ten times in a row. |

---

## The record the mouth writes

Same discipline as the ear, because the agentic pole must be able to read one conversation from one
log:

```json
{"t0":15.982,"t1":18.204,"dir":"out","text":"...","voice":"cfg:default","lang":"fr",
 "first_audible_ms":214,"interrupted":true,"interrupted_at":17.610}
```

`dir` distinguishes an inbound excerpt (the ear) from an outbound one (the mouth) in the same file.
`first_audible_ms` is the mouth's own E5: the number that decides whether the architecture is honest.

---

## Anti-goals for the mouth

- A voice that cannot be interrupted.
- A cloned voice chosen before the identity question is settled (`../IDEAS.md`, open questions).
- Beautifying the voice before the first-audible latency has been measured.
- A second synthesis engine before the first one's numbers exist.
- Waiting for a full sentence before emitting anything.
- Speaking at a volume that wakes the room.

---

## The third law, in this pole: a model that knows when it does not know (KπX, 2026-10-20)

The guiding line of every model brick of π (`../../IDEAS.md`, section 0): not a model that always
runs toward a minimum, but one that says, with each output, whether it knows, hesitates, or does
not know, and which kind of not-knowing it is in.

For the mouth: a voice that can hesitate honestly. When the content it carries is unsure (a
misheard question, a low-confidence transcript), the mouth must be able to say so in the voice
itself (a short "if I understood well"), and a refusal to speak because it is unsure is an event
(`M5`), never a silence.

### What the ear offers the mouth (2026-10-20, ear v2; a fact about the ear, nothing decided here)

The ear now publishes `question` lines on its port (`voice/ear/src/π_ear/port.py`, PORT 2): a
name it does not know, or a reading it could not choose, with the choices. And it takes answers
through one inbox (`port.Inbox(memory).answer(question_id, text=... | choice=...)`), the same
line whether typed or spoken. A mouth that reads a question aloud, and an arbiter that writes the
answer it hears, are enough for a correction by voice; the ear needs nothing more.
