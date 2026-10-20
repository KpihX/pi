# vision/: ideas

> The pole that answers: *how to show it live, in the same clock as the voice.*
> Status: brainstorm only. **The implementation already exists elsewhere; this directory
> points at it and defines how it plugs into π.**

---

## Do not duplicate: the visual pole already has a home

`../../../KpihX-Labs/𝛱/live/` is an existing, running project: *"Immersive voice-first platform"*
(`live.kpihx-labs.com`, git `kpihx-labs/live`). Its design already answers the hard part
of this pole. Note the deliberate **cross-container reference**: π now lives in
`Explore/` (the unfinished, the experimental) while `𝛱/live/` stays a product in
`KpihX-Labs/` (what ships). This is the one place where the two poles of the workspace
genuinely touch, and the direction is one-way: π points at the product, the product
does not depend on π.

| Layer | Role | Who writes it |
|---|---|---|
| **Scene IR** | the machine-readable state of the lesson: what exists (axes, curves, regions), what is in focus, what must stay true, and when the voice should stress each part | LLM / pedagogy, as patches |
| **Discourse bindings** | "this word means highlight that" | the Conductor |
| **Live Motor** | always-on motion between ideas | the engine only |

```text
  slow  (semantic, ~1-5 Hz)   Scene IR + patches on interrupt
                                       |
                                       v
  fast  (motion, ~60 Hz)      interpolation, pointer, camera, idle motion
                                       ^
  clock (the Conductor)       ONE timeline: phonemes <-> beats <-> subtitles
```

And its stated architecture bet, verbatim: **"slow semantics + fast motor + one
Conductor clock"**, with the explicit refusal of closing the loop through a pixel-video
model. Manim is export polish (YouTube), deliberately *not* in the live dialogue loop.

**This is the same two-clock split that the whole of π needs**, and it is already
designed. `vision/` must not become a second implementation.

---

## The law, applied to the visual pole (KπX, 2026-10-18, `../IDEAS.md` section 0)

π is 100% stream, 100% live. For this pole that means four things:

| Consequence | Detail |
|---|---|
| **Motion is the fast clock** | something moves at all times, on its own clock, never awaiting semantics |
| **Semantics is the slow clock** | a patch may arrive late; motion never waits for it |
| **The renderer never decides** | it accepts or refuses a patch, cheaply. Whether to show is the arbiter's call |
| **One clock binds the modalities** | voice, beats and subtitles share one timeline, so a visual and a sentence stay one event |

**What this directory owns is the clock contract**: a data contract rather than a renderer, so a
live visual can attach to a live voice with no batch step in between.

---

## The second law, in this pole: the visual decision is learned and corrected (KπX, 2026-10-19)

**No constant, no threshold, no hand-set value in any decision path, in any pole of π. Ever.** The
visual pole is exposed here more than the others, because motion tempts you into a drawer of numbers:
a fade duration, a hold time, a minimum on-screen time, a "relevant enough to show" score, a rate of
patches per second. Each one is a decision about his attention taken by a developer.

**The direction is not data science.** Classic practice builds an animation library, tunes it, ships
it. π does the opposite: **a model with good bases, alive and human**, imperfect by construction,
corrected **one case at a time**, in place, from use. A visual that came too late, or that no one
needed, is a labelled case, not a styling opinion.

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

**Open, and deliberately not decided: the form of a correction.** How KπX says "that came too late",
"nothing was needed here", "wrong figure for this sentence", and how that case is written down, is a
question to think through before anything is fixed. The clock contract is untouched by it: the
renderer still accepts or refuses a patch cheaply, the Conductor still owns one timeline, and the
learned half owns "show it now" and "do not show it".

---

## So what belongs here

The **interface between the visual pole and π**, nothing else.

1. **The clock contract.** The Conductor holds one timeline where phonemes, beats and
   subtitles share a clock. π must feed it, not compete with it: the same utterance that
   the vocal pole transcribes is the one that drives the timeline.
2. **A typed port, like every other pole.** `VisualPort`: given a semantic intent and the
   current state, produce Scene IR patches. Providers (Live Motor, Manim export, a static
   figure, Napkin) behind a registry, selected by configuration, never hardcoded.
3. **The latency rule applies here too.** Motion is local and fast; semantics may be slow
   and announced. A visual that arrives after the sentence it illustrates is noise, the
   same way a stale narration is noise in opencode-web-stream.
4. **"Show it when it helps."** The vision states it explicitly: it speaks by generating
   real-time animations *in the sentence*, which it may launch **if asked, or if it judges
   it relevant**. That judgement is the arbiter (`../agentic/`), not the renderer. The
   renderer must be able to accept or refuse a patch, cheaply.
5. **Interruption applies to the visual channel too.** If he cuts in, both the voice and
   the motion must yield.

---

## Open questions

- **Who owns the clock?** If the Conductor is authoritative for the voice as well as the
  motion, then π's TTS scheduling and the visual timeline are the same problem, and that
  is a real architectural decision, not a detail.
- **Is a headless visual pole acceptable?** For a backend-first project the answer is
  yes: Scene IR is a data structure. It can be produced, stored and replayed with no
  renderer attached at all, which is exactly the "integrable, not dependent" test.
- **What is the minimal proof?** One spoken sentence and one Scene IR patch that moves on
  the same clock. No styling, no polish.

---

## The third law, in this pole: a model that knows when it does not know (KπX, 2026-10-20)

The guiding line of every model brick of π (`../IDEAS.md`, section 0): not a model that always
runs toward a minimum, but one that says, with each output, whether it knows, hesitates, or does
not know, and which kind of not-knowing it is in.

For the visual pole: a figure shown on an uncertain reading must look uncertain (or not be shown),
and the renderer must be able to say "I have no figure for this" instead of producing a
plausible but wrong one.
