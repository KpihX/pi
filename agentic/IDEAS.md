# agentic/: ideas

> The pole that answers: *should I act now, how, and through which agent.*
> Status: brainstorm only.

---

## The core object: the arbiter

Every project in the constellation built this piece, and they disagree about who owns it
(full comparison in `../IDEAS.md` section 5). The measured conclusion:

- atlas routed `Hey Pavel...` to *respond* with address probability **0.24**, then spoke.
- `Stop stop assistant stop talking` **still produced a sentence**.
- Fix: the Speaker holds an **independent veto**, and `speak: false` is a legal answer
  from a component that is not the router.

**Rule: the arbiter declares silence by default, and any veto is independent of the
router that proposed the action.**

---

## The law, applied to the arbiter (KπX, 2026-10-18, `../IDEAS.md` section 0)

π is 100% stream, 100% live. For this pole the law has one sharp edge:

> **The arbiter runs continuously. It is never "decide, then act", it is
> "decide, act, observe, decide".**

| Consequence | Detail |
|---|---|
| **The arbiter is stateful and always on** | it holds a running judgement, not a per-turn verdict. Silence is its resting state, and it re-evaluates at the clock rate of the input, not at the rate of turns |
| **The veto must be cheap** | the independent veto (`PBS-001`, `PBS-002`) is only usable if it can be produced **inside the reaction window**: on the fast clock, with no round trip and no LLM call |
| **No batch preparation** | a decision may not require a completed prior phase (no "once training finishes", no "after the index is built"). If the arbiter needs it, the arbiter degrades **visibly** instead (`PBS-008`) |
| **Actions are stream-shaped** | a launched pipeline is a live object emitting progress events, never a blocking call. Announce, then deliver (`PBS-004`) |
| **The two clocks** | the *decision* is fast. A model used as judge (Jev, or the runtime LLM) may be slow, and must therefore never hold the fast path. The slow judge **advises**; the fast rule **protects** |

**The boundary to write down, explicitly:** which half of the arbiter is allowed to be slow.
Everything that must land inside the reaction window is code on the fast clock. Everything
else may be a model on the slow clock, and its output arrives as an **advisory patch** to
the running state, never as a gate. Mixing the two halves is exactly how atlas's Speaker
ended up with an independent veto bolted on after two real failures.

---

## The invariant that unifies the corpus

Three projects, three phrasings of one rule:

| Project | Phrasing |
|---|---|
| opencode-web-stream | *"Mechanism in the code, judgement in the model."* Banned in code: tool lists, stop-word lists, confidence thresholds, intent patterns |
| kairos | *"Whether to speak is decided by code from these signals, never by a model."* |
| atlas | timing and floor decided by the engine; meaning decided by Jev; the Speaker vetoes |

**Synthesis to adopt: code owns signal, timing, ordering and the hard reflex; the model
owns meaning. The boundary between the two is written down, tested, and never blurred.**
The single admitted exception, in every project: **barge-in must be local**, because
waiting for a round trip makes interruption feel broken.

---

## The second law, in this pole: the arbiter is learned and corrected (KπX, 2026-10-19)

**No constant, no threshold, no hand-set value in any decision path, in any pole of π. Ever.** This
pole already bans thresholds in code (the invariant above is literally a list of banned thresholds:
tool lists, stop words, confidence thresholds, intent patterns). The second law goes one step
further and removes the last hiding place: there is no **arbitration threshold** either. "Should I
speak" is not a score compared to a number, it is a judgement with parameters, and those parameters
are learned.

**The direction is not data science.** Classic practice cleans a dataset, trains, deploys, then
finds out in the field whether it works, and a correction costs a new dataset and a new training run.
π does the opposite: **a model with good bases, alive and human**, imperfect by construction,
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

**Open, and deliberately not decided: the form of a correction.** How KπX says "you should have
stayed silent there", "you should have spoken", "that was the wrong agent", and how that case is
written down, is a question to think through before anything is fixed. What the law does fix is the
outcome: those two atlas failures become cases the arbiter learns from, while the independent veto
stays in code, because *"stop"* must never wait for a model.

---

## Ideas to test

1. **The arbiter as a port.** One contract, several possible implementations
   (rule-only, model-only, model + veto, biometric prior + model). Compare them on
   recorded sessions, not on opinion.
2. **Biometrics as a non-semantic prior.** The vocal gate (see `../voice/`) gives a
   hard, physical input to the arbiter that no prompt can fake. This is the cheapest
   real fix for `PBS-001`.
3. **Mission identity.** One stable ID from request to acknowledgement to execution to
   integration to spoken delivery, so equivalent work is reused instead of relaunched
   (`PBS-004`). Idempotency as a first-class property.
4. **Announce, then deliver.** Acknowledge immediately, report progress meaningfully,
   present the final result once (`PBS-004`). Deep work never blocks the conversation.
5. **A strict output grammar.** opencode-web-stream's streamer answers in tagged blocks
   (`<quiet/>`, `<say>`, `<reply>`, `<clarify>`, `<inject>`, `<control>`, `<learn>`) and
   **anything unparseable becomes silence**. A malformed decision must never be spoken.
6. **Ask when two readings lead to two different actions.** Clarify, do not guess. Both
   web-stream and meeting-colab encode this as an explicit outcome, not a fallback.
7. **No keyword routing.** `colab` already states it. A wake phrase is a workaround for
   missing identity, not a design.
8. **The full-duplex runtime as the actuator.** `colab` (Voice x ACP x tmux,
   *"Sovereign, local-first, no keyword routing"*) and opencode-web-stream both already
   do this: the voice layer drives agent sessions. `QwenAudio/qwen-audio-agent` is the
   same shape built on ACP and is worth reading line by line.

---

## The transparent-pipeline requirement

From the stated vision: *"it launches agentic pipelines behind the scenes, transparently,
so that I can come back later and resume the thread in whatever session it opened, or not."*

That decomposes into hard requirements:

| Requirement | Consequence |
|---|---|
| launch without a human in the loop | the arbiter must be able to act, not just propose |
| transparency | every launch, tool call and result is persisted and inspectable |
| resumability | the session identifier is durable, and the state is canonical server-side, never in the UI |
| non-blocking | the conversation continues while the pipeline runs |
| non-dependency | the pipeline must not be tied to one runtime; ACP is the neutral bus |

**Resumability is the test.** If the session cannot be reattached and inspected
afterwards, the transparency requirement is not met.

---

## Anti-goals

- An LLM deciding the floor *and* nothing able to contradict it.
- A pipeline nobody can inspect afterwards.
- State that lives in a UI.
- Silent failure. A visible degraded state beats a hidden one (`PBS-008`: TTS was down
  while speeches were recorded as played).
