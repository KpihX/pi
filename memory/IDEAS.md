# memory/: ideas

> The pole that answers: *what is true, what changed, what do I know about him.*
> Status: brainstorm only.

---

## The law, already proven three times in our own code

**A transcript is not a memory.** Three independent projects hit the same wall and found
the same exit: keep a typed canonical state, and treat the human-readable document as a
**projection** you can rebuild.

| Project | The failure | The exit |
|---|---|---|
| atlas | `PBS-010`: 28 turns produced 19 note rewrites, one fact repeated 6 times under synthesis, tracked points, questions, actions and work state | typed facts stored **separately** from the rendered notes; repeated evidence strengthens an item instead of adding a bullet |
| live-listen | a journal of summaries inflates without bound | **one progressive summary, overwritten**; previous updates are deliberately discarded |
| meeting-colab | projections could drift from the event history | append-only ledger is authoritative, **projections are always replayable** |

**Rule: memory = maintained state + disposable projection. Never an append of turns.**

---

## 100% stream, 100% live: what it means for memory (KπX, 2026-10-18, `../IDEAS.md` section 0)

π is 100% stream, 100% live, and memory obeys it at the level that matters most: **there is
no "build the memory" step.**

The corpus above already proves the hard half. `meeting-colab`'s append-only ledger with
replayable projections is a *stream* design. `live-listen`'s single progressive summary is
a *stream* design. What the law adds:

| Consequence | Detail |
|---|---|
| **The write path is the stream** | every event is absorbed as it arrives: no ingestion pass, no reindex window, no "the memory is being rebuilt" state. A state that is periodically rebuilt is a batch system with extra steps |
| **The read path composes at the instant of the decision** | "what should I recall right now" must be answered *inside* the reaction window. Recall that arrives late is narration, not memory |
| **The projection is always current** | a projection rebuilt from the ledger is never stale, because the ledger is the truth and the projection is disposable. Zero-deletion applies to the ledger, never to attention |
| **Belief updates are online** | a repeated fact strengthens an item (`PBS-010`) as it happens, not in a nightly pass |
| **No batch import, ever** | importing a corpus into memory is legal only as a *stream replay* that goes through the same write path with the same provenance rule. Otherwise the imported facts have a different nature from the lived ones, and the ledger loses its meaning |

**The consequence for the two-clock split (section 0 of the root):** memory sits on the
**fast** clock for reads and on both for writes. Writing an observation is immediate and
cheap; consolidating it (a stronger belief, a re-estimated centroid, a folded scenario) may
be slow and is allowed to lag. But the ledger entry is never deferred until the slow pass.

---

## The second law, in this pole: memory is learned and corrected (KπX, 2026-10-19)

**No constant, no threshold, no hand-set value in any decision path, in any pole of π. Ever.** For
memory this is the sharpest of the four poles, because a memory system is usually a pile of numbers
nobody admits are decisions: a recency half-life, a decay rate, a salience score, a similarity
cut-off, a confidence floor, a TTL chosen in advance. Each one is a judgement about what matters,
made by a developer instead of by the system and by him.

**The direction is not data science.** Classic practice builds a corpus, fits a model, deploys, then
hopes. π does the opposite: **a model with good bases, alive and human**, imperfect by construction,
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

**Open, and deliberately not decided: the form of a correction.** How KπX says "this is wrong",
"remember this", "forget that", and how that case is written down, is a question to think through
before anything is fixed. What is already graved, in his words: a correction is said once, and after
it the system does not repeat the error.

---

## The three questions the pole must answer

| Question | Nature | Storage idea |
|---|---|---|
| **What is true now?** | facts about the world and about him | typed facts with provenance and a last-seen timestamp |
| **What changed?** | deltas, decisions, commitments | append-only event log, replayable |
| **What should I recall right now?** | retrieval, relevance, budget | bounded render from the state at the moment of a decision |

The third one is the one everyone forgets, and it is the one that decides whether the
system feels intelligent or merely well-informed.

---

## Ideas to test

1. **Facts vs prose.** Store `(subject, predicate, object, confidence, source, first_seen,
   last_seen)`, then render prose on demand. Reconstruction from evidence beats
   summarisation of summaries.
2. **Self-augmenting profile, borrowed from the voice pole.** The enrollment embedding
   improves from daily use (ICASSP 2026 self-augmentation). The same shape applies to a
   memory of a person: a stable vector of what is known, refined by use, with an explicit
   update rule instead of a rewrite.
3. **A budget per render.** Every projection gets a size ceiling, because an unbounded
   memory is the same as no memory (`PBS-021`: public state exceeded one megabyte per
   snapshot).
4. **Provenance and TTL on every claim.** meeting-colab already does this for provider
   output: advisory evidence with provenance, confidence and expiry. Apply it to memory
   items too, so unverified recall can be marked or expire.
5. **Forgetting as a first-class operation.** Zero-deletion applies to logs, not to
   attention. Something must decide what leaves the working set.
6. **Local only.** A memory of a person is the most sensitive artefact in the system.
   It never leaves the machine, and it is never summarised into a prompt sent to a
   third party (see atlas `PBS-007`: no internal IDs or machine structure in spoken
   output).

---

## The scenarios that will define the pole

These are the situations the design must survive, taken from the stated vision:

- "I tell it: go on my phone, record this voice note, send it to yourself over Telegram,
  download it, transcribe it, classify it, tell me what it is about, then we discuss it."
  A pipeline of five tools and one conversation, launched transparently, resumable later.
- "I am lying down. I ask it to find the PDF of a course and we talk about it."
  Retrieval plus a spoken summary, in a natural conversation.
- "It should know when I need it, and stay silent the rest of the time."
  That is the arbiter, but it is fed by memory: what he was doing, what he asked for
  yesterday, what is still pending.
- "It learns to know me and understand me on its own, always on."
  Long-horizon, self-updating, private.

---

## Relationship to LiveMem

`LiveMem` is registered in the kernel as a brainstorm (brain-inspired living field:
Hebbian learning, multi-modal cortices, continuous flux, an internal thought loop) and its
recorded path `$HOME/Work/AI/livemem/` **no longer exists on this machine**. Before
building the memory pole on top of it, decide whether it is the substrate or a discarded
sketch. If it is the substrate, this directory absorbs it; if not, this directory starts
from the three proven patterns above.

---

## The third law, in this pole: a model that knows when it does not know (KπX, 2026-10-20)

The guiding line of every model brick of π (`../IDEAS.md`, section 0): not a model that always
runs toward a minimum, but one that says, with each output, whether it knows, hesitates, or does
not know, and which kind of not-knowing it is in.

For memory: every belief carries its own confidence and its provenance, and the memory knows the
difference between "I never heard about this" and "I heard two contradicting things". Recall that
is unsure must say so ("I think you said..."), and the slow clock spends its time on the
beliefs it is least sure of.
