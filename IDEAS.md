# π, the ambient agent

> The agent is called **π**. The project is `π/`.
> Backend first. No GUI.
> **Correction 2026-10-18 (KπX): the rest of this line was my misreading and is superseded.**
> The original said "no code structure, no codebase to set up, no framework chosen". What was
> actually said: *do not rush into a frozen architecture, brainstorm first, build POCs, mini
> projects*. A POC is allowed to be a package, and the ear's POC now is one (`voice/ear/`, uv
> package `π-ear`). What survives from the original line: no GUI, and no framework decision
> before measurement.
> English only. Related umbrella: `../../KpihX-Labs/𝛱/` (strategy, capital, business vectors).

---

## 0. The law: 100% stream, 100% live (KπX, 2026-10-18)

**π is never in batch. Not one phase, not one component.** The whole project runs as a
permanently live process on the same clock as the human, and every pole obeys this law
without exception.

**The trap this law kills, named explicitly:** *train first, then calibrate, then infer*.
Three sequential phases is a chatbox assumption wearing research clothes. It produces
systems that are useless until a setup completes, that freeze the moment the world moves,
and that must be re-prepared whenever anything changes. π refuses the sequence.

```text
   REJECTED (sequential, batch)        THE LAW (superposed, stream)
   -----------------------------       ---------------------------
   collect  -> train                   the stream never stops
   enroll   -> calibrate                    |
   deploy   -> infer                   every observation is at once
                                           - inference   (a decision now)
                                           - exploration (is this new?)
                                           - learning    (update the state)
                                           - exploitation (use what is known)

   "useful after setup"                useful from second zero
```

**Four consequences, all testable:**

1. **No phase gate.** There is no `setup`, no `enrollment step`, no `calibration run`.
   An action that requires a previous phase to have completed is an architectural defect.
2. **Useful from second zero, with an empty state.** At t=0 π knows nothing. It must still
   be correct, honest and useful: it says "unknown", it does not invent, and it improves
   monotonically as the stream feeds it.
3. **Exploration and exploitation are the same act.** Every inference is also a
   measurement of the world; every measurement updates the state; the updated state
   serves the next inference. There is no "research phase" followed by a "product phase",
   and no "learning phase" followed by a "use phase".
4. **Learning and inference are superposed, but they do not share one clock.** This is
   the precision the law needs, and getting it wrong is how "online learning" projects
   die:

```text
   CLOCK     HORIZON            WHAT IS LEARNED                      BUDGET
   ---------------------------------------------------------------------------------
   fast      ms to seconds      the STATE: which cluster, which       must fit inside the
                                name, which sound, and the running    reaction window.
                                statistics that set the boundary      No round trip. No GPU.

   slow      hours to days      the MODELS: an embedding refined,     may be expensive,
                                a classifier adapted, a threshold     may lag, never blocks
                                re-estimated                           the fast clock.
   ---------------------------------------------------------------------------------
   RULE: the fast loop never waits for the slow loop.
         The slow loop consumes what the fast loop recorded.
         A system where inference stalls to retrain has gone back to batch.
```

**Calibration is not eliminated, it is dissolved.** A decision boundary must exist; what
disappears is the *ceremony* around it. The boundary comes from the stream itself: relative
margins between candidates, per-cluster observed statistics, and a growing cohort used for
score normalisation. The user never performs a setup, and the system still has a threshold.
"No calibration" means *no calibration phase*, never *no criterion*.

**What the law forbids**

- A capability that exists only in batch form. Every capability needs a streaming form.
  If it has none, it is not a π capability yet.
- A user-facing setup, enrollment or calibration ritual.
- A "prepare the system" step between observation and use.
- Blocking the live path on a long computation, a model load, a network round trip, or a
  retraining pass.
- Any design whose correctness depends on having seen the whole file first.

**What the law does NOT forbid**

- **Recording the stream.** A recorded artefact is not a phase, it is evidence.
- **Benchmarks and offline measurement.** A benchmark is a *probe* used to compare
  options, not a stage the system passes through. It may be slow, manual and batch; the
  system may not.
- **Expensive work**, as long as it sits on the slow clock and never blocks the fast one.

### The second law: nothing is tuned, everything is learned (KπX, 2026-10-19)

**No constant, no threshold, no hand-set value in any decision path, in any pole of π. Ever.** A
number written by hand is a decision the model should have taken. Every threshold of the first ear
POC (the identity descriptor, the pitch estimator, the voicing gate, the identity boundary) is
therefore an architectural defect, not a tuning detail, and the list lives in
`voice/ear/TODO.md` with the line counts.

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

Three structural consequences:

1. **A correction is a first-class input**: an exact case, appended, never rewritten. The state obeys
   it at once; the weights absorb it later.
2. **The parameters live in the weights, on disk.** A pole's state *is* its checkpoint, reloaded at
   launch. No configuration file holds a decision, and no flag changes a judgement.
3. **It is enforceable, not promised.** The check must fail when a numeric constant reappears in a
   decision path. A rule that is not checked is a rule that will be broken.

**Open, and deliberately not decided: the form of a correction.** How a pole lets KπX say "you got
this one wrong", and how that case is written down, is a question to think through before anything is
fixed. No format is written into this project until it is.

**Why this is not perfectionism.** A hand-set constant is a place the system cannot learn, and
therefore a place where the same error will happen again forever. A model corrected once does not
repeat that error, which is the only real definition of a system that improves with use.

### The third law: every model knows when it does not know (KπX, 2026-10-20)

**The guiding line of every model brick of π, in every pole: a self-aware model.** Not a model that
always runs toward a minimum and answers with the same assurance whether it is right or lost, but
one that carries, with each output, an honest account of its own state: *I know*, *I am unsure*,
*I do not know*, *I did not understand*. Being able to say "I don't know" is not a failure mode to
be trained away; it is the first output of a model one can trust.

In engineering terms, a brick is lawful only if:

1. **Its doubt is an output, not a log.** Every decision travels with a measure of its own
   confidence that a consumer can read (in the ear: `kind = unknown`, `margin` in nats, `p`), and
   that measure is honest: high where the model is often right, low where it is often wrong.
2. **"I don't know" is a first-class answer**, never a fallback or an exception: `unknown`, a
   candidate, a clarification asked, a silence. A model forced to always pick its best guess hides
   its ignorance inside its answer.
3. **It knows what kind of not-knowing it is in.** "I have never met this" (missing knowledge, it
   should learn) is not "this is truly ambiguous" (noise, two readings are equally possible), and
   neither is "my sensor cannot see this" (a 6 ms tail, a mixture, a whisper). Each calls for a
   different act: learn, ask, or abstain.
4. **It uses its doubt.** It asks when two readings lead to two different actions, it revisits a
   decision when new evidence lands, and it spends its slow clock on what it was least sure of.
5. **It is checked.** Like the second law: a brick whose stated confidence is never compared with
   what actually happened (the corrections are exactly that comparison) is not lawful yet.

Why this is the same paradigm, not a fourth idea: the human model of the second law is corrected
one case at a time, and you can only correct well a model that tells you where it hesitates. A
model that is sure of everything makes every correction a surprise; a model that says "here I am
not sure" makes the correction a conversation. That is what trust is made of.


---

## 1. The problem we are actually solving

A chatbox rests on three equations. All three are false the moment the input is a
continuous stream instead of a submitted message.

| Chatbox equation | What it becomes | Evidence from our own projects |
|---|---|---|
| **the transcript is the state** | a trace, not the state | atlas `PBS-010`: 28 turns produced 19 note rewrites, one fact repeated 6 times. live-listen : one progressive summary, overwritten, deliberately *not* a journal |
| **the turn is the work** | work outlives the turn | atlas `PBS-004`: a mission needs **one stable ID** from request to acknowledgement to execution to integration to spoken delivery. `PBS-005`: 28 turns launched 56 phantom workers |
| **the interface is the agent** | the interface is a lens | atlas: *the browser is replaceable, it owns no business truth*. opencode-web-stream deliberately ships **no** OpenCode plugin: the daemon survives restarts and serves every project at once |

So the real subject is not *how do I answer*. It is:

> **Once nothing is a turn, "should I act?" becomes the central object of the system.**

In a chatbox there is nothing to decide: the human spoke, so you reply. In an ambient
system the default is **silence**, and the product is the judgement that breaks it.

---

## 2. The tmux analogy (the whole architecture in one idea)

tmux solved the shell's fundamental problem by refusing to let the work live in the window.

```text
  before tmux                         with tmux
  -----------                         ---------
  window = session = state            server (persistent)  <- holds the state
  close the window                    client (terminal)    <- a view you can drop
    -> kill the work                  attach / detach, many clients, one session
```

The shell did not get smarter. The **substance was moved out of the form**. The terminal
became a replaceable attached client, and "which terminal am I using" stopped being an
architectural question.

π does the same thing to the agent:

```text
  the agent       = a server-side process that never stops
  a client        = chat UI, terminal, phone, web page, desktop   (all attachable)
  a device        = microphone, speaker, screen, camera           (I/O resources, like a tty)
```

Two consequences, and they are the reason for "backend first":

1. A client must **never own state**. Kill the UI, the agent does not blink.
2. The hot path must **not cross a client boundary**. tmux's split is cheap because the
   pty is local; π's barge-in must be cheap for the same reason. Anything that cannot
   wait for a round trip stays in the process.

---

## 3. Heterogeneous inside, homogeneous outside

The inside is allowed to be a zoo: forked models, merged research code, five runtimes,
several languages, whatever wins on measurement. The outside is **one contract**.

- **Ports in, API out.** Each pole is a port with a typed boundary behind it. Providers
  (STT, TTS, embedding, LLM, tool) live in registries and are selected by configuration,
  never hardcoded in domain logic. This is atlas's proven shape:
  `DecisionPort` / `STTPort` / `TTSPort` / `ToolPort` / `StorePort` behind one engine.
- **0 magic, 100% transparency.** If a decision was taken, it is persisted and
  inspectable. atlas puts every agent, task, tool call and information flow on a canvas.
- **Integrable, not dependent.** π plugs into any host system as a whole module and
  depends on none. Same module on the workstation and on the phone.
- **Latency is a design constraint, not a tuning afterthought.** Measured, printed,
  and owned by the architecture. Numbers in `voice/ear/RESEARCH.md` section 8.

---

## 4. The four poles

| Pole | Question it answers | Directory | Existing evidence |
|---|---|---|---|
| **Vocal** | **the ear**: who is speaking, is it me, is it addressed to me, what was said. **The mouth**: what to say, in which voice, and whether to speak at all | `voice/` (`ear/`, `tts/`) | live-listen, `colab`, atlas STT/TTS, opencode-web-stream |
| **Memory** | what is true, what changed, what do I know about him | `memory/` | atlas typed memory, live-listen single summary, meeting-colab ledger + replayable projections, LiveMem |
| **Agentic** | should I act now, how, through which agent | `agentic/` | `colab` (ACP), opencode-web-stream streamer, atlas floor manager, kairos `decide/` |
| **Visual** | how to show it live, in the same clock as the voice | `vision/` | `../../KpihX-Labs/𝛱/live/` (Scene IR + Live Motor + Conductor) |

The vocal pole is **bidirectional**, which is why it carries two research dossiers
instead of one. They have opposite constraints:

- **The ear** (`voice/ear/`): survive a very low signal, and answer *whose* voice it is.
  Its enemy is invention, a model that hallucinates as energy drops.
- **The mouth** (`voice/tts/`): be believable, and be stoppable in the middle of a
  phrase. Its enemy is latency (a sentence that arrives after it was useful) and monotony.

**Whether to speak at all** does not belong to either of them: it is the arbiter's
decision (section 5). The mouth must accept `speak: false`, and must accept a cancel in
the middle of a word, exactly as the ear must accept being ignored.

The node that binds them is **the arbiter**.

---

## 5. The arbiter (the piece to get right)

Every project in the constellation invented this piece, and they disagree on who owns it:

| Project | Who decides to speak | Trade |
|---|---|---|
| kairos | code, from signals, *never a model* | predictable, brittle on meaning |
| opencode-web-stream | the model; code only does timing and one reflex | flexible, needs a strict output schema |
| atlas | a router decides, the Speaker holds an **independent veto** | fixed after two real failures |

Measured failures that settle the argument (atlas `PBS-001`, `PBS-002`):

- `Hey Pavel...` was routed to *respond* with address probability **0.24**, then spoken.
- `Stop stop assistant stop talking, I'm talking to Pavel` **still produced a sentence**.

**Rule to adopt:** the arbiter needs an **independent veto**, and a **non-semantic prior**
(biometrics, energy, room state) beats a semantic one for deciding *is this addressed to me*.
That is where the vocal pole stops being a feature and becomes architecture.

### Lead to explore (KπX, 2026-10-18): Jev for the real-time decision layer

**Jev** (TypeSafe System One: natural language and application state turned into typed
judgements and probabilities, `P(true | context)`) looks like a strong candidate for
**handling the real-time decision side inside the different poles**: should I speak, is
this addressed to me, is this worth surfacing now, should this visual patch be applied,
is this worth remembering. atlas already used it in exactly that role, for the meaning
half of the arbiter, which is why the question is worth reopening instead of guessing.

Status: **a lead to explore, not a decision.** Add it to the comparison of arbiter
implementations in `agentic/IDEAS.md` (rule-only, model-only, model + veto, biometric
prior + model, **and Jev as the judge**) and measure it on recorded sessions before
deciding whether it wins or loses.

---

## 6. Non-goals, right now

- No GUI, no front-end. The interface is a lens we attach later.
- No code structure, no package, no `pyproject.toml`, no git repo yet.
- No framework decision. Architectures are tested *before* one is chosen.
- Nothing that needs a "setup" before it can be tried.

**Corrected 2026-10-18 (KπX).** "No code structure" meant *do not lock an architecture before the
question is answered*, never *never build anything*. Measured state today:

- `π/` is a git repository (`git init`, branch `master`: this machine is the single infra allowed
  to commit on master).
- The ear's POC is a real uv package at `voice/ear/` (`π-ear`, module `π_ear`; the distribution name
  stays `pi-ear` because the packaging standard is ASCII-only): `pyproject.toml`, `src/`, `tests/`,
  `Makefile`. Dependencies are added with `uv add` only, never hand-written into the dependency list.
- The repository is named `pi` on both remotes (`github.com/kpihx-labs/pi` and
  `gitlab.com/kpihx-labs/pi`), for the same ASCII reason, and this time it is a measured one: asked
  for `π`, GitHub did not refuse, it **silently created a repository named `-`** (deleted at once).
  π is the name of the local directory and of the thing itself, never of the registry entry.
- Nothing else in this list is relaxed: no GUI, no framework chosen before measurement, no setup
  ritual, and the law of section 0 (100% stream, 100% live) is what the POC was built to obey.

---

## 7. How we will work

1. **One pole at a time.** Today: the vocal pole.
2. **Measure before choosing.** A number beats an intuition. live-listen calibrates its
   RMS gate on real chunks from 2026-09-23 instead of guessing.
3. **Keep the artefact that caused the failure.** A regression test must hold the real
   artefact, never a paraphrase of it.
4. **Write the finding into the pole's `IDEAS.md` the moment it is learned.**
5. **Fork, merge, borrow.** The mandate is to solve the fundamental problem, not to
   write original code.
6. **One pole, one ideas file.** Graved 2026-10-18: every pole owns an `IDEAS.md` next to its
   `RESEARCH.md`, and a pole is never left with a blank `RESEARCH.md` and no `IDEAS.md`. Current
   map: `voice/IDEAS.md` is the pole hub, `voice/ear/IDEAS.md` owns the ear,
   `voice/tts/IDEAS.md` owns the mouth, and the live ear test plan is `voice/ear/TODO.md`.
7. **Kernel hands off. Standing order, graved 2026-10-18 (KπX).** Do not touch `k-context`, or
   any file outside `π/`, without an explicit request from KπX in the current conversation.
   No action that was not explicitly requested. This order is recorded here because `π/` is the
   only zone this project is authorised to write to.

   **Extended the same day (KπX).** Even inside `π/`, no action is taken without an explicit green
   light for that action. The single exception, and it is a duty rather than a permission: when a
   misreading of KπX's intent is discovered, the wrong statement is corrected in the right `.md`
   file immediately, without waiting, in append-only form (the original stays, quoted, and the
   correction follows it). Analysis is never authorisation to act.
