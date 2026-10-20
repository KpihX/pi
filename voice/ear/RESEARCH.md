# voice/ear/RESEARCH.md

> Research dossier for the vocal pole. Compiled 2026-10-18 from live sources.
> Every number below cites its source. Nothing here is a decision; decisions go in `IDEAS.md`.
> Machine of reference: `kpihx-ubuntu`, Intel Core Ultra 7 155H, Intel Arc iGPU.
>
> **RESTARTED 2026-10-18 (evening) on KπX order.** Part 0 below is the new spine: *learning in the
> stream*: online / lifelong learning, architectures with a **persistable internal state**, and
> where reinforcement learning actually belongs. Verdict on the first pass: it studied the wrong
> object, not the wrong model. It studied a **frozen** system (embedding + cosine + threshold +
> cohort of enrolled speakers), which is precisely what the law of π rejects. Nothing is deleted:
> the first pass stays below as the **baseline**, which is the contrast case that makes the
> paradigm legible. Read Part 0 first.

---

# PART 0: RESTART, learning in the stream (2026-10-18, evening)

## R0.1 The diagnosis, in one table

| | Part 1 (first pass) | Part 0 (what was actually asked for) |
|---|---|---|
| The model | frozen after training | keeps changing while it listens |
| The voiceprint | enrolled in advance | grows out of what it hears |
| The threshold | a constant, hand-set | learned, and it moves |
| The boundary | a cohort of other, known speakers | derived from the gallery itself |
| The act | `train -> calibrate -> infer` | one act, superposed |
| The state | implicit, buried in the weights | **an explicit artifact, on disk** |
| The output | `[timestamp] text` | events: text + speaker + voice physics |
| Cold start | useless until enrolled | useful from second zero, with an empty state |

Part 1 is not wrong. It is the **baseline**: the best that can be done while keeping the model
frozen. Everything below is what changes once the state becomes a first-class object.

## R0.2 The law, restated as an engineering constraint: two clocks

From `../IDEAS.md` section 0. Learning is continuous, but it is not one loop.

| Clock | What lives there | Budget | If it is late | Where it is stored |
|---|---|---|---|---|
| **fast** | what the system knows *right now*: centroids, candidates, sound prototypes, per-entity score statistics, engagement state | inside the reaction window (ms to s) | unacceptable | memory, mirrored to disk **on change**, not per frame |
| **slow** | the weights themselves: consolidation, re-estimation, forgetting | seconds to minutes, during idle | perfectly fine | one checkpoint file |

This split is not a concession to the law. It is the law, and it is also the standard answer to
catastrophic forgetting (R0.6). The fast clock is plastic and cheap and reversible. The slow clock
is expensive and must never block the fast one.

## R0.3 "Learn in the stream" = three families, not mutually exclusive

| Family | Where the learning lands | Cost | Survives a restart | Concrete members |
|---|---|---|---|---|
| **A. Adapt the state, not the weights** | memory and disk: centroids, candidates, prototypes, boundary statistics | microseconds | **yes, by construction** | online speaker inventory (`2610.11278`), cohort-free gallery calibration (`2610.11272`), adaptive admission threshold, Personal VAD read-out, sticky HDP-HMM (`0905.2592`) |
| **B. Adapt the weights at inference (plasticity)** | the parameters themselves | one backward pass per update (or per bias-only update) | yes, if the checkpoint is written | test-time training (TTT, Sun 2020; speech: `2309.10930`), TENT (entropy minimisation), **TTT-online** (`2508.01847`), fast weights (Hinton 2016), SRWM (Irie, ICML 2022), Automating Continual Learning (ICLR 2024) |
| **C. Architectures whose state IS the memory** | a recurrent state tensor, serialisable | one pass, linear in time | yes, if the state is serialised | Transformer-XL and compressive memory, Recurrent Memory Transformer, Mamba-1/2/3 (`2603.15569`), xLSTM (NeurIPS 2024), RWKV, Neural Turing Machine / DNC, Emformer and cache-aware streaming conformer (`2312.17279`) |

**The order of ambition is A, then B, then C.** A is a data structure, it works tomorrow, and it
delivers the behaviour KπX described. B is scientifically the most interesting and the most
likely to break. C is a research bet that changes what "the model" means.

## R0.4 Family A, in detail: the state is the learned object

This is the family that answers "it knows me, it notices new voices, it does not ask me to
calibrate anything".

**A.1 The inventory is built online, nobody is enrolled in advance.** Online speaker inventory
(`2610.11278`, NAVER, ICASSP 2027 submission). Three mechanisms, and they are the whole idea:

1. a **candidate pool** holds embeddings that do not clearly belong to any registered speaker;
2. a speaker is **registered only once a candidate has accumulated K mutually compatible
   embeddings** (evidence-gated registration): one noisy segment cannot create a phantom speaker;
3. a **commit gate**: while a candidate is undecided, the label is held back instead of guessed.

Measured: macro speaker-count error **225% -> 21%**, macro DER **17.59% -> 15.95%**, mean added
latency **0.109 s**. That last number is the point: the online inventory is not slower than the
frozen pipeline, it is 109 ms slower than a pipeline that was allowed to cheat by knowing the
answer in advance.

**A.2 The threshold calibrates itself.** Same paper:
`tau_t = min(tau_base + lambda * N_t, tau_max)`, where `N_t` is the number of registered speakers.
The admission threshold **grows as the inventory grows**. There is no constant to tune, which is
exactly "no calibration phase" without giving up a criterion.

**A.3 The cohort trap, measured.** Score calibration without an external cohort (`2610.11272`,
same group). The obvious move, "use the other speakers I already know as a normalisation cohort",
**collapses**: applying Z-norm on the enrolled cohort drops performance from **85.26% to 41.72%**.
The working path is affine-invariant terms computed on the gallery (score dispersion and
enrollment-profile deviance), or a session-level GMM over each cluster. This is the trap that
would have been built in week one without reading the paper.

**A.4 The mathematical ancestor.** The sticky HDP-HMM (`0905.2592`, Fox, Sudderth, Jordan,
Willsky): a non-parametric prior under which the number of speakers is not fixed; "sticky" adds a
self-transition bias that suppresses spurious speaker switches. The modern candidate-pool
inventory is a computational simplification of the same idea: **a distribution over the number of
speakers, updated as evidence arrives**.

**A.5 What A gives you for free.** No enrollment, no calibration, no train/calibrate/infer
trilogy, an explicit state (a dictionary of centroids plus a set of candidates), instant reset
(delete the file), and full inspectability (open the JSON and read what it believes).

## R0.5 Family B, in detail: the model that keeps learning

**B.1 Test-time training, and its online form.** TTT (Sun et al., ICML 2020) trains a model on
two objectives, the main task plus a self-supervised one, and at test time updates the model on
the self-supervised objective only. Its speech application (`2309.10930`, 2023) is the first TTT
work in the speech domain and it targets **speaker identification** and emotion detection under
distribution shift; the self-supervised task used is masked autoencoding, and TTT beats TENT,
linear probing and plain fine-tuning. Crucial refinement for us, from speech enhancement
(`2508.01847`, Interspeech 2025): **TTT-online** does not discard the updated parameters after the
sample, it **uses them as the initialisation for the next sample**:

```
theta*(i-1)  ->  init  ->  adapt on x_i  ->  theta*(i)  ->  permanent
```

That single line is "a model that lives forever": the weights carry the history. The same paper
gives the cheap variant, TTT-online-batch-bias, which adapts **bias parameters only**, and reports
it as the computationally efficient alternative. That is the realistic entry point on a laptop.
It also reports the concrete win that matters for us: on a domain shift it cut the PESQ drop from
**8.01% to 2.09%**.

**B.2 Fast weights: memory without addressing.** Hinton's fast associative memory (2016): the
recurrent state is influenced by a second, rapidly decaying weight matrix
`A(t) = lambda*A(t-1) + eta*h(t)*h(t)^T`, so recent hidden states are re-injected into the current
state, weighted by their scalar product with it. Two properties are worth quoting verbatim
because they are the design brief:

- "in a fast associative memory there is **no need to decide where or when to write to memory and
  where or when to read from memory**": the memory is updated all the time, writes are
  superimposed;
- the effect "is equivalent to attending to past hidden vectors in proportion to their scalar
  product with the current hidden vector", i.e. it is attention with a fixed, cheap rule instead
  of a learned one.

This is the escape route from the Neural Turing Machine family, whose addressability is an
engineering burden rather than a feature.

**B.3 Self-referential weight matrices.** SRWM (`2202.05780`, Irie, Schlag, Csordás, Schmidhuber,
ICML 2022): the network does not merely use its weights, it **generates the updates of its own
weights at runtime**, via self-invented key/value patterns and self-generated learning rates,
using outer products and the delta rule. "The weight matrix of a neural network is its program;
the program of a traditional network is fixed after training, whereas a self-referential network
can keep rapidly modifying all of itself during runtime." The continual-learning application is
*Automating Continual Learning* (Irie, Csordás, Schmidhuber, ICLR 2024 submission). Related
formal result (`2023.emnlp-main.588`): linear transformers / fast weight programmers are RNN-like
processors with a **fixed-size state**, and the recurrent and self-referential extensions
overcome limits of the plain variant, for example generalising on the parity task.

**Honest scale warning.** The published results for B.3 are few-shot classification and
procedurally generated RL games. Nobody has shipped an SRWM ear. Treat it as the long-term bet,
not next week.

## R0.6 Family C, in detail: the state is a tensor you can write to disk

- **State space models.** Mamba and its line keep a **constant-size state** with linear-time
  updates; the state is the memory. Mamba-3 (`2603.15569`, ICLR 2026) is explicitly written from
  an "inference-first perspective" and adds a **complex-valued state update** whose stated purpose
  is better **state tracking**, the precise weakness of pure linear models. Mamba-3 matches
  Mamba-2 perplexity with **half the state size**.
- **Why the SSM alone is not enough for π.** xLSTM (NeurIPS 2024) states it plainly: models
  without memory mixing, i.e. Transformers and state space models, **cannot track state** in the
  way an RNN can (parity-like tasks), and this is a fundamental expressivity gap, not a tuning
  gap. For an ear that must maintain a discriminative, evolving representation of who is present,
  this matters. xLSTM's answer is exponential gating plus a **matrix memory** updated by a
  stabilised delta rule, with linear decoding cost.
- **Audio evidence.** SSAMBA (`2405.11831`): self-supervised audio Mamba, **92.7% faster batch
  inference and 95.4% more memory-efficient** than the audio spectrogram transformer at tiny size
  with 22k input tokens, while beating it on most tasks. Audio Mamba (`2406.03344`): the first
  purely SSM, attention-free audio classifier, comparable to the AST on six benchmarks. So the
  family is live for audio, not speculative.
- **Streaming with an explicit cache.** The Emformer / stateful conformer line (`2312.17279`) and
  NVIDIA NeMo's cache-aware streaming inference, including a multitalker streaming example, show
  the production shape of "state carried across chunks": the state is an object that is passed in
  and passed out. That is the same contract the fast clock needs.

**What C buys and costs.** It buys a state that is a fixed tensor (serialise it, ship it, restart
from it) with linear cost per second. It costs a training pipeline, and it replaces the encoder we
have. It is a candidate for "the model that lives forever", not for the first prototype.

## R0.7 The output: leaving `[timestamp] text` behind

KπX's spec: speakers identified by their embedding hash (known or new), plus, for every excerpt,
numeric metadata (was the voice high or low, intensity, decibels), all of it near-zero latency and
cheap to store, as a file the agentic pole can read.

**The precedent exists, and it is very recent.** Streaming speaker-attributed ASR
(VibeVoice-ASR-Streaming, `2609.02812`, Microsoft technical report, September 2026), stated in its
own words: the system "must output both the words **and their speaker labels as the conversation
unfolds**", and "each sentence is attributed to a speaker **when it is emitted**, rather than
after the recording ends". Two adjacent works confirm the direction is active: *Beyond Fresh
Starts: Stateful Inference for Streaming ASR in Conversational Voice Agents* (`2608.22101`, August
2026), and the underlying cache-based inference work (`2312.17279`).

**The record.** Every field a number, nothing prose. One line of JSONL per emitted excerpt:

```json
{"t0":12.480,"t1":14.930,"text":"...",
 "speaker":{"id":"spk_3f9a","kind":"known","cos":0.81,"evidence":47,"since":"2026-10-18T21:04Z"},
 "voice":{"f0_hz":112.4,"f0_pctl":12,"rms_dbfs":-31.7,"spl_rel_db":6.2,
          "formants_hz":[520,1290,2440],"hnr_db":14.1,"speech_rate_per_s":4.6,"voiced_ratio":0.72},
 "candidates":[{"id":"cand_7","cos":0.58}],
 "decided_at":15.400}
```

Three properties that come straight from the law:

- `decided_at` is **later** than `t1`. The record is written immediately, the attribution may be
  revised when the evidence lands. Undecided is a legitimate value, not an error.
- `kind` is `known`, `new`, or `unknown`. A new speaker is identified by the hash of its
  candidate centroid, not by a name somebody typed.
- `f0_pctl` is a percentile computed **against the speaker's own observed distribution**, held in
  the state, not against an absolute frequency grid. A bass at 120 Hz can sit in the top decile
  *for him*. Note the deliberate absence of any categorical label such as "grave" or "aigu": the
  record carries numbers, and any interpretation of them happens downstream, by whoever needs it.
  A label written by the ear is a decision the ear had no right to make.

**Storage split (this is what makes it cheap).** Events go to an append-only JSONL, a few hundred
bytes per excerpt. Embeddings are **not** in the event log: they live in a hash-keyed store, one
row per speaker or candidate, never one per utterance. Ten hours of conversation is therefore
single-digit megabytes of events, plus a handful of kilobytes of embeddings.

**The hash.** Two different objects, often confused:

| Which hash | How | Why |
|---|---|---|
| **identity key** | cryptographic hash of the quantised centroid, e.g. 16 characters of a blake2b digest | stable, short, cheap, no model needed, comparable across machines |
| **learned code** | binary speaker embedding: i-vector to binary codes by locality-sensitive hashing, then by learned Hamming-distance projections (`1510.05937`), or end-to-end deep hashing with additive margin (DAMH, `fan19_interspeech`, VoxCeleb2) | fixed-size codes with cosine preserved under Hamming, built for retrieval over tens of thousands of speakers, and usable as a privacy-preserving form (`TR2013-082`, secure binary embeddings) |

Start with the identity key; the learned code is the upgrade when retrieval scale or privacy
demands it. Both are hash-keyed, so the rest of the architecture does not change.

**The voice physics, cheaply.** Real-time pitch and formant tracking is a solved desktop problem,
not an open research question: InFormant is a shipped real-time pitch and formant tracker, and
`parselmouth` exposes the Praat analyses (pitch, formants, intensity, harmonicity) programmatically.
For the fast clock: `f0` by autocorrelation or YIN (nearly free, and Whisper-class STT already
computes the frames), formants by LPC (Burg), intensity by RMS mapped to dBFS, and HNR read off
the same autocorrelation. The expensive variant is a neural pitch tracker (CREPE); it is not
needed to start.

## R0.8 Reinforcement learning: where it belongs, and where it does not

**Yes, for the timing and engagement policy.** This is the layer KπX described as "it must know
when I need it".

- **DuplexPO** (`2607.07148`, July 2026): argues the apparent trade-off between intelligence and
  conversational dynamics is not fundamental, and **decouples when to speak from what to say**.
  The timing policy is trained with a Factorized Conversational Dynamics Reward that gives
  fine-grained temporal credit assignment for turn initiation, backchanneling, yielding to barge-in
  and regularised participation, optimised with a GRPO-style objective. Reported wins: timely
  backchannels, smooth turn-taking, barge-in handling, **without** degrading reasoning.
- **ASPIRin** (`2604.10065`, April 2026): projects the text vocabulary onto a coarse binary state
  (active speech versus inactive silence) before applying GRPO with rule-based rewards, precisely
  so that the interactivity objective cannot degrade the linguistic foundation. Rewards are
  computed by comparing the agent's voice activity against the user's, using ASR timestamps;
  the interruption score penalises speaking while the user is talking.
- **SpeakRL** (`2512.13159`, IWSDS 2026): rewards *proactive* behaviour, asking a clarification at
  the right moment instead of answering blindly; +20.14 points of task completion over the base
  model without increasing the number of conversation turns.

**Yes, for the memory policy.** Where the state on disk stops being data and becomes a skill:

- **Agentic memory as tools** (`2603.07670` survey, reporting AgeMem): store, retrieve, update,
  summarise and discard are exposed as callable operations and the whole pipeline is optimised with
  RL; the learned tactics are exactly the ones a human would name, summarise intermediates
  **before** the context fills up, and discard records that are semantically redundant.
- **File operations as memory, learned** (`2609.34422`): a policy trained with RL writes and reads
  `state.json` and continuation files across a context boundary. The ablation is what makes it
  relevant: blanking the memory files costs **14.1 points** of success, transplanting
  task-mismatched content costs 11.5. The written state is causally load-bearing. It is not
  decoration, and it is not a log.

**No, for perception.** There is no usable reward signal for "this embedding is correct". You
cannot reinforcement-learn a speaker embedding into existence; you supervise it or you
self-supervise it (which is exactly what family B does). Saying this plainly matters more than
hedging: putting RL on the perception path is where this project would quietly die.

**And therefore, the forced order.** RL needs a reward model. The reward model needs a signal.
The signal is the event log of R0.7. So: **log first, then the policy, then RL**. Not the reverse.

## R0.9 The hard problems, named rather than hidden

**Catastrophic forgetting.** The comprehensive survey (IEEE TPAMI 2024, `wang2024survey`) frames
the entire field as one objective: a proper **stability-plasticity trade-off** under resource
constraints. Evidence in our own modality: continual speaker adaptation for TTS (`2103.14512`)
shows serial fine-tuning forgets earlier speakers, and that **experience replay prevents it to a
great extent, compared with EWC**: replay beats regularisation, measured, in speech.

**Single-pass, non-i.i.d., any-time inference is harder still.** Streaming lifelong learning with
any-time inference (`2301.11892`) defines the setting we actually live in: one sample per step,
single pass, class-incremental, evaluable at any moment. Its own finding is sobering: batch
methods collapse, and even replay-based online methods "suffer from information loss badly". The
working recipe there is a Bayesian fast parameter update from a single example, plus snapshot
self-distillation, plus a selected rehearsal buffer.

**The law is the mitigation.** Fast clock: a plastic, cheap, reversible state, per utterance.
Slow clock: weight updates during idle, i.e. consolidation, which is what biology does with sleep
replay (the CACM piece on forgetting: "one strategy that biology enforces in our learning strategy
is dreaming"). KπX's law of π is, without having been designed as such, the standard engineering
answer to continual learning. That convergence is worth more than any single model choice.

**Reset is a feature, and it is free.** Because the fast state is a file, "forget everything" is a
file operation. KπX asked for exactly that, and family A gives it by construction.

## R0.10 The stack, in build order

| Layer | First choice | Upgrade path | What changes versus Part 1 |
|---|---|---|---|
| voice activity | Silero VAD v5 (already installed) | same | nothing |
| identity encoder | CAM++ ONNX (or ECAPA) | a self-supervised encoder adapted online | **the encoder is now a sensor, not the system** |
| identity **state** | candidate pool, evidence-gated registration, adaptive threshold (`2610.11278`) | sticky HDP-HMM style non-parametric prior | **this is the new centre of gravity** |
| boundary | gallery-derived, affine-invariant (`2610.11272`) | session GMM per cluster | no external cohort, ever |
| non-speech | YAMNet (521 AudioSet classes, own 1024-d embedding) | same | same inventory, other instrument |
| voice physics | YIN f0, LPC formants, RMS to dBFS, HNR | CREPE | new: the event carries numbers |
| transcript | Whisper (installed), sliding window | streaming speaker-attributed ASR class (`2609.02812`) | text joins speaker and physics in one record |
| persistence | JSONL events plus hash-keyed embedding store | versioned centroids with explicit checkpoints | **new: the state is an artifact** |
| plasticity | TTT-online-bias (`2508.01847`) | full TTT-online, then SRWM | new: weights that move |
| state architecture | none | Mamba / xLSTM state serialisation | new: the long bet |
| policy | hand-set gate | RL on the event log (DuplexPO-style) | new: RL last, never first |

## R0.11 The smallest experiment that tests the paradigm

The first pass's reflex was to download a model. The paradigm is not tested by a download. It is
tested by three falsifiable questions, in this order:

1. **Log without deciding.** Record ten minutes of real life into the R0.7 record, with
   `kind:"unknown"` everywhere and no decision logic at all. Question answered: *is the record
   cheap enough, and does it read well to a human afterwards?*
2. **Run the inventory over that log, single-pass.** Candidates appear, gather evidence, commit,
   sometimes merge. Question answered: *does the threshold move on its own?* and *how long does a
   new voice take to become a speaker?*
3. **Try to break it.** A cough, a door, a television, and KπX's own voice recorded on a different
   microphone. Question answered honestly: *does it invent a speaker, and does it fuse someone
   else into KπX?*

Only after those three does a threshold value mean anything. That is the difference between
experimentation and configuration.

## R0.12 Limits of this dossier, stated up front

- Most of the cited work is **not a product**: the online inventory is an ICASSP submission, the
  streaming speaker-attributed ASR is a technical report, TTT-online is an Interspeech paper on
  speech enhancement. None of them ships as an ear.
- **I/O is the hidden cost** of "state on disk". The rule is to persist on change, not per frame,
  and to run the write on the slow clock. A checkpoint policy is required, not optional.
- **Float drift breaks naive hashing.** A centroid moves, so its hash moves. The identity key must
  be computed on a **committed, versioned** centroid, and the version must appear in the record.
  Otherwise a speaker silently changes identity during the night. This is a real trap, found here
  rather than in production.
- The whisper / low-volume regime from Part 1 section 5 is untouched by all of this, and it is
  where family B becomes the credible lever: a self-supervised objective is exactly what adapts to
  a regime shift without labels.

## R0.13 Sources (Part 0)

1. Online speaker inventory, NAVER, ICASSP 2027 submission, 2026-10: https://arxiv.org/abs/2610.11278
2. Score calibration without an external cohort, 2026-10: https://arxiv.org/abs/2610.11272
3. Sticky HDP-HMM for speaker diarization, Fox et al.: https://arxiv.org/html/0905.2592v4
4. TTT, Sun et al., ICML 2020; TTT++: https://proceedings.neurips.cc/paper/2021/file/b618c3210e934362ac261db280128c22-Paper.pdf
5. Test-Time Training for Speech, 2023: https://arxiv.org/html/2309.10930
6. Test-Time Training for Speech Enhancement (TTT-online), Interspeech 2025: https://arxiv.org/abs/2508.01847
7. Fast weights, Hinton, 2016: https://www.cs.toronto.edu/~hinton/absps/FastWeights.pdf
8. Self-referential weight matrix, Irie et al., ICML 2022: https://proceedings.mlr.press/v162/irie22b.html
9. Automating Continual Learning, Irie, Csordás, Schmidhuber, ICLR 2024: https://openreview.net/forum?id=5twh6pM4SR
10. Computational power of linear transformers and self-referential extensions, EMNLP 2023: https://aclanthology.org/2023.emnlp-main.588
11. Fast weight programming and linear transformers, review, 2025: https://arxiv.org/pdf/2508.08435
12. Mamba-3, ICLR 2026: https://arxiv.org/abs/2603.15569
13. xLSTM, NeurIPS 2024: https://papers.nips.cc/paper_files/paper/2024/file/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Paper-Conference.pdf
14. Mamba core repository: https://github.com/state-spaces/mamba
15. SSAMBA: https://arxiv.org/pdf/2405.11831v2 ; Audio Mamba: https://arxiv.org/abs/2406.03344
16. Stateful conformer, cache-based inference: https://arxiv.org/html/2312.17279v3
17. VibeVoice-ASR-Streaming technical report, 2026-09: https://arxiv.org/pdf/2609.02812v1
18. Beyond Fresh Starts: stateful inference for streaming ASR, 2026-08: https://arxiv.org/html/2608.22101v1
19. NeMo cache-aware multitalker streaming example: https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py
20. Binary speaker embedding: https://arxiv.org/abs/1510.05937 ; deep hashing for speaker retrieval: https://www.isca-archive.org/interspeech_2019/fan19_interspeech.pdf ; secure binary embeddings: https://www.merl.com/publications/docs/TR2013-082.pdf
21. DuplexPO, 2026-07: https://arxiv.org/html/2607.07148v1 ; ASPIRin, 2026-04: https://www.alphaxiv.org/abs/2604.10065 ; SpeakRL, IWSDS 2026: https://arxiv.org/abs/2512.13159
22. Memory for autonomous LLM agents (AgeMem), 2026: http://arxiv.org/html/2603.07670v1 ; file operations as memory via RL, 2026: https://arxiv.org/pdf/2609.34422v2
23. Continual learning survey, IEEE TPAMI 2024: https://ieeexplore.ieee.org/document/10444954
24. Streaming lifelong learning with any-time inference: https://arxiv.org/abs/2301.11892
25. Continual speaker adaptation for TTS: https://arxiv.org/html/2103.14512v2
26. Catastrophic forgetting, CACM: https://cacm.acm.org/news/forget-the-catastrophic-forgetting/
27. Real-time pitch and formant tracking (InFormant): https://github.com/in-formant/in-formant ; Praat analyses in Python: https://parselmouth.readthedocs.io/en/stable/api_reference.html

---

# PART 1: BASELINE, the frozen paradigm (first pass, 2026-10-18 morning)

> Kept verbatim. This is what was written before the paradigm question was asked, and it remains
> the correct reference for the *models* (which encoder, which VAD, which runtime), and the
> wrong reference for the *architecture*. Read it as the contrast case.

---

## 1. "Voice detection" is not one problem, it is four

Almost every product on the market answers only #1, and answers #3 as a guess.

| # | The question | State of the art | Cost |
|---|---|---|---|
| 1 | is there sound at all | energy / RMS | free |
| 2 | is it speech | **Silero VAD v5** (neural, far better than an energy gate) | microjoules |
| 3 | **is it MY voice** | **speaker embedding + Personal VAD conditioned on it** | ~7 M params |
| 4 | what was said | ASR, and **whispering is a genuine acoustic shift** | heavy |

Our own `live-listen` today implements #1 (an RMS gate calibrated at `0.002` on real
chunks) and #4, with #2 available but switched off. **#3 is the missing piece and the
whole subject of this dossier.**

---

## 2. Layer 1-2: voice activity detection

- `whisper-cli` ships a native neural VAD (Silero) behind `--vad`, `--vad-model`,
  `--vad-threshold`. It is a real speech / non-speech separator.
- The GGML VAD model **is now installed on this machine**:
  `~/.local/share/whisper-cpp/ggml-silero-v5.1.2.bin` (885 KB, dated 2026-09-30).
  live-listen's README still claims it is absent; the README is stale.
- `live-listen` only needs `--vad-model <path> --vad-threshold 0.5` to activate it,
  with no code change (`vad_model: null` by default).
- Silero VAD is also available as ONNX and is what `sherpa-onnx` and `decibri` embed.
- **Known limit:** Silero separates *speech* from *non-speech*. It does not tell you
  whose speech it is, and it does not help with music or jingles carrying real energy
  (the `Radio-Canada x7` failure was real audio with real energy).

---

## 3. Layer 3: speaker verification (the voiceprint)

### 3.1 Model landscape, VoxCeleb1-O, EER lower is better

| Model | Params | EER (Vox-O) | Notes |
|---|---|---|---|
| x-vector / TDNN | 4 M | ~1.5 % | historic baseline |
| Res2Net | 4.03 M | 1.56 % | 3D-Speaker table |
| **CAM++** | **7.2 M** | **0.65 %** | best accuracy-per-cost, faster inference than ECAPA |
| ECAPA-TDNN | 20.8 M | 0.86 % | the reference everyone compares to |
| ERes2NetV2 | 17.8 M | 0.61 % | tuned for **short** utterances (2 to 3 s) |
| ERes2Net-large | 22.46 M | 0.52 % | best of the open set, 3x the cost of CAM++ |

Sources: 3D-Speaker README (Alibaba, Apache-2.0), CAM++ paper arXiv 2303.00332,
ERes2NetV2 arXiv 2406.02167. A separate 2025 study reports CAM++ 0.73 / ECAPA 0.86 /
ERes2Net-Large 0.57, so the ordering is stable across sources even when the absolute
numbers move by a few hundredths.

On CN-Celeb the ordering changes: ERes2NetV2 6.14 % and CAM++ 6.78 % beat ECAPA 8.01 %.
**Conclusion: CAM++ is the default, ERes2NetV2 is the fallback when utterances are short.**

### 3.2 The task, in one line

```text
enroll:  3-10 s of the owner  ->  embedding vector (192 or 512 floats)  ->  stored locally
runtime: N ms of audio        ->  embedding  ->  cosine similarity vs stored  ->  score
```

Text-independent, so **no passphrase**. The vector *is* the voiceprint.

### 3.3 Adaptation with short enrollment

- **ICASSP 2026, arXiv 2601.12769**: *Adaptive Speaker Embedding Self-Augmentation for
  Personal VAD with Short Enrollment Speech*. Augments the enrollment embedding by
  additive fusion of keyframe embeddings from the mixed speech, plus a long-term
  adaptation rule that refines the embedding during detection. Result: **after five
  iterative updates, short enrollment matches full-length enrollment performance**,
  despite the practical limit that enrollment is often a single wake-up word.
  This is the mechanism that makes "it learns to know me as it is used" concrete.

### 3.4 Caveats to hold in mind

- Whispered enrollment is harder: SV on normal-vs-whisper trials reaches **EER 22.77 %**
  on CHAINS with legacy architectures, though a 2026 post-processing model improves this
  substantially (AUC 98.16 % on normal-vs-whisper). The paper also measures the energy
  gap: **RMS 0.046 normal vs 0.0138 whispered**, i.e. 3.3x lower.
- Ageing, illness, and microphones all drift the embedding. Hence adaptation, not a
  frozen vector.
- **Compression hurts.** A 2025 study on neural audio codecs shows SV performance
  degrades under encoding, so do not put a lossy codec between mic and embedding.

---

## 4. Personal VAD: the gate that opens only for one person

Personal VAD (Google, Odyssey 2020, arXiv 1908.04284) is not a VAD. It is a
**three-class per-frame classifier**: SILENCE / TARGET-SPEAKER / NON-TARGET SPEAKER.

| Variant | Params | ACC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| PVAD 1.0 | 2.3 M | 0.82 | 0.86 | 0.89 | 0.82 |
| PVAD 2.0 | 10.4 M | 0.84 | 0.88 | 0.91 | 0.85 |
| USEF-TP (2025, embedding-free) | 15.1 M | 0.97 | 0.98 | 0.98 | 0.97 |

Architecture ET (Embedding-Conditioned Training), from the open-source reproduction
(`personal-vad` on PyPI, **Apache-2.0**, own from-scratch implementation with zero
proprietary dependencies, trained on the open 8-language MLS corpus):

```text
input = [ 40-D log-Mel  ||  256-D enrolled d-vector ]   = 296-D
      -> 2-layer LSTM, 64 units, unidirectional
      -> 3 classes          total  129,795 params  =  262 KB quantized TFLite
```

The same model with `p_0 = 0.2` probability of zeroing the embedding during training
also works as a plain enrollment-less VAD. **One model, two modes.**

Google's stated motivation is our exact case:

> "*gating the inputs to a streaming on-device speech recognition system, such that it
> only triggers for the target user, which helps reduce the computational cost and
> battery consumption, especially in scenarios where a keyword detector is unpreferable*"

**That sentence is π's vocal pole, written in 2019.**

Related: **TargetVoice** (Interspeech 2025) does the harder version, *target speaker
extraction*: a 10 MB encoder builds an embedding from **a single 3 s enrollment**, fused
into a 12 MB extraction block at 6G MACs, enough for real-time on constrained hardware.
That is the escalation path if we ever want to *separate* the owner's voice from a room
rather than merely detect it.

---

## 5. Layer 4: what was said, at whisper volume

### 5.1 The measured damage

| Test set | Model | Normal | Whispered |
|---|---|---|---|
| AISHELL6-Whisper (zh) | Whisper-v3 | 3.95 % CER | **18.93 % CER** |
| AISHELL6-Whisper (zh) | Whisper-v3 fine-tuned | 1.62 % | 6.69 % |
| AISHELL6-Whisper (zh) | Seed-ASR | 0.65 % | 1.58 % |
| AISHELL6-Whisper (zh) | **Whisper-Aware LLM (2026)** | **0.63 %** | **1.31 %** |
| wTIMIT (en) | Mamba SSM + SSL | - | 5.97 % (US) / 16.58 % (IR) / 21.67 % (SG) |
| CHAINS (en) | WavLM fine-tuned | - | **9.22 % WER** |

Source: arXiv 2608.10836, arXiv 2506.16969, arXiv 2407.21211.

### 5.2 Why whispering is a different regime, not a lower volume

No vocal-fold vibration, therefore: vowel formants shift upward, low-frequency energy on
voiced consonants collapses, and **spectral flatness increases**. Whisper's own README
position: it is trained predominantly on normal voiced speech.

### 5.3 The hallucination trap (we already hit it)

Whisper invents text when there is little to transcribe. `live-listen` documents
`Sous-titrage Societe Radio-Canada` repeated 7 times, with **real energy**, therefore
invisible to any energy gate. This failure mode gets **worse** as input energy drops.
The 2026 Whisper-Aware LLM paper quantifies it: hallucination rate **over 25 % down to
4.5 %** purely from teaching the model to perceive and react to whisper uncertainty.

**Consequence: gain, a neural VAD, a per-speaker gate and a repetition guard are not
optional polish at whisper volume. They are the difference between a transcript and
fabrication.**

### 5.4 The three remedies, in order of cost

1. **Pre-processing**: super-resolution / whisper-to-normal voice conversion before ASR.
   Model-agnostic, no training.
2. **Adaptation**: fine-tune a small model on whispered corpora. Datasets exist:
   **wTIMIT**, **CHAINS**, **AISHELL6-Whisper**. A Mamba SSM trained on *little* whisper
   data handles both regimes at once; WavLM fine-tuned reaches 9.22 % on CHAINS.
3. **Uncertainty-aware decoding**: the Whisper-Aware LLM approach, best accuracy and best
   hallucination behaviour, but it is an Audio-LLM rather than a small on-device model.

---

## 6. The physical ladder (what a microphone can and cannot do)

This distinction must be kept honest, because it separates engineering from research.

| Rung | Sound present | Solvable with a microphone |
|---|---|---|
| normal speech | yes, voiced | trivially |
| very soft voice | yes, voiced | gain + AGC + sensitive VAD + ASR |
| **whisper** | yes, **unvoiced** | yes, via one of the three remedies in 5.4 |
| **silent articulation** | **no** | **no. Requires a second, non-acoustic sensor.** |

For the last rung the literature is real but hardware-bound:

| Technology | Result | Source |
|---|---|---|
| sEMG on the face/neck (AlterEgo) | ~92 % on digits and short phrases | MIT Media Lab |
| sEMG microneedle array (2025) | **WER 8.5 %** on 1,396 words | Sensors and Actuators Reports |
| Ultrasound under the jaw (SottoVoce) | controls an unmodified Alexa by silent mouthing | ACM / arXiv 2303.01758 |
| Wi-Fi backscatter (Lip-Siri, 2026) | 85.61 % word accuracy, **36.87 % WER** continuous | arXiv 2601.18177 |
| Visual lip reading | mature, needs a camera facing the user | many |

**Reading of the owner's memory** ("Google responded when I barely moved my lips"):
that was almost certainly the *whisper* rung, not the silent rung, because a sound *was*
produced. A phone mic in near-field with AGC already reaches 3.3x below normal RMS.
So the goal is reachable acoustically; **true silence is a sensor project, not an
ASR project, and belongs to a later port.**

---

## 7. Who spoke when: diarization (the meeting case)

The meeting scenario ("in a meeting, only take the floor if it is me") needs *who*, not
just *not me*.

DER, lower is better, no collar:

| Dataset | pyannote 3.1 | DiariZen Large-s80-v2 |
|---|---|---|
| AMI-SDM | 22.4 | **13.9** |
| AliMeeting far-field | 24.4 | **10.8** |
| NOTSOFAR-1 | - | 16.7 |
| VoxConverse | 11.3 | **9.1** |

DiariZen by speaker count (DER): 1 speaker **2.3**, 2 speakers 11.4, 3 speakers 10.3,
4 speakers 12.7, **5+ speakers 7.1**. Note it is *better* in crowded rooms than in
dyads, because it was trained and pruned for that regime.

Independent ETH benchmark (arXiv 2509.26177) over 196.6 h, 5 languages: pyannoteAI
(commercial) **11.2 %**, **DiariZen 13.3 %** open-source best, Sortformer v2-streaming
7.0 % on AliMeeting but far weaker elsewhere. DiariZen architecture: structurally pruned
**WavLM-Large** encoder + Conformer backend with powerset classification + VBx
clustering with PLDA, multi-channel WavLM support added 2026-01-31.

**Practical note:** diarization is *offline* in most toolkits, including sherpa-onnx.
For a live meeting that is a latency problem. Keep it out of the hot path.

---

## 8. The latency budget (the number that decides the architecture)

Measured, from our own corpus and from the industry:

| Source | Constraint |
|---|---|
| atlas `PBS-006` | current spoken responses took **9 to 21 s**; target stated in-repo: **median under 1.2 s** |
| opencode-web-stream | speech confirmed at **320 ms**, end-of-utterance hangover **850 ms**, injection delay **1.5 s**, false barge-in 2.2 s, beat window 2.5 s |
| UAF paper (arXiv 2604.19221) | human comfort zone for full-duplex: **200 to 500 ms** |
| barge-in, all three projects | must stop within **one frame**, no round trip |

So the budget is hierarchical, and it is a design constraint not a tuning goal:

```text
  barge-in / stop          < 1 frame        local, non-negotiable, no model
  gate + identity          ~10s of ms       local, cheap, always on
  first audible sound      target < 1.2 s    semantic, streamed, model allowed
  deep work                seconds to hours  async, announced, never blocking
```

**Design rule: anything the human perceives as "reaction" must be local. Anything that
requires judgement may be remote or slow, as long as it is announced.**

---

## 9. Two architectural schools, and the wake-word inversion

### 9.1 Cascade vs speech-to-speech

Picovoice's current framing, and it matches our needs: the cascade (VAD -> gate -> ASR ->
LLM -> TTS) survives because **every stage boundary is an inspection point**, which is
where policy, redaction, provenance and audit live. Full speech-to-speech removes those
boundaries and has to reinvent debugging, guardrails, compliance and evaluation.
Our projects are built *on* those boundaries (meeting-colab's provenance and TTL,
atlas's typed ports). **Stay cascade. Borrow S2S only if it wins on measured latency.**

Counter-current to watch: **UAF** (arXiv 2604.19221), a unified audio front-end LLM that
does AEC + ANS + VAD + speaker recognition + turn-taking in one generative framework. It
beats TEN-VAD, Silero-VAD and FSMN-VAD on speaker-aware low-SNR conditions. It is
research, not a dependency, but it is the honest signal that the cascade has a ceiling.

### 9.2 The power cascade, and why the wake word is the real prize

Industry standard multi-stage wakeup (Ambiq, and the pattern behind every assistant):
**a cheap always-on VAD triggers a small KWS model, which triggers the expensive model**.
Numbers from the ultra-low-power end: under 200 KB RAM, **~500 microjoules per inference**,
KWS around **50 ms**. `nanowakeword` states the same trade explicitly: the gate filters
**~99 % of audio** before anything expensive happens, and this is exactly why Google
Assistant and Alexa keep a tiny model on-device.

Then the inversion that this whole dossier is aimed at:

> "OK Google" exists because (a) the machine cannot tell you from anyone else, and
> (b) it cannot afford to run ASR continuously. **Both premises die** once a
> speaker-conditioned gate exists. **The voiceprint itself becomes the wake word.**

The trigger stops being an acoustic beacon (a shout the whole room hears) and becomes a
**state decision**: this is the enrolled voice, and this is addressed to the system, and
here it is useful. Which is precisely the arbiter problem from `../IDEAS.md` section 5.

### 9.3 Prior art that does exactly this

**`QwenAudio/qwen-audio-agent`** is the closest existing thing to π's vocal + agentic
poles, and the top clone candidate:

> "*A realtime full-duplex voice runtime for AI coding agents, talk to Claude Code,
> Codex, OpenCode and more like a phone call... delegating deep work to your backend
> agent through the **open ACP protocol**, then announcing results back into the
> conversation. It uses sherpa-onnx keyword spotting with the **3M-parameter zh-en
> transducer model** to implement a local voice wake word: the app sleeps on idle but
> **keeps the microphone open**, and the wake phrase brings the session back with
> **no cloud calls**.*"

Other references worth reading (not necessarily using):
`agalue/sherpa-voice-assistant` (Go, sherpa-onnx + Ollama + wake word + interrupt),
`Fono` (GPL-3.0, one binary: STT + TTS + local LLM + wake word + **speaker ID**),
`Open-LLM-VTuber` (hands-free with interruption, Live2D),
`reindent/jauvex` (Electron client, a nice pattern: it answers **in three beats**, one
quick word, then what it understood, then a summary of the agent's answer),
`voicute/onnx-wakeword` (pure offline KWS runtime).

---

## 10. Anti-spoofing: the "0 trust" obligation

If the architecture says "react only to my voiceprint", then **playing a recording of the
owner's voice is an attack**. This is not paranoia, it is the direct consequence of the
design.

- **ASVspoof 5** (IEEE TASLP, 2026, arXiv 2601.03944) is the reference: 53 teams,
  crowdsourced data from **~2000 speakers**, **32 attack algorithms**, seven
  speaker-disjoint partitions, adversarial attacks included for the first time.
  Two tracks: standalone countermeasure (CM), and **SASV** (spoofing-robust ASV), which
  must accept *only* genuine target trials.
- Honest state of the art: results "degrade under adversarial attacks and under neural
  encoding/compression".
- **Scope note:** ASVspoof 5 covers *post-sensor* logical-access attacks. A **replay**
  (someone holding a speaker to the microphone) is an earlier, simpler stage and needs
  its own, cheaper detector.

**Minimum viable posture for π:** no decision that matters (unlocking, sending, deleting,
spending) may rest on the voiceprint alone. The voiceprint gates *attention*; it must not
gate *authority*.

---

## 11. The runtime pivot: why `sherpa-onnx`

If we want "same thing on the phone and on the PC" without stitching five projects
together, there is one obvious candidate.

| Property | Value |
|---|---|
| Licence | **Apache-2.0** |
| Runtime | **ONNX Runtime, no Internet connection**, next-gen Kaldi |
| Functions | ASR (streaming **and** offline), **TTS**, **speaker diarization**, **speaker identification**, **speaker verification**, spoken language ID, audio tagging, **VAD (Silero)**, **keyword spotting**, punctuation, **speech enhancement (GTCRN, DPDFNet)**, source separation |
| Platforms | embedded, Android, iOS, HarmonyOS, Raspberry Pi, RISC-V, RK NPU, Axera NPU, Ascend NPU, x86_64, websocket server/client, WebAssembly |
| Bindings | **12 languages**: C++, C, Python, JavaScript, Java, C#, Kotlin, Swift, Go, Dart, Rust, Pascal |
| Ref models | `sherpa-onnx-pyannote-segmentation-3-0` (5.7 MB, int8 1.5 MB), `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`, `nemo_en_titanet_small.onnx`, `sherpa-onnx-reverb-diarization-v1` |

**One runtime, one contract, two platforms, zero SaaS.** Caveat to verify by measurement:
sherpa-onnx's diarization is exposed *offline*; VAD, KWS and ASR are streaming. So the
live path is VAD -> KWS/identity -> streaming ASR, with diarization as an offline pass.

Escape hatch if a specific model is missing: `3D-Speaker` (CAM++, ERes2Net) exports ONNX
directly, and `whisper.cpp` already handles ASR and diarization locally
(`--diarize` for stereo, `--tinydiarize` with a `tdrz` model).

---

## 12. Current state of this machine (2026-10-18)

Already installed and working, nothing to set up:

| Asset | Path / state |
|---|---|
| Whisper models | `~/.local/share/whisper-cpp/`: large-v3-turbo-q5_0 (574 MB), small, small.en, tiny, tiny.en |
| **Silero VAD (GGML)** | `~/.local/share/whisper-cpp/ggml-silero-v5.1.2.bin` (885 KB), **installed, unused** |
| `whisper-cli` | supports `--vad`, `--diarize`, `--tinydiarize`, `--ov-e-device` (OpenVINO) |
| Local LLM server | OpenVINO on `127.0.0.1:11437`, serving `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-vl-2b-instruct` |
| TTS | `piper` and `k-piper` in `~/.local/bin` |
| Capture | PipeWire (`pw-record`, `wpctl`) |
| GPU | Intel Arc iGPU present (`/dev/dri/card1`, `renderD128`) |

Absent and needed for layer 3: any speaker-embedding model, any ONNX runtime for it,
no `sherpa-onnx`, no `speaches`.

**Rule from `~/.agents/AGENTS.md`: no `pip`, `uv` only, and no new installs without
an explicit request.** Slice A needs no install at all.

---

## 13. Sources

- Personal VAD, Google, Odyssey 2020, https://arxiv.org/abs/1908.04284
- `personal-vad` open reproduction (Apache-2.0), https://pypi.org/project/personal-vad
- Short-enrollment PVAD, ICASSP 2026, https://arxiv.org/abs/2601.12769
- USEF-TP, joint TSE + PVAD, https://arxiv.org/abs/2501.03612
- TargetVoice, Interspeech 2025, https://www.isca-archive.org/interspeech_2025/pallala25_interspeech.pdf
- CAM++, https://arxiv.org/html/2303.00332 ; ERes2NetV2, https://arxiv.org/html/2406.02167
- 3D-Speaker toolkit, https://github.com/modelscope/3D-Speaker
- Whispered ASR, Whisper-Aware LLM, https://arxiv.org/html/2608.10836v1
- Whispered ASR, Mamba SSM, https://arxiv.org/html/2506.16969v2
- Whispered ASR, WavLM on CHAINS, https://arxiv.org/html/2407.21211v2
- Whispered SV + RMS gap, https://arxiv.org/html/2604.20229
- SV under neural codecs, https://arxiv.org/pdf/2509.02771
- DiariZen, https://github.com/BUTSpeechFIT/DiariZen ; explained, https://arxiv.org/abs/2604.21507
- Diarization benchmark, ETH, https://arxiv.org/pdf/2509.26177v1
- pyannote Community-1, https://www.pyannote.ai/blog/community-1
- ASVspoof 5, https://arxiv.org/abs/2601.03944
- sherpa-onnx, https://github.com/k2-fsa/sherpa-onnx
- qwen-audio-agent (voice + ACP for coding agents), https://github.com/QwenAudio/qwen-audio-agent
- UAF, unified audio front-end LLM, https://arxiv.org/pdf/2604.19221v2
- Wake-word cascade and power budget, https://ambiq.com/blog/practical-speech-interfaces-using-ai-a-word-is-worth-a-thousand-pictures
- Cascade vs speech-to-speech, https://picovoice.ai/guide/voice-agents/future-of-voice-agents/
- Deployment cascade for wake word, https://github.com/arcosoph/nanowakeword
- SottoVoce, ultrasound silent speech, https://arxiv.org/pdf/2303.01758v1
- Lip-Siri, Wi-Fi backscatter silent speech, https://arxiv.org/pdf/2601.18177
- AlterEgo / sEMG review, https://pmc.ncbi.nlm.nih.gov/articles/PMC7922488
