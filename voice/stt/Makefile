.DEFAULT_GOAL := help
SHELL := /bin/zsh

UV := $(shell command -v uv 2>/dev/null || echo uv)
JQ := $(shell command -v jq 2>/dev/null || echo jq)
HF := $(shell command -v hf 2>/dev/null || echo hf)

# ===========================================================================
# π-ear: the streaming ear of π. Setup, IN THIS ORDER:
#
#   1. make sync      build the base env: numpy + sherpa-onnx + typer.
#                     uv resolves the graph itself, never a hand-written pin.
#   2. make models    pull the only download of the POC: the CAM++ speaker
#                     embedding ONNX, 27.4 MB. Needed from v1 (who is speaking);
#                     v0 (the state moving on disk) needs no download at all.
#   3. make rec       PANE 1: the recorder. Endless one-second raw chunks.
#                     Started once, NEVER stopped: it is the ear's memory.
#   4. make run       PANE 2: the ear. Follows the chunks and keeps
#                     $(STATE) live, appends $(EVENTS).
#   5. make watch     PANE 3: watch the state move.
#
#   at any time: make test (no microphone, no model) and make check (ruff)
#   then: make pause / resume / snapshot / diff / reset / replay / clean
#
#   THE DURABLE CORPUS: while the ear runs, every chunk it ingests is copied to
#   $(CORPUS_DIR) (bounded, default one hour), because $(ROOT) is RAM and the
#   audio directory is emptied as it is consumed. Without it a session survives
#   only as numbers and can never be run again. make corpus shows it,
#   make replay-corpus replays a recorded session with other parameters.
#
# SYSTEM PACKAGES THAT WOULD NEED sudo: deliberately NO make target for them.
# Elevation never goes through a Makefile or an agent shell, it goes through an
# interactive tmux pane (k-tmux). Comment only, on purpose:
#
#   - a DIGITAL loopback, to capture a video's audio without the room noise:
#       sudo apt install pipewire-pulse     # provides pactl
#       then `pactl list short sources` lists the monitor sources.
#     Nothing is blocked today: pactl is ABSENT from this machine, and the
#     microphone already captures the voice AND the speakers in ONE stream,
#     which is exactly the mixture the POC has to survive.
#   - arecord, ffmpeg, sox, watch, jq, pw-record: all present, nothing to do.
# ===========================================================================

# ---- runtime root: throwaway by design, and it is tmpfs (RAM) on this box ----
ROOT ?= /tmp/π
STATE := $(ROOT)/state.json
EVENTS := $(ROOT)/events.jsonl
AUDIO := $(ROOT)/audio
MODELS ?= $(HOME)/.cache/π/models

# ---- capture contract: must match what `π-ear run` is told ------------------
DEVICE ?= plughw:0,6
RATE ?= 16000
CHANNELS ?= 1
CHUNK_S ?= 1

# ---- inventory knobs (same defaults as src/π_ear/config.py) ----------------
# The admission criterion is a MEASUREMENT, not a counter: TAU_FLOOR is the cosine
# the closest voice must clear, Z_MARGIN is how many sigma above the observed
# margin noise the winner must be. Nothing grows with the inventory any more.
WINDOW_S ?= 3.0
K_EVIDENCE ?= 3
TAU_FLOOR ?= 0.5
Z_MARGIN ?= 3.0
VOICED_FLOOR ?= 0.25

# The durable corpus: the audio of a session, bounded, in the home, not in RAM.
CORPUS_DIR ?= $(HOME)/.cache/π/corpus
CORPUS_MAX_CHUNKS ?= 3600

EAR_ARGS := --root $(ROOT) --device $(DEVICE) --rate $(RATE) --chunk-s $(CHUNK_S) \
	--window-s $(WINDOW_S) --k-evidence $(K_EVIDENCE) \
	--tau-floor $(TAU_FLOOR) --z-margin $(Z_MARGIN) \
	--voiced-floor $(VOICED_FLOOR) \
	--corpus-dir $(CORPUS_DIR) --corpus-max-chunks $(CORPUS_MAX_CHUNKS)

# Extra flags for `make run`, e.g. make run RUN_FLAGS=--keep-audio
RUN_FLAGS ?=

.PHONY: help sync sync-cpu sync-xpu sync-cuda models check test rec run replay \
	watch events state status pause resume reset snapshot diff corpus replay-corpus clean

help:  ## Show the setup order and every target
	@echo "π-ear setup order: sync -> models -> rec + run + watch (three panes)"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

sync:  ## [setup 1] Base env: numpy + sherpa-onnx + typer, resolved by uv
	@$(UV) sync
	@echo "base env ready"

sync-cpu:  ## [unused today] Base + CPU torch. No-op: the cpu extra is empty
	@$(UV) sync --extra cpu
	@echo "cpu extra is empty: nothing installed (see pyproject.toml)"

sync-xpu:  ## [unused today] Base + Intel Arc torch. No-op while xpu is empty
	@$(UV) sync --extra xpu
	@echo "xpu extra is empty: nothing installed (see pyproject.toml)"

sync-cuda:  ## [unused today] Base + CUDA torch. No-op while cuda is empty
	@$(UV) sync --extra cuda
	@echo "cuda extra is empty: nothing installed (see pyproject.toml)"

models:  ## [setup 2] Pull the CAM++ speaker-embedding ONNX (27.4 MB), once
	@mkdir -p $(MODELS)
	$(HF) download csukuangfj/speaker-embedding-models \
	  3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx --local-dir $(MODELS)
	@echo "model ready: $(MODELS)/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
	# The English / VoxCeleb variant (44.8 MB), to compare on real French
	# voices. Kept as a comment, not a target, on purpose: nothing needs it yet.
	#   hf download csukuangfj/speaker-embedding-models \
	#     3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx --local-dir $(MODELS)

check:  ## Lint then format then lint again (ruff)
	@$(UV) run ruff check --select I --fix src tests
	@$(UV) run ruff format src tests
	@$(UV) run ruff check src tests
	@echo "make check passed cleanly"

test:  ## Run the test suite: no microphone, no model, no download
	@$(UV) run pytest

rec:  ## [setup 3, PANE 1] Recorder: endless 1 s raw chunks. NEVER stopped.
	@mkdir -p $(AUDIO)
	@echo "recording $(DEVICE) $(RATE)Hz $(CHANNELS)ch -> $(AUDIO)/*.raw"
	arecord -D $(DEVICE) -f S16_LE -r $(RATE) -c $(CHANNELS) -t raw \
	  --use-strftime --max-file-time $(CHUNK_S) \
	  $(AUDIO)/%Y%m%d-%H%M%S.raw

run:  ## [setup 4, PANE 2] Ear: follows the chunks, keeps the state live
	@$(UV) run π-ear run $(EAR_ARGS) $(RUN_FLAGS)

replay:  ## Re-run the whole backlog from scratch (needs a run --keep-audio)
	@$(UV) run π-ear replay --root $(ROOT)

corpus:  ## What the durable corpus holds, and how much space it uses
	@echo "corpus: $(CORPUS_DIR)"
	@ls -l --time-style=+%H:%M:%S $(CORPUS_DIR) 2>/dev/null | tail -3
	@du -sh $(CORPUS_DIR) 2>/dev/null || echo "  empty (nothing ingested yet)"

replay-corpus:  ## Re-run a whole recorded session from the durable corpus
	@$(UV) run π-ear replay --root $(ROOT) --audio-dir $(CORPUS_DIR)

watch:  ## [setup 5, PANE 3] Watch the state on disk move
	@watch -n 0.5 "$(JQ) . $(STATE)"

events:  ## Follow the append-only event log (one line per window)
	@tail -f $(EVENTS)

state:  ## Print the current state once
	@$(JQ) . $(STATE)

status:  ## Summarise the state, human readable
	@$(UV) run π-ear ctl status --root $(ROOT)

pause:  ## Freeze the ear. The recorder keeps going: the audio becomes a backlog.
	@$(UV) run π-ear ctl pause --root $(ROOT)

resume:  ## Unfreeze: the ear drains the backlog and catches up
	@$(UV) run π-ear ctl resume --root $(ROOT)

reset:  ## Ask the ear to archive its state and restart from the backlog
	@$(UV) run π-ear ctl reset --root $(ROOT)

snapshot:  ## Freeze a comparable copy of the state
	@$(UV) run π-ear snapshot --root $(ROOT)

diff:  ## Diff the two most recent snapshots: the visible proof that it moved
	@$(UV) run π-ear snapshot --diff --root $(ROOT)

clean:  ## Wipe the runtime root (audio, state, snapshots). Trash-aware `rm`.
	@rm -rf $(ROOT)
	@echo "$(ROOT) wiped"
