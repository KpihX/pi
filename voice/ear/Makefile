.DEFAULT_GOAL := help
SHELL := /bin/bash

UV := $(shell command -v uv 2>/dev/null || echo uv)
JQ := $(shell command -v jq 2>/dev/null || echo jq)

# ===========================================================================
# π-ear v1: the streaming ear of π. Setup, IN THIS ORDER:
#
#   1. make sync      the env: numpy + sherpa-onnx (+ its onnxruntime) + typer
#   2. make fetch     the models (Silero, CAM++, Whisper tiny, Piper voices) and
#                     the public test corpora: ~1.1 GB once, ~850 MB kept, ~/.cache/π
#   3. make rec       PANE 1: the recorder, endless 1 s chunks. NEVER stopped.
#   4. make run       PANE 2: the ear. One line per decision, state live on disk.
#   5. make watch     PANE 3: the state moving.
#
#   then: make status / events / correct / pause / resume / snapshot / diff /
#         reset / corpus / replay-corpus / test / check / base
#
# Nothing here sets a judgement: there is no threshold to pass, because the ear
# has none (tests/test_law.py). The variables below are places and formats.
# sudo is never needed and never goes through a Makefile.
# ===========================================================================

ROOT ?= /tmp/π
MEMORY ?= $(HOME)/.cache/π/ear
CORPUS_DIR ?= $(HOME)/.cache/π/corpus
STATE := $(ROOT)/state.json
EVENTS := $(ROOT)/events.jsonl
AUDIO := $(ROOT)/audio

DEVICE ?= plughw:0,6
RATE := 16000
CHANNELS := 1
CHUNK_S := 1

RUN_FLAGS ?=
EAR := $(UV) run π-ear

.PHONY: help sync fetch base check test test-fast rec run watch events state status \
	pause resume reset snapshot diff correct undo why probe corpus replay-corpus clean

help:  ## Show the setup order and every target
	@echo "π-ear setup order: sync -> fetch -> rec + run + watch (three panes)"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

sync:  ## [setup 1] The env, resolved by uv
	@$(UV) sync

fetch:  ## [setup 2] Models + public corpora into ~/.cache/π (~1.1 GB, once)
	@$(EAR) fetch all

base:  ## Re-measure the base weights on public voices and sounds the tests never use
	@$(EAR) base

check:  ## ruff: imports, format, lint
	@$(UV) run ruff check --select I --fix src tests
	@$(UV) run ruff format src tests
	@$(UV) run ruff check src tests

test:  ## Every test: the law, the maths, the model, the server, the simulated rooms
	@$(UV) run pytest -q

test-fast:  ## Only what needs no model and no download (< 1 s)
	@$(UV) run pytest -q tests/test_law.py tests/test_sphere.py tests/test_model.py tests/test_io.py

rec:  ## [setup 3, PANE 1] Recorder: endless 1 s raw chunks. NEVER stopped.
	@mkdir -p $(AUDIO)
	arecord -D $(DEVICE) -f S16_LE -r $(RATE) -c $(CHANNELS) -t raw \
	  --use-strftime --max-file-time $(CHUNK_S) $(AUDIO)/%Y%m%d-%H%M%S.raw

run:  ## [setup 4, PANE 2] The ear: follows the chunks, keeps the state live
	@$(EAR) run --root $(ROOT) --memory $(MEMORY) --corpus-dir $(CORPUS_DIR) $(RUN_FLAGS)

watch:  ## [setup 5, PANE 3] Watch the state on disk move
	@watch -n 0.5 "$(JQ) '{now, model, speakers}' $(STATE)"

events:  ## Follow the append-only record
	@tail -f $(EVENTS)

state:  ## Print the state once
	@$(JQ) . $(STATE)

status:  ## Human-readable summary
	@$(EAR) ctl status --root $(ROOT)

correct:  ## One correction: make correct C="different 812 901" (see README)
	@$(EAR) correct $(C) --memory $(MEMORY)

undo:  ## Forget one correction exactly: make undo N=3 (its `line` in the record)
	@$(EAR) undo $(N) --memory $(MEMORY)

why:  ## What the ear decided for one window, and why: make why ROW=812
	@$(EAR) why $(ROW) --root $(ROOT) --memory $(MEMORY)

probe:  ## The corpus under the decisions: make probe FROM=84 TO=163 [ASR=--asr]
	@$(EAR) probe $(CORPUS_DIR) --from $(FROM) --to $(TO) --root $(ROOT) $(ASR)

pause:  ## Freeze the ear; the recorder keeps going and the audio becomes a backlog
	@$(EAR) ctl pause --root $(ROOT)

resume:  ## Drain the backlog and catch up
	@$(EAR) ctl resume --root $(ROOT)

reset:  ## Forget everything learned (archived under $(ROOT)/archive, never deleted)
	@$(EAR) ctl reset --root $(ROOT)

snapshot:  ## Freeze a comparable copy of the state
	@$(EAR) snapshot --root $(ROOT)

diff:  ## Diff the two last snapshots: the visible proof that it moved
	@$(EAR) snapshot --diff --root $(ROOT)

corpus:  ## What the durable corpus holds
	@ls $(CORPUS_DIR) 2>/dev/null | tail -3; du -sh $(CORPUS_DIR) 2>/dev/null || echo "empty"

replay-corpus:  ## Replay the recorded corpus into a fresh memory (nothing learned is touched)
	@$(EAR) replay $(CORPUS_DIR) --root $(ROOT)/replay

clean:  ## Wipe the live root (the memory and the corpus are elsewhere and survive)
	@rm -rf $(ROOT)
