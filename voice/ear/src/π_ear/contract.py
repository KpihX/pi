"""The contract: every number in this package that is NOT a decision.

The second law of π forbids a hand-set value in any decision path. It does not
forbid a format. Three kinds of number live here, and nowhere else:

* the **capture contract**: 16 kHz mono s16le in one-second chunks. That is
  what the recorder writes and what every model downstream was trained on;
* the **sensor's own operating points**: the frozen encoder and the frozen VAD
  are used the way their authors published them, exactly like their weights.
  ``ENCODER_WINDOW_S`` is the one value chosen here, and it was measured, not
  guessed (``voice/ear/IDEAS.md``, "Sensor choice"): CAM++ on 1.0 s of speech
  gives an EER of 11.8 % on real LibriSpeech voices, 0.0 % on 2.0 s. Below that
  length an embedding is a glimpse, decided but never learned from;
* **resource bounds**: how much disk a store may take, how often the loop
  wakes up. They limit cost, they never choose an answer.

``tests/test_law.py`` holds the line: the decision modules (``model.py``)
contain no numeric literal other than 0 and 1 and compare nothing against a
constant. Every number they use comes from the weights on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- capture --------------------------------------------------------------
RATE = 16000
CHANNELS = 1
CHUNK_S = 1
SAMPLE_BYTES = 2
DEVICE = os.environ.get("PI_EAR_DEVICE", "plughw:0,6")

# --- sensors (frozen, published operating points) --------------------------
ENCODER = os.environ.get("PI_EAR_ENCODER", "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
ENCODER_WINDOW_S = 2.0
VAD_MODEL = "silero_vad.onnx"
# Measured 2026-10-19 (voice/ear/IDEAS.md, "Transcription"): WER on French speech
# tiny 16 %, base 9 %, small 3 %; small costs 0.22 s per second of audio with the
# language given, 0.54 s without, and runs off the decision loop.
WHISPER = os.environ.get("PI_EAR_WHISPER", "small")
WHISPER_DIR = f"sherpa-onnx-whisper-{WHISPER}"
WHISPER_LANGUAGE = os.environ.get("PI_EAR_LANGUAGE", "")  # empty: whisper detects it

# --- formats -----------------------------------------------------------------
KEY_BYTES = 8  # a speaker key is 16 hex characters

# --- resources --------------------------------------------------------------
IDLE_SLEEP_S = 0.1
PERSIST_EVERY_S = 1.0
CORPUS_MAX_CHUNKS = 3600
THREADS = 2

# --- places -----------------------------------------------------------------
HOME = Path(os.path.expanduser(os.environ.get("PI_EAR_HOME", "~/.cache/π")))
ROOT = Path(os.environ.get("PI_EAR_ROOT", "/tmp/π"))
MODELS = HOME / "models"
DATA = HOME / "data"
MEMORY = HOME / "ear"
CORPUS = HOME / "corpus"
WEIGHTS = Path(__file__).parent / "weights"


def chunk_bytes() -> int:
    """A complete chunk, in bytes: what "the recorder moved on" means."""
    return RATE * CHANNELS * CHUNK_S * SAMPLE_BYTES
