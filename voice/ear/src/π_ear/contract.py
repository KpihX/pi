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
# What was said (asr.py): international by default, measured 2026-10-20 (IDEAS.md,
# "v2, the what"). FAST hears first and is re-run on the speech in progress, so the
# text appears while it is said; SLOW hears again in the silences. Any registry name,
# any "module:factory", or "none".
FAST = os.environ.get("PI_EAR_FAST", "parakeet-v3")
SLOW = os.environ.get("PI_EAR_SLOW", "whisper-turbo")
LANGUAGE = os.environ.get("PI_EAR_LANGUAGE", "")  # empty: the recognisers detect it
# Voice out of music before the slow reading (Q6). Off: measured 2026-10-20, the
# sherpa-onnx Spleeter int8 stems came out 130 times louder than their input and
# uncorrelated with the voice (WER 3 % -> 18 % at 0 dB music), while Parakeet and
# Whisper turbo read through the music on their own (3 % and 5 % at 0 dB). The
# slot stays: "sherpa-onnx-spleeter-2stems-int8", or any separator that works.
SEPARATOR = os.environ.get("PI_EAR_SEPARATOR", "none")
# kept for the base: its archive carries the LibriSpeech test files
WHISPER_DIR = "sherpa-onnx-whisper-tiny"

# --- formats -----------------------------------------------------------------
KEY_BYTES = 8  # a speaker key is 16 hex characters
TEXT_GRAM = 3  # characters per n-gram of the text language model (language.py)

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
LIVE = Path(
    os.path.expanduser(os.environ.get("PI_EAR_LIVE", str(DATA / "live")))
)  # never committed
WEIGHTS = Path(__file__).parent / "weights"


def chunk_bytes() -> int:
    """A complete chunk, in bytes: what "the recorder moved on" means."""
    return RATE * CHANNELS * CHUNK_S * SAMPLE_BYTES
