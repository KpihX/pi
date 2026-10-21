"""The recognisers: *what* was said. Frozen sensors, pluggable, measured.

Every recogniser is a callable ``samples -> Hypothesis``: a text, its words with
their times and their log-probabilities when the model gives them, and the
language. Nothing here decides which reading is right; that is ``words.py``.

International by default, measured 2026-10-20 on real and synthetic speech
(``voice/ear/IDEAS.md``, "v2, the what"):

    recogniser            WER en   WER fr   s per s of audio   words invented on noise
    parakeet-v3 (fast)     3 %      6 %         0.16                2 / 10
    whisper-turbo (slow)   2 %      9 %         1.14                8 / 10
    whisper-small          2 %     19 %         0.89                6 / 10
    canary-180m            1 %     15 %         0.25                2-9 / 10

Parakeet TDT 0.6b v3 (25 European languages, punctuation, word times) hears
first; Whisper turbo (99 languages) hears again in the silences. Two different
models on purpose: where they disagree is where the ear does not know.

Anything else plugs in by name (``REGISTRY``) or by import path
(``PI_EAR_FAST=mypackage.module:factory``): a factory takes the models directory
and returns a callable ``samples -> Hypothesis``, optionally ``language=`` too.
"""

from __future__ import annotations

import importlib
import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from π_ear import contract


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    t0: float
    t1: float
    logp: float | None  # mean token log-probability, when the recogniser gives one


@dataclass(frozen=True, slots=True)
class Hypothesis:
    text: str
    words: tuple[Word, ...] = field(default_factory=tuple)
    language: str | None = None
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "source": self.source,
            "words": [[w.text, round(w.t0, 2), round(w.t1, 2), w.logp] for w in self.words],
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> Hypothesis | None:
        if not raw:
            return None
        words = tuple(Word(t, a, b, p) for t, a, b, p in raw.get("words", []))
        return cls(raw["text"], words, raw.get("language"), raw.get("source", ""))


Recognizer = Callable[[np.ndarray], Hypothesis]


def takes_language(recognizer) -> bool:
    try:
        parameters = inspect.signature(recognizer).parameters
    except (TypeError, ValueError):
        return False
    return "language" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def read(recognizer, samples: np.ndarray, language: str | None = None) -> Hypothesis:
    """Call any recogniser; the language goes only to those that take one."""
    if language is not None and takes_language(recognizer):
        return recognizer(samples, language=language)
    return recognizer(samples)


def words_of(tokens, times, logps, end: float) -> tuple[Word, ...]:
    """Tokens to words: a token that starts with a space (or ``▁``) starts a word."""
    words: list[Word] = []
    current, start, scores = "", 0.0, []

    def at(index: int) -> float:
        return times[index] if index < len(times) else end

    for index, token in enumerate(tokens):
        begins = token.startswith((" ", "▁")) or not current
        if begins and current.strip():
            words.append(_word(current, start, at(index), scores))
            current, scores = "", []
        if begins and not current:
            start = at(index)
        current += token.replace("▁", " ")
        if logps is not None and index < len(logps):
            scores.append(float(logps[index]))
    if current.strip():
        words.append(_word(current, start, end, scores))
    return tuple(words)


def words_of_segments(texts, starts, end: float) -> tuple[Word, ...]:
    """Whisper's sentences with their start times, to words: each sentence's words share
    its span evenly (sherpa-onnx's Whisper gives segment times, not token times). Coarse,
    but enough to give each piece of a long reading the words said during it."""
    words: list[Word] = []
    bounds = list(starts) + [end]
    for text, t0, t1 in zip(texts, bounds, bounds[1:], strict=False):
        said = text.split()
        step = (max(t1, t0) - t0) / max(len(said), 1)
        words += [Word(w, t0 + i * step, t0 + (i + 1) * step, None) for i, w in enumerate(said)]
    return tuple(words)


def _word(text: str, t0: float, t1: float, scores: list[float]) -> Word:
    return Word(text.strip(), float(t0), float(t1), float(np.mean(scores)) if scores else None)


class Sherpa:
    """Any sherpa-onnx offline recogniser, wrapped into a ``Recognizer``.

    ``language``: the model's own setting (``whisper.language`` or Canary's source and
    target), changed in place per call and back (measured: no cost), so one loaded
    model serves every voice in its own language. Models without one ignore it.
    """

    def __init__(self, name: str, recognizer, language: str | None = None, setting=None):
        self.name = name
        self.recognizer = recognizer
        self.language = language
        self.setting = setting  # config -> the object and attribute(s) that hold the language
        self.current = language or ""
        self.lock = threading.Lock()  # one model, shared: its language is set per call

    def _set(self, language: str) -> None:
        if self.setting is None or language == self.current:
            return
        config = self.recognizer.config
        for holder, attribute in self.setting(config):
            setattr(holder, attribute, language)
        self.recognizer.recognizer.set_config(config)
        self.current = language

    def __call__(self, samples: np.ndarray, language: str | None = None) -> Hypothesis:
        with self.lock:
            return self._read(samples, language)

    def _read(self, samples: np.ndarray, language: str | None) -> Hypothesis:
        self._set(language or self.language or "")
        stream = self.recognizer.create_stream()
        stream.accept_waveform(contract.RATE, samples)
        self.recognizer.decode_stream(stream)
        result = stream.result
        logps = list(result.ys_log_probs) if len(result.ys_log_probs) else None
        end = len(samples) / contract.RATE
        if len(result.timestamps) or not len(result.segment_timestamps):
            words = words_of(list(result.tokens), list(result.timestamps), logps, end)
        else:
            words = words_of_segments(
                list(result.segment_texts), list(result.segment_timestamps), end
            )
        language = (result.lang or "").strip("<|>") or self.language
        return Hypothesis(result.text.strip(), words, language, self.name)


def _parakeet(models: Path) -> Sherpa:
    import sherpa_onnx

    root = models / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    return Sherpa(
        "parakeet-v3",
        sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(root / "encoder.int8.onnx"),
            decoder=str(root / "decoder.int8.onnx"),
            joiner=str(root / "joiner.int8.onnx"),
            tokens=str(root / "tokens.txt"),
            num_threads=contract.THREADS,
            model_type="nemo_transducer",
        ),
    )


def _whisper(size: str):
    def build(models: Path) -> Sherpa:
        import sherpa_onnx

        root = models / f"sherpa-onnx-whisper-{size}"
        return Sherpa(
            f"whisper-{size}",
            sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(root / f"{size}-encoder.int8.onnx"),
                decoder=str(root / f"{size}-decoder.int8.onnx"),
                tokens=str(root / f"{size}-tokens.txt"),
                language=contract.LANGUAGE,
                num_threads=contract.THREADS,
                enable_segment_timestamps=True,
            ),
            contract.LANGUAGE or None,
            lambda config: [(config.model_config.whisper, "language")],
        )

    return build


def _canary(models: Path) -> Sherpa:
    """Canary needs its language: the ear's ``PI_EAR_LANGUAGE``, English otherwise."""
    import sherpa_onnx

    root = models / "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8"
    language = contract.LANGUAGE or "en"
    return Sherpa(
        "canary-180m",
        sherpa_onnx.OfflineRecognizer.from_nemo_canary(
            encoder=str(root / "encoder.int8.onnx"),
            decoder=str(root / "decoder.int8.onnx"),
            tokens=str(root / "tokens.txt"),
            src_lang=language,
            tgt_lang=language,
            num_threads=contract.THREADS,
        ),
        language,
        lambda config: [
            (config.model_config.canary, "src_lang"),
            (config.model_config.canary, "tgt_lang"),
        ],
    )


REGISTRY: dict[str, tuple[str, Callable[[Path], Recognizer]]] = {
    "parakeet-v3": ("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", _parakeet),
    "whisper-turbo": ("sherpa-onnx-whisper-turbo", _whisper("turbo")),
    "whisper-small": ("sherpa-onnx-whisper-small", _whisper("small")),
    "whisper-base": ("sherpa-onnx-whisper-base", _whisper("base")),
    "canary-180m": ("sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8", _canary),
}
# The longest audio a model reads whole: Whisper's encoder sees 30 s, and sherpa-onnx
# drops the rest ("we process only the first 30 seconds"; measured live: a 70.9 s run
# came back with the words of its first 30 s). The scribe cuts pieces strictly shorter.
WINDOW_S = {"whisper-turbo": 30.0, "whisper-small": 30.0, "whisper-base": 30.0}


def window_s(recognizer) -> float | None:
    """The longest audio this recogniser reads whole, if it has one."""
    return getattr(recognizer, "window_s", None) or WINDOW_S.get(getattr(recognizer, "name", ""))


def available(name: str, models: Path) -> bool:
    if not name or name == "none":
        return False
    if ":" in name:
        return True
    return name in REGISTRY and (models / REGISTRY[name][0]).exists()


_LOADED: dict[tuple[str, str], Recognizer] = {}
_LOADING = threading.Lock()


def build(name: str, models: Path) -> Recognizer | None:
    """A recogniser by registry name or ``module:factory``; None if absent or ``none``.

    One instance per model and process: Whisper turbo is a gigabyte, and every ear that
    loaded its own (tests, a reset) ran this container out of memory (measured)."""
    if not available(name, models):
        return None
    with _LOADING:
        key = (name, str(models))
        if key not in _LOADED:
            if ":" in name:
                module, factory = name.split(":", 1)
                _LOADED[key] = getattr(importlib.import_module(module), factory)(models)
            else:
                _LOADED[key] = REGISTRY[name][1](models)
        return _LOADED[key]
