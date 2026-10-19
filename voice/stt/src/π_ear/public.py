"""Public voices and sounds, turned into streams the ear can be tested on.

Three uses, one module:

* ``fit_base``: the base weights, the "good bases" of the ear, measured on
  public voices that the tests never use (FSDD jackson, nicolas and theo; the
  two LibriSpeech speakers shipped with Whisper's test files);
* ``Scene``: a recorded-room simulation, written as one-second raw chunks in a
  directory, exactly what ``arecord`` produces, with its ground truth;
* ``score``: what the ear decided, against that truth.
"""

from __future__ import annotations

import collections
import csv
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from π_ear import contract
from π_ear.fetch import BASE_FOLDS, TEST_FOLDS
from π_ear.model import Model, fit_weights

BASE_FSDD = ("jackson", "nicolas", "theo")
TEST_FSDD = ("yweweler", "george", "lucas")


def load(path: Path) -> np.ndarray:
    """Any audio file to 16 kHz mono float32, through ffmpeg."""
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "quiet",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(contract.RATE),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0


def fsdd(data: Path, speaker: str, takes: range) -> list[np.ndarray]:
    """One FSDD speaker, ten digits per take: one utterance of about four seconds per take."""
    return [
        np.concatenate(
            [load(data / "fsdd" / f"{digit}_{speaker}_{take}.wav") for digit in range(10)]
        )
        for take in takes
    ]


def libri(data: Path) -> dict[str, np.ndarray]:
    return {
        f"libri{name.split('-')[0]}": load(data / "libri" / name)
        for name in sorted(p.name for p in (data / "libri").glob("*.ogg"))
    }


def esc50(data: Path, folds=TEST_FOLDS) -> dict[str, list[np.ndarray]]:
    """The indoor sound clips of these folds. The tests only ever hear fold 5."""
    rows = list(csv.DictReader((data / "esc50" / "esc50.csv").open()))
    out: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for row in rows:
        path = data / "esc50" / row["filename"]
        if path.exists() and row["fold"] in folds:
            out[row["category"]].append(load(path))
    return out


def windows(x: np.ndarray) -> list[np.ndarray]:
    n = int(contract.ENCODER_WINDOW_S * contract.RATE)
    return [x[i : i + n] for i in range(0, len(x) - n + 1, n)]


def fit_base(sensor, data: Path, models: Path):
    def recording(x):
        return [sensor.embed(w) for w in windows(x)]

    groups = {s: [recording(u) for u in fsdd(data, s, range(0, 20))] for s in BASE_FSDD}
    tests = models / "sherpa-onnx-whisper-tiny" / "test_wavs"
    groups["libri1089"] = [recording(load(tests / "0.wav"))]
    groups["libri1221"] = [recording(load(tests / "1.wav")), recording(load(tests / "8k.wav"))]
    sizes = {k: sum(len(r) for r in v) for k, v in groups.items()}
    weights = fit_weights(groups, "")
    # What the VAD lets through from 20 indoor sound classes, exactly as the ear
    # would see it: full encoder windows, cut live by the same sensor.
    model = Model(weights, lambda row: None)
    heard = 0
    for clips in esc50(data, BASE_FOLDS).values():
        for clip in clips:
            pieces, _ = sensor.feed(clip)
            tail, _ = sensor.feed(np.zeros(contract.RATE, dtype=np.float32))
            for piece in pieces + tail:
                if piece.full:
                    model.hear_noise(sensor.embed(piece.samples))
                    heard += 1
    weights.noise = model.noise_weights()
    weights.source = (
        f"{contract.ENCODER}; voices: FSDD {'/'.join(BASE_FSDD)} + LibriSpeech 1089/1221, "
        f"windows {sizes}; noise: ESC-50 folds {'/'.join(BASE_FOLDS)}, {heard} windows "
        f"in {len(weights.noise)} clusters"
    )
    return weights


def synth(
    models: Path, speakers: list[int], sentences: list[str], cache: Path | None = None
) -> dict[str, list[np.ndarray]]:
    """Piper LibriTTS-R voices at 16 kHz. VITS samples noise, so the audio is cached:
    a test must hear the same voices every time it runs."""
    import hashlib

    import sherpa_onnx
    from scipy.signal import resample_poly

    cache = cache or contract.DATA / "tts"
    cache.mkdir(parents=True, exist_ok=True)
    root = models / "vits-piper-en_US-libritts_r-medium"
    tts = None

    def engine():
        return sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(root / "en_US-libritts_r-medium.onnx"),
                        tokens=str(root / "tokens.txt"),
                        data_dir=str(root / "espeak-ng-data"),
                    ),
                    num_threads=contract.THREADS,
                )
            )
        )

    out = {}
    for sid in speakers:
        clips = []
        for sentence in sentences:
            digest = hashlib.blake2b(sentence.encode(), digest_size=6).hexdigest()
            path = cache / f"{sid}-{digest}.npy"
            if not path.exists():
                tts = tts or engine()
                audio = tts.generate(sentence, sid=sid, speed=1.0)
                x = np.asarray(audio.samples, dtype=np.float32)
                np.save(path, resample_poly(x, contract.RATE, audio.sample_rate).astype(np.float32))
            clips.append(np.load(path))
        out[f"tts{sid}"] = clips
    return out


@dataclass
class Scene:
    """A simulated room: turns of speech and sounds over a noise floor, with its truth."""

    seed: int = 0
    floor_dbfs: float = -60.0
    audio: list[np.ndarray] = field(default_factory=list)
    truth: list[tuple[float, float, str]] = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.length = 0

    def _floor(self, n: int) -> np.ndarray:
        level = 10 ** (self.floor_dbfs / 20)
        return (self.rng.standard_normal(n) * level).astype(np.float32)

    def silence(self, seconds: float) -> Scene:
        n = int(seconds * contract.RATE)
        self.audio.append(self._floor(n))
        self.length += n
        return self

    def add(self, label: str, x: np.ndarray, gain_db: float = 0.0) -> Scene:
        peak = float(np.max(np.abs(x))) or 1.0
        y = x / peak * 0.5 * 10 ** (gain_db / 20) + self._floor(len(x))
        t0 = self.length / contract.RATE
        self.audio.append(y.astype(np.float32))
        self.length += len(x)
        self.truth.append((t0, self.length / contract.RATE, label))
        return self

    def write(self, directory: Path) -> Path:
        """One-second s16le chunks named like arecord's strftime pattern."""
        directory.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(np.concatenate(self.audio), -1, 1) * 32767).astype("<i2")
        step = contract.RATE * contract.CHUNK_S
        for index, start in enumerate(range(0, len(pcm), step)):
            (directory / f"20261018-{index:06d}.raw").write_bytes(
                pcm[start : start + step].tobytes()
            )
        return directory

    def label_at(self, t0: float, t1: float) -> str:
        """The truth label covering most of [t0, t1], or 'silence'."""
        best, cover = "silence", 0.0
        for a, b, label in self.truth:
            overlap = min(b, t1) - max(a, t0)
            if overlap > cover:
                best, cover = label, overlap
        return best


def score(events: list[dict], scene: Scene, voices: set[str]) -> dict:
    """Purity, fragmentation and phantoms of the speaker attribution."""
    obs = [e for e in events if "row" in e]
    by_truth: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_speaker: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for e in obs:
        truth = scene.label_at(e["t0"], e["t1"])
        key = e["speaker"] if e["kind"] in ("known", "new") else e["kind"]
        by_truth[truth][key] += 1
        if e["kind"] in ("known", "new"):
            by_speaker[e["speaker"]][truth] += 1
    owner = {spk: c.most_common(1)[0][0] for spk, c in by_speaker.items()}
    voiced = [e for e in obs if scene.label_at(e["t0"], e["t1"]) in voices and e["learned"]]
    attributed = [e for e in voiced if e["kind"] in ("known", "new")]
    correct = [e for e in attributed if owner[e["speaker"]] == scene.label_at(e["t0"], e["t1"])]
    return {
        "speakers": len(by_speaker),
        "phantoms": sorted(s for s, who in owner.items() if who not in voices),
        "clusters_per_voice": {v: sum(1 for s in owner.values() if s == v) for v in sorted(voices)},
        "purity": len(correct) / max(len(attributed), 1),
        "coverage": len(attributed) / max(len(voiced), 1),
        "by_truth": {k: dict(c) for k, c in by_truth.items()},
    }


def room(data: Path, seed: int, sounds: dict | None = None, varied: bool = True):
    """Six real voices taking turns, indoor sounds between turns, over a noise floor.

    With ``varied``, every utterance gets a random gain (-12..0 dB) and the room a
    random floor (-70..-50 dBFS): the same voice, near and far, in a quiet or a
    noisy room. Returns the scene and the set of voice labels.
    """
    import random

    rng = random.Random(seed)
    sounds = sounds if sounds is not None else esc50(data)
    utterances = {}
    for key, x in libri(data).items():
        third = len(x) // 3
        utterances[key] = [x[i * third : (i + 1) * third] for i in range(3)]
    for speaker in TEST_FSDD:
        start = rng.randrange(0, 40) if varied else 20
        utterances[speaker] = fsdd(data, speaker, range(start, start + 6))
    turns = [(k, u) for k, us in utterances.items() for u in us]
    rng.shuffle(turns)
    noises = [(c, x) for c, xs in sounds.items() for x in xs]
    rng.shuffle(noises)
    floor = rng.uniform(-70, -50) if varied else -60.0
    scene = Scene(seed=seed, floor_dbfs=floor).silence(2)
    for index, (key, u) in enumerate(turns):
        scene.add(key, u, gain_db=rng.uniform(-12, 0) if varied else 0.0)
        scene.silence(rng.uniform(0.4, 2.0))
        if index % 2 == 0 and noises:
            category, x = noises.pop()
            scene.add(f"noise:{category}", x, gain_db=rng.uniform(-12, 0) if varied else -6.0)
            scene.silence(rng.uniform(0.3, 1.5))
    return scene, set(utterances)
