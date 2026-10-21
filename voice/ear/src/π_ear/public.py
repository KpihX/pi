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


def place(rng):
    """One place a voice is heard in: a room, a microphone, a distance, a noise floor.

    The public voices of the base were each recorded in one place, so on their own
    they cannot say how far a voice drifts between conditions (``kappa_c``). Each
    session of the base is therefore heard through one simulated place: a
    reverberant room (RT60 0.15-0.7 s), a microphone band (high-pass 50-300 Hz,
    low-pass 3.5-7.5 kHz), a gain (-12..0 dB) and a noise floor (SNR 15-35 dB).
    The ranges describe ordinary rooms and laptop or phone microphones; the
    corrections take over from there (``same`` joins conditions and teaches
    kappa_c). Returns ``audio -> audio`` with the place fixed, the noise drawn anew.
    """
    from scipy.signal import butter, fftconvolve, sosfilt

    rate = contract.RATE
    rt60 = rng.uniform(0.15, 0.7)
    tail = np.arange(int(rate * rt60)) / rate
    rir = rng.standard_normal(tail.size) * np.exp(-6.9 * tail / rt60) * 0.3
    rir[0] = 1.0
    band = butter(
        2, [rng.uniform(50, 300), rng.uniform(3500, 7500)], "bandpass", fs=rate, output="sos"
    )
    gain = 10 ** (rng.uniform(-12, 0) / 20)
    snr = 10 ** (rng.uniform(15, 35) / 10)

    def hear(x: np.ndarray) -> np.ndarray:
        y = sosfilt(band, fftconvolve(x, rir)[: len(x)])
        y = y / (np.max(np.abs(y)) or 1.0) * 0.5 * gain
        power = float(np.mean(np.square(y)))
        return (y + rng.standard_normal(len(y)) * np.sqrt(power / snr)).astype(np.float32)

    return hear


def condition(x: np.ndarray, rng) -> np.ndarray:
    """One recording heard in a place of its own."""
    return place(rng)(x)


SESSION = 5  # FSDD takes per simulated place
PLACES = 4  # 20 FSDD takes per base voice, 5 per place


def fit_base(sensor, data: Path, models: Path):
    # Reproducible: the base is a measurement. One generator per part, so that
    # changing one part does not redraw the others.
    rng = np.random.default_rng(0)

    # Four places, each heard by every base voice: one room, one microphone, one
    # noise floor per place. A voice's recordings in different places teach how it
    # drifts (kappa_c); different voices of one place teach how close a shared
    # channel brings them (kappa_in); within one recording, kappa.
    tests = models / "sherpa-onnx-whisper-tiny" / "test_wavs"
    first = load(tests / "0.wav")
    half = len(first) // 2
    libri = {
        "libri1089": [first[:half], first[half:]],
        "libri1221": [load(tests / "1.wav"), load(tests / "8k.wav")],
    }
    groups: dict[str, list] = {}
    places = []
    for k in range(PLACES):
        here = place(rng)
        place_of = []
        takes = {
            s: np.concatenate(fsdd(data, s, range(k * SESSION, (k + 1) * SESSION)))
            for s in BASE_FSDD
        }
        takes.update({s: recs[k] for s, recs in libri.items() if k < len(recs)})
        for s, x in takes.items():
            groups.setdefault(s, []).append([sensor.embed(w) for w in windows(here(x))])
            place_of.append((s, len(groups[s]) - 1))
        places.append(place_of)
    sizes = {k: sum(len(r) for r in v) for k, v in groups.items()}
    weights = fit_weights(groups, "", places)
    # What the VAD lets through from 20 indoor sound classes, exactly as the ear
    # would see it: full encoder windows, cut live by the same sensor.
    model = Model(weights, lambda row: None)
    heard, noise = 0, []
    # Each clip twice: as recorded, and through a place of its own (a door heard
    # through a reverberant room and a phone microphone came out of the VAD as a
    # person, measured in a room passed through ``place``).
    rooms = np.random.default_rng(3)
    for clips in esc50(data, BASE_FOLDS).values():
        for clip in (c for x in clips for c in (x, place(rooms)(x))):
            pieces, _ = sensor.feed(clip)
            tail, _ = sensor.feed(np.zeros(contract.RATE, dtype=np.float32))
            for piece in pieces + tail:
                if piece.full and not piece.shadow:
                    noise.append(sensor.embed(piece.samples))
                    model.hear_noise(noise[-1])
                    heard += 1
    weights.noise = model.noise_weights()
    weights.calibration = calibrate_who(weights, groups, noise, np.random.default_rng(4))
    weights.spectra = base_spectra(data, models, np.random.default_rng(1))
    weights.reader = base_reader(data, models, np.random.default_rng(2))
    weights.source = (
        f"{contract.ENCODER}; voices: FSDD {'/'.join(BASE_FSDD)} + LibriSpeech 1089/1221, "
        f"heard in {PLACES} simulated places, "
        f"windows {sizes}; noise: ESC-50 folds {'/'.join(BASE_FOLDS)}, {heard} windows "
        f"(as recorded and through simulated places) in {len(weights.noise)} clusters; source: person/device on "
        f"{weights.spectra['person']['n']}+{weights.spectra['device']['n']} recordings; "
        f"reader: {weights.reader.get('runs', 0)} public runs, "
        f"{weights.reader.get('episodes', 0)} correction episodes"
    )
    return weights


def calibrate_who(weights, groups: dict, noise: list, rng) -> list[list[float]]:
    """Q1 from day one: how often a margin of so many nats is right, on public windows.

    The base's own recordings (each place a session) and the noise windows it
    heard, replayed through a model in a random order of recordings, the truth
    known. A decision is right when it puts the window with the voice the
    window's cluster ends up owning, or, for the first windows of a voice and
    for noise, when it does not put it with anyone else. Corrections add theirs.
    """
    from π_ear.model import VOICE, Model

    vectors: list[np.ndarray] = []
    labels: list[str] = []
    model = Model(weights, lambda row: vectors[row])
    turns = [(v, i) for v, recordings in groups.items() for i in range(len(recordings))]
    turns += [("noise", i) for i in range(len(noise))]
    decided = []
    for index in rng.permutation(len(turns)):
        voice, i = turns[index]
        windows = groups[voice][i] if voice != "noise" else [noise[i]]
        previous = None
        for x in windows:
            row = len(vectors)
            vectors.append(np.asarray(x, dtype=np.float64))
            labels.append(voice)
            d = model.observe(row, vectors[row], float(row), True, previous, f"base-{i}")
            decided.append((row, d))
            previous = row
    owner: dict[str, str] = {}
    for key, cluster in model.clusters.items():
        if cluster.kind in VOICE and cluster.members:
            names = [labels[r] for r in cluster.members]
            owner[key] = max(set(names), key=names.count)
    out, heard = [], set()  # heard: the voices already decided as someone
    for row, d in decided:
        truth = labels[row]
        voiced = d["kind"] in ("known", "new")
        if voiced:
            right = owner.get(model.where.get(row)) == truth
        else:  # unknown or noise: right if the window is noise or its voice's first
            right = truth == "noise" or truth not in heard
        if voiced:
            heard.add(truth)
        out.append([round(d["margin"], 4), float(right)])
    return out


def loudspeaker(x: np.ndarray, rng) -> np.ndarray:
    """The same voice played by a device: a small driver (no bass), a cone resonance,
    a little compression. The room comes after, from ``condition``."""
    from scipy.signal import butter, sosfilt

    rate = contract.RATE
    y = sosfilt(butter(2, rng.uniform(150, 450), "highpass", fs=rate, output="sos"), x)
    centre = rng.uniform(1500, 4000)
    peak = sosfilt(butter(2, [centre / 1.4, centre * 1.4], "bandpass", fs=rate, output="sos"), y)
    y = y + peak * (10 ** (rng.uniform(3, 9) / 20) - 1)
    drive = rng.uniform(1.0, 3.0)
    y = np.tanh(drive * y / (np.max(np.abs(y)) or 1.0)) / np.tanh(drive)
    y = sosfilt(butter(2, rng.uniform(5000, 7500), "lowpass", fs=rate, output="sos"), y)
    return y.astype(np.float32)


def base_spectra(data: Path, models: Path, rng) -> dict:
    """Mean spectral shapes of public recordings heard live (person) or replayed (device)."""
    from π_ear.sense import spectrum
    from π_ear.source import Source

    source = Source()
    tests = models / "sherpa-onnx-whisper-tiny" / "test_wavs"
    recordings = [u for s in BASE_FSDD for u in fsdd(data, s, range(20, 30))]
    recordings += [load(tests / f) for f in ("0.wav", "1.wav", "8k.wav")]
    for x in recordings:
        for name, audio in (
            ("person", condition(x, rng)),
            ("device", condition(loudspeaker(x, rng), rng)),
        ):
            shapes = [spectrum(w) for w in windows(audio)] or [spectrum(audio)]
            source.add(name, np.mean(shapes, axis=0))
    return source.to_dict()


EN_BASE = [
    "Could you turn the lights off in the kitchen before you leave?",
    "I left my keys on the table next to the window this morning.",
    "The train to the airport was delayed by almost forty minutes.",
    "We should call your sister and ask her about the weekend.",
    "My computer keeps crashing every time I open that program.",
    "Please write down the address so we do not forget it again.",
]
FR_BASE = [
    "Est-ce que tu peux fermer la porte de la cuisine en partant ?",
    "J'ai oublié mon parapluie dans le bus ce matin.",
    "Le train pour Lyon a encore quarante minutes de retard.",
    "On devrait appeler ta sœur pour lui parler du week-end.",
    "Mon ordinateur plante chaque fois que j'ouvre ce logiciel.",
    "Note bien l'adresse pour qu'on ne l'oublie plus.",
]


def base_reader(data: Path, models: Path, rng) -> dict:
    """The reader's base weights: public pieces heard by both recognisers, truth known.

    English and French sentences of Piper voices, heard in three regimes, because the
    live test showed the reader must not learn one (2026-10-20: on a television heard
    by a laptop, Silero's mean on real speech was 0.3-0.7, where clean speech gives
    0.82-0.94, and "I did not understand" took 106 real sentences):

    * ``clean``: as synthesised;
    * ``room``: through a simulated place (``place``);
    * ``tv``: played by a loudspeaker (``loudspeaker``), in a place, over a bed of
      indoor sounds at 0-15 dB under it (ESC-50 folds 1-3, looped).

    And the indoor sounds alone in each regime, as the VAD lets them through: their
    truth is "nothing was said". Each piece carries its regime's own mean of Silero
    on speech, as the ear does live (``Context.regime``).
    """
    from π_ear import asr
    from π_ear.language import FAST, SLOW, Languages
    from π_ear.sense import Sensor, speech_probability
    from π_ear.words import Context, Heard, fit_reader, mean_logp

    fast, slow = asr.build(contract.FAST, models), asr.build(contract.SLOW, models)
    if fast is None:
        return {}
    sensor = Sensor(models)
    sounds = [x for _, clips in sorted(esc50(data, BASE_FOLDS).items()) for x in clips]

    def bed(n: int) -> np.ndarray:
        """Indoor sounds one after another, as long as asked."""
        out, size = [], 0
        while size < n:
            out.append(sounds[int(rng.integers(len(sounds)))])
            size += len(out[-1])
        return np.concatenate(out)[:n]

    def regime(name: str, x: np.ndarray) -> np.ndarray:
        if name == "clean":
            return x
        if name == "room":
            return condition(x, rng)
        y = condition(loudspeaker(x, rng), rng)
        b = bed(len(y))
        level = np.mean(y * y) / (np.mean(b * b) or 1.0) / 10 ** (rng.uniform(0, 15) / 10)
        return (y + b * np.sqrt(level)).astype(np.float32)

    def runs_of(x: np.ndarray) -> list[np.ndarray]:
        _, found = sensor.feed(x)
        _, tail = sensor.feed(np.zeros(contract.RATE, dtype=np.float32))
        return [r.samples for r in found + tail]

    def spoken(x: np.ndarray) -> float:
        """Silero's mean over what the VAD let through, as the ear computes it on a piece."""
        inside = runs_of(x)
        return float(np.mean(speech_probability(models, np.concatenate(inside) if inside else x)))

    REGIMES = ("clean", "room", "tv")
    pieces: list[tuple[str, np.ndarray, str, str | None, str | None]] = []
    for language, sentences, speakers in (
        ("en", EN_BASE, [100, 300, 600]),
        ("fr", FR_BASE, [0, 1]),
    ):
        for speaker, clips in synth(models, speakers, sentences, language=language).items():
            for x, truth in zip(clips, sentences, strict=True):
                for name in REGIMES:
                    pieces.append((name, regime(name, x), truth, f"{speaker}", language))
    for clip in sounds[:: max(1, len(sounds) // 40)]:
        for name in REGIMES:
            for run in runs_of(regime(name, clip)):
                pieces.append((name, run, "", None, None))
    speech = [(name, spoken(x)) for name, x, truth, _, _ in pieces]
    means = {
        name: float(
            np.mean(
                [
                    s
                    for (n, s), (_, _, t, _, _) in zip(speech, pieces, strict=True)
                    if n == name and t
                ]
            )
        )
        for name in REGIMES
    }
    readings = [
        (heard_by(contract.FAST, fast, x), heard_by(contract.SLOW, slow, x) if slow else None)
        for _, x, _, _, _ in pieces
    ]
    confidence = {
        name: float(
            np.mean(
                [
                    mean_logp(f)
                    for (n, _, t, _, _), (f, _) in zip(pieces, readings, strict=True)
                    if n == name and t and f.words
                ]
            )
        )
        for name in REGIMES
    }
    heard = []
    for (name, x, truth, voice, language), (_, s), (f, g) in zip(
        pieces, speech, readings, strict=True
    ):
        context = Context(voice, s, means[name], len(x) / contract.RATE, confidence[name])
        heard.append(Heard(f, g, truth, context, language))
    out = fit_reader(heard)
    # how often the slow recogniser's language detection is right, at every length it
    # reads, not only a sentence's: one voice's sentences, then every voice's of a
    # language, one after another, doubling until its window is full, in each regime. Measured at 2-4 s alone, the slope of
    # reliability in length was free, and three detections of one voice on the live
    # replay turned it over (longer came out less reliable).
    window = asr.window_s(slow) if slow else None
    texts = Languages()
    texts.load({"words": out["languages"]["words"]})
    if slow is not None and window:
        by_voice: dict[tuple, list] = {}
        for name, x, _, voice, language in pieces:
            if language:
                by_voice.setdefault((name, voice, language), []).append(x)
                # and every voice of a language in turn, to reach the window's length
                by_voice.setdefault((name, "", language), []).append(x)
        for (_, _, language), clips in sorted(by_voice.items()):
            k = 2
            while k <= len(clips):
                x = np.concatenate(clips[:k])
                if len(x) / contract.RATE >= window:
                    break
                detected = heard_by(contract.SLOW, slow, x).language
                if detected:
                    out["languages"]["base"].append(
                        [len(x) / contract.RATE, float(detected == language), SLOW]
                    )
                written = texts.of_text(heard_by(contract.FAST, fast, x).text)
                if written:
                    out["languages"]["base"].append(
                        [
                            len(x) / contract.RATE,
                            float(max(written, key=written.get) == language),
                            FAST,
                        ]
                    )
                k *= 2
    out["regime"] = means["clean"]
    out["confidence"] = confidence["clean"]
    out["regimes"] = {"speech": means, "confidence": confidence}
    return out


def heard_by(name: str, recognizer, x: np.ndarray):
    """A recogniser's reading of public audio, cached: the base is measured once."""
    import hashlib
    import json

    from π_ear.asr import Hypothesis

    cache = contract.DATA / "heard"
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.blake2b(np.asarray(x, dtype=np.float32).tobytes(), digest_size=10).hexdigest()
    path = cache / f"{name.replace(':', '_')}-{digest}.json"
    if path.exists():
        return Hypothesis.from_dict(json.loads(path.read_text()))
    heard = recognizer(x)
    path.write_text(json.dumps(heard.to_dict()))
    return heard


VOICES = {
    "en": ("vits-piper-en_US-libritts_r-medium", "en_US-libritts_r-medium.onnx"),
    "fr": ("vits-piper-fr_FR-upmc-medium", "fr_FR-upmc-medium.onnx"),
}


def synth(
    models: Path,
    speakers: list[int],
    sentences: list[str],
    cache: Path | None = None,
    language: str = "en",
) -> dict[str, list[np.ndarray]]:
    """Piper voices at 16 kHz (LibriTTS-R for English, UPMC for French). VITS samples
    noise, so the audio is cached: a test must hear the same voices every run."""
    import hashlib

    import sherpa_onnx
    from scipy.signal import resample_poly

    cache = cache or contract.DATA / "tts"
    cache.mkdir(parents=True, exist_ok=True)
    folder, onnx = VOICES[language]
    root = models / folder
    tts = None

    def engine():
        return sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(root / onnx),
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
            path = cache / (
                f"{sid}-{digest}.npy" if language == "en" else f"{language}{sid}-{digest}.npy"
            )
            if not path.exists():
                tts = tts or engine()
                audio = tts.generate(sentence, sid=sid, speed=1.0)
                x = np.asarray(audio.samples, dtype=np.float32)
                np.save(path, resample_poly(x, contract.RATE, audio.sample_rate).astype(np.float32))
            clips.append(np.load(path))
        out[f"tts{sid}" if language == "en" else f"tts-{language}{sid}"] = clips
    return out


def chunk_name(index: int) -> str:
    """The name arecord gives the chunk ``index`` seconds after 2026-10-18 00:00:00."""
    from datetime import datetime, timedelta

    return (datetime(2026, 10, 18) + timedelta(seconds=index)).strftime("%Y%m%d-%H%M%S.raw")


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
            (directory / chunk_name(index)).write_bytes(pcm[start : start + step].tobytes())
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
    said, right = 0.0, 0.0
    for e in events:
        for a, b, who in e.get("segments", []) if e.get("kind") == "turns" else []:
            truth = scene.label_at(a, b)
            if truth in voices and who in owner:
                said += b - a
                right += (b - a) * (owner[who] == truth)
    return {
        "speakers": len(by_speaker),
        "phantoms": sorted(s for s, who in owner.items() if who not in voices),
        "owners": owner,
        "clusters_per_voice": {v: sum(1 for s in owner.values() if s == v) for v in sorted(voices)},
        "purity": len(correct) / max(len(attributed), 1),
        "coverage": len(attributed) / max(len(voiced), 1),
        "turns": right / said if said else None,  # Q4: who spoke when, by the half window
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
