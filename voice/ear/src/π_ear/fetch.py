"""Every download of the package, in one list, and the public corpora it is tested on.

Models (sherpa-onnx releases on GitHub, ~1.5 GB downloaded, ~1.3 GB kept):

* Silero VAD and CAM++ (the sensors of *who*), pyannote segmentation-3.0 (two voices
  at once, ``overlap.py``), Whisper base (a voice's language over its pooled speech,
  ``lid.py``);
* Parakeet TDT 0.6b v3 (25 languages) and Whisper turbo (99 languages): the
  fast and the slow reader of *what* (``asr.py``); Whisper tiny is kept for the
  LibriSpeech test files of its archive;
* whatever ``PI_EAR_FAST`` and ``PI_EAR_SLOW`` name from the registry instead
  (``asr.REGISTRY``), and Spleeter if ``PI_EAR_SEPARATOR`` asks for it;
* Piper voices, LibriTTS-R (904 English voices) and UPMC (French): synthetic
  speech whose text is known, for the base and the tests.

Public corpora (raw.githubusercontent.com, ~120 MB), never committed:

* ESC-50 (Piczak 2015, CC BY-NC): 20 indoor classes, 16 clips of folds 1-3
  (heard by the base) and 8 of fold 5 (the tests). Coughs, doors,
  keyboard, laughter: the sounds that made v0 invent a person;
* FSDD (Jackson et al., CC BY-SA 4.0): six real speakers, 3000 recordings;
* three LibriSpeech excerpts (CC BY 4.0, as mirrored by librosa);
* the pyannote two-speaker sample with its RTTM (MIT);
* the sherpa-onnx four-speaker Mandarin test file;
* three music tracks mirrored by librosa (CC BY / CC0), to mix under speech.
"""

from __future__ import annotations

import csv
import io
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from π_ear import contract
from π_ear.asr import REGISTRY

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
RAW = "https://raw.githubusercontent.com"

MODELS = {
    contract.VAD_MODEL: f"{RELEASES}/asr-models/silero_vad.onnx",
    contract.ENCODER: f"{RELEASES}/speaker-recongition-models/{contract.ENCODER}",
}
ARCHIVES = {
    contract.WHISPER_DIR: f"{RELEASES}/asr-models/{contract.WHISPER_DIR}.tar.bz2",
    **{
        REGISTRY[name][0]: f"{RELEASES}/asr-models/{REGISTRY[name][0]}.tar.bz2"
        for name in (contract.FAST, contract.SLOW)
        if name in REGISTRY
    },
    **(
        {contract.SEPARATOR: f"{RELEASES}/source-separation-models/{contract.SEPARATOR}.tar.bz2"}
        if contract.SEPARATOR != "none"
        else {}
    ),
    **(
        {contract.LID: f"{RELEASES}/asr-models/{contract.LID}.tar.bz2"}
        if contract.LID != "none"
        else {}
    ),
    **(
        {
            contract.SEGMENTATION: f"{RELEASES}/speaker-segmentation-models/"
            f"{contract.SEGMENTATION}.tar.bz2"
        }
        if contract.SEGMENTATION != "none"
        else {}
    ),
    "vits-piper-en_US-libritts_r-medium": f"{RELEASES}/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2",
    "vits-piper-fr_FR-upmc-medium": f"{RELEASES}/tts-models/vits-piper-fr_FR-upmc-medium.tar.bz2",
}
MUSIC = (
    "442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime.ogg",
    "147793__setuniman__sweet-waltz-0i-22mi.ogg",
    "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra).ogg",
)

ESC50_CLASSES = (
    "coughing sneezing door_wood_knock door_wood_creaks keyboard_typing mouse_click footsteps "
    "clapping clock_tick breathing laughing drinking_sipping glass_breaking can_opening "
    "pouring_water water_drops toilet_flush vacuum_cleaner washing_machine dog"
).split()
BASE_FOLDS = ("1", "2", "3")  # the ear is born having heard these
TEST_FOLDS = ("5",)  # and is tested on these, never seen
FSDD_SPEAKERS = ("jackson", "nicolas", "theo", "yweweler", "george", "lucas")
LIBRI = ("5703-47212-0000.ogg", "3436-172162-0000.ogg", "198-209-0000.ogg")


def _get(url: str, target: Path) -> Path:
    if target.exists() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    for attempt in range(4):  # mirrors drop connections under parallel load
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                tmp.write_bytes(response.read())
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    tmp.replace(target)
    return target


def models(root: Path = contract.MODELS, echo=print) -> None:
    for name, url in MODELS.items():
        echo(f"model   {_get(url, root / name)}")
    for name, url in ARCHIVES.items():
        if (root / name).exists():
            echo(f"model   {root / name}")
            continue
        with urllib.request.urlopen(url, timeout=600) as response:
            tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:bz2").extractall(
                root, filter="data"
            )
        for heavy in (root / name).glob("*coder.onnx"):  # only the int8 models are used
            heavy.unlink()
        echo(f"model   {root / name}")


def data(root: Path = contract.DATA, echo=print) -> None:
    jobs: list[tuple[str, Path]] = []
    meta = _get(f"{RAW}/karolpiczak/ESC-50/master/meta/esc50.csv", root / "esc50" / "esc50.csv")
    rows = list(csv.DictReader(meta.open()))
    for category in ESC50_CLASSES:
        mine = [r for r in rows if r["category"] == category]
        base = [r for r in mine if r["fold"] in BASE_FOLDS][:16]
        test = [r for r in mine if r["fold"] in TEST_FOLDS][:8]
        for row in base + test:
            name = row["filename"]
            jobs.append((f"{RAW}/karolpiczak/ESC-50/master/audio/{name}", root / "esc50" / name))
    fsdd = f"{RAW}/Jakobovski/free-spoken-digit-dataset/master/recordings"
    for speaker in FSDD_SPEAKERS:
        for digit in range(10):
            for take in range(50):
                name = f"{digit}_{speaker}_{take}.wav"
                jobs.append((f"{fsdd}/{name}", root / "fsdd" / name))
    for name in LIBRI:
        jobs.append((f"{RAW}/librosa/data/main/audio/{name}", root / "libri" / name))
    for name in MUSIC:
        jobs.append((f"{RAW}/librosa/data/main/audio/{name}", root / "music" / name))
    pyannote = f"{RAW}/pyannote/pyannote-audio/develop/tutorials/assets"
    jobs.append((f"{pyannote}/sample.wav", root / "sample.wav"))
    jobs.append((f"{pyannote}/sample.rttm", root / "sample.rttm"))
    jobs.append(
        (f"{RELEASES}/speaker-segmentation-models/0-four-speakers-zh.wav", root / "zh4.wav")
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: _get(*job), jobs))
    for directory in ("esc50", "fsdd", "libri", "music"):
        echo(f"data    {root / directory}: {len(list((root / directory).iterdir()))} files")
