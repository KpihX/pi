"""The machine's own sound, end to end: two channels, one ear (``contract.OUTPUT``).

A laptop plays an English podcast through its own speakers (a browser, ``apps.jsonl``
says so) while a person in the room speaks French, partly over it. The two recorders
started apart: the microphone hears the machine a third of a second after the
monitor captured it. The same scene is heard with the output on and off.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import needs_data

from π_ear import contract, echo
from π_ear.ear import Ear, Places
from π_ear.io import read_json, read_jsonl
from π_ear.lexicon import tokens
from π_ear.live import recall
from π_ear.public import chunk_name, loudspeaker, place, synth

pytestmark = [
    needs_data,
    pytest.mark.skipif(not echo.available(), reason="no Speex echo canceller"),
]

R = contract.RATE
PODCAST = [
    "the river was quiet that morning and the boats stayed close to the shore",
    "nobody in the village remembered a winter as long as that one",
    "she wrote the address on the back of the envelope and left before noon",
    "the old bridge was closed for repairs until the end of the month",
]
SAID = [
    "je voudrais qu'on reprenne le chapitre trois demain matin",
    "envoie le compte rendu à toute l'équipe avant midi",
    "on se retrouve devant la bibliothèque à quatorze heures",
]
OFFSET_S = 0.33


def _write(directory, x):
    directory.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    for k, start in enumerate(range(0, len(pcm), R)):
        (directory / chunk_name(k)).write_bytes(pcm[start : start + R].tobytes())


def _scene(tmp_path):
    rng = np.random.default_rng(11)
    podcast = list(synth(contract.MODELS, [200], PODCAST).values())[0]
    person = list(synth(contract.MODELS, [1], SAID, language="fr").values())[0]
    gap = np.zeros(int(0.8 * R), np.float32)
    # played twice: the canceller learns the offset on the first stretch it hears
    # (contract.ECHO_ESTIMATE_S of the machine's sound), then converges
    played = np.concatenate(
        [np.zeros(2 * R, np.float32)] + [c for p in podcast * 2 for c in (p, gap)]
    )
    plays = len(played)  # how long the machine plays
    n = plays + 8 * R
    played *= 10 ** (-22 / 20) / np.sqrt(np.mean(played[played != 0] ** 2))
    played = np.concatenate([played, np.zeros(n - len(played), np.float32)])
    heard = place(rng)(loudspeaker(played, rng)).astype(np.float32)
    heard = np.concatenate([np.zeros(int(OFFSET_S * R), np.float32), heard])[:n]
    room, voice, spans = place(rng), np.zeros(n, np.float32), []
    starts = [plays * 0.55, plays * 0.75, plays + 1.5 * R]
    for i, clip in enumerate(person):
        at = int(starts[i])  # the first two over the podcast, the last after it
        y = room(clip).astype(np.float32)[: n - at]
        voice[at : at + len(y)] += y
        spans.append((at / R, (at + len(y)) / R, SAID[i]))
    talking = voice != 0
    heard *= 10 ** (-30 / 20) / np.sqrt(np.mean(heard[talking] ** 2) + 1e-20)
    voice *= 10 ** (-30 / 20) / np.sqrt(np.mean(voice[talking] ** 2))
    mic = heard + voice + (rng.standard_normal(n) * 10 ** (-65 / 20)).astype(np.float32)
    places = Places(
        root=tmp_path / "live",
        memory=tmp_path / "memory",
        audio=tmp_path / "mic",
        output=tmp_path / "out",
        corpus=None,
    )
    _write(places.audio_dir, mic)
    _write(places.output_dir, played)
    places.root.mkdir(parents=True, exist_ok=True)
    with places.apps.open("w") as f:
        for k in range(n // R):
            if np.any(played[k * R : (k + 1) * R]):
                app = {"app": "Firefox", "media": "Le podcast", "playing": True}
                f.write(json.dumps({"chunk": chunk_name(k)[:-4], "apps": [app]}) + "\n")
    return places, spans


def _hear(places, monkeypatch, output: str):
    from π_ear import asr

    fast = asr.build(contract.FAST, contract.MODELS)
    monkeypatch.setattr(contract, "OUTPUT", output)
    monkeypatch.setattr(
        "π_ear.ear.asr.build", lambda name, models: fast if name == contract.FAST else None
    )
    ear = Ear(places, from_start=True, live=False)
    ear.run(drain=True)
    events = [e for e in read_jsonl(places.events)[0] if e.get("session") == ear.session]
    return ear, events


def _texts(events, channel):
    final = {}
    for e in events:
        if e.get("kind") == "text" and e.get("channel") == channel:
            final[e["piece"]] = e
    return sorted(final.values(), key=lambda e: e["t0"])


def _person_recall(events, spans, over_machine: bool):
    texts = _texts(events, "mic")
    wanted = spans[:2] if over_machine else spans[2:]
    truth = [w for _, _, s in wanted for w in tokens(s)]
    written = [
        w
        for e in texts
        if any(e["t0"] < b and a < e["t1"] for a, b, _ in wanted)
        for w in tokens(e["text"])
    ]
    return recall(truth, written)


def test_the_room_is_heard_over_what_the_machine_plays(tmp_path, monkeypatch):
    places, spans = _scene(tmp_path)
    ear, events = _hear(places, monkeypatch, "on")
    machine = [e for e in events if "row" in e and e["channel"] == "machine"]
    assert machine and all(e["apps"] == ["Firefox"] for e in machine if e["kind"] != "noise")
    # what the machine played is read from the machine itself: clean
    podcast = [w for s in PODCAST for w in tokens(s)]
    read = [w for e in _texts(events, "machine") for w in tokens(e["text"])]
    assert recall(podcast, read) > 0.8, read
    assert all(e["apps"] == ["Firefox"] for e in _texts(events, "machine"))
    # the machine was taken out of the microphone: the offset was learned from the stream
    state = read_json(places.state)
    assert state["output"]["on"] and abs(state["output"]["echo"]["lag_s"] - OFFSET_S) < 0.05
    # and the person is read over it, where with the output off they were not
    with_output = _person_recall(events, spans, over_machine=True)
    alone_after = _person_recall(events, spans, over_machine=False)
    off_places, _ = _scene(tmp_path / "off")
    _, off_events = _hear(off_places, monkeypatch, "off")
    assert not [e for e in off_events if e.get("channel") == "machine"]
    without_output = _person_recall(off_events, spans, over_machine=True)
    assert with_output > without_output, (with_output, without_output)
    assert alone_after > 0.5  # the canceller does nothing to the room once the machine is silent
    # the decisions over the machine say how much of the window was the machine
    over = [e for e in events if "row" in e and e["channel"] == "mic" and e["echo"] is not None]
    assert over and max(e["echo"] for e in over) > 0.5


def test_no_machine_recorder_costs_the_microphone_nothing(tmp_path, monkeypatch):
    """Output on, but `make rec-out` not running: no chunk of the machine ever comes.
    The microphone is not held back waiting for it, and the state says why nothing
    is cancelled."""
    places, _ = _scene(tmp_path)
    for path in places.output_dir.iterdir():
        path.unlink()
    monkeypatch.setattr(contract, "OUTPUT", "on")
    ear = Ear(places, from_start=True, live=False, asr=False)
    while ear.tick():
        assert not ear.waiting
    state = ear.state()
    assert "make rec-out" in state["output"]["degraded"]
    assert ear.counters["chunks"] == len(list(places.audio_dir.iterdir()))
