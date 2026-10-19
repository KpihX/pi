"""The ear as a running server: live, pausable, restartable, correctable.

These tests drive the real loop on real sensors (Silero, CAM++), with a
simulated recorder that writes one-second chunks the way ``arecord`` does,
half a chunk at a time.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
from conftest import needs_data

from π_ear import contract
from π_ear.ear import Ear, Places, correction
from π_ear.io import read_json, read_jsonl
from π_ear.public import Scene, fsdd, libri

pytestmark = needs_data


@pytest.fixture(scope="module")
def two_voices():
    voices = libri(contract.DATA)
    a = voices["libri5703"]
    b = np.concatenate(fsdd(contract.DATA, "george", range(30, 34)))
    later = np.concatenate(fsdd(contract.DATA, "george", range(40, 44)))
    return a, b, later


def _scene(a, b, seed=0) -> Scene:
    scene = Scene(seed=seed).silence(1.5)
    half_a, half_b = len(a) // 2, len(b) // 2
    for label, x in (("A", a[:half_a]), ("B", b[:half_b]), ("A", a[half_a:]), ("B", b[half_b:])):
        scene.add(label, x).silence(1.0)
    return scene


def _places(tmp_path, **extra) -> Places:
    return Places(root=tmp_path / "live", memory=tmp_path / "memory", **extra)


def _replay(places: Places) -> Ear:
    ear = Ear(places, from_start=True, live=False, asr=False)
    ear.run(drain=True)
    return ear


def _speakers(ear: Ear) -> list[str]:
    return sorted(c.key for c in ear.model.clusters.values() if c.kind == "speaker")


def test_live_with_a_recorder_writing_as_it_goes(tmp_path, two_voices):
    """Decisions arrive while the recorder is still recording; every chunk is read once."""
    places = _places(tmp_path, corpus=tmp_path / "corpus")
    places.ensure()
    pcm = (np.clip(np.concatenate(_scene(*two_voices[:2]).audio), -1, 1) * 32767).astype("<i2")
    step = contract.RATE
    chunks = [pcm[i : i + step] for i in range(0, len(pcm), step)]
    first_event_at: list[float] = []

    def recorder():
        for index, chunk in enumerate(chunks):
            path = places.audio_dir / f"20261018-{index:06d}.raw"
            half = len(chunk) // 2
            path.write_bytes(chunk[:half].tobytes())  # arecord is mid-chunk
            time.sleep(0.02)
            with open(path, "ab") as handle:
                handle.write(chunk[half:].tobytes())
            if places.events.exists() and not first_event_at:
                first_event_at.append(index)
        # one more chunk opened and never finished: the live gate must not read it
        (places.audio_dir / "20261018-999999.raw").write_bytes(b"\0\0" * 100)

    ear = Ear(places, live=True, asr=False)
    listening = threading.Thread(target=ear.run, kwargs={"max_seconds": len(chunks) * 0.05 + 8})
    listening.start()
    time.sleep(0.5)  # the ear is up before the recorder: a live start begins "now"
    recorder()
    listening.join()
    state = read_json(places.state)
    assert ear.counters["chunks"] == len(chunks)
    assert first_event_at and first_event_at[0] < len(chunks) - 3, "the ear waited for the end"
    assert len(_speakers(ear)) == 2
    assert [p.name for p in places.audio_dir.iterdir()] == ["20261018-999999.raw"]
    assert state["corpus"]["chunks"] == len(chunks)


def test_pause_is_a_backlog_and_resume_drains_it(tmp_path, two_voices):
    places = _places(tmp_path, audio=tmp_path / "chunks", corpus=None)
    _scene(*two_voices[:2]).write(places.audio_dir)
    places.ensure()
    places.pause.touch()
    ear = Ear(places, from_start=True, live=False, asr=False)
    ear.run(max_seconds=0.5)
    state = read_json(places.state)
    assert state["paused"] is True and state["audio"]["chunks_seen"] == 0
    places.pause.unlink()
    ear.run(drain=True)
    assert ear.tailer.backlog() == 0 and len(_speakers(ear)) == 2


def test_the_loop_sleeps_instead_of_spinning(tmp_path, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("π_ear.ear.time.sleep", lambda s: sleeps.append(s))
    Ear(_places(tmp_path, corpus=None), live=True, asr=False).run(max_seconds=0.2)
    assert sleeps and all(s == contract.IDLE_SLEEP_S for s in sleeps)


def test_a_restart_keeps_every_name_and_every_row(tmp_path, two_voices):
    """v0 gave the same film voice a new key after each restart. Here the memory
    is read back: the next day, new recordings of the same two voices get the
    same two keys, and rows and revision continue."""
    a, b, later = two_voices
    first = _places(tmp_path, audio=tmp_path / "day1", corpus=None)
    _scene(a, b, seed=1).write(first.audio_dir)
    ear = _replay(first)
    keys, rows, revision = _speakers(ear), len(ear.store), ear.revision
    assert len(keys) == 2
    second = _places(tmp_path, audio=tmp_path / "day2", corpus=None)
    _scene(a, later, seed=2).write(second.audio_dir)
    again = _replay(second)
    assert _speakers(again) == keys
    assert len(again.store) > rows and again.revision > revision
    sessions = {e["session"] for e in read_jsonl(first.events)[0] if "session" in e}
    assert len(sessions) == 2, "every line carries its session"


def test_a_correction_written_while_running_is_obeyed(tmp_path, two_voices):
    places = _places(tmp_path, audio=tmp_path / "chunks", corpus=None)
    _scene(*two_voices[:2]).write(places.audio_dir)
    ear = _replay(places)
    events = [e for e in read_jsonl(places.events)[0] if e.get("kind") in ("known", "new")]
    row = events[-1]["row"]
    correction(places, "name", [row], label="KπX")
    ear.run(max_seconds=ear.clock() + 0.5)
    state = read_json(places.state)
    assert "KπX" in [s["label"] for s in state["speakers"]]
    assert ear.counters["corrections"] == 1
    logged = [e for e in read_jsonl(places.events)[0] if e.get("kind") == "correction"]
    assert logged and logged[0]["label"] == "KπX"


def test_reset_archives_and_restarts_from_the_base(tmp_path, two_voices):
    places = _places(tmp_path, audio=tmp_path / "chunks", corpus=None)
    _scene(*two_voices[:2]).write(places.audio_dir)
    ear = _replay(places)
    assert _speakers(ear)
    places.reset.touch()
    for chunk in places.audio_dir.iterdir():
        chunk.unlink()
    ear.run(max_seconds=ear.clock() + 0.3)
    assert not places.reset.exists() and not _speakers(ear)
    assert list((places.root / "archive").glob("*model.json"))


def test_state_is_readable_json_with_the_numbers_downstream_needs(tmp_path, two_voices):
    places = _places(tmp_path, audio=tmp_path / "chunks", corpus=None)
    _scene(*two_voices[:2]).write(places.audio_dir)
    _replay(places)
    state = json.loads(places.state.read_text())
    assert {"kappa", "kappa0", "alpha", "speakers"} <= set(state["model"])
    line = next(e for e in read_jsonl(places.events)[0] if e.get("kind") == "known")
    assert {
        "row",
        "session",
        "t0",
        "t1",
        "decided_at",
        "speaker",
        "p",
        "runner_up",
        "rms_dbfs",
    } <= set(line)
    assert line["decided_at"] >= line["t1"]
