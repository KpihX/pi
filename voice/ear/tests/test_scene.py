"""Where a voice comes from (``scene.py``): room, remote, media, or π itself."""

from __future__ import annotations

import numpy as np

from π_ear.contract import MOUTH_APP
from π_ear.scene import Scene, conversation_evidence


def _at(second):
    return range(second, second + 1)


def _timeline(scene, seconds, machine, room, apps=("Firefox",)):
    """``machine`` and ``room`` speak at the seconds their masks say."""
    for s in range(seconds):
        if machine[s]:
            scene.heard("A", "machine", _at(s), apps=list(apps))
        if room[s]:
            scene.heard("B", "mic", _at(s), echo=0.0)
    scene.settle(seconds)


def test_a_voice_that_waits_for_the_room_is_a_conversation():
    """A meeting: A speaks, stops, B answers, A again: they take turns."""
    scene = Scene()
    machine = [(s // 4) % 2 == 0 for s in range(120)]
    room = [not m and s % 4 != 3 for s, m in enumerate(machine)]
    _timeline(scene, 120, machine, room)
    p = scene.of("A")
    assert p["remote"] > p["media"] and p["remote"] > p["room"] + p["self"]
    assert scene.person("A") and scene.person("B")


def test_a_voice_that_does_not_wait_for_anyone_is_a_recording():
    """A video with pauses: it talks on, stops, starts again on its own clock; the room
    talks over it whenever it wants, no more in its silences than while it plays."""
    rng = np.random.default_rng(0)
    scene = Scene()
    machine = list(rng.uniform(size=240) < 0.7)
    room = list(rng.uniform(size=240) < 0.4)
    _timeline(scene, 240, machine, room)
    p = scene.of("A")
    assert scene.conversation("A") < 0
    assert p["media"] > p["remote"] and not scene.person("A")


def test_a_voice_that_never_stops_says_nothing_of_its_turns():
    """A video that never pauses: there is no silence of it to compare with, the
    evidence is exactly none, remote and media stay as likely. Whether it is a person
    is then a coin: off ½ only by the room's one pseudo-window (the Beta prior)."""
    rng = np.random.default_rng(0)
    scene = Scene()
    _timeline(scene, 120, [True] * 120, list(rng.uniform(size=120) < 0.4))
    p = scene.of("A")
    assert scene.conversation("A") == 0
    assert abs(p["remote"] - p["media"]) < 1e-9
    people = p["room"] + p["remote"]
    assert abs(people - 1 / 2) < 1 / 120
    assert conversation_evidence(0, 0, 0, 0) == 0


def test_a_voice_nobody_ever_answers_is_not_a_conversation():
    """It pauses, and nobody in the room ever takes the turn: media is the better
    account (a conversation would have the room speak in its silences)."""
    scene = Scene()
    _timeline(scene, 120, [(s // 5) % 2 == 0 for s in range(120)], [False] * 120)
    p = scene.of("A")
    assert scene.conversation("A") < 0 and p["media"] > p["remote"]
    assert not scene.person("A")


def test_what_the_machine_played_is_not_in_the_room():
    """The facts: a voice out of the machine is not in the room; its residual in the
    microphone, mostly the machine's (echo), does not make it so."""
    scene = Scene()
    for s in range(30):
        scene.heard("A", "machine", _at(s), apps=["Firefox"])
        scene.heard("A", "mic", _at(s), echo=0.9)
        scene.heard("B", "mic", _at(s), echo=0.05)
    p, q = scene.of("A"), scene.of("B", device=0.1)
    assert p["room"] < 0.25 and q["room"] > 0.5
    # B never came out of the machine: what is left of "device" is its spectrum's word
    assert scene.of("B")["room"] > p["room"]


def test_a_voice_only_heard_through_the_microphone_is_placed_by_its_spectrum():
    scene = Scene()
    for s in range(20):
        scene.heard("P", "mic", _at(s))
    assert scene.of("P", device=0.9)["room"] < 0.2
    assert scene.of("P", device=0.1)["room"] > 0.8


def test_what_pi_says_is_pi():
    scene = Scene()
    for s in range(10):
        scene.heard("π", "machine", _at(s), apps=[MOUTH_APP])
    p = scene.of("π")
    assert max(p, key=p.get) == "self" and not scene.person("π")


def test_an_application_learns_what_it_plays_from_one_correction():
    """KπX says once that a voice of the meeting application is a person: the next
    voice out of that application starts as a person, before any turn is taken."""
    scene = Scene()
    for s in range(10):
        scene.heard("A", "machine", _at(s), apps=["Meet"])
        scene.heard("C", "machine", _at(100 + s), apps=["Meet"])
        scene.heard("V", "machine", _at(200 + s), apps=["Firefox"])
    before = scene.of("C")["remote"]
    scene.tell("A", "remote")
    assert scene.of("A") == {"room": 0, "remote": 1, "media": 0, "self": 0}
    assert scene.of("C")["remote"] > before
    assert scene.of("V")["remote"] == before  # another application learned nothing


def test_the_scene_round_trips():
    scene = Scene()
    _timeline(scene, 40, [(s // 4) % 2 == 0 for s in range(40)], [(s // 4) % 2 for s in range(40)])
    scene.tell("B", "room")
    again = Scene()
    again.load(scene.to_dict())
    assert again.of("A") == scene.of("A") and again.of("B") == scene.of("B")
