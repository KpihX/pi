"""Which application plays what (``device.py``): what the sound server says, kept per second."""

from __future__ import annotations

import time

from π_ear.device import Apps, follow, playing
from π_ear.io import read_jsonl

# the shape `pactl -f json list sink-inputs` prints (pipewire-pulse 1.0), trimmed
LISTING = [
    {
        "index": 71,
        "corked": False,
        "properties": {
            "application.name": "Firefox",
            "application.process.binary": "firefox",
            "media.name": "PSG - Barça | Canal+",
        },
    },
    {
        "index": 88,
        "corked": True,
        "properties": {"application.name": "Meet", "media.name": "Meet - Groupe de TD"},
    },
    {"index": 90, "properties": {"node.name": "π-mouth"}},
]


def test_the_sound_server_s_listing_is_read_as_names_and_playing_or_not():
    apps = playing(LISTING)
    assert [a["app"] for a in apps] == ["Firefox", "Meet", "π-mouth"]
    assert [a["playing"] for a in apps] == [True, False, True]
    assert apps[0]["media"] == "PSG - Barça | Canal+" and apps[0]["binary"] == "firefox"


def test_a_line_per_second_while_something_plays_and_one_when_it_stops(tmp_path):
    answers = [LISTING[:1]] * 3 + [[]] * 3
    now = [time.mktime((2026, 10, 21, 21, 30, 0, 0, 0, -1))]

    def ask():
        return answers.pop(0) if answers else []

    def sleep(seconds):
        now[0] += 1

    written = follow(tmp_path / "apps.jsonl", ask=ask, clock=lambda: now[0], sleep=sleep, seconds=6)
    lines = read_jsonl(tmp_path / "apps.jsonl")[0]
    assert written == len(lines) == 4  # three seconds playing, then one saying it stopped
    assert [line["chunk"] for line in lines][:2] == ["20261021-213000", "20261021-213001"]
    assert lines[-1]["apps"] == []


def test_what_played_during_a_stretch_includes_what_was_already_playing(tmp_path):
    path = tmp_path / "apps.jsonl"
    path.write_text(
        '{"chunk": "20261021-213000", "apps": [{"app": "Firefox", "playing": true}]}\n'
        '{"chunk": "20261021-213010", "apps": [{"app": "Firefox", "playing": true},'
        ' {"app": "Meet", "playing": true}]}\n'
        '{"chunk": "20261021-213020", "apps": []}\n'
    )
    apps = Apps(path)
    from π_ear.io import chunk_index

    t = chunk_index("20261021-213000")
    assert apps.during(t + 3, t + 5) == ["Firefox"]
    assert apps.during(t + 8, t + 12) == ["Firefox", "Meet"]
    assert apps.during(t + 21, t + 25) == []
