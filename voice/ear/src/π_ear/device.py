"""Which application plays what on this machine, second by second.

What the machine plays is captured as one stream (its output's monitor, ``make
rec-out``): the mix of every application. Which applications were playing is a
fact the sound server knows; it is written next to the chunks, one line per second,
keyed by the chunk the recorder writes in that second (``apps.jsonl``):

    {"chunk": "20261021-213004", "apps": [{"app": "Firefox", "media": "PSG - Barça",
     "binary": "firefox", "playing": true}]}

The ear reads it to say, of a voice that came out of the machine, which application
it came out of: a meeting, a video, a call, or π's own mouth (``contract.MOUTH_APP``).
Two applications playing at once are both named: the mix cannot be split by name.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from π_ear.io import append_jsonl


def playing(listing: list[dict]) -> list[dict]:
    """The streams of a ``pactl -f json list sink-inputs`` listing, as the ear names them."""
    out = []
    for stream in listing:
        props = stream.get("properties", {})
        out.append(
            {
                "app": props.get("application.name") or props.get("node.name") or "?",
                "binary": props.get("application.process.binary"),
                "media": props.get("media.name"),
                "playing": not stream.get("corked", False),
            }
        )
    return out


def listing() -> list[dict] | None:
    """What the sound server says is playing, or None if it cannot be asked."""
    try:
        raw = subprocess.run(
            ["pactl", "-f", "json", "list", "sink-inputs"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout
        return json.loads(raw or "[]")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def follow(path: Path, ask=listing, clock=time.time, sleep=time.sleep, seconds=None) -> int:
    """Write one line per second while the recorder runs; return the lines written.
    A line is written only when the answer changes or a new second starts with
    something playing: an idle machine writes nothing."""
    written, last = 0, None
    start = clock()
    while seconds is None or clock() - start < seconds:
        now = clock()
        found = ask()
        if found is not None:
            apps = playing(found)
            chunk = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
            if apps or last:
                append_jsonl(path, {"chunk": chunk, "apps": apps})
                written += 1
            last = apps
        sleep(max(0.0, 1.0 - (clock() - now)))
    return written


class Apps:
    """The ear's reading of ``apps.jsonl``: chunk second -> what was playing."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.by_second: dict[int, list[dict]] = {}

    def poll(self) -> None:
        from π_ear.io import chunk_index, read_jsonl

        lines, self.offset = read_jsonl(self.path, self.offset)
        for line in lines:
            self.by_second[chunk_index(line["chunk"])] = line["apps"]

    def during(self, first: int, last: int) -> list[str]:
        """The applications playing at any second of [first, last], by name."""
        self.poll()
        names: list[str] = []
        known = sorted(k for k in self.by_second if k <= last)
        # a second without a line repeats the last line before it (lines are written on change)
        start = max((k for k in known if k <= first), default=None)
        for k in known:
            if (start is not None and k >= start) or first <= k:
                for a in self.by_second[k]:
                    if a["playing"] and a["app"] not in names:
                        names.append(a["app"])
        return names
