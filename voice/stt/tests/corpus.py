"""Replaying a recorded session through the inventory, with no audio at all.

The files in ``tests/artefacts/`` are the real event logs of the sessions that
broke things, copied out of ``/tmp`` before the sweep. A log line carries the four
numbers the identity vector is built from, so a whole session can be pushed through
a *new* inventory and compared with what the old one did. That is how a fix gets
proven on the failure itself instead of on a synthetic mood.

Measured on the hum-then-video session, 2026-10-18, before the fix: 56 registered
speakers for two real voices, one voice present in eight separate entries, a median
best-minus-runner-up margin of 0.0005, and an admission threshold pinned at its
ceiling. The same 414 windows produce one entry after the fix.
"""

from __future__ import annotations

import json
from pathlib import Path

from π_ear.config import Config
from π_ear.inventory import Inventory

ARTEFACTS = Path(__file__).parent / "artefacts"
HUM_THEN_VIDEO = ARTEFACTS / "2026-10-18-hum-then-video.events.jsonl"
FIRST_RUN = ARTEFACTS / "2026-10-18-first-run.events.jsonl"


def windows(path: Path) -> list[dict]:
    """One event per chunk: the last one, which describes the completed window.

    The old log carried about ten events per chunk, because a file being written
    was read as it grew. Replaying all of them would replay the bug instead of the
    audio.
    """
    last: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        last[event["chunk"]] = event
    return [last[name] for name in sorted(last)]


def replay(path: Path, cfg: Config) -> Inventory:
    """Push a recorded session through a fresh inventory."""
    inventory = Inventory(cfg)
    for event in windows(path):
        inventory.tick(dict(event["voice"]), t=event["t_s"])
    return inventory
