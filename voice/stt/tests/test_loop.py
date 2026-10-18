"""Two regressions in the server loop, both measured on the real ear.

Both have the same shape: a control that looked implemented and was not. Neither
one is visible in a state file, which is exactly why they lasted through every
green test run before them.
"""

from __future__ import annotations

import json

import numpy as np

from π_ear.config import Config
from π_ear.live import Ear


def _fill(cfg: Config, chunks: int = 3) -> None:
    """A frozen directory of complete chunks, so the tailer has work to do."""
    cfg.ensure()
    rng = np.random.default_rng(7)
    frames = cfg.expected_chunk_bytes // 2
    for index in range(chunks):
        samples = (rng.standard_normal(frames) * 3000).astype(np.int16)
        (cfg.audio_dir / f"20260101-00000{index}.raw").write_bytes(samples.tobytes())


def test_the_loop_sleeps_when_there_is_nothing_to_read(tmp_path, monkeypatch):
    """The sleep was unreachable: it sat after the `break`, so a live ear spun.

    Measured 2026-10-19 on the real ear: 399 ticks of CPU per 400 over four
    seconds, one core pinned at 99.9%, while the analysis itself costs 28 ms.
    The loop's whole job every tenth of a second is to notice that nothing is new.
    """
    slept: list[float] = []
    monkeypatch.setattr("π_ear.live.wait", lambda seconds: slept.append(seconds))
    cfg = Config(root=tmp_path, idle_sleep_s=0.01, corpus=False)
    Ear(cfg).run(max_seconds=0.3)
    assert slept, "the loop never waited: it spun instead"


def test_pause_stops_the_cursor_and_the_state_says_so(tmp_path):
    """PAUSE did not pause: the loop waited, then fell through to the consumer.

    Measured 2026-10-19: with the PAUSE file present, the cursor advanced anyway
    and the state was rewritten with paused=False. The ear lied about its own
    state, and no test had ever asserted the contract.
    """
    cfg = Config(root=tmp_path, idle_sleep_s=0.01, corpus=False)
    _fill(cfg)
    cfg.pause_file.touch()
    Ear(cfg, from_start=True, keep_audio=True).run(max_seconds=0.3)
    payload = json.loads(cfg.state_file.read_text(encoding="utf-8"))
    assert payload["paused"] is True
    assert payload["audio"]["chunks_seen"] == 0
    assert payload["inventory"]["counters"]["windows"] == 0
