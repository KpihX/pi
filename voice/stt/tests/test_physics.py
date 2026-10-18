"""The pitch detector, and the artefact a real run exposed.

Written after running `make run` on the real microphone at 23:45 on 2026-10-18:
every window reported `f0 = 457.1 Hz`, silence included, and a phantom speaker was
registered from room and keyboard noise. 457.1 is 16000/35, the first lag of the
pitch search range: when nothing is periodic, the maximum of the autocorrelation
falls on the boundary and the detector announced the highest frequency it could
express. The tests below hold that failure down.
"""

from __future__ import annotations

import numpy as np
from synthetic import voice

from π_ear.config import Config
from π_ear.inventory import Inventory
from π_ear.physics import analyse_window


def test_white_noise_yields_no_pitch():
    """Loud broadband noise: energy yes, period no."""
    cfg = Config()
    rng = np.random.default_rng(0)
    signal = (0.05 * rng.standard_normal(cfg.rate)).astype(np.float32)
    features = analyse_window(signal, cfg)
    assert features["f0_hz"] == 0.0
    assert features["rms_dbfs"] > cfg.level_floor_db  # the level gate is not what saves us


def test_transient_bursts_yield_no_pitch():
    """Keyboard-like clicks: broadband, sharp, repeating, and not periodic."""
    cfg = Config()
    rng = np.random.default_rng(1)
    total = cfg.rate
    signal = 0.0005 * rng.standard_normal(total)
    burst = int(0.004 * cfg.rate)
    for start in range(0, total - burst, int(0.11 * cfg.rate)):
        signal[start : start + burst] += 0.08 * rng.standard_normal(burst)
    features = analyse_window(signal.astype(np.float32), cfg)
    assert features["f0_hz"] == 0.0


def test_a_real_voice_still_gets_a_pitch():
    """The fix must not deafen the ear: a voice keeps its pitch."""
    cfg = Config()
    features = analyse_window(voice(1.0, cfg.rate, 120.0, 3), cfg)
    assert 100.0 < features["f0_hz"] < 140.0
    assert features["voiced_ratio"] > 0.5


def test_noise_bursts_register_no_speaker(tmp_path):
    """The end of the chain: transients must never become a person."""
    cfg = Config(root=tmp_path)
    rng = np.random.default_rng(2)
    inventory = Inventory(cfg)
    for index in range(10):
        signal = np.zeros(cfg.rate, dtype=np.float32)
        burst = (0.08 * rng.standard_normal(int(0.004 * cfg.rate))).astype(np.float32)
        signal[: burst.size] = burst
        decision = inventory.tick(analyse_window(signal, cfg), t=float(index))
        assert decision["kind"] == "silent"
    assert inventory.registered == {}
    assert inventory.candidates == {}
