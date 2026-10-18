"""Synthetic audio, so the mechanism can be tested without a microphone.

A voice appears, disappears, a second voice appears, then the first one comes
back. A working inventory ends with two registered speakers and no phantom from
the silence. Keeping this in ``tests/`` is the point: the package ships the
system, the tests build the world it is checked against.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from π_ear.config import Config
from π_ear.inventory import Inventory
from π_ear.physics import analyse_window


def voice(seconds: float, rate: int, f0: float, seed: int) -> np.ndarray:
    """A crude but honest synthetic voice: harmonic stack, pitch drift, noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    signal = np.zeros_like(t)
    for harmonic, weight in enumerate((1.0, 0.5, 0.33, 0.25, 0.15), start=1):
        signal += weight * np.sin(2 * np.pi * f0 * harmonic * t)
    # a stationary tone is too easy: add a syllable envelope and a slow drift
    signal *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * t)
    signal += 0.02 * rng.standard_normal(t.size)
    signal /= max(1e-9, float(np.abs(signal).max()))
    return (signal * 0.25).astype(np.float32)


def silence(seconds: float, rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.0002 * rng.standard_normal(int(seconds * rate))).astype(np.float32)


def session(cfg: Config) -> list[tuple[str, np.ndarray]]:
    """Two speakers, one second each window, one gap, then the first returns.

    One second per segment on purpose: with ``window_s = 1.0`` every analysis
    window is homogeneous, so the test measures the inventory and not the
    cross-fade between two voices.
    """
    return [
        ("voice_A", voice(1.0, cfg.rate, 110.0, 1)),
        ("voice_A", voice(1.0, cfg.rate, 108.0, 2)),
        ("voice_A", voice(1.0, cfg.rate, 112.0, 3)),
        ("voice_B", voice(1.0, cfg.rate, 215.0, 4)),
        ("voice_B", voice(1.0, cfg.rate, 218.0, 5)),
        ("voice_B", voice(1.0, cfg.rate, 212.0, 6)),
        ("silence", silence(1.0, cfg.rate, 7)),
        ("silence", silence(1.0, cfg.rate, 8)),
        ("voice_A", voice(1.0, cfg.rate, 111.0, 9)),
        ("voice_A", voice(1.0, cfg.rate, 109.0, 10)),
    ]


def run_session(cfg: Config, echo=None) -> dict:
    """Feed the synthetic session through the real sliding-window pipeline."""
    inventory = Inventory(cfg)
    buffer: list[np.ndarray] = []
    window_frames = int(cfg.rate * cfg.window_s)
    clock = 0.0
    decisions: list[tuple[str, str, str | None]] = []

    for label, samples in session(cfg):
        buffer.append(samples)
        total = sum(int(part.size) for part in buffer)
        while len(buffer) > 1 and total - int(buffer[0].size) >= window_frames:
            total -= int(buffer.pop(0).size)
        features = analyse_window(np.concatenate(buffer), cfg)
        clock += len(samples) / cfg.rate
        decision = inventory.tick(features, clock)
        decisions.append((label, decision["kind"], decision.get("id")))
        if echo:
            echo(
                f"  truth={label:8s} -> {decision['kind']:9s} id={decision.get('id')} "
                f"cos={decision.get('cos_best')} tau={decision.get('tau')} "
                f"f0={features['f0_hz']:.1f}Hz voiced={features['voiced_ratio']:.2f}"
            )

    return {
        "registered": len(inventory.registered),
        "candidates": len(inventory.candidates),
        "decisions": decisions,
        "inventory": inventory,
    }


def write_chunks(cfg: Config, plan: list[tuple[str, np.ndarray]] | None = None) -> Path:
    """Persist the session as real one-second raw chunks, for pipeline tests."""
    cfg.ensure()
    plan = plan or session(cfg)
    base = int(time.time())
    step = cfg.rate * cfg.chunk_s
    index = 0
    for _, samples in plan:
        for start in range(0, samples.size, step):
            piece = samples[start : start + step]
            if piece.size < step:
                continue
            name = time.strftime("%Y%m%d-%H%M%S", time.localtime(base + index))
            (piece * 32767.0).astype("<i2").tofile(cfg.audio_dir / f"{name}.raw")
            index += 1
    return cfg.audio_dir
