"""Unit tests for the inventory: the claims the mechanism rests on."""

from __future__ import annotations

import pytest
from synthetic import run_session, voice

from π_ear.config import Config
from π_ear.inventory import Entry, Inventory, centroid_key
from π_ear.physics import analyse_window, cosine, identity_vector


def features_for(cfg: Config, f0: float, seconds: float = 3.0) -> dict:
    samples = voice(seconds, cfg.rate, f0, seed=int(f0))
    return analyse_window(samples, cfg)


def test_key_is_stable_and_versioned():
    vec = identity_vector({"f0_hz": 120.0, "centroid_hz": 900.0, "rms_dbfs": -30.0, "zcr": 0.05})
    assert centroid_key(vec, 1) == centroid_key(vec.copy(), 1)
    assert len(centroid_key(vec, 1)) == 16
    # a re-commitment must be a different key, never a silent mutation
    assert centroid_key(vec, 1) != centroid_key(vec, 2)


def test_one_observation_cannot_create_a_speaker(tmp_path):
    cfg = Config(root=tmp_path, k_evidence=3, corpus=False)
    inventory = Inventory(cfg)
    decision = inventory.tick(features_for(cfg, 120.0), t=1.0)
    assert decision["kind"] == "unknown"
    assert inventory.registered == {}
    assert len(inventory.candidates) == 1


def test_registration_needs_the_evidence_bar(tmp_path):
    cfg = Config(root=tmp_path, k_evidence=3, corpus=False)
    inventory = Inventory(cfg)
    kinds = [
        inventory.tick(features_for(cfg, f0, seconds=3.0), t=float(index + 1))
        for index, f0 in enumerate((120.0, 121.0, 119.0))
    ]
    assert kinds[0]["kind"] == "unknown"
    assert kinds[1]["kind"] == "candidate"
    assert kinds[2]["kind"] == "new"
    assert len(inventory.registered) == 1
    assert inventory.candidates == {}


def test_the_bar_is_measured_from_the_sensor_not_counted_from_the_inventory(tmp_path):
    """What this replaced. The old criterion was
    ``tau = min(tau_base + lam * N_registered, tau_max)``: every registration moved
    the boundary, so the boundary could not hold anything. Measured 2026-10-19 on
    379 seconds of real audio: 56 identities for two voices, with tau pinned at its
    ceiling. The bar now moves with the noise this sensor produces, and registering
    a speaker does not move it at all."""
    cfg = Config(root=tmp_path, corpus=False)
    inventory = Inventory(cfg)
    assert inventory.margin_required == 0.0  # nothing observed yet: the floor decides
    assert inventory.tau == cfg.tau_floor

    inventory.margins.extend([0.010, 0.020, 0.015, 0.012, 0.018, 0.011, 0.013, 0.017])
    assert inventory.sigma > 0.0
    assert inventory.margin_required == pytest.approx(cfg.z_margin * inventory.sigma)

    before = inventory.margin_required
    for index in range(5):
        key = f"placeholder_{index}"
        inventory.registered[key] = Entry(
            id=key, kind="registered", vec=identity_vector(features_for(cfg, 100.0 + index))
        )
    assert inventory.registered  # five speakers in the inventory now
    assert inventory.margin_required == before


def test_one_voice_never_multiplies_into_many_entries(tmp_path):
    """Twelve windows of one voice: one entry, carrying twelve observations.

    This is the measured failure in one test. Before the invariant existed, a
    candidate crossed the evidence bar every three windows and became a new
    identity: a minute of one voice held twenty of them, and the log could not say
    which of the twenty was actually the person."""
    cfg = Config(root=tmp_path, k_evidence=3, corpus=False)
    inventory = Inventory(cfg)

    for index in range(12):
        inventory.tick(features_for(cfg, 120.0 + index * 0.3), t=float(index))

    assert len(inventory.registered) == 1
    only = next(iter(inventory.registered.values()))
    assert only.evidence == 12


def test_a_candidate_that_is_already_a_known_voice_is_absorbed(tmp_path):
    """The invariant, isolated, on a geometry the test controls.

    A candidate that passes the test a window passes IS that voice, so promoting it
    must absorb it into the entry that already describes it, never create a second
    key. The absorbed key is kept in ``absorbed``: a re-attribution after the fact
    is normal in this record, losing the old key would not be."""
    cfg = Config(root=tmp_path, corpus=False)
    inventory = Inventory(cfg)
    features = features_for(cfg, 120.0)
    key = "known_voice"
    inventory.registered[key] = Entry(
        id=key, kind="registered", vec=identity_vector(features), evidence=3
    )

    candidate = inventory._new_candidate(identity_vector(features), features, 1.0)  # noqa: SLF001
    for _ in range(3):
        candidate.absorb(identity_vector(features))

    promoted, merged = inventory._promote(candidate)  # noqa: SLF001

    assert merged is True
    assert promoted.id == key
    assert len(inventory.registered) == 1
    assert candidate.id in inventory.absorbed
    assert inventory.absorbed[candidate.id] == key
    assert promoted.evidence > 3  # the evidence of both entries, not a fresh start


def test_two_voices_this_sensor_can_separate_stay_two_entries(tmp_path):
    """The invariant must not collapse voices that are genuinely different.

    The pair is measured here, never assumed: its cosine is 0.854 in this space, so
    a floor of 0.9 is what says "these two are not the same person". With the
    default floor of 0.5 they would merge, and that is exactly what v0 does with
    real voices: a synthetic 110 Hz voice and a 215 Hz voice sit at 0.962, which is
    why the pole needs CAM++ and not a better threshold."""
    cfg = Config(root=tmp_path, tau_floor=0.9, z_margin=0.0, corpus=False)
    inventory = Inventory(cfg)
    low = {
        "f0_hz": 100.0,
        "centroid_hz": 600.0,
        "rms_dbfs": -20.0,
        "zcr": 0.03,
        "voiced_ratio": 0.7,
    }
    high = {
        "f0_hz": 380.0,
        "centroid_hz": 6000.0,
        "rms_dbfs": -40.0,
        "zcr": 0.30,
        "voiced_ratio": 0.7,
    }
    assert cosine(identity_vector(low), identity_vector(high)) == pytest.approx(0.854, abs=0.01)

    # the default evidence bar: three mutually compatible observations
    assert [inventory.tick(low, t=float(step))["kind"] for step in range(3)] == [
        "unknown",
        "candidate",
        "new",
    ]
    assert len(inventory.registered) == 1

    # the second voice is 0.854 away, below the floor: it is not that person
    assert [inventory.tick(high, t=float(step))["kind"] for step in range(3, 6)] == [
        "unknown",
        "candidate",
        "new",
    ]

    assert len(inventory.registered) == 2
    assert inventory.separation is not None
    assert inventory.separation == pytest.approx(0.854, abs=0.01)
    assert inventory.separation < cfg.tau_floor


def test_silence_creates_no_phantom(tmp_path):
    cfg = Config(root=tmp_path)
    inventory = Inventory(cfg)
    quiet = {"rms_dbfs": -95.0, "f0_hz": 0.0, "centroid_hz": 120.0, "zcr": 0.4, "voiced_ratio": 0.0}
    for index in range(10):
        decision = inventory.tick(quiet, t=float(index))
        assert decision["kind"] == "silent"
    assert inventory.registered == {}
    assert inventory.candidates == {}


def test_synthetic_session_is_speaker_consistent(tmp_path):
    cfg = Config(root=tmp_path, window_s=1.0, corpus=False)
    result = run_session(cfg)
    labels = result["decisions"]
    inventory = result["inventory"]
    # the first speaker must be recognised again after the gap
    assert labels[0][1] == "unknown"
    assert labels[-1][1] == "known"
    assert labels[-1][2] == labels[2][2]
    # and no entry may be admitted as another one: whatever the count, it has to be
    # defensible. With this placeholder vector the count is 1, because 110 Hz and
    # 215 Hz sit at a cosine of 0.962: they look like one voice, and saying so is
    # the honest answer, not a regression.
    assert len(inventory.registered) >= 1
    for key in inventory.registered:
        admitted, *_ = inventory._match(inventory.registered[key].vec)  # noqa: SLF001
        assert admitted in (None, key), f"{key} is admitted as {admitted}: two keys, one voice"
