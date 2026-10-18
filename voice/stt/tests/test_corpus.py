"""The failure corpus, replayed: the fix has to survive the session that broke it.

These tests are the reason ``tests/artefacts/`` exists. They run on real numbers
from a real microphone, on two real voices (a hummed note and a film), and they
compare the inventory against what it did before the fix, which is recorded in the
docstrings so the failure is never anecdotal.
"""

from __future__ import annotations

from corpus import FIRST_RUN, HUM_THEN_VIDEO, replay, windows

from π_ear.config import Config


def test_the_real_session_no_longer_multiplies_speakers(tmp_path):
    """Before the fix: 56 registered speakers, 8 entries for one voice, every entry
    at a cosine of 0.9999 from the others, and no threshold able to repair it."""
    cfg = Config(root=tmp_path, corpus=False)
    assert len(windows(HUM_THEN_VIDEO)) > 350

    inventory = replay(HUM_THEN_VIDEO, cfg)

    assert inventory.counters["registered_new"] <= 2
    assert len(inventory.registered) <= 2


def test_the_first_recording_behaves_the_same(tmp_path):
    cfg = Config(root=tmp_path, corpus=False)

    inventory = replay(FIRST_RUN, cfg)

    assert inventory.counters["registered_new"] <= 2


def test_the_replay_keeps_the_silence_out_of_the_inventory(tmp_path):
    cfg = Config(root=tmp_path, corpus=False)

    inventory = replay(HUM_THEN_VIDEO, cfg)

    assert inventory.counters["silent"] > 0
    for entry in inventory.registered.values():
        assert entry.evidence >= cfg.k_evidence


def test_the_two_entries_this_sensor_holds_are_never_indistinguishable(tmp_path):
    """The invariant, on real data: no entry may be admitted as another one."""
    cfg = Config(root=tmp_path, corpus=False)

    inventory = replay(HUM_THEN_VIDEO, cfg)

    for key in inventory.registered:
        admitted, *_ = inventory._match(inventory.registered[key].vec)  # noqa: SLF001
        assert admitted in (None, key), f"{key} is admitted as {admitted}: one voice, two keys"
