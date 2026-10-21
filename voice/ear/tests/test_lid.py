"""A voice's language over all it said (``lid.py``): its speech pooled, detected once."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_data

from π_ear import contract, lid
from π_ear.ear import base_weights_path
from π_ear.io import read_json
from π_ear.language import Languages
from π_ear.public import synth

pytestmark = [
    needs_data,
    pytest.mark.skipif(not lid.available(contract.MODELS), reason="no language identifier"),
]
SAID = [
    "je voudrais qu'on reprenne le chapitre trois demain matin",
    "envoie le compte rendu à toute l'équipe avant midi",
    "on se retrouve devant la bibliothèque à quatorze heures",
]


def test_a_voice_s_speech_is_pooled_until_the_window_is_full_then_detected_once():
    clips = list(synth(contract.MODELS, [1], SAID, language="fr").values())[0]
    pooled = lid.Pooled(contract.MODELS)
    found = [pooled.add("v", clip) for _ in range(6) for clip in clips]
    told = [f for f in found if f is not None]
    assert found[0] is None and len(told) >= 1  # nothing before the window is full
    assert all(language == "fr" and seconds == 30.0 for language, seconds in told)
    assert sum(len(p) for p in pooled.heard["v"]) < pooled.window  # the rest waits
    assert pooled.add("w", clips[0]) is None  # another voice has its own pool


def test_a_pooled_detection_moves_a_voice_as_its_base_reliability_says():
    """Measured on the base (``public.measure_pooled``): one pooled detection of 30 s
    counts nearly as one whole reading, so it moves a new voice's language at once."""
    base = read_json(base_weights_path(contract.ENCODER))["reader"]["languages"]
    assert any(row[2] == lid.POOLED for row in base["base"])
    languages = Languages()
    languages.load(base)
    languages.refit()
    right = languages.detection_right(30.0, None, lid.POOLED)
    assert right is not None and right > 1 - right
    languages.heard("guest", "en", "", True, 30.0, lid.POOLED)
    languages.heard("kpihx", "fr", "", True, 30.0, lid.POOLED)
    languages.refit()
    assert max(languages.of_voice("guest"), key=languages.of_voice("guest").get) == "en"
    assert max(languages.of_voice("kpihx"), key=languages.of_voice("kpihx").get) == "fr"
    assert np.isfinite(right)
