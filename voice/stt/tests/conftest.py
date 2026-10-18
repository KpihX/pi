"""Test-wide settings. A test must never touch the operator's home directory.

Measured 2026-10-19: two tests called ``Config.ensure()`` while the durable corpus was
enabled by default, so simply running the suite created ``$HOME/.cache/π/corpus`` in the
operator's home. The corpus is a real, bounded copy of real audio and belongs to a real
session, never to a test run.

The first fix was to switch the corpus off for every test, and it was too blunt. The
command line reads its defaults from the environment, so the CLI test that asks for a
corpus through ``--corpus-dir`` got a corpus that was silently off and its assertion
failed: disabling a feature is not the same thing as isolating it. The default directory
now points inside pytest's own temporary tree instead. The corpus stays on, exactly as in
production, and the operator's home is never written to.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _corpus_never_touches_home(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("PI_EAR_CORPUS_DIR", str(tmp_path_factory.mktemp("corpus")))
