"""The entry point must be tested, not only the library.

This file exists because of a real failure. Every unit test built its Config from
a ``Path`` object, while the command line hands a ``str`` over: ``make run``
crashed on ``self.root / "audio"`` and not a single test noticed. A pure function
tested only through its Python API says nothing about the way a user reaches it.

Everything below therefore goes through the command line, exactly as a user does.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from π_ear.cli import app
from π_ear.config import Config

runner = CliRunner()


def test_config_normalises_a_string_root(tmp_path):
    cfg = Config(root=str(tmp_path))
    assert isinstance(cfg.root, Path)
    cfg.ensure()
    assert cfg.audio_dir.is_dir()


def test_run_accepts_a_string_root(tmp_path):
    result = runner.invoke(app, ["run", "--root", str(tmp_path), "--max-seconds", "0.2"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "state.json").is_file()


def test_run_accepts_every_flag_the_makefile_passes(tmp_path):
    """The Makefile is how this is actually started, so its flags are the real
    interface. Keep this list in sync with EAR_ARGS in the Makefile: when the two
    drift, `make run` breaks and no unit test notices."""
    result = runner.invoke(
        app,
        [
            "run",
            "--root",
            str(tmp_path),
            "--device",
            "plughw:0,6",
            "--rate",
            "16000",
            "--chunk-s",
            "1",
            "--window-s",
            "3.0",
            "--k-evidence",
            "3",
            "--tau-floor",
            "0.5",
            "--z-margin",
            "3.0",
            "--voiced-floor",
            "0.25",
            "--corpus-dir",
            str(tmp_path / "corpus"),
            "--corpus-max-chunks",
            "60",
            "--max-seconds",
            "0.2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "corpus").is_dir()
    assert (tmp_path / "state.json").is_file()


def test_no_corpus_leaves_the_home_alone(tmp_path):
    result = runner.invoke(
        app, ["run", "--root", str(tmp_path), "--no-corpus", "--max-seconds", "0.2"]
    )
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "corpus").exists()


def test_ctl_status_accepts_a_string_root(tmp_path):
    result = runner.invoke(app, ["ctl", "status", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "no state yet" in result.output


def test_snapshot_accepts_a_string_root(tmp_path):
    (tmp_path / "state.json").write_text("{}\n", encoding="utf-8")
    result = runner.invoke(app, ["snapshot", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert list((tmp_path / "snapshots").glob("*.json"))
