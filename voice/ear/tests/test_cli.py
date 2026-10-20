"""The command line, as an operator types it."""

from __future__ import annotations

import json

from conftest import needs_data
from typer.testing import CliRunner

from π_ear.cli import app
from π_ear.public import Scene, libri

runner = CliRunner()


def test_status_without_a_state_says_so(tmp_path):
    result = runner.invoke(app, ["ctl", "status", "--root", str(tmp_path)])
    assert result.exit_code == 1 and "no state yet" in result.output


def test_pause_and_resume_are_files(tmp_path):
    runner.invoke(app, ["ctl", "pause", "--root", str(tmp_path)])
    assert (tmp_path / "PAUSE").exists()
    runner.invoke(app, ["ctl", "resume", "--root", str(tmp_path)])
    assert not (tmp_path / "PAUSE").exists()


def test_a_correction_is_one_appended_line(tmp_path):
    for args in (["same", "3", "7"], ["name", "3", "--label", "KπX"]):
        result = runner.invoke(app, ["correct", *args, "--memory", str(tmp_path)])
        assert result.exit_code == 0, result.output
    lines = [json.loads(x) for x in (tmp_path / "corrections.jsonl").read_text().splitlines()]
    assert [x["kind"] for x in lines] == ["same", "name"] and lines[1]["label"] == "KπX"


def test_name_without_a_label_is_refused(tmp_path):
    result = runner.invoke(app, ["correct", "name", "3", "--memory", str(tmp_path)])
    assert result.exit_code != 0


@needs_data
def test_replay_then_status_then_correct_by_speaker(tmp_path):
    from π_ear import contract

    scene = Scene(seed=0).silence(1).add("A", libri(contract.DATA)["libri5703"]).silence(1)
    scene.write(tmp_path / "chunks")
    root, memory = str(tmp_path / "live"), str(tmp_path / "memory")
    result = runner.invoke(
        app, ["replay", str(tmp_path / "chunks"), "--root", root, "--memory", memory, "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output.splitlines()[-1])["speakers"] == 1
    status = runner.invoke(app, ["ctl", "status", "--root", root])
    assert "speaker" in status.output
    key = next(
        c["key"]
        for c in json.loads((tmp_path / "memory" / "model.json").read_text())["model"]["clusters"]
        if c["kind"] == "speaker"
    )
    result = runner.invoke(
        app, ["correct", "name", "--speaker", key, "--label", "A", "--memory", memory]
    )
    assert result.exit_code == 0 and '"label": "A"' in result.output


@needs_data
def test_from_to_why_undo_and_probe(tmp_path):
    from π_ear import contract

    scene = Scene(seed=0).silence(2).add("A", libri(contract.DATA)["libri5703"]).silence(2)
    scene.write(tmp_path / "chunks")
    root, memory = str(tmp_path / "live"), str(tmp_path / "memory")
    runner.invoke(
        app, ["replay", str(tmp_path / "chunks"), "--root", root, "--memory", memory, "--quiet"]
    )

    result = runner.invoke(
        app,
        [
            "correct",
            "name",
            "--from",
            "4",
            "--to",
            "8",
            "--label",
            "A",
            "--root",
            root,
            "--memory",
            memory,
        ],
    )
    assert result.exit_code == 0, result.output
    line = json.loads((tmp_path / "memory" / "corrections.jsonl").read_text().splitlines()[0])
    assert line["rows"] and line["span"]["from"] == 4.0

    result = runner.invoke(app, ["why", str(line["rows"][0]), "--root", root, "--memory", memory])
    assert result.exit_code == 0 and "then:" in result.output and "behind=" in result.output

    result = runner.invoke(app, ["undo", "1", "--memory", memory])
    last = json.loads((tmp_path / "memory" / "corrections.jsonl").read_text().splitlines()[-1])
    assert result.exit_code == 0 and last == {**last, "kind": "undo", "line": 1}

    result = runner.invoke(
        app, ["probe", str(tmp_path / "chunks"), "--from", "0", "--to", "6", "--root", root]
    )
    assert result.exit_code == 0, result.output
    body = result.output.splitlines()
    assert len([b for b in body if "20261018-" in b]) == 6 and "speech probability mean" in body[-1]
