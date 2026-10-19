"""The command line. Thin on purpose.

π-ear run        the ear, live, following the recorder's directory
π-ear ctl        pause | resume | reset | status
π-ear correct    same | different | not_voice | name   ROWS...  (one case, obeyed now)
π-ear snapshot   freeze the state, or --diff the last two
π-ear replay     a recorded directory (the corpus), from a fresh or the current memory
π-ear fetch      the models and the public corpora
π-ear base       re-measure the base weights on public voices
"""

from __future__ import annotations

import enum
import json
import shutil
from pathlib import Path

import typer

from π_ear import contract
from π_ear.ear import Ear, Places, base_weights_path, correction
from π_ear.io import diff_last_two, read_json, write_atomic
from π_ear.io import snapshot as take_snapshot

app = typer.Typer(add_completion=False, help="π-ear: the streaming ear of π.")

ROOT = typer.Option(None, help="live root (default /tmp/π)")
MEMORY = typer.Option(None, help="what the ear has learned (default ~/.cache/π/ear)")


def _places(root=None, memory=None, **extra) -> Places:
    kwargs = {k: v for k, v in dict(root=root, memory=memory, **extra).items() if v is not None}
    return Places(**kwargs)


@app.command()
def run(
    root: str = ROOT,
    memory: str = MEMORY,
    audio_dir: str = typer.Option(None, help="read chunks here instead of <root>/audio"),
    from_start: bool = typer.Option(False, "--from-start", help="ingest the backlog first"),
    corpus: bool = typer.Option(True, "--corpus/--no-corpus", help="keep a durable copy"),
    corpus_dir: str = typer.Option(None, help="where the durable corpus lives"),
    asr: bool = typer.Option(True, "--asr/--no-asr", help="transcribe finished speech runs"),
    max_seconds: float = typer.Option(None, help="stop after N seconds"),
):
    """Follow the recorder forever. The state on disk is live; Ctrl+C stops."""
    places = _places(root, memory, audio=audio_dir)
    if not corpus:
        places.corpus = None
    elif corpus_dir:
        places.corpus = Path(corpus_dir)
    ear = Ear(places, from_start=from_start, asr=asr)
    typer.echo(f"π-ear  audio={places.audio_dir}  memory={places.memory}  rows={len(ear.store)}")
    typer.echo(f"       weights={ear.model.base.source}")
    try:
        ear.run(max_seconds=max_seconds, echo=typer.echo)
    except KeyboardInterrupt:
        ear.persist()
        typer.echo("\nstopped, memory saved")


@app.command()
def replay(
    audio_dir: str = typer.Argument(..., help="a directory of recorded chunks (e.g. the corpus)"),
    root: str = ROOT,
    memory: str = typer.Option(None, help="memory to replay into (default: a fresh one in <root>)"),
    asr: bool = typer.Option(False, "--asr/--no-asr", help="transcribe while replaying"),
    quiet: bool = typer.Option(False, "--quiet", help="no line per decision"),
):
    """Run a recorded session through the ear, to the end, then stop."""
    places = _places(root, memory, audio=audio_dir)
    if memory is None:
        places.memory = places.root / "replay-memory"
        shutil.rmtree(places.memory, ignore_errors=True)
        places.events.unlink(missing_ok=True)
    places.corpus = None
    ear = Ear(places, from_start=True, live=False, asr=asr)
    ear.run(drain=True, echo=None if quiet else typer.echo)
    typer.echo(json.dumps(ear.state()["model"]))


class Action(enum.StrEnum):
    pause = "pause"
    resume = "resume"
    reset = "reset"
    status = "status"


@app.command()
def ctl(action: Action, root: str = ROOT):
    """Control files, not signals: the state says what the ear is doing."""
    places = _places(root)
    places.ensure()
    if action is Action.pause:
        places.pause.touch()
        typer.echo("paused: the recorder keeps recording, the audio becomes a backlog")
    elif action is Action.resume:
        places.pause.unlink(missing_ok=True)
        typer.echo("resumed: the backlog is drained")
    elif action is Action.reset:
        places.reset.touch()
        typer.echo("reset requested: everything learned is archived, the ear starts from its base")
    else:
        state = read_json(places.state)
        if not state:
            typer.echo(f"no state yet at {places.state}")
            raise typer.Exit(code=1)
        typer.echo(
            f"revision {state['revision']}  session {state['session']}  stream {state['stream_s']}s  "
            f"paused {state['paused']}  backlog {state['audio']['backlog_chunks']}"
        )
        typer.echo(f"model    {json.dumps(state['model'])}")
        for s in state["speakers"]:
            typer.echo(
                f"  {s['id']:18s} {s['kind']:8s} {s['label'] or '':12s} evidence={s['evidence']:<5d}"
                f" silent_for={s['silent_for_s']}s"
            )
        typer.echo(f"counters {json.dumps(state['counters'])}")


class Kind(enum.StrEnum):
    same = "same"
    different = "different"
    not_voice = "not_voice"
    name = "name"


@app.command()
def correct(
    kind: Kind,
    rows: list[int] = typer.Argument(None, help="row numbers, as printed in events.jsonl"),
    speaker: list[str] = typer.Option(None, help="every row of this speaker key (repeatable)"),
    label: str = typer.Option(None, help="for `name`: what to call this voice"),
    memory: str = MEMORY,
):
    """Say once what was wrong. The running ear obeys at once and learns from it.

    \b
      π-ear correct same 812 840            these two windows are one voice
      π-ear correct different 812 901       these two are two voices
      π-ear correct not_voice --speaker KEY that "speaker" was never a voice
      π-ear correct name 812 --label KπX    call this voice KπX
    """
    places = _places(memory=memory)
    rows = list(rows or [])
    if speaker:
        saved = read_json(places.checkpoint, {}) or {}
        clusters = {c["key"]: c for c in saved.get("model", {}).get("clusters", [])}
        for key in speaker:
            if key not in clusters:
                raise typer.BadParameter(f"no speaker {key} in {places.checkpoint}")
            rows += clusters[key]["members"]
    if not rows:
        raise typer.BadParameter("give rows, or --speaker")
    if kind is Kind.name and not label:
        raise typer.BadParameter("`name` needs --label")
    line = correction(places, kind.value, rows, label)
    typer.echo(f"written: {json.dumps(line)}")


@app.command()
def snapshot(
    tag: str = typer.Option(None), diff: bool = typer.Option(False, "--diff"), root: str = ROOT
):
    """Freeze a comparable copy of the state, or diff the last two."""
    places = _places(root)
    if diff:
        typer.echo(diff_last_two(places.snapshots))
    else:
        typer.echo(f"snapshot: {take_snapshot(places.state, places.snapshots, tag)}")


@app.command()
def fetch(what: str = typer.Argument("all", help="models | data | all")):
    """Download the models and the public test corpora (~470 MB, once)."""
    from π_ear import fetch as downloads

    if what in ("models", "all"):
        downloads.models(echo=typer.echo)
    if what in ("data", "all"):
        downloads.data(echo=typer.echo)


@app.command()
def base():
    """Re-measure the base weights on public voices the tests never use."""
    from π_ear.public import fit_base
    from π_ear.sense import Sensor

    weights = fit_base(Sensor(contract.MODELS), contract.DATA, contract.MODELS)
    target = base_weights_path(contract.ENCODER)
    write_atomic(target, weights.to_dict())
    typer.echo(f"{target}\n{weights.source}")


def main() -> None:
    app()
