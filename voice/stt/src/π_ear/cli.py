"""The command line. Thin on purpose: every flag maps to one Config field.

Two entry points, and they are exactly the two servers of the architecture plus
the two things you need while watching a live system: freeze it, compare it.

    π-ear run       the ear: reads the chunks, keeps the state live on disk
    π-ear ctl       pause | resume | reset | status
    π-ear snapshot  freeze the state, or diff the last two
    π-ear replay    re-run the whole backlog with fresh parameters
"""

from __future__ import annotations

import enum
import json

import typer

from π_ear.config import Config
from π_ear.live import Ear
from π_ear.state import archive, diff_last_two, read_json
from π_ear.state import snapshot as take_snapshot

app = typer.Typer(add_completion=False, help="π-ear: the streaming ear of π.")


class CtlAction(enum.StrEnum):
    pause = "pause"
    resume = "resume"
    reset = "reset"
    status = "status"


def _config(**overrides) -> Config:
    """Build a Config, ignoring the flags that were not given (None)."""
    clean = {key: value for key, value in overrides.items() if value is not None}
    return Config(**clean)


def _echo_config(cfg: Config) -> None:
    typer.echo("π-ear")
    for key, value in cfg.as_dict().items():
        typer.echo(f"  {key:16s} {value}")
    typer.echo(f"  {'state_file':16s} {cfg.state_file}")
    typer.echo(f"  {'events_file':16s} {cfg.events_file}")
    typer.echo("")


@app.command()
def run(
    root: str = typer.Option(None, help="runtime root (default /tmp/π)"),
    device: str = typer.Option(None, help="ALSA capture device (must match the recorder)"),
    rate: int = typer.Option(None, help="sample rate in Hz (the model contract is 16000)"),
    chunk_s: int = typer.Option(None, help="chunk duration, must match arecord --max-file-time"),
    window_s: float = typer.Option(None, help="analysis window in seconds"),
    k_evidence: int = typer.Option(None, help="observations needed before registering a speaker"),
    tau_floor: float = typer.Option(None, help="min cosine to the closest voice"),
    z_margin: float = typer.Option(
        None, help="how many sigma above the observed margin noise the winner must be"
    ),
    voiced_floor: float = typer.Option(
        None, help="min voiced ratio before a window counts as speech"
    ),
    from_start: bool = typer.Option(
        False, "--from-start", help="ingest the backlog instead of skipping it"
    ),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="do not delete consumed chunks"),
    audio_dir: str = typer.Option(None, help="read chunks from here instead of <root>/audio"),
    corpus: bool | None = typer.Option(
        None, "--corpus/--no-corpus", help="copy ingested chunks into the durable corpus"
    ),
    corpus_dir: str = typer.Option(None, help="where the durable corpus lives"),
    corpus_max_chunks: int = typer.Option(None, help="how many chunks the corpus keeps"),
    log_silence: bool = typer.Option(
        False, "--log-silence", help="log every window, silent ones included"
    ),
    max_seconds: float = typer.Option(None, help="stop after N seconds (tests, demos)"),
):
    """Follow the chunks directory forever and keep the state live."""
    cfg = _config(
        root=root,
        device=device,
        rate=rate,
        chunk_s=chunk_s,
        window_s=window_s,
        k_evidence=k_evidence,
        tau_floor=tau_floor,
        z_margin=z_margin,
        voiced_floor=voiced_floor,
        audio_dir_override=audio_dir,
        corpus=corpus,
        corpus_dir=corpus_dir,
        corpus_max_chunks=corpus_max_chunks,
    )
    cfg.ensure()
    _echo_config(cfg)
    typer.echo(f"waiting for chunks in {cfg.audio_dir}  (Ctrl+C stops)\n")
    ear = Ear(cfg, from_start=from_start, keep_audio=keep_audio)
    try:
        ear.run(max_seconds=max_seconds, log_silence=log_silence, echo=typer.echo)
    except KeyboardInterrupt:
        ear._persist(paused=False)
        typer.echo("\nstopped")


@app.command()
def ctl(
    action: CtlAction = typer.Argument(help="pause | resume | reset | status"),
    root: str = typer.Option(None, help="runtime root (default /tmp/π)"),
):
    """Freeze, unfreeze, wipe, or read the state. Files, not signals."""
    cfg = _config(root=root)
    cfg.ensure()
    if action is CtlAction.pause:
        cfg.pause_file.touch()
        typer.echo(f"paused: {cfg.pause_file} exists, the recorder keeps recording")
    elif action is CtlAction.resume:
        cfg.pause_file.unlink(missing_ok=True)
        typer.echo("resumed: the ear will drain the backlog")
    elif action is CtlAction.reset:
        cfg.reset_file.touch()
        typer.echo(f"reset requested: {cfg.reset_file} exists, the ear will archive and restart")
    else:
        payload = read_json(cfg.state_file)
        if not payload:
            typer.echo(f"no state yet at {cfg.state_file}")
            raise typer.Exit(code=1)
        inventory = payload.get("inventory", {})
        typer.echo(f"utc            {payload.get('utc')}")
        typer.echo(f"revision       {payload.get('revision')}")
        typer.echo(f"clock_s        {payload.get('clock_s')}")
        typer.echo(f"paused         {payload.get('paused')}")
        typer.echo(f"backlog        {payload.get('audio', {}).get('backlog_chunks')} chunks")
        typer.echo(f"tau_floor      {inventory.get('tau_now')}")
        typer.echo(
            f"margin_now     {inventory.get('margin_now')}  (sigma {inventory.get('margin_sigma')})"
        )
        typer.echo(
            f"separation     {inventory.get('separation')}  "
            "(max cosine between two entries: 1.0 = indistinguishable)"
        )
        typer.echo(f"registered     {len(inventory.get('registered', []))}")
        for entry in inventory.get("registered", []):
            typer.echo(
                f"  {entry['id']}  evidence={entry['evidence']:3d} "
                f"f0={entry['f0_med_hz']:6.1f}Hz level={entry['level_med_dbfs']:7.1f}dB "
                f"silent_for={entry['silent_for_s']}s"
            )
        typer.echo(f"candidates     {len(inventory.get('candidates', []))}")
        for entry in inventory.get("candidates", []):
            typer.echo(f"  {entry['id']}  evidence={entry['evidence']} f0={entry['f0_med_hz']}Hz")
        absorbed = inventory.get("absorbed", {})
        typer.echo(f"absorbed       {len(absorbed)} keys merged into a surviving entry")
        typer.echo(f"counters       {json.dumps(inventory.get('counters', {}))}")


@app.command()
def snapshot(
    tag: str = typer.Option(None, help="suffix for the snapshot file"),
    diff: bool = typer.Option(False, "--diff", help="show the diff between the last two snapshots"),
    root: str = typer.Option(None, help="runtime root (default /tmp/π)"),
):
    """Freeze the state into a comparable copy, or compare the last two."""
    cfg = _config(root=root)
    cfg.ensure()
    if diff:
        typer.echo(diff_last_two(cfg.snapshots_dir))
        return
    target = take_snapshot(cfg.state_file, cfg.snapshots_dir, tag=tag)
    typer.echo(f"snapshot: {target}")


@app.command()
def replay(
    window_s: float = typer.Option(None, help="analyse the backlog with a different window"),
    k_evidence: int = typer.Option(None, help="and a different evidence bar"),
    keep_audio: bool = typer.Option(True, "--keep-audio/--no-keep-audio", help="keep the backlog"),
    audio_dir: str = typer.Option(None, help="replay this directory (the corpus) instead"),
    max_seconds: float = typer.Option(None, help="stop after N seconds"),
    root: str = typer.Option(None, help="runtime root (default /tmp/π)"),
):
    """Re-run a whole backlog from scratch, with fresh parameters, and compare.

    A frozen directory is not a live one: nothing there is being written, so the
    completeness gate is off and the last, legitimately partial chunk of a
    recorded session is analysed instead of skipped.
    """
    cfg = _config(root=root, window_s=window_s, k_evidence=k_evidence, audio_dir_override=audio_dir)
    cfg.ensure()
    backlog = sorted(cfg.audio_dir.glob("*.raw"))
    if not backlog:
        typer.echo(f"nothing to replay in {cfg.audio_dir} (was the ear run with --keep-audio?)")
        raise typer.Exit(code=1)
    archive(cfg.archive_dir, cfg.state_file, cfg.events_file)
    typer.echo(f"replaying {len(backlog)} chunks, previous state archived in {cfg.archive_dir}\n")
    ear = Ear(cfg, from_start=True, keep_audio=keep_audio, strict_chunks=False)
    ear.run(max_seconds=max_seconds, echo=typer.echo)


def main() -> None:
    app()
