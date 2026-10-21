"""The command line. Thin on purpose.

π-ear run        the ear, live, following the recorder's directory
π-ear ctl        pause | resume | reset | status
π-ear correct    same | different | not_voice | name | person | device | text
π-ear undo       one correction, forgotten exactly (what it taught is replayed without it)
π-ear questions  what the ear is asking; π-ear answer  answers one
π-ear why        ROW: what the ear decided for that window, and what it would say now
π-ear probe      a span of recorded audio, second by second: level, speech probability, decisions
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

from π_ear import contract, port
from π_ear.ear import Ear, Places, base_weights_path, load_memory
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
            source = s["source"]
            typer.echo(
                f"  {s['id']:18s} {s['label'] or '':12s} {source['is'] or '?':6s} "
                f"conditions={s['conditions']:<3d} evidence={s['evidence']:<5d}"
                f" silent_for={s['silent_for_s']}s"
            )
        for c in state.get("candidates", []):
            typer.echo(
                f"  {c['id']:18s} candidate              evidence={c['evidence']:<5d}"
                f" waiting={c['waiting_s']}s weight={c['weight']}"
            )
        typer.echo(
            f"floor    {state['audio'].get('floor_dbfs')} dBFS (this session's non-speech level)"
        )
        reader = state.get("reader", {})
        if reader.get("fast"):
            typer.echo(
                f"what     {reader['fast']} + {reader.get('slow')}  pieces={reader.get('pieces')}"
                f"  slow behind={reader.get('behind_s')}s at {reader.get('slow_seconds_per_second')}"
                f" s/s  here: speech {reader.get('regime')}, token logp {reader.get('confidence')}"
                f"  questions={reader.get('questions')} errors={reader.get('errors')}"
            )
        corrections = state.get("corrections", {})
        if corrections:
            typer.echo(
                f"corrections seen={corrections['seen']} applied={corrections['applied']}"
                f" undone={corrections['undone']}"
            )
            for r in corrections.get("refused", []):
                typer.echo(f"  REFUSED line {r['line']} ({r['kind']}): {r['why']}")
        typer.echo(f"counters {json.dumps(state['counters'])}")


class Kind(enum.StrEnum):  # `undo` and `answer` have their own commands
    same = "same"
    different = "different"
    not_voice = "not_voice"
    name = "name"
    person = "person"
    device = "device"
    text = "text"


def _rows_between(places: Places, start: float, end: float, session: str | None) -> list[int]:
    """The rows of a session whose window overlaps [start, end] stream seconds."""
    session = session or port.latest_session(places.events)
    return [
        e.row
        for e in port.read(places.events, session)
        if isinstance(e, port.Decision) and e.t0 < end and start < e.t1
    ]


@app.command()
def correct(
    kind: Kind,
    rows: list[int] = typer.Argument(None, help="row numbers, as printed in events.jsonl"),
    start: float = typer.Option(None, "--from", help="every row from this stream second..."),
    end: float = typer.Option(None, "--to", help="...to this one (of --session, default latest)"),
    session: str = typer.Option(None, help="the session --from/--to refer to"),
    speaker: list[str] = typer.Option(None, help="every row of this voice (repeatable)"),
    label: str = typer.Option(None, help="for `name`: what to call this voice"),
    text: str = typer.Option(None, help="for `text`: what was really said"),
    root: str = ROOT,
    memory: str = MEMORY,
):
    """Say once what was wrong. The running ear obeys at once and learns from it.

    \b
      π-ear correct same 812 840                  these windows are one person
      π-ear correct different 812 901             two people
      π-ear correct not_voice --speaker KEY       that "voice" was never a voice
      π-ear correct name 812 --label KπX          call this voice KπX
      π-ear correct device --speaker KEY          that voice comes out of a loudspeaker
      π-ear correct text --from 163 --to 170 --text "le cours de Galois"
    """
    places = _places(root, memory)
    rows = list(rows or [])
    extra = {}
    if (start is None) != (end is None):
        raise typer.BadParameter("--from and --to go together")
    if start is not None:
        found = _rows_between(places, start, end, session)
        if not found:
            raise typer.BadParameter(f"no decided window between {start} s and {end} s")
        rows += found
        extra["span"] = {
            "from": start,
            "to": end,
            "session": session or port.latest_session(places.events),
        }
    if speaker:
        model, _, _ = load_memory(places)
        for key in speaker:
            voice = model.voices.get(key)
            if voice is None:
                raise typer.BadParameter(f"no voice {key} in {places.checkpoint}")
            rows += [r for c in voice.conditions for r in model.clusters[c].members]
    if not rows:
        raise typer.BadParameter("give rows, --from/--to, or --speaker")
    if kind is Kind.name and not label:
        raise typer.BadParameter("`name` needs --label")
    if kind is Kind.text and text is None:
        raise typer.BadParameter("`text` needs --text")
    line = port.Inbox(places.memory).correct(kind.value, sorted(set(rows)), label, text, **extra)
    typer.echo(f"written: {json.dumps(line, ensure_ascii=False)}")


@app.command()
def undo(
    line: int = typer.Argument(..., help="the correction's line number (`line` in the record)"),
    memory: str = MEMORY,
):
    """Forget one correction. The running ear replays what it taught without it."""
    written = port.Inbox(_places(memory=memory).memory).undo(line)
    typer.echo(f"written: {json.dumps(written)}")


@app.command()
def questions(root: str = ROOT, session: str = typer.Option(None)):
    """What the ear is asking: names it does not know, readings it could not choose."""
    places = _places(root)
    open_ = port.open_questions(places.events, session)
    if not open_:
        typer.echo("no open question")
    for q in open_:
        options = "  ".join(f"[{i}] {c}" for i, c in enumerate(q.choices))
        typer.echo(f"{q.id}  {q.about:5s} {q.doubt:14s} rows={q.rows[:6]}  {options}")


@app.command()
def answer(
    question: str = typer.Argument(..., help="the question's id, as `questions` prints it"),
    text: str = typer.Argument(None, help="the name, or what was really said"),
    choice: int = typer.Option(None, help="or the index of one of the question's choices"),
    memory: str = MEMORY,
):
    """Answer a question of the ear. The same call serves a voice through the arbiter."""
    if text is None and choice is None:
        raise typer.BadParameter("give the answer's text, or --choice")
    written = port.Inbox(_places(memory=memory).memory).answer(question, text, choice)
    typer.echo(f"written: {json.dumps(written, ensure_ascii=False)}")


@app.command()
def why(row: int, root: str = ROOT, memory: str = MEMORY):
    """What the ear decided for this window, and how it would explain it now."""
    places = _places(root, memory)
    model, store, _ = load_memory(places)
    if row >= len(store):
        raise typer.BadParameter(f"row {row} does not exist ({len(store)} rows)")
    then = next(
        (e for e in port.read(places.events) if isinstance(e, port.Decision) and e.row == row),
        None,
    )
    if then:
        typer.echo(
            f"row {row}  session {then.session}  [{then.t0}-{then.t1}] s  {then.rms_dbfs} dBFS"
            f"  rel {then.rel_db} dB  learned={then.learned}"
        )
        typer.echo(
            f"  then: {then.kind} {then.label or then.speaker or ''}  p={then.p}"
            f"  margin={then.margin} nats  trust={then.trust}  evidence={then.evidence}"
            f"  source={then.source} ({then.source_p})"
        )
    where = model.where.get(row)
    typer.echo(
        f"  now : in {where or 'no cluster'}"
        + (f" ({model.clusters[where].kind})" if where else "")
    )
    previous = model.where.get(model.after.get(row)) if row in model.after else None
    for rank, option in enumerate(model.explain(store(row), previous)[:3], 1):
        typer.echo(
            f"    {rank}. {option['id'] or 'someone new':18s} {option['kind']:9s}"
            f" {option['label'] or '':10s} evidence={option['evidence']:<5d}"
            f" behind={option['behind_nats']:.1f} nats"
        )


@app.command()
def probe(
    audio_dir: str = typer.Argument(None, help="recorded chunks (default: the corpus)"),
    start: float = typer.Option(0.0, "--from", help="stream second where the span starts"),
    end: float = typer.Option(None, "--to", help="stream second where it ends"),
    session: str = typer.Option(None, help="map seconds through this session's first chunk"),
    asr: bool = typer.Option(
        False, "--asr", help="also read the whole span with both recognisers, ungated"
    ),
    root: str = ROOT,
):
    """Look under the decisions: Silero's own probability, second by second.

    Answers "was it not speech, or speech the detector missed?" from the audio
    itself, without touching the memory.
    """
    import numpy as np

    from π_ear.io import load_raw
    from π_ear.sense import rms_dbfs, speech_probability

    places = _places(root)
    directory = Path(audio_dir) if audio_dir else contract.CORPUS
    chunks = sorted(directory.glob("*.raw"))
    if not chunks:
        raise typer.BadParameter(f"no chunk in {directory}")
    session = session or port.latest_session(places.events)
    opening = next(
        (e for e in port.read(places.events, session) if isinstance(e, port.Session)), None
    )
    names = [c.name for c in chunks]
    origin = names.index(opening.first_chunk) if opening and opening.first_chunk in names else 0
    first = origin + int(start)
    last = min(len(chunks), origin + int(end if end is not None else len(chunks)))
    span = chunks[first:last]
    if not span:
        raise typer.BadParameter("the span holds no chunk")
    audio = [load_raw(c) for c in span]
    probs = speech_probability(contract.MODELS, np.concatenate(audio))
    per = len(probs) / max(len(span), 1)
    decided = (
        [e for e in port.read(places.events, session) if isinstance(e, port.Decision)]
        if opening
        else []
    )
    typer.echo(f"{directory}  chunks {first}..{last - 1}  session {session if opening else '-'}")
    typer.echo("  t(s)  chunk                  dBFS   speech p mean / max   ear")
    for index, (chunk, x) in enumerate(zip(span, audio, strict=True)):
        p = probs[int(index * per) : int((index + 1) * per)]
        t = start + index
        said = [
            f"{e.kind}:{(e.label or e.speaker or '-')[:8]}"
            for e in decided
            if e.t0 <= t + 1 and t < e.t1
        ]
        typer.echo(
            f"  {t:5.0f}  {chunk.name:22s} {rms_dbfs(x):6.1f}   {p.mean() if p.size else 0:6.3f} /"
            f" {p.max() if p.size else 0:5.3f}       {' '.join(sorted(set(said)))}"
        )
    typer.echo(
        f"  span: speech probability mean {probs.mean():.3f}, max {probs.max():.3f},"
        f" over {len(probs)} frames of 32 ms"
    )
    if asr:
        from π_ear import asr as recognisers

        for name in (contract.FAST, contract.SLOW):
            heard = recognisers.build(name, contract.MODELS)
            if heard is not None:
                typer.echo(f"  {name} (ungated): {heard(np.concatenate(audio)).text!r}")


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
