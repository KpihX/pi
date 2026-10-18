"""End-to-end tests: chunks on disk in, live state and event log out."""

from __future__ import annotations

import json

from synthetic import voice, write_chunks

from π_ear.config import Config
from π_ear.ingest import ChunkTailer
from π_ear.live import Ear


def test_one_voice_does_not_multiply_speakers_through_the_pipeline(tmp_path):
    """End to end, on the real pipeline: ten windows of one voice, one entry.

    The count is the whole point. Before the invariant, the same ten windows made
    three entries on a good day and twenty on a bad one, and nothing in the state
    said which of them was the person."""
    cfg = Config(root=tmp_path, window_s=1.0, idle_sleep_s=0.01, corpus=False)
    plan = [("voice_A", voice(1.0, cfg.rate, 110.0, index)) for index in range(1, 11)]
    write_chunks(cfg, plan)

    ear = Ear(cfg, from_start=True, keep_audio=False)
    ear.run(max_seconds=0.4)

    payload = json.loads(cfg.state_file.read_text(encoding="utf-8"))
    assert payload["inventory"]["counters"]["registered_new"] == 1
    assert len(payload["inventory"]["registered"]) == 1


def test_tailer_reads_each_chunk_exactly_once(tmp_path):
    cfg = Config(root=tmp_path)
    write_chunks(cfg)
    tailer = ChunkTailer(cfg.audio_dir, cfg.chunk_s, cfg.gap_factor, from_start=True)
    first = tailer.poll()
    assert len(first) == 10  # ten one-second segments, one chunk per second
    assert tailer.poll() == []
    assert tailer.chunks_seen == len(first)


def test_a_chunk_still_being_written_is_not_read(tmp_path):
    """The measured 10x, in one test.

    Reading a growing file made every poll analyse one more observation of the SAME
    second of voice: 4179 events for 379 seconds of audio, the evidence bar diluted
    tenfold, one core at 99.9%."""
    cfg = Config(root=tmp_path, corpus=False)
    plan = [
        ("voice_A", voice(1.0, cfg.rate, 110.0, 1)),
        ("voice_A", voice(1.0, cfg.rate, 112.0, 2)),
    ]
    write_chunks(cfg, plan)
    names = sorted(path.name for path in cfg.audio_dir.glob("*.raw"))
    assert len(names) == 2

    newest = cfg.audio_dir / names[-1]
    complete = newest.read_bytes()
    newest.write_bytes(complete[: len(complete) // 2])  # the recorder is still writing

    tailer = ChunkTailer(
        cfg.audio_dir,
        cfg.chunk_s,
        cfg.gap_factor,
        from_start=True,
        expected_bytes=cfg.expected_chunk_bytes,
    )
    assert [chunk.name for chunk in tailer.poll()] == [names[0]]
    assert tailer.chunks_seen == 1

    newest.write_bytes(complete)  # the recorder moved on: the chunk is complete
    second = tailer.poll()
    assert [chunk.name for chunk in second] == [names[1]]
    assert second[0].samples.size * 2 == cfg.expected_chunk_bytes
    assert tailer.chunks_seen == 2
    assert tailer.poll() == []


def test_tailer_skips_the_backlog_on_a_fresh_start(tmp_path):
    cfg = Config(root=tmp_path)
    write_chunks(cfg)
    tailer = ChunkTailer(cfg.audio_dir, cfg.chunk_s, cfg.gap_factor, from_start=False)
    assert tailer.poll() == []


def test_prune_never_deletes_the_chunk_being_read(tmp_path):
    cfg = Config(root=tmp_path)
    write_chunks(cfg)
    tailer = ChunkTailer(cfg.audio_dir, cfg.chunk_s, cfg.gap_factor, from_start=True)
    chunks = tailer.poll()
    removed = ChunkTailer.prune(cfg.audio_dir, keep_name=chunks[-1].name)
    remaining = sorted(path.name for path in cfg.audio_dir.glob("*.raw"))
    assert remaining == [chunks[-1].name]
    assert removed == len(chunks) - 1


def test_ear_writes_a_live_state_and_an_append_only_log(tmp_path):
    cfg = Config(root=tmp_path, idle_sleep_s=0.01, corpus=False)
    write_chunks(cfg)
    ear = Ear(cfg, from_start=True, keep_audio=False)
    ear.run(max_seconds=0.4)

    payload = json.loads(cfg.state_file.read_text(encoding="utf-8"))
    assert payload["revision"] > 0
    assert payload["paused"] is False
    assert payload["capture"]["rate"] == cfg.rate
    assert "inventory" in payload and "tau_now" in payload["inventory"]
    assert payload["audio"]["chunks_seen"] == 10

    lines = cfg.events_file.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert len(events) == payload["revision"]
    for event in events:
        assert "regime" not in event["voice"]
        assert "f0_hz" in event["voice"] and "rms_dbfs" in event["voice"]
        assert set(event["speaker"]) >= {"id", "kind", "cos_best", "tau", "margin", "margin_req"}
        assert event["decided_at"].endswith("Z")
        # the field that used to be called `frames` carried a sample count. Two
        # names for two things: a reader can now tell them apart without guessing,
        # and no downstream consumer can trust the wrong unit for a year.
        assert "frames" not in event
        assert cfg.rate <= event["samples"] <= cfg.window_s * cfg.rate
        assert event["frames_analysed"] > 0

    # the audio is consumed, not accumulated: /tmp is RAM on this machine. At
    # most the chunk currently being read survives the prune, never the history.
    remaining = list(cfg.audio_dir.glob("*.raw"))
    assert len(remaining) <= 1, remaining


def test_the_ear_keeps_a_durable_corpus(tmp_path):
    """Measured 2026-10-19: six minutes of real test (a hum, then a film) survived
    only as numbers, because the audio directory is RAM and is emptied as it is
    consumed. Nothing could be replayed against another identity vector. The corpus
    is what makes a session runnable again, and it must not cost RAM to exist."""
    corpus = tmp_path / "corpus"
    cfg = Config(root=tmp_path, idle_sleep_s=0.01, corpus=True, corpus_dir=corpus)
    write_chunks(cfg)

    ear = Ear(cfg, from_start=True, keep_audio=False)
    ear.run(max_seconds=0.4)

    kept = sorted(path.name for path in corpus.glob("*.raw"))
    remaining = sorted(path.name for path in cfg.audio_dir.glob("*.raw"))
    assert len(kept) == 10, kept  # the whole session, durably, outside /tmp
    assert len(remaining) <= 1, remaining  # and RAM still gives it back

    payload = json.loads(cfg.state_file.read_text(encoding="utf-8"))
    assert payload["audio"]["corpus_chunks"] == 10
    assert payload["audio"]["corpus_bytes"] > 0
