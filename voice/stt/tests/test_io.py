"""The disk, the transport and the controls: what survived from v0, still held down."""

from __future__ import annotations

import numpy as np

from π_ear import contract
from π_ear.io import Corpus, EmbeddingStore, Tailer, append_jsonl, read_jsonl, write_atomic


def _chunk(directory, name, seconds=1.0, value=1000):
    samples = np.full(int(contract.RATE * seconds), value, dtype="<i2")
    (directory / name).write_bytes(samples.tobytes())


def test_tailer_reads_in_order_once_and_waits_for_a_growing_chunk(tmp_path):
    for i in range(3):
        _chunk(tmp_path, f"20261018-00000{i}.raw")
    _chunk(tmp_path, "20261018-000003.raw", seconds=0.4)  # the recorder is still writing it
    tailer = Tailer(tmp_path, contract.chunk_bytes(), from_start=True)
    assert [c.name for c in tailer.poll()] == [f"20261018-00000{i}.raw" for i in range(3)]
    assert tailer.poll() == []
    _chunk(tmp_path, "20261018-000003.raw")  # now complete
    _chunk(tmp_path, "20261018-000004.raw", seconds=0.2)
    assert [c.name for c in tailer.poll()] == ["20261018-000003.raw"]


def test_a_frozen_directory_reads_its_partial_last_chunk(tmp_path):
    _chunk(tmp_path, "a.raw")
    _chunk(tmp_path, "b.raw", seconds=0.3)
    assert len(Tailer(tmp_path, 0, from_start=True).poll()) == 2


def test_a_live_start_skips_the_backlog_and_prune_drops_only_what_was_read(tmp_path):
    _chunk(tmp_path, "a.raw")
    tailer = Tailer(tmp_path, contract.chunk_bytes(), from_start=False)
    assert tailer.poll() == []
    _chunk(tmp_path, "b.raw")
    _chunk(tmp_path, "c.raw")
    assert [c.name for c in tailer.poll()] == ["b.raw", "c.raw"]
    _chunk(tmp_path, "d.raw", seconds=0.1)
    tailer.prune()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d.raw"]


def test_jsonl_is_read_from_an_offset_and_never_half_a_line(tmp_path):
    path = tmp_path / "c.jsonl"
    append_jsonl(path, {"a": 1})
    lines, offset = read_jsonl(path)
    assert lines == [{"a": 1}]
    with open(path, "a") as handle:
        handle.write('{"b": 2')  # a writer caught mid-line
    assert read_jsonl(path, offset) == ([], offset)
    with open(path, "a") as handle:
        handle.write("}\n")
    assert read_jsonl(path, offset)[0] == [{"b": 2}]


def test_state_is_atomic(tmp_path):
    write_atomic(tmp_path / "s.json", {"x": 1})
    assert not list(tmp_path.glob("*.tmp"))


def test_embedding_rows_survive_a_restart(tmp_path):
    store = EmbeddingStore(tmp_path / "e.f32", 4)
    rows = [store.append(np.eye(4)[i]) for i in range(3)]
    again = EmbeddingStore(tmp_path / "e.f32", 4)
    assert rows == [0, 1, 2] and len(again) == 3
    assert np.allclose(again(2), np.eye(4)[2])


def test_corpus_is_bounded(tmp_path):
    corpus = Corpus(tmp_path / "corpus", max_chunks=2)
    for name in ("a.raw", "b.raw", "c.raw"):
        _chunk(tmp_path, name)
        corpus.keep(type("C", (), {"name": name, "path": tmp_path / name})())
    assert sorted(p.name for p in (tmp_path / "corpus").iterdir()) == ["b.raw", "c.raw"]
