"""The ear server: one loop, no batch step, no end-of-session processing.

The move this module makes, and the reason the POC exists at all: there is no
moment where the audio is "finished". New chunks arrive in a directory, each one
slides the analysis window by one second, and the state on disk is rewritten with
whatever is known *at that instant*. Stopping the server does not produce a
result, it just stops showing you the current one.

Pause semantics, deliberately chosen:

* ``PAUSE`` present -> the ear stops consuming chunks. The recorder keeps
  recording, so the audio is not lost, it is converted into a backlog.
* ``RESUME`` (file removed) -> the ear drains the backlog and catches up. The
  cursor was never advanced, so nothing was skipped.

The alternative (keep consuming but freeze the state) would silently throw the
paused audio away. A backlog is honest, it is measurable, and it is visible in
the state as ``backlog_chunks``.
"""

from __future__ import annotations

import shutil
import time
from collections import deque

import numpy as np

from π_ear.ingest import Chunk, ChunkTailer, wait
from π_ear.inventory import Inventory
from π_ear.physics import analyse_window
from π_ear.state import append_event, archive, utc_now, write_atomic


class Ear:
    def __init__(
        self,
        cfg,
        from_start: bool = False,
        keep_audio: bool = False,
        strict_chunks: bool = True,
    ):
        self.cfg = cfg
        self.keep_audio = keep_audio
        # A live directory has a file being written right now, and reading it while
        # it grows was the measured 10x. A frozen directory has no such file, so a
        # replay does not want the gate: the last chunk of a recorded session is
        # legitimately partial and must be seen.
        self.strict_chunks = strict_chunks
        self.inventory = Inventory(cfg)
        self.tailer = ChunkTailer(
            cfg.audio_dir,
            cfg.chunk_s,
            cfg.gap_factor,
            from_start=from_start,
            expected_bytes=cfg.expected_chunk_bytes if strict_chunks else 0,
        )
        self.corpus_written = 0
        self.buffer: deque[np.ndarray] = deque()
        self.window_frames = int(cfg.rate * cfg.window_s)
        self.min_frames = int(cfg.rate * min(cfg.window_s, 1.0))
        self.revision = 0
        self.started = time.monotonic()
        # -inf, not 0.0: the very first persist must always happen, otherwise the
        # state file does not exist during the first second of a run and the
        # throttled guard keeps blocking it.
        self.last_persist = float("-inf")
        self.last_paused = False
        self.paused_since: float | None = None
        self.silent_run_logged = False
        self.last_chunk: str | None = None
        self.last_features: dict = {}

    # -- the loop -------------------------------------------------------------
    def run(self, max_seconds: float | None = None, log_silence: bool = False, echo=None):
        self.cfg.ensure()
        while True:
            self._handle_reset()
            now = self._clock()
            # The session bound is checked first, once, at the top of the iteration.
            # It used to sit at the bottom, behind the pause branch, so a bounded
            # run could never end while paused.
            if max_seconds is not None and now >= max_seconds:
                break
            if self.cfg.pause_file.exists():
                if self.paused_since is None:
                    self.paused_since = now
                    self._persist(paused=True)
                    if echo:
                        echo(f"[pause] t={now:.1f}s  backlog={self.tailer.backlog()} chunks")
                else:
                    self._persist(paused=True)
                # `continue` is load-bearing, and it was missing. Measured
                # 2026-10-19: PAUSE did not pause. Without this line the loop waited
                # and then fell straight through to the consumer, so the cursor
                # advanced while paused and the state was rewritten with
                # paused=False with the PAUSE file sitting right there: the ear lied
                # about its own state, which is the one thing this pole may not do.
                wait(self.cfg.idle_sleep_s)
                continue
            if self.paused_since is not None:
                if echo:
                    echo(f"[resume] t={now:.1f}s  catching up {self.tailer.backlog()} chunks")
                self.paused_since = None
            self._consume(echo=echo, log_silence=log_silence)
            self._persist(paused=False)
            # `now` is already the session clock (elapsed seconds since start), so
            # it is compared directly. Subtracting `started` a second time made
            # the bound unreachable, and that is what hung the end-to-end test.
            #
            # And the sleep is HERE, at the end of every iteration, always. It used
            # to sit after the `break` above, which made it dead code no reader
            # could ever reach: a live ear spun as fast as the CPU allowed, one core
            # pinned at 99.9%, for a loop whose only job every tenth of a second is
            # to notice that there is nothing new. The analysis costs 28 ms, measured
            # on real audio: the spin was never the work, it was the absence of a
            # line.
            wait(self.cfg.idle_sleep_s)

    # -- internals ------------------------------------------------------------
    def _clock(self) -> float:
        """One clock for the whole system: seconds since this ear started."""
        return round(time.monotonic() - self.started, 2)

    def _handle_reset(self) -> None:
        if not self.cfg.reset_file.exists():
            return
        archive(self.cfg.archive_dir, self.cfg.state_file, self.cfg.events_file)
        self.inventory = Inventory(self.cfg)
        self.tailer = ChunkTailer(
            self.cfg.audio_dir,
            self.cfg.chunk_s,
            self.cfg.gap_factor,
            from_start=True,
            expected_bytes=self.cfg.expected_chunk_bytes if self.strict_chunks else 0,
        )
        self.buffer.clear()
        self.revision = 0
        self.started = time.monotonic()
        self.cfg.reset_file.unlink()

    def _consume(self, echo=None, log_silence: bool = False) -> None:
        for chunk in self.tailer.poll():
            self.last_chunk = chunk.name
            self.buffer.append(chunk.samples)
            self._trim()
            self._keep(chunk)
            if not self.keep_audio:
                ChunkTailer.prune(self.cfg.audio_dir, keep_name=chunk.name)
            self._analyse(log_silence=log_silence, echo=echo)

    def _keep(self, chunk: Chunk) -> None:
        """Copy the chunk into the durable corpus before the audio directory drops it.

        Measured 2026-10-19: six minutes of real test (a hum, then a film) survived
        only as numbers, because the audio was deleted as it was consumed and there
        was nowhere to replay it from. The corpus is that somewhere: a bounded copy
        in the home directory, never in RAM, so the exact session can be run again
        against a different identity vector.
        """
        if not self.cfg.corpus:
            return
        target = self.cfg.corpus_dir / chunk.path.name
        if not target.exists():
            shutil.copyfile(chunk.path, target)
            self.corpus_written += 1
        self._trim_corpus()

    def _trim_corpus(self) -> None:
        """Keep the corpus bounded: the oldest chunks go, the recent window stays."""
        files = sorted(self.cfg.corpus_dir.glob("*.raw"))
        excess = len(files) - int(self.cfg.corpus_max_chunks)
        if excess <= 0:
            return
        for path in files[:excess]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _trim(self) -> None:
        total = sum(int(part.size) for part in self.buffer)
        while len(self.buffer) > 1 and total - int(self.buffer[0].size) >= self.window_frames:
            total -= int(self.buffer.popleft().size)

    def _analyse(self, log_silence: bool = False, echo=None) -> None:
        total = sum(int(part.size) for part in self.buffer)
        if total < self.min_frames:
            return
        window = np.concatenate(list(self.buffer))
        t = self._clock()
        features = analyse_window(window, self.cfg)
        decision = self.inventory.tick(features, t)

        voiced = decision["kind"] != "silent"
        if not voiced and not log_silence:
            if not self.silent_run_logged:
                self._emit(features, decision, t, window.size, echo=echo)
                self.silent_run_logged = True
            return
        self.silent_run_logged = False
        self._emit(features, decision, t, window.size, echo=echo)

    def _emit(self, features: dict, decision: dict, t: float, samples: int, echo=None) -> None:
        speaker_id = decision.get("id")
        pitch_pct = (
            self.inventory.f0_percentile(speaker_id, features["f0_hz"]) if speaker_id else None
        )
        rel_db = None
        entry = self.inventory.registered.get(speaker_id) or self.inventory.candidates.get(
            speaker_id
        )
        if entry is not None and entry.levels:
            rel_db = round(features["rms_dbfs"] - float(np.median(entry.levels)), 1)

        self.revision += 1
        event = {
            "revision": self.revision,
            "t_s": t,
            # decided_at exists from the very first line, per the record
            # contract: the excerpt is emitted immediately and its attribution
            # may be revised later. A revision after the fact is normal.
            "decided_at": utc_now(),
            "chunk": self.last_chunk,
            "window_s": self.cfg.window_s,
            # `samples` is exactly what it says: the number of samples in the
            # analysis window. The field used to be named `frames` and carried a
            # sample count, which is the kind of thing a downstream reader gets
            # wrong once and then trusts for a year. 48944 samples is a 3.06 s
            # window, not 48944 frames.
            "samples": samples,
            "frames_analysed": features["frames"],
            "speaker": {
                "id": speaker_id,
                "kind": decision["kind"],
                "cos_best": decision.get("cos_best"),
                "cos_second": decision.get("cos_second"),
                "tau": decision.get("tau"),
                "margin": decision.get("margin"),
                "margin_req": decision.get("margin_req"),
                "evidence": decision.get("evidence"),
            },
            "voice": {
                "f0_hz": features["f0_hz"],
                "f0_pct_in_speaker": pitch_pct,
                "rms_dbfs": features["rms_dbfs"],
                "rel_db": rel_db,
                "centroid_hz": features["centroid_hz"],
                "zcr": features["zcr"],
                "voiced_ratio": features["voiced_ratio"],
            },
        }
        append_event(self.cfg.events_file, event)
        self.last_features = {
            "t_s": t,
            **features,
            "speaker_id": speaker_id,
            "kind": decision["kind"],
            "cos_best": decision.get("cos_best"),
            "cos_second": decision.get("cos_second"),
            "tau": decision.get("tau"),
            "f0_pct_in_speaker": pitch_pct,
            "rel_db": rel_db,
        }
        if echo:
            echo(
                f"[{t:8.1f}s] {decision['kind']:9s} "
                f"id={str(speaker_id)[:16]:16s} "
                f"cos={self._fmt(decision.get('cos_best'))} "
                f"tau={decision.get('tau')} "
                f"f0={features['f0_hz']:6.1f}Hz "
                f"pct={self._fmt(pitch_pct)} "
                f"lvl={features['rms_dbfs']:7.1f}dB "
                f"voiced={features['voiced_ratio']:.2f}"
            )

    @staticmethod
    def _fmt(value) -> str:
        return "  none" if value is None else f"{value:.3f}"

    def _persist(self, paused: bool) -> None:
        now = self._clock()
        if paused and self.paused_since is None:
            self.paused_since = now
        # Throttled both ways (one write per second at most), but a pause or a
        # resume transition is written immediately: the state has to show it.
        transition = paused != self.last_paused
        if not transition and (now - self.last_persist) < self.cfg.persist_every_s:
            return
        self.last_persist = now
        self.last_paused = paused
        now_features = self._last_features()
        # Counted once, not twice: two globs of the whole corpus per state write is
        # the kind of waste that only shows up once the corpus is a thousand chunks.
        corpus_chunks, corpus_bytes = self._corpus_stats()
        payload = {
            "revision": self.revision,
            "utc": utc_now(),
            "clock_s": now,
            "uptime_s": now,
            "paused": paused,
            "paused_since_s": self.paused_since,
            "capture": self.cfg.as_dict(),
            "audio": {
                "chunks_seen": self.tailer.chunks_seen,
                "backlog_chunks": self.tailer.backlog(),
                "last_chunk": self.last_chunk,
                "gaps": self.tailer.gaps[-5:],
                "audio_dir_bytes": self._dir_bytes(),
                "keep_audio": self.keep_audio,
                "corpus_dir": str(self.cfg.corpus_dir) if self.cfg.corpus else None,
                "corpus_chunks": corpus_chunks,
                "corpus_bytes": corpus_bytes,
            },
            "now": now_features,
            "inventory": self.inventory.as_dict(now),
        }
        write_atomic(self.cfg.state_file, payload)

    def _last_features(self) -> dict:
        return self.last_features

    def _dir_bytes(self) -> int:
        total = 0
        for path in self.cfg.audio_dir.glob("*.raw"):
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                pass
        return total

    def _corpus_stats(self) -> tuple[int, int]:
        """How many chunks the durable corpus holds, and how many bytes."""
        if not self.cfg.corpus:
            return 0, 0
        count = 0
        total = 0
        for path in self.cfg.corpus_dir.glob("*.raw"):
            count += 1
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                count -= 1
        return count, total
