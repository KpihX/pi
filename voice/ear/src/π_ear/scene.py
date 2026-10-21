"""Where a voice comes from: the scene a person in the room would understand.

"Person or device" (``source.py``) is the wrong question for an ambient ear. A person
who is there tells at least four kinds of voice apart, and answers them differently:

    room    a person in the room              someone to listen to and answer
    remote  a person reached through a device someone too: a meeting, a call
    media   something recorded                context: what is being watched
    self    π itself                          never someone else

Some of it is a fact, the rest is learned, nothing is written by hand:

* **through the machine** is a fact when the machine's own sound is captured
  (``echo.py``): a voice heard on the machine's channel was played by this machine.
  A voice's microphone windows count for the room only for the share of their energy
  the machine did not explain (``echo``). Beta-counted, one pseudo-window each way;
* **through another device** (a phone, a television): the voice's long-term spectrum
  against people and loudspeakers (``source.py``);
* **self**: the share of the voice's machine windows played by π's own mouth
  (``contract.MOUTH_APP``), a fact the sound server gives;
* **remote or media**, among the voices that come out of a device: a conversation takes
  turns, a recording does not wait for anyone. Over the seconds the ear heard, does
  the room speak less while the voice speaks (rate p1) than while it is silent (p2)?
  *Media*: one rate, p1 = p2 ~ U(0, 1). *Conversation*: p1 ~ U(0, 1) and p2 ~ U(p1, 1),
  so that both give p1 the same prior and only the comparison counts. Their Bayes
  factor is the evidence, with no parameter: exactly one when the voice never stopped
  (nothing to compare it with), and for media when the room never answers anything (a
  voice nobody answers is not a conversation). A first form compared p1 with a plug-in
  rate of the room, 1/2 before any silence of the voice: a video that never paused,
  talked over by the room 40 % of the time, came out a conversation;
* **what an application plays**, remote or media, is learned per application from what
  KπX says of its voices (a correction), never from a list of names: before any, an
  application is as likely a call as a video.

A correction (``room``, ``remote``, ``media``) is final for its voice, and teaches the
applications that voice came out of.

The law: no numeric literal but 0 and 1 in this module (``tests/test_law.py``).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import betaincc, betaln

from π_ear.contract import MOUTH_APP

KINDS = ("room", "remote", "media", "self")
PEOPLE = ("room", "remote")


def _logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def conversation_evidence(n1: int, k1: int, n2: int, k2: int) -> float:
    """log P(turns | the room speaks less while the voice speaks) - log P(turns | it
    does not care): ``k1`` of ``n1`` seconds of the voice had the room speaking, ``k2``
    of the ``n2`` seconds without it. Conversation: p1 ~ U(0,1), p2 ~ U(p1,1); media:
    p1 = p2 ~ U(0,1) (the binomial coefficients are common to both and cancel)."""
    if not n1:
        return 0.0
    b = (k2 + 1, n2 - k2 + 1)

    def log_g(p: float) -> float:
        # p1's likelihood, times P(p2 > p1 | the seconds without the voice), over (1 - p1)
        tail = max(float(betaincc(*b, p)), math.ulp(0))
        return k1 * math.log(p) + (n1 - k1 - 1) * math.log1p(-p) + math.log(tail)

    peak = minimize_scalar(lambda p: -log_g(p), bounds=(0, 1), method="bounded")
    top = -float(peak.fun)
    area, _ = quad(lambda p: math.exp(log_g(p) - top), 0, 1, points=[float(peak.x)])
    live = betaln(*b) + top + math.log(max(area, math.ulp(0)))
    media = betaln(k1 + k2 + 1, n1 + n2 - k1 - k2 + 1)
    return live - media


class Scene:
    def __init__(self):
        # per voice: windows heard through the machine, room-weighted microphone windows,
        # machine windows played by π's mouth, the applications it came out of
        self.voices: dict[str, dict] = defaultdict(
            lambda: {"machine": 0.0, "mic": 0.0, "mouth": 0.0, "apps": Counter()}
        )
        self.told: dict[str, str] = {}
        self.device: dict[str, float] = {}  # voice -> its spectrum's P(a loudspeaker)
        self.app_told: dict[str, Counter] = defaultdict(Counter)  # app -> remote/media/self
        # the turns: per second still open, who spoke on which channel; per voice, the
        # seconds it spoke (n) and those the room spoke in too (k); in all, the same
        self.seconds: dict[int, dict] = {}
        self.talk: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.total = [0, 0]  # seconds heard, and those the room spoke in
        self.settled: int | None = None  # every second before this one is counted

    # -- evidence -----------------------------------------------------------------------
    def heard(
        self,
        voice: str,
        channel: str,
        seconds: range,
        echo: float | None = None,
        apps: list[str] | None = None,
        learned: bool = True,
        device: float | None = None,
    ) -> None:
        """A window of ``voice``, on ``channel``, over ``seconds``. Only a learned window
        counts as evidence of where it comes from (a shadow is the same audio again);
        every window marks its seconds as spoken, for the turns. ``device``: what the
        voice's spectrum says now, P(a loudspeaker), when it says anything."""
        v = self.voices[voice]
        if device is not None:
            self.device[voice] = device
        for second in seconds:
            slot = self.seconds.setdefault(second, {"machine": set(), "mic": set()})
            slot["machine" if channel == "machine" else "mic"].add(voice)
        if not learned:
            return
        room = 1 - (echo or 0)  # the part of a microphone window the machine did not explain
        if channel == "machine":
            v["machine"] += 1
            for app in apps or []:
                v["apps"][app] += 1
            v["mouth"] += 1 if MOUTH_APP in (apps or []) else 0
        else:
            v["mic"] += room

    def settle(self, before: int) -> None:
        """Count the turns of every second before ``before`` (no window will join them);
        a second nobody spoke in counts too: it is a second the room could have."""
        first = min(self.seconds, default=before) if self.settled is None else self.settled
        if before <= first:
            return
        self.total[0] += before - first
        for second in sorted(s for s in self.seconds if s < before):
            slot = self.seconds.pop(second)
            room = {v for v in slot["mic"] if self._in_room(v)}
            self.total[1] += 1 if room else 0
            for voice in slot["machine"] | slot["mic"]:
                others = room - {voice}
                self.talk[voice][0] += 1
                self.talk[voice][1] += 1 if others else 0
        self.settled = before

    def tell(self, voice: str, kind: str) -> None:
        """A correction: ``voice`` is ``kind``; its applications learn what they play."""
        self.told[voice] = kind
        for app in self.voices[voice]["apps"]:
            self.app_told[app][kind] += 1

    # -- the posterior --------------------------------------------------------------------
    def _in_room(self, voice: str) -> bool:
        """Did a voice of the microphone speak in the room? Unless it is shown to come
        out of a device: this machine (its windows, not one window's share: a person over
        a loud video is half the machine's energy, and still the room speaking), or
        another loudspeaker (its spectrum: the video heard through the air)."""
        machine, device = self._machine(voice), self.device.get(voice)
        return machine < 1 - machine and (device is None or device < 1 - device)

    def _machine(self, voice: str) -> float:
        """P(through this machine): its machine windows against its room-weighted ones."""
        v = self.voices[voice]
        return (v["machine"] + 1) / (v["machine"] + v["mic"] + 1 + 1)

    def _app_remote(self, voice: str) -> float:
        """P(remote | not room, not self) from what its applications were said to play."""
        apps = self.voices[voice]["apps"]
        if not apps:
            return 1 / (1 + 1)
        total = sum(apps.values())
        p = 0.0
        for app, n in apps.items():
            said = self.app_told.get(app, Counter())
            p += n / total * (said["remote"] + 1) / (said["remote"] + said["media"] + 1 + 1)
        return p

    def conversation(self, voice: str) -> float:
        """The turn-taking evidence for ``voice``, in nats (positive: a conversation)."""
        n, k = self.talk.get(voice, (0, 0))
        # the seconds without it: when the room spoke, it was someone else than this voice
        rest_n, rest_k = self.total[0] - n, self.total[1] - k
        return conversation_evidence(n, k, max(rest_n, 0), max(min(rest_k, rest_n), 0))

    def of(self, voice: str | None, device: float | None = None) -> dict[str, float] | None:
        """P(kind) for ``voice``; ``device``: P(a loudspeaker | its spectrum), when known."""
        if voice is None or voice not in self.voices:
            return None
        if voice in self.told:
            return {k: float(k == self.told[voice]) for k in KINDS}
        v = self.voices[voice]
        machine = self._machine(voice)
        through_device = (1 - machine) * (device if device is not None else 1 / (1 + 1))
        room = (1 - machine) - through_device
        self_ = machine * (v["mouth"] + 1) / (v["machine"] + 1 + 1) if v["mouth"] else 0.0
        out_of_device = machine + through_device - self_
        z = _logit(self._app_remote(voice)) + self.conversation(voice)
        remote = out_of_device / (1 + math.exp(-z))
        return {"room": room, "remote": remote, "media": out_of_device - remote, "self": self_}

    def person(self, voice: str | None, device: float | None = None) -> bool | None:
        """Is ``voice`` more probably a person (in the room or reached) than not?"""
        p = self.of(voice, device)
        if p is None:
            return None
        people = sum(p[k] for k in PEOPLE)
        return people > 1 - people

    # -- persistence ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "voices": {k: {**v, "apps": dict(v["apps"])} for k, v in self.voices.items()},
            "told": self.told,
            "device": self.device,
            "app_told": {k: dict(v) for k, v in self.app_told.items()},
            "talk": dict(self.talk),
            "total": self.total,
            "settled": self.settled,
        }

    def load(self, raw: dict) -> None:
        for k, v in raw.get("voices", {}).items():
            self.voices[k] = {**v, "apps": Counter(v.get("apps", {}))}
        self.told = dict(raw.get("told", {}))
        self.device = dict(raw.get("device", {}))
        self.app_told = defaultdict(
            Counter, {k: Counter(v) for k, v in raw.get("app_told", {}).items()}
        )
        self.talk = defaultdict(
            lambda: [0, 0], {k: list(v) for k, v in raw.get("talk", {}).items()}
        )
        self.total = list(raw.get("total", [0, 0]))
        self.settled = raw.get("settled")
