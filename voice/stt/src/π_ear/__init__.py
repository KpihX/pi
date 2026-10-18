"""π-ear: the streaming ear of π.

A POC, not a framework. Two servers, one directory of raw audio chunks, one state
file rewritten atomically, one append-only event log. There is no "render the
session first" step anywhere in this package, and that absence is the design.

Layout of the pole this package lives in:

    π/voice/stt/          the ear's memory: IDEAS.md, RESEARCH.md, TODO.md
    π/voice/stt/src/      the system (this package)
    π/voice/stt/tests/    the world it is checked against

The distribution name stays ASCII (`pi-ear`), because the packaging standard
forbids anything else: uv answers "Names must start and end with a letter or
digit and may only contain -, _, ., and alphanumeric characters". The module
itself is π, through `[tool.uv.build-backend] module-name = "π_ear"`.
"""

from π_ear.cli import app

__all__ = ["app", "main"]


def main() -> None:
    app()
