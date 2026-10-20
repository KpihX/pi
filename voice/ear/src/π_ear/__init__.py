"""π-ear: the streaming ear of π.

A frozen sensor (Silero VAD, CAM++), one learned model on top of it, a state on
disk that is the model's own checkpoint, and an append-only record. There is no
setup step, no calibration step and no "process the session afterwards" step.

The distribution name stays ASCII (``pi-ear``) because the packaging standard
forbids anything else; the module is π through ``module-name = "π_ear"``.
"""


def main() -> None:
    from π_ear.cli import app

    app()
