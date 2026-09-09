"""Console timing for robot I/O diagnostics."""
from contextlib import contextmanager
from datetime import datetime, timezone
import time


def trace(step, operation, event, detail=""):
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    print(f"[robot-io] {timestamp} mono={time.monotonic():.6f} "
          f"step={step} operation={operation} event={event} {detail}".rstrip(), flush=True)


@contextmanager
def timed_operation(step, operation):
    trace(step, operation, "start")
    started = time.monotonic()
    try:
        yield
    except Exception as exc:
        elapsed = (time.monotonic() - started) * 1000
        trace(step, operation, "error", f"elapsed_ms={elapsed:.3f} error={str(exc)!r} type={type(exc).__name__}")
        raise
    else:
        trace(step, operation, "end", f"elapsed_ms={(time.monotonic() - started) * 1000:.3f}")


def read_observation(robot, step, phase):
    # This measures the whole RobotEnv observation call, not an individual ZED grab.
    with timed_operation(step, f"get_observation:{phase}"):
        return robot.get_observation()
