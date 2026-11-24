"""Observe GNU scheduler and estimator errors, with explicit error propagation."""

import threading
import time


def wait_graph_monitored(tb, timeout, failure):
    """Return (naturally_completed, error), stopping the graph on estimator error.

    A Python block exception must *not* escape the GNU work thread: that
    aborts its worker while the header-demux can wait for parser feedback.
    """
    errors = []

    def waiter():
        try:
            tb.wait()
        except BaseException as exc:
            errors.append(repr(exc))

    tb.start()
    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    deadline = time.monotonic() + float(timeout)
    problem = None
    timed_out = False
    while thread.is_alive():
        problem = failure()
        if problem is not None:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        thread.join(min(0.05, remaining))
    # Read failure again: the work thread may have finished at the same instant.
    if problem is None:
        problem = failure()
    if problem is not None or timed_out:
        tb.stop()
        thread.join(5.0)
    if thread.is_alive():
        raise RuntimeError("GNU Radio scheduler did not stop after abort")
    if errors:
        raise RuntimeError("GNU Radio scheduler error: " + str(errors))
    return (problem is None and not timed_out), problem
