"""Minimal `showvis` stub used by Neatprogram.py.
Original likely provided a visualization window per-genome; this
class provides a small interface used by the training script.
"""

class showvis:
    def __init__(self):
        # flag used by Neatprogram.py
        self.go = True

    def draw(self):
        # no-op drawing method
        return
