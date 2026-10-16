"""Read-only native capture metadata with explicitly rebound state/output."""


class StateReference:
    def __init__(self, original, handle):
        self.original = original
        self._param_ref = self.handle = handle

    def __getattr__(self, name):
        return getattr(self.original, name)


class Operation:
    def __init__(self, original, output):
        self.original, self.out = original, output

    def __getattr__(self, name):
        return getattr(self.original, name)
