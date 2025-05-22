class Device:
    def __init__(self, name):
        self.name = name

    def allocate(self, nbytes): ...  # noqa: E704
    def free(self, ptr): ...  # noqa: E704
    def memcpy(self, dst, src, nbytes, kind): ...  # noqa: E704
    def synchronize(self): ...  # noqa: E704
