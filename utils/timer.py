import time
from .logger import PrintLogger
from contextlib import contextmanager
class Timer:
    def __init__(self):
        pass

    @staticmethod
    @PrintLogger()
    @contextmanager
    def timing(msg=None):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration = end - start
            if msg is not None:
                print(f"{msg} {duration:.6f} seconds")
            else:
                print(f"duration:{duration:.6f} seconds")
