import functools
from typing import Callable


class PrintLogger:
    def __init__(self, prefix: str=None):
        self.prefix = prefix

    def __call__(self, fnc: Callable):
        @functools.wraps(fnc)
        def wrapper(*args, **kwargs):
            print(f'[{self.prefix}] Calling {fnc.__name__}')
            return fnc(*args, **kwargs)
        return wrapper