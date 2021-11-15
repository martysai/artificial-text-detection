from os import environ
from typing import Callable


def skip_github(test_func: Callable[..., None]) -> Callable[..., None]:
    def wrapper(*args, **kwargs):
        if environ.get('GITHUB_ACTIONS'):
            pass
        test_func(*args, **kwargs)
    return wrapper
