from os import environ
from typing import Callable

import torch.cuda as cuda


def skip_github(test_func: Callable[..., None]) -> Callable[..., None]:
    def wrapper(*args, **kwargs):
        if environ.get('GITHUB_ACTIONS'):
            pass
        test_func(*args, **kwargs)
    return wrapper


def skip_non_gpu(test_func: Callable[..., None]) -> Callable[..., None]:
    def wrapper(*args, **kwargs):
        if not cuda.is_available():
            pass
        test_func(*args, **kwargs)
    return wrapper
