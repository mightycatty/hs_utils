import logging
import telebot
from telebot import apihelper
from typing import Any
import io
import pickle


def line_profile(func):
    """
    function decoderator for line-wise profile
    usage:
        from evaluation_utils import line_profile
        @line_profile
        some_fn()
    reference:
        https://github.com/rkern/line_profiler#kernprof
    :param func:
    :return:
    """
    from functools import wraps
    import line_profiler
    prof = line_profiler.LineProfiler()
    @wraps(func)
    def newfunc(*args, **kwargs):
        try:
            pfunc = prof(func)
            return pfunc(*args, **kwargs)
        finally:
            prof.print_stats(1e-3)
    return newfunc


class EasyDict(dict):
    """
    Convenience class that behaves like a dict but allows access with the attribute syntax.
    one can use this to as a simple kwargs passed to a function
    eg:
        input_kwargs = EasyDick()
        input_kwargs.a = 1
        input_kwargs.b = 2
        some_fn(**input_kwargs)
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def is_pickleable(obj: Any) -> bool:
    """
    as named
    :param obj:
    :return:
    """
    try:
        with io.BytesIO() as stream:
            pickle.dump(obj, stream)
        return True
    except:
        return False



