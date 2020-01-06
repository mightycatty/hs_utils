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