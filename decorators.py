import logging
import os
import time
from functools import wraps
from multiprocessing import Pool

import cloudpickle


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


def timethis(func):
    '''
    Decorator that reports the execution time.
    '''

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result

    return wrapper


# TODO: BUGs, "failed call to cuInit: CUDA_ERROR_NOT_INITIALIZED: initialization error"
class RunAsCUDASubprocess:
    """
    transparent gpu management for tensorflow and other gpu-required applications.
    1. make desired number of gpus visible to tensorflow
    2. completely release gpu resource when tf session closed
    3. select desired number of gpus with at least fraction of memory

    Credit to ed-alertedh:
        https://gist.github.com/ed-alertedh/85dc3a70d3972e742ca0c4296de7bf00
    """

    def __init__(self, num_gpus=0, memory_fraction=0.95, verbose=False):
        self._num_gpus = num_gpus
        self._memory_fraction = memory_fraction
        if not verbose:
            logging.getLogger('py3nvml.utils').setLevel(logging.ERROR)  # mute py3nvml logging info

    @staticmethod
    def _subprocess_code(num_gpus, memory_fraction, fn, args):
        # set the env vars inside the subprocess so that we don't alter the parent env
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see tensorflow issue #152
        try:
            import py3nvml
            num_grabbed = py3nvml.grab_gpus(num_gpus, gpu_fraction=memory_fraction)
        except Exception as e:
            print(e)
            print('\n try "pip install py3nvml" and try again')
            exit(0)
            # either CUDA is not installed on the system or py3nvml is not installed (which probably means the env
            # does not have CUDA-enabled packages). Either way, block the visible devices to be sure.
            # num_grabbed = 0
            # os.environ['CUDA_VISIBLE_DEVICES'] = ""

        assert num_grabbed == num_gpus, 'Could not grab {} GPU devices with {}% memory available'.format(
            num_gpus,
            memory_fraction * 100)
        if os.environ['CUDA_VISIBLE_DEVICES'] == "":
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # see tensorflow issues: #16284, #2175

        # using cloudpickle because it is more flexible about what functions it will
        # pickle (lambda functions, notebook code, etc.)
        return cloudpickle.loads(fn)(*args)

    def __call__(self, f):
        def wrapped_f(*args):
            with Pool(1) as p:
                return p.apply(RunAsCUDASubprocess._subprocess_code,
                               (self._num_gpus, self._memory_fraction, cloudpickle.dumps(f), args))

        return wrapped_f
