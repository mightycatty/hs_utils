from typing import Any
import io
import pickle


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



