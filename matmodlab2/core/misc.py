"""Misc. functions"""
import os
from contextlib import contextmanager


def is_listlike(item):
    """Is item list like?"""
    try:
        [x for x in item]
        return not is_stringlike(item)
    except TypeError:
        return False


def is_stringlike(item):
    """Is item string like?"""
    try:
        item + "string"
        return True
    except TypeError:
        return False


def is_scalarlike(item):
    """Is item scalar like?"""
    try:
        float(item)
        return True
    except TypeError:
        return False


# Copied from six.py: https://github.com/benjaminp/six
def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""

    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get("__slots__")
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop("__dict__", None)
        orig_vars.pop("__weakref__", None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)

    return wrapper


@contextmanager
def working_dir(dirname):
    cwd = os.getcwd()
    os.chdir(dirname)
    yield dirname
    os.chdir(cwd)
