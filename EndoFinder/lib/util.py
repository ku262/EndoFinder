
import argparse
import inspect
from typing import Callable
from distutils.util import strtobool


def call_using_args(function: Callable, args: argparse.Namespace):
    """Calls the callable using arguments from an argparse container."""
    signature = inspect.signature(function)
    arguments = {key: getattr(args, key) for key in signature.parameters}
    return function(**arguments)


def parse_bool(bool_str):
    return bool(strtobool(bool_str))
