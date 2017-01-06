import sys
from argparse import ArgumentParser
import logging

try:
    import mml_userenv
except ImportError:
    mml_userenv = None

class SimulationEnvironment:
    SQA = False
    verbosity = 1
    notebook = False
    parent_process = False
    loglevel = logging.WARNING
environ = SimulationEnvironment()

# Look for options in the user's environment file
if mml_userenv is not None:
    for item in dir(mml_userenv):
        if item.startswith('_'):
            continue
        setattr(environ, item, getattr(mml_userenv, item))

# Get some commands that can change the environment
p = ArgumentParser(add_help=False)
p.add_argument('--verbosity', default=None, type=int)
args, extra = p.parse_known_args()
if args.verbosity is not None:
    v = min(max(args.verbosity, 0), 3)
    level = {0: logging.ERROR,
             1: logging.WARNING,
             2: logging.INFO,
             3: logging.DEBUG}[v]
    environ.loglevel = level
sys.argv[:] = [sys.argv[0]] + extra
