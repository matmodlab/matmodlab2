import os
import sys
import logging
import warnings
from argparse import ArgumentParser

__all__ = ['environ']

def load_user_env():
    user_env = {}
    filename = 'mml_userenv.py'
    candidate_files = []
    if os.getenv('MML_USERENV'):
        for item in os.getenv('MML_USERENV').split(os.pathsep):
            if os.path.isfile(item):
                candidate_files.append(item)
            elif os.path.isdir(item):
                if os.path.isfile(os.path.join(item, filename)):
                    candidate_files.append(os.path.join(item, filename))
    if os.path.isfile(os.path.join(os.getcwd(), filename)):
        candidate_files.append(os.path.join(os.getcwd(), filename))

    for user_env_file in candidate_files:
        # User environment file found - load it!
        with open(user_env_file) as fh:
            code = compile(fh.read(), user_env_file, 'exec')
            file_env = {}
            ns = {'my_filename': user_env_file,
                  'os': os, 'sys': sys}
            exec(code, ns, file_env)
            for item in file_env:
                if item.startswith('_'):
                    continue
                if item == 'sys_path':
                    # Modify the system path
                    if os.path.isdir(file_env[item]):
                        sys.path.insert(0, file_env[item])
                    else:
                        msg = 'sys_path variable {0!r} is not a directory'
                        warnings.warn(msg.format(file_env[item]))
                    continue
                user_env[item] = file_env[item]
    return user_env

class SimulationEnvironment:
    SQA = False
    verbosity = 1
    notebook = False
    parent_process = False
    loglevel = logging.WARNING
environ = SimulationEnvironment()

# Look for options in the user's environment file
for (key, val) in load_user_env().items():
    setattr(environ, key, val)

try:
    # Check to see if we are running in a Jupyter notebook
    get_ipython()
    print('Setting up the Matmodlab notebook environment')
    environ.notebook = True
except NameError:
    pass

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
