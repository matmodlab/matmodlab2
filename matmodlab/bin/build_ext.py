#!/usr/bin/env python
import os
import re
import sys
import glob
import shutil
import logging
import tempfile
from argparse import ArgumentParser
from subprocess import Popen, STDOUT
from contextlib import contextmanager

# distutils
import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.system_info import get_info
from numpy.distutils.core import setup

from matmodlab.core.logio import get_logger
from matmodlab.core.environ import environ

D = os.path.dirname(os.path.realpath(__file__))
UD = os.path.join(D, '../umat')
environ.loglevel = logging.DEBUG
logger = get_logger('build-umat')

lapack_lite = os.path.join(UD, 'blas_lapack-lite.f')
assert os.path.isfile(lapack_lite)
lapack_lite_obj = os.path.splitext(lapack_lite)[0] + '.o'
aba_utils = os.path.join(UD, 'aba_utils.f90')
umat_pyf = os.path.join(UD, 'umat.pyf')
aba_sdvini = os.path.join(UD, 'aba_sdvini.f90')
mml_io = os.path.join(UD, 'mml_io.f90')

def which(name):
    for path in os.getenv('PATH', '').split(os.pathsep):
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, name)):
            return os.path.join(path, name)
    return None

def build_extension_module(name, sources, include_dirs=None,
                           user_ics=False, fc=None):

    fc = fc or which('gfortran')
    if fc is None:
        raise OSError('Fortran compiler not found')

    # Check source files
    for source_file in sources:
        if not os.path.isfile(source_file):
            raise OSError('{0!r}: file not found'.format(source_file))
    sources.append(mml_io)

    include_dirs = include_dirs or []

    umat = name.lower() == 'umat'
    if umat:
        name = '_umat'
        sources.extend([aba_utils, umat_pyf])
        if not user_ics:
            sources.append(aba_sdvini)

    if any(' ' in x for x in sources):
        logger.warning('File paths with spaces are known to fail to build')

    options = {}

    # Build a "lite" version of lapack
    options['extra_objects'] = [lapack_lite_obj]
    options['extra_compile_args'] = ['-fPIC', '-shared']
    options['include_dirs'] = include_dirs + [D]

    # Explicitly add this python distributions lib directory. This
    # shouldn't be necessary, but on some RHEL systems I have found that
    # it is
    d = os.path.join(os.path.dirname(sys.executable), '../lib')
    assert os.path.isdir(d)
    options["library_dirs"] = [d]

    if not os.path.isfile(lapack_lite_obj):
        stat = build_blas_lapack()
        if stat != 0:
            logger.error('failed to build blas_lapack, dependent '
                         'libraries will not be importable')

    cwd = os.getcwd()
    package_path = cwd
    build_dir = os.path.join(package_path, 'build')
    had_build_dir = os.path.isdir(build_dir)

    config = Configuration(name, parent_package='', top_path='',
                           package_path=package_path)
    config.add_extension(name, sources=sources, **options)

    fexec = '--f77exec={0} --f90exec={0}'.format(fc)
    argv = ['./setup.py', 'config_fc'] + fexec.split()
    fflags = ['-Wno-unused-dummy-argument']
    if os.getenv('FCFLAGS'):
        fflags.extend(os.environ['FCFLAGS'].split())
    fflags = ' '.join(fflags)
    fflags = '--f77flags={0!r} --f90flags={0!r}'.format(fflags).split()
    argv.extend(fflags)
    argv.extend('build_ext -i'.split())

    # build the extension modules with distutils setup
    logger.info('building extension module {0!r}... '.format(name),
                extra={'continued':1})
    failed = 0

    # change sys.argv for distutils
    hold = [x for x in sys.argv]
    sys.argv = [x for x in argv]
    logfile = None
    try:
        if environ.notebook:
            from IPython.utils import io
            with io.capture_output() as captured:
                setup(**config.todict())
        else:
            logfile = os.path.join(package_path, 'build.log')
            with stdout_redirected(to=logfile), merged_stderr_stdout():
                setup(**config.todict())
        logger.info('done')
    except:
        logger.info('failed')
        failed = 1
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        sys.argv = [x for x in hold]

    if failed:
        logger.error('Failed to build')
        raise SystemExit('{0}: failed to build'.format(name))

    if not had_build_dir:
        shutil.rmtree(build_dir)
    if logfile is not None:
        os.remove(logfile)
    for item in glob.glob(os.path.join(package_path, '*.so.dSYM')):
        if os.path.isdir(item):
            shutil.rmtree(item)
        else:
            os.remove(item)



def build_blas_lapack(fc=None):
    fc = fc or which('gfortran')
    if fc is None:
        raise OSError('Fortran compiler not found')
    logger.info('building blas_lapack-lite... ', extra={'continued':1})
    cmd = [fc, '-fPIC', '-shared', '-O3', lapack_lite, '-o' + lapack_lite_obj]
    proc = Popen(cmd, stdout=open(os.devnull, 'a'), stderr=STDOUT)
    proc.wait()
    if proc.returncode == 0:
        logger.info('done')
    else:
        logger.info('no')
    return proc.returncode

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """From:  http://stackoverflow.com/questions/4675728/
                        redirect-stdout-to-a-file-in-python/22434262#22434262

    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def merged_stderr_stdout():  # $ exec 2>&1
    return stdout_redirected(to=sys.stdout, stdout=sys.stderr)


def main():
    p = ArgumentParser()
    p.add_argument('name')
    p.add_argument('sources', nargs='+')
    args = p.parse_args()

    build_extension_module(args.name, args.sources)

if __name__ == '__main__':
    main()
