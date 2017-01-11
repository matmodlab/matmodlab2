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

from matmodlab.core.logio import get_logger
from matmodlab.core.environ import environ
from matmodlab.core.misc import is_listlike

ext_support_dir = os.path.dirname(os.path.realpath(__file__))
aba_support_dir = os.path.join(ext_support_dir, '../umat')

# "Lite" version of blas/lapack
lapack_lite = os.path.join(ext_support_dir, 'blas_lapack-lite.f')
lapack_lite_obj = os.path.splitext(lapack_lite)[0] + '.o'
assert os.path.isfile(lapack_lite)

# Fortran I/O
mml_io = os.path.join(ext_support_dir, 'mml_io.f90')
assert os.path.isfile(mml_io)

# Abaqus related files
aba_sdvini = os.path.join(aba_support_dir, 'aba_sdvini.f90')
assert os.path.isfile(aba_sdvini)
aba_utils = os.path.join(aba_support_dir, 'aba_utils.f90')
assert os.path.isfile(aba_utils)
umat_pyf = os.path.join(aba_support_dir, 'umat.pyf')
assert os.path.isfile(umat_pyf)

class ExtensionNotBuilt(Exception):
    pass

def which(name):
    """Find the executable name on PATH"""
    for path in os.getenv('PATH', '').split(os.pathsep):
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, name)):
            return os.path.join(path, name)
    return None

def clean_f2py_tracks(paths, dirs_to_remove):
    for dirname in paths:
        if not os.path.isdir(dirname):
            continue
        for pat in ('*.so.dSYM', '*-f2pywrappers2.*', '*module.c'):
            for item in glob.glob(os.path.join(dirname, pat)):
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
    for dirname in dirs_to_remove:
        shutil.rmtree(dirname)

def build_extension_module(name, sources, include_dirs=None, verbose=False,
                           user_ics=False, fc=None):
    """Build the fortran extension module (material model)

    Parameters
    ----------
    name : str
        The name of the extension module to build
    sources : list of str
        List of source files
    include_dirs : list of str
        List of extra include directories
    verbose : bool
        Write output to stdout if True, otherwise suppress stdout
    user_ics : bool
        List of source files includes source defining subroutine SDVINI.
        Applicable only for Abaqus umats.
    fc : str
        Fortran compiler

    Notes
    -----
    To build abaqus umats, the name must be 'umat'

    """
    the_loglevel = environ.loglevel
    environ.loglevel = logging.DEBUG
    logger = get_logger('build-ext')
    fc = fc or which('gfortran')
    if fc is None:
        raise OSError('Fortran compiler not found')

    # Check source files
    for source_file in sources:
        if not os.path.isfile(source_file):
            raise OSError('{0!r}: file not found'.format(source_file))

    if name != '_matfuncs_sq3':
        sources.append(mml_io)

    if lapack_lite not in sources:
        sources.append(lapack_lite)

    include_dirs = include_dirs or []

    umat = name.lower() == 'umat'
    if umat:
        # Build the umat module - add some Abaqus utility files
        name = '_umat'
        sources.extend([aba_utils, umat_pyf])
        if not user_ics:
            sources.append(aba_sdvini)
        include_dirs = include_dirs + [aba_support_dir]

    if any(' ' in x for x in sources):
        logger.warning('File paths with spaces are known to fail to build')

    command = ['f2py', '-c']

    # Build the fortran flags argument
    fflags = ['-Wno-unused-dummy-argument', '-fPIC', '-shared']
    if os.getenv('FCFLAGS'):
        fflags.extend(os.environ['FCFLAGS'].split())
    command.extend(['--f77flags={0!r}'.format(' '.join(fflags)),
                    '--f90flags={0!r}'.format(' '.join(fflags))])
    command.extend(['--include-paths', ':'.join(include_dirs)])
    command.extend(['-m', name])
    command.extend(sources)

    logger.info('building extension module {0!r}... '.format(name),
                extra={'continued':1})

    logfile = None
    if verbose:
        # Call directly - LOTS of output!
        p = Popen(command)
        p.wait()
    elif environ.notebook:
        from IPython.utils import io
        with io.capture_output() as captured:
            p = Popen(command)
            p.wait()
    else:
        logfile = os.path.join(os.getcwd(), 'build.log')
        with stdout_redirected(to=logfile), merged_stderr_stdout():
            p = Popen(command)
            p.wait()
    logger.info('done')

    if logfile is not None and logfile != sys.stdout:
        os.remove(logfile)

    # Return the loglevel back to what it was
    environ.loglevel = the_loglevel

    if p.returncode != 0:
        logger.error('Failed to build')
        raise ExtensionNotBuilt(name)

    return

def _build_blas_lapack(logger, fc):
    logger.info('building blas_lapack-lite... ', extra={'continued':1})
    cmd = [fc, '-fPIC', '-shared', '-O3', lapack_lite, '-o' + lapack_lite_obj]
    proc = Popen(cmd, stdout=open(os.devnull, 'a'), stderr=STDOUT)
    proc.wait()
    if proc.returncode == 0:
        logger.info('done')
    else:
        logger.info('failed')
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

def build_extension_module_as_subprocess(name, sources,
                                         include_dirs=None, verbose=False,
                                         user_ics=False, fc=None):
    """Build the extension module, but call as a subprocess.

    Parameters
    ----------
    Same as build_extension_module

    Notes
    -----
    This function exists since distutils can only be initialized once and we want to run build several different extensions
    """
    build_extension_module(name, sources, include_dirs=include_dirs,
                           verbose=verbose, user_ics=user_ics, fc=fc)

def build_mml_matrix_functions():
    """Build the mml linear algebra library"""
    name = '_matfuncs_sq3'
    mfuncs_pyf = os.path.join(ext_support_dir, 'matrix_funcs.pyf')
    mfuncs_f90 = os.path.join(ext_support_dir, 'matrix_funcs.f90')
    dgpadm_f = os.path.join(ext_support_dir, 'dgpadm.f')
    sources = [mfuncs_pyf, mfuncs_f90, lapack_lite, dgpadm_f]
    package_path = os.path.join(ext_support_dir, '../core')
    command = ['f2py', '-c']
    command.extend(sources)
    p = Popen(command, cwd=package_path)
    p.wait()
    if p.returncode != 0:
        raise ExtensionNotBuilt(name)
    return 0

def main():
    p = ArgumentParser()
    p.add_argument('name')
    p.add_argument('sources', nargs='*')
    p.add_argument('--include-dirs', action='append', default=None)
    p.add_argument('--verbose', action='store_true', default=False)
    p.add_argument('--package-path', default=None)
    p.add_argument('--user-ics', action='store_true', default=False)
    p.add_argument('--fc', default=False)
    args = p.parse_args()
    if args.name == 'matfuncs':
        return build_mml_matrix_functions()
    if not args.sources:
        raise ValueError('Missing sources argument')
    build_extension_module(args.name, args.sources,
                           include_dirs=args.include_dirs,
                           verbose=args.verbose,
                           user_ics=args.user_ics,
                           fc=args.fc)

if __name__ == '__main__':
    main()
