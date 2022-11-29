#!/usr/bin/env python3
import os
import re
import sys
import glob
import shutil
import logging
from argparse import ArgumentParser
from subprocess import Popen, STDOUT
from contextlib import contextmanager

from matmodlab2.core.logio import get_logger
from matmodlab2.core.environ import environ

ext_support_dir = os.path.dirname(os.path.realpath(__file__))
aba_support_dir = os.path.join(ext_support_dir, "../umat")

# "Lite" version of blas/lapack
lapack_lite = os.path.join(ext_support_dir, "blas_lapack-lite.f")
lapack_lite_obj = os.path.splitext(lapack_lite)[0] + ".o"
assert os.path.isfile(lapack_lite)

# Fortran I/O
mml_io = os.path.join(ext_support_dir, "mml_io.f90")
assert os.path.isfile(mml_io)

# Abaqus related files
aba_sdvini = os.path.join(aba_support_dir, "aba_sdvini.f90")
assert os.path.isfile(aba_sdvini)
aba_utils = os.path.join(aba_support_dir, "aba_utils.f90")
assert os.path.isfile(aba_utils)
umat_pyf = os.path.join(aba_support_dir, "umat.pyf")
assert os.path.isfile(umat_pyf)
vumat_pyf = os.path.join(aba_support_dir, "vumat.pyf")
assert os.path.isfile(vumat_pyf)
uhyper_pyf = os.path.join(aba_support_dir, "uhyper.pyf")
assert os.path.isfile(uhyper_pyf)
tensalg_f90 = os.path.join(aba_support_dir, "tensalg.f90")
assert os.path.isfile(tensalg_f90)
uhyper_wrap_f90 = os.path.join(aba_support_dir, "uhyper_wrap.f90")
assert os.path.isfile(uhyper_wrap_f90)


class ExtensionNotBuilt(Exception):
    pass


def which(name):
    """Find the executable name on PATH"""
    for path in os.getenv("PATH", "").split(os.pathsep):
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, name)):
            return os.path.join(path, name)
    return None


def clean_f2py_tracks(dirname):
    if not os.path.isdir(dirname):
        return
    for pat in ("*.so.dSYM", "*-f2pywrappers2.*", "*module.c"):
        for item in glob.glob(os.path.join(dirname, pat)):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)


def build_extension_module(name, *files, **kwds):
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
    fc : str
        Fortran compiler

    """
    include_dirs = kwds.pop("include_dirs", [])
    verbose = kwds.pop("verbose", False)
    fc = kwds.pop("fc", None)
    cwd = kwds.pop("cwd", None)

    the_loglevel = environ.loglevel
    environ.loglevel = logging.DEBUG
    logger = get_logger("build-ext")
    fc = fc or which("gfortran")
    if fc is None:
        raise OSError("Fortran compiler not found")

    # Check source files
    files = list(files)
    for file in files:
        if not os.path.isfile(file):
            raise OSError("{0!r}: file not found".format(file))

    if name != "_matfuncs_sq3":
        files.append(mml_io)

    # We'll add the object file back in
    if lapack_lite in files:
        files.remove(lapack_lite)

    # Everyone gets lapack!
    if lapack_lite_obj not in files:
        files.append(lapack_lite_obj)

    if not os.path.isfile(lapack_lite_obj):
        _build_blas_lapack(logger, fc)

    include_dirs = include_dirs or []

    if any(" " in x for x in files):
        logger.warning("File paths with spaces are known to fail to build")

    command = ["f2py", "-c"]

    # Build the fortran flags argument
    fflags = ["-Wno-unused-dummy-argument", "-fPIC", "-shared"]
    if os.getenv("FCFLAGS"):
        fflags.extend(os.environ["FCFLAGS"].split())
    command.extend(
        [
            "--f77flags={0!r}".format(" ".join(fflags)),
            "--f90flags={0!r}".format(" ".join(fflags)),
        ]
    )
    command.extend(["--include-paths", ":".join(include_dirs)])
    command.extend(["-m", name])
    command.extend(files)

    logger.info(
        "building extension module {0!r}... ".format(name), extra={"continued": 1}
    )

    logfile = None
    cwd = cwd or os.getcwd()
    if verbose:
        # Call directly - LOTS of output!
        p = Popen(command, cwd=cwd)
        p.wait()
    elif environ.notebook:
        from IPython.utils import io

        with io.capture_output():
            p = Popen(command, cwd=cwd)
            p.wait()
    else:
        logfile = os.path.join(cwd, "build.log")
        with stdout_redirected(to=logfile), merged_stderr_stdout():
            p = Popen(command, cwd=cwd)
            p.wait()
    logger.info("done")

    if logfile is not None and logfile != sys.stdout:
        os.remove(logfile)

    # Return the loglevel back to what it was
    environ.loglevel = the_loglevel

    clean_f2py_tracks(cwd)

    if p.returncode != 0:
        logger.error(f"Failed to build {name}")
        raise ExtensionNotBuilt(name)

    return 0


def _build_blas_lapack(logger, fc):
    logger.info("building blas_lapack-lite... ", extra={"continued": 1})
    cmd = [fc, "-fPIC", "-shared", "-O3", lapack_lite, "-o" + lapack_lite_obj]
    proc = Popen(cmd, stdout=open(os.devnull, "a"), stderr=STDOUT, cwd=ext_support_dir)
    proc.wait()
    if proc.returncode == 0:
        logger.info("done")
    else:
        logger.info("failed")
    return proc.returncode


def fileno(file_or_fd):
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
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
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, "wb") as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def merged_stderr_stdout():  # $ exec 2>&1
    return stdout_redirected(to=sys.stdout, stdout=sys.stderr)


def build_umat(*files, **kwds):
    """Build the umat extension module (material model)"""
    clean_f2py_tracks(aba_support_dir)
    sdvini_defined = has_sdvini(*files)
    files = list(files) + [aba_utils, umat_pyf]
    if not sdvini_defined:
        files.append(aba_sdvini)
    include_dirs = kwds.pop("include_dirs", None) or []
    include_dirs.append(aba_support_dir)
    return build_extension_module("_umat", *files, include_dirs=include_dirs, **kwds)


def build_vumat(*files, **kwds):
    """Build the vumat extension module (material model)"""
    clean_f2py_tracks(aba_support_dir)
    files = list(files) + [aba_utils, vumat_pyf]
    include_dirs = kwds.pop("include_dirs", None) or []
    include_dirs.append(aba_support_dir)
    return build_extension_module("_vumat", *files, include_dirs=include_dirs, **kwds)


def build_uhyper(*files, **kwds):
    """Build the uhyper extension module (material model)"""
    clean_f2py_tracks(aba_support_dir)
    sdvini_defined = has_sdvini(*files)
    files = list(files) + [aba_utils, uhyper_pyf, tensalg_f90, uhyper_wrap_f90]
    if not sdvini_defined:
        files.append(aba_sdvini)
    include_dirs = kwds.pop("include_dirs", None) or []
    include_dirs.append(aba_support_dir)
    return build_extension_module("_uhyper", *files, include_dirs=include_dirs, **kwds)


def build_mml_matrix_functions():
    """Build the mml linear algebra library"""
    name = "_matfuncs_sq3"
    mfuncs_pyf = os.path.join(ext_support_dir, "matrix_funcs.pyf")
    mfuncs_f90 = os.path.join(ext_support_dir, "matrix_funcs.f90")
    dgpadm_f = os.path.join(ext_support_dir, "dgpadm.f")
    files = [mfuncs_pyf, mfuncs_f90, lapack_lite, dgpadm_f]
    package_path = os.path.join(ext_support_dir, "../core")
    command = ["f2py", "-c"]
    command.extend(files)
    p = Popen(command, cwd=package_path)
    p.wait()
    if p.returncode != 0:
        raise ExtensionNotBuilt(name)
    return 0


def has_sdvini(*files):
    for file in files:
        with open(file) as fh:
            if re.search("(?i)\w+subroutine\s+sdvini", fh.read()):
                return True
    return False


def main():
    p = ArgumentParser()
    p.add_argument("name")
    p.add_argument("files", nargs="*")
    p.add_argument("--include-dirs", action="append", default=None)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--package-path", default=None)
    p.add_argument("--fc", default=False)
    args = p.parse_args()
    if args.name == "matfuncs":
        return build_mml_matrix_functions()
    if not args.files:
        raise ValueError("Missing source files")
    kwds = {
        "include_dirs": args.include_dirs,
        "verbose": args.verbose,
        "fc": args.fc,
        "cwd": args.package_path,
    }
    files = list(args.files or [])
    if args.name == "umat":
        return build_umat(*files, **kwds)
    elif args.name == "uhyper":
        return build_uhyper(*files, **kwds)
    elif args.name == "vumat":
        return build_vumat(*files, **kwds)
    else:
        return build_extension_module(args.name, *files, **kwds)


if __name__ == "__main__":
    main()
