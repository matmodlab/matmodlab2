from __future__ import print_function
import os
import sys
import time
import shutil
import logging
import datetime
import traceback
import subprocess
import numpy as np

from ..core.environ import environ
from ..core.logio import get_logger, add_filehandler, splash
from .tabular import TabularWriter

IOPT = 0
LASTEVALD = None
BIGNUM = 1.E+20
MAXITER = 50
TOL = 1.E-06
MAX_ERR = None

class Optimizer(object):
    def __init__(self, job, func, xinit, method='simplex',
                 maxiter=MAXITER, tolerance=TOL, descriptors=None,
                 funcargs=[], Ns=10, dryrun=0, keep_intermediate=True,
                 halt_on_err=False):
        environ.raise_e = True
        environ.no_cutback = True
        global IOPT
        IOPT = 0
        self.job = job
        self.func = func
        self.ran = False
        self.dryrun = dryrun
        self.halt_on_err = halt_on_err

        d = os.path.realpath(os.getcwd())
        self.directory = d
        self.rootd = os.path.join(d, job + ".eval")
        self.output = os.path.join(self.rootd, job + '.edb')

        if not isinstance(descriptors, (list, tuple)):
            descriptors = [descriptors]
        self.descriptors = descriptors
        self.nresp = len(descriptors)

        if not isinstance(funcargs, (list, tuple)):
            funcargs = [funcargs]
        self.funcargs = [x for x in funcargs]

        # Number of evaluations per dimension for brute force optimizations.
        self.Ns = int(round(max(Ns, 2.0)))

        # check method
        valid = ('simplex', 'powell', 'cobyla', 'brute')
        if method.lower() not in valid:
            msg = 'Unknown optimization method {0!r}. '.format(method)
            msg += 'Choose from {0}'.format(','.join(valid))
            raise ValueError(msg)
        self.method = method.lower()

        # set up logger
        if os.path.isdir(self.rootd):
            shutil.rmtree(self.rootd)
        os.makedirs(self.rootd)

        # basic logger
        logfile = os.path.join(self.rootd, self.job + '.log')
        logger = get_logger('optimize')
        add_filehandler(logger, logfile)
        splash(logger)

        # individual sims only log to file and not the console
        environ.parent_process = 1

        # check xinit
        self.names = []
        self.idata = []
        self.bounds = []
        for x in xinit:
            if not isinstance(x, OptimizeVariable):
                raise TypeError("all xinit must be of type OptimizeVariable")

            self.names.append(x.name)
            self.idata.append(x.initial_value)

            if x.bounds is not None:
                if self.method in ('simplex', 'powell'):
                    logger.warn('optimization method does not support bounds')
                    x.bounds = None
            self.bounds.append(x.bounds)

        if self.method in ('simplex', 'powell'):
            self.bounds = None

        if maxiter <= 0:
            logger.warn("maxiter < 0, setting to default value")
            maxiter = MAXITER
        self.maxiter = maxiter

        if tolerance <= 0:
            logger.warn("tolerance < 0, setting to default value")
            tolerance = TOL
        self.tolerance = tolerance

        self.tabular = TabularWriter(self.output, job)
        self.timing = {}

        # write summary to the log file
        str_pars = "\n".join("  {0}={1:.2g}".format(name, self.idata[i])
                             for (i, name) in enumerate(self.names))
        resp = "\n".join("  {0}".format(it) for it in self.descriptors)
        summary = """
summary of optimization job input
------- -- ------------ --- -----
Job:    {0}
Method: {1}
Variables: {2:d}
{3}
Response descriptors:
{4}
""".format(self.job, self.method, len(self.names), str_pars, resp)
        logger.info(summary)

    def run(self):
        """Run the optimization job

        Set up directory to run the optimization job and call the minimizer

        """
        import scipy.optimize
        logger = logging.getLogger('optimize')

        self.timing["start"] = time.time()
        logger.info("{0}: Starting optimization jobs...".format(self.job))

        # optimization methods work best with number around 1, here we
        # normalize the optimization variables and save the multiplier to be
        # used when the function gets called by the optimizer.
        xfac = []
        for ival in self.idata:
            mag = eval("1.e" + "{0:12.6E}".format(ival).split("E")[1])
            xfac.append(mag)
            continue
        xfac = np.array(xfac)
        x0 = self.idata / xfac

        if self.bounds is not None:
            # user has specified bounds on the parameters to be optimized. Here,
            # we convert the bounds to inequality constraints (for cobyla) and
            # normalized bounds (for brute).
            lcons, ucons = [], []
            normalized_bounds = []
            for ibnd, bound in enumerate(self.bounds):
                lbnd, ubnd = bound
                lcons.append(lambda z, idx=ibnd, bnd=lbnd: z[idx]-bnd/xfac[idx])
                ucons.append(lambda z, idx=ibnd, bnd=ubnd: bnd/xfac[idx]-z[idx])
                normalized_bounds.append((lbnd/xfac[ibnd], ubnd/xfac[ibnd]))
                continue
            cons = lcons + ucons

        args = (self.func, self.funcargs, self.rootd, self.halt_on_err,
                self.job, self.names, self.descriptors, self.tabular, xfac)

        if self.dryrun:
            # do a dry run of the function
            err = run_job(x0, *args)
            if err == np.nan:
                s = 'Optimization dry run failed'
                logger.error(s)
            else:
                s = 'Optimization dry run successful'
                logger.info(s)
            if environ.notebook:
                print(s)
            self.dryrun_error = err
            return

        if self.method == 'simplex':
            xopt = scipy.optimize.fmin(
                run_job, x0, xtol=self.tolerance, ftol=self.tolerance,
                maxiter=self.maxiter, args=args, disp=0)

        elif self.method == 'powell':
            xopt = scipy.optimize.fmin_powell(
                run_job, x0, xtol=self.tolerance, ftol=self.tolerance,
                maxiter=self.maxiter, args=args, disp=0)

        elif self.method == 'cobyla':
            xopt = scipy.optimize.fmin_cobyla(
                run_job, x0, cons, consargs=(), args=args, disp=0)

        elif self.method == 'brute':
            xopt = scipy.optimize.brute(
                run_job, normalized_bounds, args=args, Ns=self.Ns, disp=0,
                finish=None)

        self.xopt = xopt * xfac

        self.timing["end"] = time.time()

        logger.info("\nOptimization jobs complete")

        self.finish()

        return

    def finish(self):
        """ finish up the optimization job """
        logger = logging.getLogger('optimize')
        self.tabular.close()
        self.ran = True
        opt_pars = "\n".join("  {0}={1:12.6E}".format(name, self.xopt[i])
                             for (i, name) in enumerate(self.names))
        opt_time = self.timing["end"] - self.timing["start"]
        summary = """
Summary of optimization results
------- -- ------------ -------
{0}: calculations completed ({1:.4f}s.)
Iterations: {2}
Optimized parameters
{3}
""".format(self.job, opt_time, IOPT, opt_pars)
        logger.info(summary)

        # write out optimized params
        with open(os.path.join(self.rootd, "params.opt"), "w") as fobj:
            for (i, name) in enumerate(self.names):
                fobj.write("{0} = {1: .18f}\n".format(name, self.xopt[i]))
        environ.parent_process = 0

        # Link directory 'final' to the last evaluation directory
        os.symlink(os.path.relpath(LASTEVALD, start=self.rootd),
                   os.path.join(self.rootd, "final"))

        if environ.notebook:
            print('\nDone')

    def todict(self):
        if not self.ran:
            return None
        return dict(zip(self.names, self.xopt))

    @property
    def duration(self):
        if not self.ran:
            return None
        return self.timing["end"] - self.timing["start"]

def catd(d, i):
    N = 3
    return os.path.join(d, "eval_{0:0{1}d}".format(i, N))

def run_job(xcall, *args):
    """Objective function

    Creates a directory to run the current job, runs the job, returns the
    value of the objective function determined.

    Returns
    -------
    error : float
        Error in job

    """
    global IOPT, LASTEVALD, MAX_ERR
    logger = logging.getLogger('optimize')
    func, funcargs, rootd, halt_on_err, job, xnames, desc, tabular, xfac = args

    IOPT += 1
    evald = catd(rootd, IOPT)
    os.mkdir(evald)
    LASTEVALD = evald

    cwd = os.getcwd()
    os.chdir(evald)
    environ.simulation_dir = evald

    # write the params.in for this run
    x = xcall * xfac
    parameters = zip(xnames, x)
    with open("params.in", "w") as fobj:
        for name, param in parameters:
            fobj.write("{0} = {1: .18f}\n".format(name, param))

    logger.info("starting job {0} with {1}... ".format(
        IOPT, ",".join("{0}={1:.2g}".format(n, p) for n, p in parameters)),
        extra={'continued':1})

    if environ.notebook:
        print('\rRunning job {0}'.format(IOPT), end='')

    try:
        err = func(x, xnames, evald, job, *funcargs)
        logger.info("done (error={0:.4e})".format(err))
        stat = 0
        if MAX_ERR is None:
            MAX_ERR = err
        MAX_ERR = max(MAX_ERR, err)
    except BaseException:
        string = traceback.format_exc()
        if not environ.notebook:
            logger.error("\nRun {0} failed with the following "
                         "exception:\n{1}".format(IOPT, string))
        else:
            logger.info("failed")

        if halt_on_err:
            logger.error("\n\nHalting optimization on error at user request.\n")
            raise  # re-raise previous error

        stat = 1
        if MAX_ERR is None:
            err = np.nan
        else:
            err = MAX_ERR

    tabular.write_eval_info(IOPT, stat, evald, parameters, ((desc[0], err),))

    os.chdir(cwd)

    return err

class OptimizeVariable(object):

    def __init__(self, name, initial_value, bounds=None):
        self.name = name
        self.ival = initial_value
        self.cval = initial_value
        self.bounds = bounds

        errors = 0
        # check bounds
        if bounds is not None:
            if not isinstance(bounds, (list, tuple, np.ndarray)):
                raise ValueError("expected bounds to be a tuple of length 2")
            if len(bounds) != 2:
                raise ValueError("expected bounds to be a tuple of length 2")
            if bounds[0] is None: bounds[0] = -BIGNUM
            if bounds[1] is None: bounds[1] =  BIGNUM

            if bounds[0] > bounds[1]:
                errors += 1
                logging.error("{0}: upper bound < lower bound".format(name))

            if bounds[1] < initial_value < bounds[0]:
                errors += 1
                logging.error("{0}: initial value not bracketed "
                              "by bounds".format(name))
            if errors:
                raise ValueError("stopping due to previous errors")

            self.bounds = np.array(bounds)

    def __repr__(self):
        return "opt{0}({1})".format(self.name, self.initial_value)

    @property
    def current_value(self):
        return self.cval

    @property
    def initial_value(self):
        return self.ival
