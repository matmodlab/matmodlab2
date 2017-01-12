import warnings
import numpy as np
from copy import deepcopy as copy
from .environ import environ
from .tensor import VOIGT
from .deformation import update_deformation

def d_from_prescribed_stress(material, t, dt, temp, dtemp, f0, f,
                             stran, d, sig, statev, v, sigspec):
    '''Determine the symmetric part of the velocity gradient given stress

    Parameters
    ----------

    Returns
    -------

    Approach
    --------
    Seek to determine the unknown components of the symmetric part of
    velocity gradient d[v] satisfying

                               P(d[v]) = Ppres[:]                      (1)

    where P is the current stress, d the symmetric part of the velocity
    gradient, v is a vector subscript array containing the components for
    which stresses (or stress rates) are prescribed, and Ppres[:] are the
    prescribed values at the current time.

    Solution is found iteratively in (up to) 3 steps
      1) Call newton to solve 1, return stress, statev, d if converged
      2) Call newton with d[v] = 0. to solve 1, return stress, statev, d
         if converged
      3) Call simplex with d[v] = 0. to solve 1, return stress, statev, d

    '''
    dsave = d.copy()

    d = newton(material, t, dt, temp, dtemp, f0, f, stran, d,
               sig, statev, v, sigspec)
    if d is not None:
        return d

    # --- didn't converge, try Newton's method with initial
    # --- d[v]=0.
    d = dsave.copy()
    d[v] = np.zeros(len(v))
    d = newton(material, t, dt, temp, dtemp, f0, f, stran, d,
               sig, statev, v, sigspec)
    if d is not None:
        return d

    # --- Still didn't converge. Try downhill simplex method and accept
    #     whatever answer it returns:
    d = dsave.copy()
    return simplex(material, t, dt, temp, dtemp, f0, f, stran, d,
                   sig, statev, v, sigspec)

def newton(material, t, dt, temp, dtemp, f0, farg, stran, darg,
           sigarg, statev_arg, v, sigspec):
    '''Seek to determine the unknown components of the symmetric part of velocity
    gradient d[v] satisfying

                               sig(d[v]) = sigspec

    where sig is the current stress, d the symmetric part of the velocity
    gradient, v is a vector subscript array containing the components for
    which stresses (or stress rates) are prescribed, and sigspec are the
    prescribed values at the current time.

    Parameters
    ----------
    material : instance
        constiutive model instance
    dt : float
        time step
    sig : ndarray
        stress at beginning of step
    statev_arg : ndarray
        state dependent variables at beginning of step
    v : ndarray
        vector subscript array containing the components for which
        stresses (or stress rates) are specified
    sigspec : ndarray
        Prescribed stress

    Returns
    -------
    d : ndarray || None
        If converged, the symmetric part of the velocity gradient, else None

    Notes
    -----
    The approach is an iterative scheme employing a multidimensional Newton's
    method. Each iteration begins with a call to subroutine jacobian, which
    numerically computes the Jacobian submatrix

                                  Js = J[v, v]

    where J[:,;] is the full Jacobian matrix J = dsig/deps. The value of
    d[v] is then updated according to

                d[v] = d[v] - Jsi*sigerr(d[v])/dt

    where

                   sigerr(d[v]) = sig(d[v]) - sigspec

    The process is repeated until a convergence critierion is satisfied. The
    argument converged is a flag indicat- ing whether or not the procedure
    converged:

    '''
    depsmag = lambda a: np.sqrt(sum(a[:3] ** 2) + 2. * sum(a[3:] ** 2)) * dt

    # Initialize
    eps = np.finfo(np.float).eps
    tol1, tol2 = eps, np.sqrt(eps)
    maxit1, maxit2, depsmax = 20, 30, .2

    sig = sigarg.copy()
    d = darg.copy()
    f = farg.copy()
    statev = copy(statev_arg)

    sigsave = sig.copy()
    statev_save = copy(statev)

    # --- Check if strain increment is too large
    if (depsmag(d) > depsmax):
        return None

    # update the material state to get the first guess at the new stress
    sig, statev, stif = material.eval(t, dt, temp, dtemp,
                                      f0, f, stran, d, sig, statev)
    sigerr = sig[v] - sigspec

    # --- Perform Newton iteration
    for i in range(maxit2):
        sig = sigsave.copy()
        statev = copy(statev_save)
        stif = material.eval(t, dt, temp, dtemp,
                             f0, f, stran, d, sig, statev)[2]
        if stif is None:
            # material models without an analytic jacobian send the Jacobian
            # back as None so that it is found numerically here. Likewise, we
            # find the numerical jacobian for visco materials - otherwise we
            # would have to convert the the stiffness to that corresponding to
            # the Truesdell rate, pull it back to the reference frame, apply
            # the visco correction, push it forward, and convert to Jaummann
            # rate. It's not as trivial as it sounds...
            statev = copy(statev_save)
            stif = numerical_jacobian(material, t, dt, temp, dtemp, f0,
                                      f, stran, d, sig, statev, v)
        else:
            stif = stif[[[i] for i in v], v]

        if environ.SQA:
            try:
                evals = np.linalg.eigvalsh(stif)
            except np.linalg.LinAlgError:
                raise RuntimeError('failed to determine elastic '
                                   'stiffness eigenvalues')
            else:
                if np.any(evals < 0.):
                    negevals = evals[np.where(evals < 0.)]
                    warnings.warn('negative eigen value[s] encountered '
                                  'in material Jacobian: '
                                  '{0} ({1:.2f})'.format(negevals, t))

        try:
            d[v] -= np.linalg.solve(stif, sigerr) / dt
        except np.linalg.LinAlgError:
            if environ.SQA:
                warnings.warn('using least squares approximation to '
                              'matrix inverse')
            d[v] -= np.linalg.lstsq(stif, sigerr)[0] / dt

        if (depsmag(d) > depsmax or  np.any(np.isnan(d)) or np.any(np.isinf(d))):
            # increment too large
            return None

        # with the updated rate of deformation, update stress and check
        sig = sigsave.copy()
        statev = copy(statev_save)
        fp, ep = update_deformation(f, d, dt, 0)
        sig, statev, stif = material.eval(t, dt, temp, dtemp,
                                          f0, fp, ep, d, sig, statev)

        sigerr = sig[v] - sigspec
        dnom = max(np.amax(np.abs(sigspec)), 1.)
        relerr = np.amax(np.abs(sigerr) / dnom)

        if i <= maxit1 and relerr < tol1:
            return d

        elif i > maxit1 and relerr < tol2:
            return d

        continue

    # didn't converge, restore restore data and exit
    return None

def simplex(material, t, dt, temp, dtemp, f0, farg, stran, darg, sigarg,
            statev_arg, v, sigspec):
    '''Perform a downhill simplex search to find sym_velgrad[v] such that

                        sig(sym_velgrad[v]) = sigspec[v]

    Parameters
    ----------
    material : instance
        constiutive model instance
    dt : float
        time step
    sig : ndarray
        stress at beginning of step
    statev_arg : ndarray
        state dependent variables at beginning of step
    v : ndarray
        vector subscript array containing the components for which
        stresses (or stress rates) are specified
    sigspec : ndarray
        Prescribed stress

    Returns
    -------
    d : ndarray
        the symmetric part of the velocity gradient

    '''
    # --- Perform the simplex search
    import scipy.optimize
    d = darg.copy()
    f = farg.copy()
    sig = sigarg.copy()
    statev = copy(statev_arg)
    args = (material, t, dt, temp, dtemp, f0, f, stran, d,
            sig, statev, v, sigspec)
    d[v] = scipy.optimize.fmin(_func, d[v], args=args, maxiter=20, disp=False)
    return d

def _func(x, material, t, dt, temp, dtemp, f0, farg, stran, darg,
          sigarg, statev_arg, v, sigspec):
    '''Objective function to be optimized by simplex

    '''
    d = darg.copy()
    f = farg.copy()
    sig = sigarg.copy()
    statev = copy(statev_arg)

    # initialize
    d[v] = x
    fp, ep = update_deformation(f, d, dt, 0)

    # store the best guesses
    sig, statev, stif = material.eval(t, dt, temp, dtemp,
                                      f0, fp, ep, d, sig, statev)

    # check the error
    error = 0.
    for i, j in enumerate(v):
        error += (sig[j] - sigspec[i]) ** 2
        continue

    return error

def numerical_jacobian(material, time, dtime, temp, dtemp, F0, F, stran, d,
                        stress, statev, v):
    '''Numerically compute material Jacobian by a centered difference scheme.

    Parameters
    ----------
    time : float
        Time at beginning of step
    dtime : float
        Time step length.  `time+dtime` is the time at the end of the step
    temp : float
        Temperature at beginning of step
    dtemp : float
        Temperature increment. `temp+dtemp` is the temperature at the end
        of the step
    F0, F : ndarray
        Deformation gradient at the beginning and end of the step
    strain : ndarray
        Strain at the beginning of the step
    d : ndarray
        Symmetric part of the velocity gradient at the middle of the step
    stress : ndarray
        Stress at the beginning of the step
    statev : ndarray
        State variables at the beginning of the step
    v : ndarray
        Array of subcomponents of Jacobian to return

    Returns
    -------
    Js : array_like
        Jacobian of the deformation J = dsig / dE

    Notes
    -----
    The submatrix returned is the one formed by the intersections of the
    rows and columns specified in the vector subscript array, v. That is,
    Js = J[v, v]. The physical array containing this submatrix is
    assumed to be dimensioned Js[nv, nv], where nv is the number of
    elements in v. Note that in the special case v = [1,2,3,4,5,6], with
    nv = 6, the matrix that is returned is the full Jacobian matrix, J.

    The components of Js are computed numerically using a centered
    differencing scheme which requires two calls to the material model
    subroutine for each element of v. The centering is about the point eps
    = epsold + d * dt, where d is the rate-of-strain array.

    History
    -------
    This subroutine is a python implementation of a routine by the same
    name in Tom Pucick's MMD driver.

    Authors
    -------
    Tom Pucick, original fortran implementation in the MMD driver
    Tim Fuller, Sandial National Laboratories, tjfulle@sandia.gov

    '''
    # local variables
    nv = len(v)
    deps =  np.sqrt(np.finfo(np.float64).eps)
    Jsub = np.zeros((nv, nv))
    dtime = 1 if dtime < 1.e-12 else dtime

    for i in range(nv):
        # perturb forward
        Dp = d.copy()
        Dp[v[i]] = d[v[i]] + (deps / dtime) / 2.
        Fp, Ep = update_deformation(F, Dp, dtime, 0)
        sigp = stress.copy()
        xp = copy(statev)
        sigp = material.eval(time, dtime, temp, dtemp,
                             F0, Fp, Ep, Dp, sigp, xp)[0]

        # perturb backward
        Dm = d.copy()
        Dm[v[i]] = d[v[i]] - (deps / dtime) / 2.
        Fm, Em = update_deformation(F, Dm, dtime, 0)
        sigm = stress.copy()
        xm = copy(statev)
        sigm = material.eval(time, dtime, temp, dtemp,
                             F0, Fm, Em, Dm, sigm, xm)[0]

        # compute component of jacobian
        Jsub[i, :] = (sigp[v] - sigm[v]) / deps

        continue

    return Jsub
