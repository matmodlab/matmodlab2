import logging
import numpy as np
from copy import deepcopy as copy
from .mmlabpack import update_deformation

class Material(object):
    """The material model base class

    Notes
    -----
    The `Material` class is a base class and is meant to be inherited by
    concrete implementations of material models. At minimum, the material model
    must provide an `eval` method that is called by the model driver to update
    the material state. See the documentation for `eval` for more details.

    For material models that require state dependent variable tracking, the
    `num_sdv` member must be set to the number of state dependent variables
    required. Optionally, the `sdv_names` member can also be set to a list of
    state dependent variable names (for output purposes). State dependent
    variables are initialized to 0. The method `sdvini` can optionally be
    defined that returns alternative values for state dependent variables. See
    the documentation for `sdvini` for more information.

    """
    name = None
    num_sdv = None
    sdv_names = None
    def sdvini(self, statev):
        """Initialize the state dependent variables

        Parameters
        ----------
        statev : ndarray or None
            If `self.num_sdv is None` than `statev` is also `None`, otherwise
            it an array of zeros `self.num_sdv` in length

        Returns
        -------
        statev : ndarray or None
            The initialized state dependent variables.

        Notes
        -----
        This base method does not need to be overwritten if a material does not
        have any state dependent variables, or their initial values should be
        zero.

        """
        return statev

    def eval(self, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, **kwds):
        """Evaluate the material model

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

        Returns
        -------
        stress : ndarray
            Stress at the end of the step
        statev : ndarray
            State variables at the end of the step
        ddsdde : ndarray
            Elastic stiffness (Jacobian) of the material

        Notes
        -----
        Each material model is responsible for returning the elastic stiffness.
        If an analytic elastic stiffness is not known, return `None` and it
        will be computed numerically.

        The input arrays `stress` and `statev` are mutable and copies are not
        passed in. DO NOT MODIFY THEM IN PLACE. Doing so can cause problems
        down stream.

        """
        raise NotImplementedError

    def numerical_jacobian(self, time, dtime, temp, dtemp, F0, F, stran, d,
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
            Fp, Ep = update_deformation(dtime, 0., F, Dp)
            sigp = stress.copy()
            xp = copy(statev)
            sigp = self.eval(time, dtime, temp, dtemp,
                             F0, Fp, Ep, Dp, sigp, xp)[0]

            # perturb backward
            Dm = d.copy()
            Dm[v[i]] = d[v[i]] - (deps / dtime) / 2.
            Fm, Em = update_deformation(dtime, 0., F, Dm)
            sigm = stress.copy()
            xm = copy(statev)
            sigm = self.eval(time, dtime, temp, dtemp,
                             F0, Fm, Em, Dm, sigm, xm)[0]

            # compute component of jacobian
            Jsub[i, :] = (sigp[v] - sigm[v]) / deps

            continue

        return Jsub
