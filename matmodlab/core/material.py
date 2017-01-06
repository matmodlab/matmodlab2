import logging
import numpy as np
from copy import deepcopy as copy
from .mmlabpack import update_deformation, VOIGT, dot, I6

class Material(object):
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

class ElasticMaterial(Material):
    name = 'pyelastic'
    def __init__(self, **parameters):
        E = parameters['E']
        assert E > 0.
        Nu = parameters['Nu']
        assert -1. < Nu < .5
        self.G = E / 2. / (1. + Nu)
        self.K = E / 3. / (1. - 2. * Nu)

    def eval(self, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, **kwds):
        K3 = 3. * self.K
        G2 = 2. * self.G
        Lam = (K3 - G2) / 3.
        # elastic stiffness
        ddsdde = np.zeros((6,6))
        ddsdde[np.ix_(range(3), range(3))] = Lam
        ddsdde[range(3),range(3)] += G2
        ddsdde[range(3,6),range(3,6)] = self.G
        # stress update
        stress = stress + dot(ddsdde, d * dtime)
        return stress, statev, ddsdde

class PlasticMaterial(Material):
    name = 'pyplastic'
    def __init__(self, **parameters):

        # Linear elastic bulk modulus
        K = parameters.get('K', 0.)

        # Linear elastic shear modulus
        G = parameters.get('G', 0.)

        # A1: Intersection of the yield surface with the sqrt(J2) axis (pure
        # shear).
        # sqrt(J2) = r / sqrt(2); r = sqrt(2*J2)
        # sqrt(J2) = q / sqrt(3); q = sqrt(3*J2)
        A1 = parameters.get('A1', 0.)

        # Pressure dependence term.
        #   A4 = -d(sqrt(J2)) / d(I1)
        # always positive
        A4 = parameters.get('A4', 0.)

        if abs(A1) <= 1.E-12:
            A1 = 1.0e99

        # Check the input parameters
        errors = 0
        if K <= 0.0:
            errors += 1
            logging.error('Bulk modulus K must be positive')
        if G <= 0.0:
            errors += 1
            logging.error('Shear modulus G must be positive')
        nu = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G)
        if nu > 0.5:
            errors += 1
            logging.error('Poisson\'s ratio > .5')
        if nu < -1.0:
            errors += 1
            logging.error('Poisson\'s ratio < -1.')
        if nu < 0.0:
            logging.warn('#negative Poisson\'s ratio')
        if A1 <= 0.0:
            errors += 1
            logging.error('A1 must be positive nonzero')
        if A4 < 0.0:
            errors += 1
            logging.error('A4 must be non-negative')
        if errors:
            raise ValueError('stopping due to previous errors')

        # Save the new parameters
        self.params = {'K': K, 'G': G, 'A1': A1, 'A4': A4}

        # Register State Variables
        self.sdv_names = ['EP_XX', 'EP_YY', 'EP_ZZ', 'EP_XY', 'EP_XZ', 'EP_YZ',
                          'I1', 'ROOTJ2', 'YROOTJ2', 'ISPLASTIC']
        self.num_sdv = len(self.sdv_names)

    def eval(self, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, **kwds):
        """Compute updated stress given strain increment """

        sigsave = np.copy(stress)

        # Define helper functions and unload params/state vars
        A1 = self.params['A1']
        A4 = self.params['A4']
        idx = lambda x: self.sdv_names.index(x.upper())
        ep = statev[idx('EP_XX'):idx('EP_YZ')+1]

        # Compute the trial stress and invariants
        stress = stress + self.dot_with_elastic_stiffness(d / VOIGT * dtime)
        i1 = self.i1(stress)
        rootj2 = self.rootj2(stress)
        if rootj2 - (A1 - A4 * i1) <= 0.0:
            statev[idx('ISPLASTIC')] = 0.0
        else:
            statev[idx('ISPLASTIC')] = 1.0

            s = self.dev(stress)
            N = np.sqrt(2.) * A4 * I6 + s / self.tensor_mag(s)
            N = N / np.sqrt(6.0 * A4 ** 2 + 1.0)
            P = self.dot_with_elastic_stiffness(N)

            # 1) Check if linear drucker-prager
            # 2) Check if trial stress is beyond the vertex
            # 3) Check if trial stress is in the vertex
            if (A4 != 0.0 and
                    i1 > A1 / A4 and
                    rootj2 / (i1 - A1 / A4) < self.rootj2(P) / self.i1(P)):
                dstress = stress - A1 / A4 / 3.0 * I6
                # convert all of the extra strain into plastic strain
                ep += self.iso(dstress) / (3.0 * self.params['K'])
                ep += self.dev(dstress) / (2.0 * self.params['G'])
                stress = A1 / A4 / 3.0 * I6
            else:
                # not in vertex; do regular return
                lamb = ((rootj2 - A1 + A4 * i1) / (A4 * self.i1(P)
                        + self.rootj2(P)))
                stress = stress - lamb * P
                ep += lamb * N

            # Save the updated plastic strain
            statev[idx('EP_XX'):idx('EP_YZ')+1] = ep

        statev[idx('I1')] = self.i1(stress)
        statev[idx('ROOTJ2')] = self.rootj2(stress)
        statev[idx('YROOTJ2')] = A1 - A4 * self.i1(stress)

        return stress, statev, None

    def dot_with_elastic_stiffness(self, A):
        return (3.0 * self.params['K'] * self.iso(A) +
                2.0 * self.params['G'] * self.dev(A))

    def tensor_mag(self, A):
        return np.sqrt(np.dot(A[:3], A[:3]) + 2.0 * np.dot(A[3:], A[3:]))

    def iso(self, sig):
        return sig[:3].sum() / 3.0 * I6

    def dev(self, sig):
        return sig - self.iso(sig)

    def rootj2(self, sig):
        s = self.dev(sig)
        return self.tensor_mag(s) / np.sqrt(2.)

    def i1(self, sig):
        return np.sum(sig[:3])
