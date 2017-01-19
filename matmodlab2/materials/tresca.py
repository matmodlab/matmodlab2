"""This model is courtesy of Scot Swan.

The model and plots from its verification tests support the publication:

<CITATION>

All attribution goes to Scot Swan for this model. Any mistakes should be
assumed to be errors on my part (Tim Fuller).

"""
import numpy as np
from ..core.logio import logger
from ..core.material import Material
from ..core.tensor import isotropic_part, deviatoric_part, invariants, dot, \
    magnitude, double_dot
epsilon = np.finfo(float).eps

class TrescaMaterial(Material):
    """Implements linear pressure dependent Tresca plasticity

    Parameters
    ----------
    **kwds : dict
        Material parameters.  Recognized parameters are the bulk
        modules `K`, shear modules `G`, yield strength in shear `A1`,
        and pressure dependence term `A4`.

    """
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

        if abs(A1) <= 1.E-12:
            A1 = 1.0e99

        # Check the input parameters
        errors = 0
        if K <= 0.0:
            errors += 1
            logger.error('Bulk modulus K must be positive')
        if G <= 0.0:
            errors += 1
            logger.error('Shear modulus G must be positive')
        nu = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G)
        if nu > 0.5:
            errors += 1
            logger.error('Poisson\'s ratio > .5')
        if nu < -1.0:
            errors += 1
            logger.error('Poisson\'s ratio < -1.')
        if nu < 0.0:
            logger.warn('negative Poisson\'s ratio')
        if A1 <= 0.0:
            errors += 1
            logger.error('A1 must be positive nonzero')
        if errors:
            raise ValueError('stopping due to previous errors')

        # Save the new parameters
        self.params = {'K': K, 'G': G, 'A1': A1}

    def yield_function(self, stress):
        """ Return the value of the yield function at a given stress. """
        _, r, theta, _ = invariants(stress, 'lode')
        return r / np.sqrt(2.) * np.cos(theta) - self.params['A1']

    def yield_function_gradient(self, stress):
        """ Return the value of the yield function at a given stress. """
        S = deviatoric_part(stress)
        d = invariants(stress, 'lode&mechanics')
        r, theta, lode, j3 = d['r'], d['theta'], d['lode'], d['j3']
        term1 = np.cos(theta) / (np.sqrt(2.) * r) * S

        if abs(theta) < epsilon:
            return term1

        # Trim the lode parameter to alleviate problems at verticies
        lode = max(-0.9999, min(0.9999, lode))

        S2 = deviatoric_part(dot(S, S))
        coef2 = r/np.sqrt(2.)*np.sin(theta)/(3.0*np.sqrt(1.0 - lode**2))
        deriv = lode * (S2 / j3 - 3. / (r ** 2) * S)
        return term1 - coef2 * deriv

    def radial_return(self, stress):
        """ A !WRONG!, but still useful, radial return. """
        ys = self.params['A1']
        S = deviatoric_part(stress)
        d = invariants(stress, 'lode&mechanics')
        rootj2, theta = d['rootj2'], d['theta']
        return (self.params['A1'] / np.cos(theta)) * S / rootj2

    def simple_return(self, stress):
        """ The two-stage return from Brannon (2009). """
        S = deviatoric_part(stress)
        d = invariants(S, 'lode&mechanics')
        r, theta = d['r'], d['theta']

        if r * np.sin(abs(theta)) > np.sqrt(2./3.) * self.params['A1']:
            # Vertex
            r_scaled = np.sqrt(2./3.) * self.params['A1'] / np.sin(abs(theta))
            S = r_scaled * S / magnitude(S)

        sigf = self.radial_return(S)
        dfdsig = self.yield_function_gradient(sigf)
        matn = dfdsig / magnitude(dfdsig)
        sigp = stress - matn * double_dot(stress - sigf, matn)
        return sigp

    def eval(self, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, **kwds):
        """ Update the stress state from the strain increment 'd'. """

        P = isotropic_part(stress)
        S = deviatoric_part(stress)
        deps = d * dtime

        # Calculate the trial stress
        P += 3. * self.params['K'] * isotropic_part(deps)
        S += 2. * self.params['G'] * deviatoric_part(deps)

        stress = P + S
        f0 = self.yield_function(stress)
        if f0 > 0.:
            # Plastic, perform the return
            stress = self.simple_return(stress)

        return stress, statev, None
