from numpy import dot, zeros, trace, array, sum, eye

from ..core.logio import logger
from ..core.material import Material
from ..core.tensor import array_rep, det, I6

class NeoHookeMaterial(Material):
    name = "neo-hooke"

    def __init__(self, **parameters):
        """Set up the Neo Hooke material """

        E = parameters.pop('E', 0.)
        assert E > 0
        Nu = parameters.pop('Nu', 0.)

        C10 = E / (4. * (1. + Nu))
        D1 = 6. * (1. - 2. * Nu) / E
        self.params = {'C10': C10, 'D1': D1}

        errors = 0
        if D1 <= 0.:
            errors += 1
            logger.error('D1 must be > 0')

        G = 2. * C10
        if G <= 0:
            errors += 1
            logger.error('2 C10  > 0')

        if errors:
            raise ValueError("stopping due to previous errors")

    def eval(self, time, dtime, temp, dtemp, F0, F,
             stran, d, stress, statev, **kwargs):
        """Compute updated stress given the updated deformation"""

        # elastic properties
        C10 = self.params['C10']
        D1 = self.params['D1']

        # elastic stiffness
        ddsdde = zeros((6,6))

        # Reshape the deformation gradient
        F = F.reshape(3,3)
        Jac = det(F)

        scale = Jac ** (-1. / 3.)
        fb = scale * F

        # deviatoric left Cauchy-Green deformation tensor
        bb = zeros(6)
        bb[0] = fb[0,0] * fb[0,0] + fb[0,1] * fb[0,1] + fb[0,2] * fb[0,2]
        bb[1] = fb[1,0] * fb[1,0] + fb[1,1] * fb[1,1] + fb[1,2] * fb[1,2]
        bb[2] = fb[2,0] * fb[2,0] + fb[2,1] * fb[2,1] + fb[2,2] * fb[2,2]
        bb[3] = fb[1,0] * fb[0,0] + fb[1,1] * fb[0,1] + fb[1,2] * fb[0,2]
        bb[4] = fb[2,0] * fb[1,0] + fb[2,1] * fb[1,1] + fb[2,2] * fb[1,2]
        bb[5] = fb[2,0] * fb[0,0] + fb[2,1] * fb[0,1] + fb[2,2] * fb[0,2]

        trbbar = sum(bb[:3]) / 3.
        eg = 2. * C10 / Jac
        ek = 2. / D1 * (2. * Jac - 1.)
        pr = 2. / D1 * (Jac - 1.)

        # cauchy stress
        stress = eg * (bb - trbbar * I6) + pr * I6

        # spatial stiffness
        eg23 = eg * 2. / 3.
        ddsdde[0,0] =  eg23 * (bb[0] + trbbar) + ek
        ddsdde[0,1] = -eg23 * (bb[0] + bb[1]-trbbar) + ek
        ddsdde[0,2] = -eg23 * (bb[0] + bb[2]-trbbar) + ek
        ddsdde[0,3] =  eg23 * bb[3] / 2.
        ddsdde[0,4] = -eg23 * bb[4]
        ddsdde[0,5] =  eg23 * bb[5] / 2.

        ddsdde[1,1] =  eg23 * (bb[1] + trbbar) + ek
        ddsdde[1,2] = -eg23 * (bb[1] + bb[2]-trbbar) + ek
        ddsdde[1,3] =  eg23 * bb[3] / 2.
        ddsdde[1,4] =  eg23 * bb[4] / 2.
        ddsdde[1,5] = -eg23 * bb[5]

        ddsdde[2,2] =  eg23 * (bb[2] + trbbar) + ek
        ddsdde[2,3] = -eg23 * bb[3]
        ddsdde[2,4] =  eg23 * bb[4] / 2.
        ddsdde[2,5] =  eg23 * bb[5] / 2.

        ddsdde[3,3] =  eg * (bb[0] + bb[1]) / 2.
        ddsdde[3,4] =  eg * bb[5] / 2.
        ddsdde[3,5] =  eg * bb[4] / 2.

        ddsdde[4,4] =  eg * (bb[0] + bb[2]) / 2.
        ddsdde[4,5] =  eg * bb[3] / 2.

        ddsdde[5,5] =  eg * (bb[1] + bb[2]) / 2.

        for i in range(6):
            for j in range(i+1, 6):
                ddsdde[i,j] = ddsdde[j,i]

        return stress, statev, ddsdde
