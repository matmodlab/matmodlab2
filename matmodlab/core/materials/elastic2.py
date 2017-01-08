from numpy import dot, zeros, ix_, eye, triu
from numpy.linalg import cholesky, LinAlgError
from matmodlab.core.logio import logger
from matmodlab.core.material import Material

class AnisotropicElasticMaterial(Material):
    name = "elastic2"

    def __init__(self, **parameters):
        """Set up the Elastic material"""
        # Define mapping from 21 elastic parameters to location
        # in parameter array
        param_names = ['C{0}{1}'.format(i+1,j+1)
                       for i in range(6) for j in range(i,6)]

        # Create the parameter array and check inputs
        self.params = zeros(len(param_names))
        for (i, name) in enumerate(param_names):
            self.params[i] = parameters.pop(name, 0.)
        if parameters:
            unused = ', '.join(list(parameters.keys()))
            logger.warning('The following parameters were '
                           'unused {0}'.format(unused))

        # check if C is positive definite
        C = self.form_stiff(self.params)
        try:
            cholesky(C)
        except LinAlgError:
            raise ValueError('elastic stiffness not positive definite')

    @staticmethod
    def form_stiff(cij):
        C = zeros((6,6))
        i = 0
        for k in range(6):
            j = 6 - k
            C[k, k:] = cij[i:i+j]
            i += j
        return C + triu(C,k=1).T

    def eval(self, time, dtime, temp, dtemp, F0, F,
             stran, d, stress, statev, **kwargs):
        """Compute updated stress given strain increment"""

        # elastic stiffness
        ddsdde = self.form_stiff(self.params)

        # stress update
        stress = dot(ddsdde, stran)

        return stress, statev, ddsdde
