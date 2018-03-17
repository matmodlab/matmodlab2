import numpy as np

from ..core.misc import is_scalarlike, is_listlike
from ..core.deformation import defgrad_from_strain
from ..core.database import COMPONENT_SEP
from ..core.tensor import SYMMETRIC_COMPONENTS, TENSOR_COMPONENTS
from .addon import AddonModel


class ExpansionModel(AddonModel):
    """Thermal expansion model"""
    name = '__expansion__'
    def __init__(self, expansion):
        """Format the thermal expansion term"""
        self.num_sdv = 15
        self.sdv_names = ['EM'+COMPONENT_SEP+x for x in SYMMETRIC_COMPONENTS]
        self.sdv_names.extend(['FM'+COMPONENT_SEP+x for x in TENSOR_COMPONENTS])
        if is_scalarlike(expansion):
            expansion = [expansion] * 3
        if not is_listlike(expansion):
            raise ValueError('Expected expansion to be array_like')
        if len(expansion) == 3:
            expansion = [x for x in expansion] + [0, 0, 0]
        if len(expansion) != 6:
            raise ValueError('Expected len(expansion) to be 3 or 6')
        self.data = np.array([float(x) for x in expansion])

    def sdvini(self, statev):
        statev = np.append(np.zeros(6), np.array([1.,0.,0.,0.,1.,0.,0.,0.,1.]))
        return statev

    def eval(self, kappa, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, initial_temp=0., **kwds):
        """Evaluate the thermal expansion model

        F0, F, strain, d are updated in place
        """
        assert len(statev) == 15

        # Determine mechanical strain
        thermal_strain = (temp + dtemp - initial_temp) * self.data
        strain -= thermal_strain

        # Updated deformation gradient
        F0[:9] = np.array(statev[6:15])
        F[:9] = defgrad_from_strain(strain, kappa, flatten=1)

        thermal_d = self.data * dtemp / dtime
        d -= thermal_d

        # Save the mechanical state to the statev
        statev[:self.num_sdv] = np.append(strain, F)

        return None

class EffectiveStressModel(AddonModel):
    """Effective stress model"""
    def __init__(self, porepres):
        """Format the thermal expansion term"""
        self.num_sdv = 1
        self.sdv_names = ['POREPRES']
        self.porepres = np.asarray(porepres)
        if len(self.porepres.shape) != 2:
            raise ValueError('pore_pres must be a 2 dimensional array with '
                             'the first column being time and the second the '
                             'associated pore pressure.')
    def get_porepres_at_time(self, time):
        return np.interp(1, self.porepres[0,:], self.porepres[1,:],
                         left=self.porepres[1,0], right=self.porepres[1,-1])
    def sdvini(self, statev):
        statev = np.array([self.get_porepres_at_time(0.)])
        return statev

    def eval(self, kappa, time, dtime, temp, dtemp,
             F0, F, strain, d, stress, statev, **kwds):
        """Evaluate the effective stress model

        """
        porepres = self.get_porepres_at_time(time+dtime/2.)
        stress[[0,1,2]] -= porepres
        statev[0] = porepres
        return None

    def posteval(self, kappa, time, dtime, temp, dtemp, F0, F, strain, d,
                 stress, statev, **kwds):
        porepres = self.get_porepres_at_time(time+dtime/2.)
        stress[[0,1,2]] += porepres
        statev[0] = porepres
        return None

