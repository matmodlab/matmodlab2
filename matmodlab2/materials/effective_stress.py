import numpy as np

from .addon import AddonModel


class EffectiveStressModel(AddonModel):
    """Effective stress model"""
    name = '__effstress__'
    def __init__(self, porepres):
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

