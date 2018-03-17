class AddonModel(object):
    def sdvini(self, statev):
        return statev
    def eval(self, kappa, time, dtime, temp, dtemp, F0, F, strain, d,
             stress, statev, initial_temp=0., **kwds):
        # All arrays must be modified in place
        return None
    def posteval(self, kappa, time, dtime, temp, dtemp, F0, F, strain, d,
                 stress, statev, initial_temp=0., **kwds):
        # All arrays must be modified in place
        return None
