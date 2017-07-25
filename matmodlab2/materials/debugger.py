import numpy as np

from ..core.logio import logger
from ..core.material import Material

class DebuggerMaterial(Material):
    """A dummy material that saves all standard inputs as state variables

    Parameters
    ----------

    *** takes no paramters ***

    """
    name = 'debugger'
    def __init__(self, **parameters):

        def symm_names(base):
            return [base + "." + _ for _ in "XX YY ZZ XY XZ YZ".split()]

        def full_names(base):
            return [base + "." + _ for _ in "XX XY XZ YX YY YZ ZX ZY ZZ".split()]

        self.sdv_names = (
                          ["TIME", "DTIME", "TEMP", "DTEMP"] +
                          full_names("F0") +
                          full_names("F1") +
                          symm_names("STRAIN") +
                          symm_names("D") +
                          symm_names("STRESS")
                         )
        self.sdv_names = ["SDV_" + _ for _ in self.sdv_names]
        self.num_sdv = len(self.sdv_names)


    def eval(self, time, dtime, temp, dtemp,
             F0, F1, strain, d, stress, statev, **kwds):

        def stick_it(what, where, expected_length):
            if len(what) != expected_length:
                raise Exception("Unexpected length: {0} {1}".format(len(what), expected_length))
            statev[where:where+len(what)] = what

        statev[self.sdv_names.index("SDV_TIME")] = time
        statev[self.sdv_names.index("SDV_DTIME")] = dtime
        statev[self.sdv_names.index("SDV_TEMP")] = temp
        statev[self.sdv_names.index("SDV_DTEMP")] = dtemp
        statev[self.sdv_names.index("SDV_DTEMP")] = dtemp
        stick_it(F0.flatten(), self.sdv_names.index("SDV_F0.XX"), 9)
        stick_it(F1.flatten(), self.sdv_names.index("SDV_F1.XX"), 9)
        stick_it(strain, self.sdv_names.index("SDV_STRAIN.XX"), 6)
        stick_it(d, self.sdv_names.index("SDV_D.XX"), 6)
        stick_it(stress, self.sdv_names.index("SDV_STRESS.XX"), 6)

        return stress, statev, None
