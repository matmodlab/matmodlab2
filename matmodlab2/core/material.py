import numpy as np
from copy import deepcopy as copy

from .misc import add_metaclass

class BaseMaterial(type):
    def __call__(cls, *args, **kwargs):
        """Called before __init__ method is called"""

        # Call the objects __init__ method (indirectly through __call__)
        obj = type.__call__(cls, *args, **kwargs)

        # Store the addon models
        obj.addon_models = []

        x = kwargs.pop('addon_model', None)
        if x is not None:
            obj.addon_models.append(x)

        x = kwargs.pop('addon_models', None)
        if x is not None:
            obj.addon_models.extend(x)

        return obj

@add_metaclass(BaseMaterial)
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
    _has_addon_models = None
    assigned = False

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

    @property
    def has_addon_models(self):
        if self._has_addon_models is None:
            self._has_addon_models = hasattr(self, 'addon_models')
        return self._has_addon_models

    def base_eval(self, kappa, time, dtime, temp, dtemp,
                  F0, F, strain, d, stress, ufield, dufield, statev,
                  initial_temp, **kwds):
        """Wrapper method to material.eval. This is called by Matmodlab so that
        addon models can first be evaluated. See documentation for eval.

        """
        num_sdv = getattr(self, 'num_sdv', None)
        i = 0 if num_sdv is None else num_sdv

        if hasattr(self, 'addon_models'):
            for model in self.addon_models:
                # Evaluate each addon model.  Each model must change the input
                # arrays in place.

                # Determine starting point in statev array
                j = i + model.num_sdv
                xv = statev[i:j]
                model.eval(kappa, time, dtime, temp, dtemp,
                           F0, F, strain, d, stress, xv,
                           initial_temp=initial_temp,
                           ufield=ufield, dufield=dufield, **kwds)
                statev[i:j] = xv
                i += j

        if num_sdv is not None:
            xv = statev[:num_sdv]
        else:
            xv = None
        sig, xv, ddsdde = self.eval(time, dtime, temp, dtemp, F0, F, strain, d,
                                    stress, xv, ufield=ufield, dufield=dufield,
                                    **kwds)

        if hasattr(self, 'addon_models'):
            for model in self.addon_models:
                # Determine starting point in statev array
                j = i + model.num_sdv
                xv = statev[i:j]
                model.posteval(kappa, time, dtime, temp, dtemp,
                               F0, F, strain, d, sig, xv,
                               initial_temp=initial_temp,
                               ufield=ufield, dufield=dufield, **kwds)
                statev[i:j] = xv
                i += j

        if num_sdv is not None:
            statev[:num_sdv] = xv

        return sig, statev, ddsdde

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

    def Expansion(self, alpha):
        if self.assigned:
            raise ValueError('Expansion model must be created before assigning '
                             'material model to MaterialPointSimulator')
        from matmodlab2.materials.expansion import ExpansionModel
        self.addon_models.append(ExpansionModel(alpha))

    def Viscoelastic(self, wlf, prony):
        if self.assigned:
            raise ValueError('Viscoelastic model must be created before assigning '
                             'material model to MaterialPointSimulator')
        from matmodlab2.materials.viscoelastic import ViscoelasticModel
        self.addon_models.append(ViscoelasticModel(wlf, prony))

    def EffectiveStress(self, porepres):
        if self.assigned:
            raise ValueError('EffectiveStress model must be created before assigning '
                             'material model to MaterialPointSimulator')
        from matmodlab2.materials.effective_stress import EffectiveStressModel
        self.addon_models.append(EffectiveStressModel(porepres))
