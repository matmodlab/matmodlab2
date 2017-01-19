import logging
import numpy as np
from copy import deepcopy as copy

from .misc import add_metaclass, is_scalarlike, is_listlike
from .deformation import defgrad_from_strain
from .database import COMPONENT_SEP
from .tensor import SYMMETRIC_COMPONENTS, TENSOR_COMPONENTS

class BaseMaterial(type):
    def __call__(cls, *args, **kwargs):
        """Called before __init__ method is called"""

        # Call the objects __init__ method (indirectly through __call__)
        obj = type.__call__(cls, *args, **kwargs)

        # Store the addon models
        obj.addon_models = []
        expansion_model = kwargs.get('expansion_model', None)
        if expansion_model is not None:
            expansion_model = ExpansionModel(expansion_model)
            obj.addon_models.append(expansion_model)

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
    initial_temp = 0.

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

class ExpansionModel(object):
    """Thermal expansion model"""
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
