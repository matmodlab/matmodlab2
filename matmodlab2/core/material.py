from collections import OrderedDict

from .misc import add_metaclass


class BaseMaterial(type):
    def __call__(cls, *args, **kwargs):
        """Called before __init__ method is called"""

        # Call the objects __init__ method (indirectly through __call__)
        obj = type.__call__(cls, *args, **kwargs)

        obj.aux_models = OrderedDict()

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
    def num_aux_sdv(self):
        return sum([x.num_sdv for x in self.aux_models.values()])

    def get_aux_model_sdv_slice(self, aux_model):
        num_sdv = getattr(self, "num_sdv", None)
        start = 0 if num_sdv is None else num_sdv
        for (name, model) in self.aux_models.items():
            if model == aux_model:
                break
            start += model.num_sdv
        else:
            raise ValueError("No such aux model: {0!r}".format(aux_model))
        end = start + aux_model.num_sdv
        return slice(start, end)

    def base_eval(
        self,
        kappa,
        time,
        dtime,
        temp,
        dtemp,
        F0,
        F,
        strain,
        d,
        stress,
        ufield,
        dufield,
        statev,
        initial_temp,
        **kwds
    ):
        """Wrapper method to material.eval. This is called by Matmodlab so that
        addon models can first be evaluated. See documentation for eval.

        """
        from matmodlab2.materials.expansion import ExpansionModel
        from matmodlab2.materials.viscoelastic import ViscoelasticModel
        from matmodlab2.materials.effective_stress import EffectiveStressModel

        num_sdv = getattr(self, "num_sdv", None)

        if ExpansionModel.name in self.aux_models:
            # Evaluate thermal expansion
            aux_model = self.aux_models[ExpansionModel.name]

            # Determine starting point in statev array
            x_slice = self.get_aux_model_sdv_slice(aux_model)

            aux_model.eval(
                kappa,
                time,
                dtime,
                temp,
                dtemp,
                F0,
                F,
                strain,
                d,
                stress,
                statev[x_slice],
                initial_temp=initial_temp,
                ufield=ufield,
                dufield=dufield,
                **kwds
            )

        if EffectiveStressModel.name in self.aux_models:
            # Evaluate effective stress model
            aux_model = self.aux_models[EffectiveStressModel.name]

            # Determine starting point in statev array
            x_slice = self.get_aux_model_sdv_slice(aux_model)

            aux_model.eval(
                kappa,
                time,
                dtime,
                temp,
                dtemp,
                F0,
                F,
                strain,
                d,
                stress,
                statev[x_slice],
                initial_temp=initial_temp,
                ufield=ufield,
                dufield=dufield,
                **kwds
            )

        # Evaluate the material model
        xv = None if num_sdv is None else statev[:num_sdv]
        sig, xv, ddsdde = self.eval(
            time,
            dtime,
            temp,
            dtemp,
            F0,
            F,
            strain,
            d,
            stress,
            xv,
            ufield=ufield,
            dufield=dufield,
            **kwds
        )

        if xv is not None:
            statev[:num_sdv] = xv

        if ViscoelasticModel.name in self.aux_models:
            # Evaluate the viscoelastic overstress model
            aux_model = self.aux_models[ViscoelasticModel.name]

            # Determine starting point in statev array
            x_slice = self.get_aux_model_sdv_slice(aux_model)

            aux_model.eval(
                kappa,
                time,
                dtime,
                temp,
                dtemp,
                F0,
                F,
                strain,
                d,
                stress,
                statev[x_slice],
                initial_temp=initial_temp,
                ufield=ufield,
                dufield=dufield,
                **kwds
            )

            # Force the use of a numerical stiffness - otherwise we would have
            # to convert the stiffness to that corresponding to the Truesdell
            # rate, pull it back to the reference frame, apply the visco
            # correction, push it forward, and convert to Jaummann rate. It's
            # not as trivial as it sounds...
            ddsdde = None

        if EffectiveStressModel.name in self.aux_models:
            # Add pore pressure back
            aux_model = self.aux_models[EffectiveStressModel.name]

            # Determine starting point in statev array
            x_slice = self.get_aux_model_sdv_slice(aux_model)

            aux_model.posteval(
                kappa,
                time,
                dtime,
                temp,
                dtemp,
                F0,
                F,
                strain,
                d,
                stress,
                statev[x_slice],
                initial_temp=initial_temp,
                ufield=ufield,
                dufield=dufield,
                **kwds
            )

        return sig, statev, ddsdde

    def eval(self, time, dtime, temp, dtemp, F0, F, strain, d, stress, statev, **kwds):
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
            raise ValueError(
                "Expansion model must be created before assigning "
                "material model to MaterialPointSimulator"
            )
        from matmodlab2.materials.expansion import ExpansionModel

        self.aux_models[ExpansionModel.name] = ExpansionModel(alpha)

    def Viscoelastic(self, wlf, prony):
        if self.assigned:
            raise ValueError(
                "Viscoelastic model must be created before assigning "
                "material model to MaterialPointSimulator"
            )
        from matmodlab2.materials.viscoelastic import ViscoelasticModel

        self.aux_models[ViscoelasticModel.name] = ViscoelasticModel(wlf, prony)

    def EffectiveStress(self, porepres):
        if self.assigned:
            raise ValueError(
                "EffectiveStress model must be created before assigning "
                "material model to MaterialPointSimulator"
            )
        from matmodlab2.materials.effective_stress import EffectiveStressModel

        self.aux_models[EffectiveStressModel.name] = EffectiveStressModel(porepres)
