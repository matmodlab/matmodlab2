from __future__ import print_function
import os
import time
import numpy as np
from copy import deepcopy as copy
from collections import OrderedDict

from .tensor import VOIGT, I9, inv, dot  # noqa
from .environ import environ
from .tensor import array_rep, unrotate, rotate  # noqa
from .material import Material
from .misc import is_stringlike, is_listlike
from .logio import logger, add_filehandler, splash
from .database import DatabaseFile, COMPONENT_SEP, groupby_names
from .stress_control import d_from_prescribed_stress, numerical_jacobian

import matmodlab2.core.linalg as la
import matmodlab2.core.deformation as dfm

continued = {"continued": 1}

__all__ = ["MaterialPointSimulator"]


class MaterialPointSimulator(object):
    """The material point simulator

    The material point simulator exercises a material model just as a finite
    element solver would.

    Parameters
    ----------
    jobid : str
        A job identifier
    initiali_temp : float [0.]
        The (optional) initial temperature
    db_fmt : str ['npz'], {'npz', 'exo'}
        The output database format.  npz is the numpy compressed storage format
        and exo is the ExodusII format
    logfile : bool [False]
        Whether to write a log file
    write_db : bool [True]
        Whether to write the database file
    ufield : ndarray [None]
        The initial user defined field.

    Notes
    -----
    The steps to creating and running a simulation with the
    `MaterialPointSimulator` are:

    1. Instantiate a `MaterialPointSimulator` simulator object, giving it a
       string `jobid`
    2. Assign a material model to the simulator
    3. Add simulation (deformation) steps to the simulator
       The steps are run as they are added.

    Output is written to either a compressed numpy file or an ExodusII database
    file. ExodusII database files can be viewed using the open source
    `tsviewer` or `ParaView`.

    Examples
    --------

    In the following simulation, the `ElasticMaterial` is exercised through a
    step of uniaxial strain and then a stress controlled step to bring the
    material to a state of zero stress.

    >>> jobid = 'Job-1'
    >>> mps = MaterialPointSimulator(jobid)
    >>> material = ElasticMaterial(E=10, Nu=.1)
    >>> mps.assign_material(material)
    >>> mps.run_step('EEEEEE', [1., 0., 0., 0., 0., 0.])
    >>> mps.run_step('SSSEEE', [0., 0., 0., 0., 0., 0.])

    """

    valid_descriptors = ["DE", "E", "S", "DS", "U", "F"]

    def __init__(
        self,
        jobid,
        initial_temp=0.0,
        db_fmt="npz",
        logfile=False,
        write_db=True,
        ufield=None,
    ):

        logger.info("Initializing the simulation")
        self.jobid = jobid
        self.ran = False

        # File I/O
        if logfile:
            add_filehandler(logger, self.jobid + ".log")
        splash(logger)
        if db_fmt not in ("npz", "exo"):
            raise ValueError("db_fmt must by npz or exo")
        self.db_fmt = db_fmt

        logger.info("Matmodlab simulation: {0!r}".format(self.jobid))

        # Create initial step
        self.initial_temp = initial_temp
        logger.info("Creating initial step... ", extra=continued)
        self.steps = self._initialize_steps(self.initial_temp, ufield)
        logger.info("done")

        # Set defaults
        self._df = None
        self._columns = None
        self._material = None
        self._elem_var_names = None
        self.db = None
        self.data = None
        self.num_ufield = len(self.steps[0].ufield)

        # Following attributes are only applicable if using add_step/run
        self._steps = []
        self.write_db = write_db

        self._initialized = False

        logger.info("Done initializing the simulation")

    def _initialize_steps(self, temp, ufield):
        """Create the initial step"""
        # begin, end = 0.0, 0.0
        components = np.zeros(6)
        descriptors = ["E"] * 6
        return [Step(0, 0, 1, descriptors, components, temp, 0, ufield)]

    def _format_descriptors_and_components(self, descriptors, components):
        """Validate the user given descriptors"""

        # Make sure components is an array. Whatever the length of the
        # components is the length of the final descriptors
        if not is_listlike(components):
            components = [components]

        # Specify 'float64' for type consistency (important for some functions).
        components = np.array(components, dtype=np.float64)

        if is_stringlike(descriptors):
            if len(descriptors) == 1:
                # Lazy typing...
                descriptors = descriptors * len(components)
        elif not is_listlike(descriptors):
            raise TypeError("descriptors must be list_like or string_like")
        descriptors = list(descriptors)

        if len(descriptors) != len(components):
            raise ValueError(
                "components and descriptors must have same number" "of entries"
            )

        for (i, descriptor) in enumerate(descriptors):
            if descriptor.upper() not in self.valid_descriptors:
                raise ValueError("Invalid descriptor {0!r}".format(descriptor))
            descriptors[i] = descriptor.upper()

        unique_descriptors = list(set(descriptors))
        if "F" in unique_descriptors:
            if len(unique_descriptors) != 1:
                raise ValueError("Cannot mix F with other descriptors")
            elif len(descriptors) != 9:
                raise ValueError("Must specify all 9 components of F")

        elif "U" in unique_descriptors:
            if len(unique_descriptors) != 1:
                raise ValueError("Cannot mix U with other descriptors")
            elif len(descriptors) != 3:
                raise ValueError("Must specify all 3 components of U")

        elif np.any(np.in1d(["E", "S", "DE", "DS"], descriptors)):
            if len(descriptors) > 6:
                raise ValueError(
                    "At most 6 components of stress/strain " "can be prescribed"
                )

        return descriptors, components

    def add_step(
        self,
        descriptors,
        components,
        increment=1.0,
        frames=1,
        scale=1.0,
        kappa=0.0,
        temperature=0.0,
        time_whole=None,
        ufield=None,
    ):
        self._steps.append(
            (
                descriptors,
                components,
                increment,
                frames,
                scale,
                kappa,
                temperature,
                time_whole,
                ufield,
            )
        )

    def run(self):
        """Run the simulation"""
        if not self._steps:
            logger.warning("No steps to run")
            return
        for step in self._steps:
            (
                descriptors,
                components,
                increment,
                frames,
                scale,
                kappa,
                temperature,
                time_whole,
                ufield,
            ) = step
            self.run_step(
                descriptors,
                components,
                increment=increment,
                frames=frames,
                scale=scale,
                kappa=kappa,
                temperature=temperature,
                time_whole=time_whole,
                ufield=ufield,
            )
        if self.write_db:
            self.dump()

    def run_step(
        self,
        descriptors,
        components,
        increment=1.0,
        frames=1,
        scale=1.0,
        kappa=0.0,
        temperature=0.0,
        time_whole=None,
        ufield=None,
    ):
        """Create a deformation step for the simulation

        Parameters
        ----------
        descriptors : string or listlike of string
            Descriptors for each component of deformation. Each `descriptor` in
            `descriptors` must be one of:
                - `E`: representing strain
                - `DE`: representing an increment in strain
                - `S`: representing stress
                - `DS`: representing an increment in stress
                - `F`: representing the deformation gradient
                - `U`: representing displacement
        components : listlike of floats
            The components of deformation. `components[i]` is interpreted as
            `descriptors[i]`. Thus, `len(components)` must equal
            `len(descriptors)`
        increment : float, optional
            The length of the step in time units, default is 1.
        frames : int, optional
            The number of discrete increments in the step, default is 1
        scale : float or listlike of float
            Scaling factor to be applied to components.  If scale
        kappa : float
            The Seth-Hill parameter of generalized strain.  Default is 0.
        temperature : float
            The temperature at the end of the step.  Default is 0.
        time_whole : float
            The whole time at the end of the step.  Default is `None`.
            If defined, the `increment` argument is ignored.
        ufield : ndarray [None]
            The value of the user defined field at the end of the step.  The
            ufield argument must have been also passed to the constructor.  The
            interpolated value at the beginning of an increment and the
            increment in ufield are passed to material models as the `ufield`
            and `dufield` keyword arguments, respectively.

        Tensor Component Ordering
        -------------------------
        Component ordering for components is:

        1. Symmetric tensors: XX, YY, ZZ, XY, YZ, XZ
        2. Unsymmetric tensors: XX, XY, XZ YX, YY, YZ ZX, ZY, ZZ
        3. Vectors: X, Y, Z

        Examples
        --------
        To create a step of uniaxial strain with magnitude .1:

        >>> obj.run_step('EEEEEE', [1., 0., 0., 0., 0., 0.], scale=.1)

        To create a step of uniaxial stress with magnitude 1e6:

        >>> obj.run_step('SSSEEE', [1., 0., 0., 0., 0., 0.], scale=1e6)

        Stress and strain (and their increments) can be mixed.  To create a step
        of uniaxial stress by holding the lateral stress components at 0. and
        deforming along the axial direction:

        >>> obj.run_step('ESSEEE', [1., 0., 0., 0., 0., 0.], scale=.1)

        To create a step of uniaxial strain, controlled by deformation gradient:

        >>> obj.run_step('FFFFFFFFF', [1.05, 0., 0., 1., 0., 0.], scale=1e6)

        Note, all 9 components of the deformation gradient must be prescribed.

        Special deformation cases are volumetric strain and pressure. Each is
        defined by prescribing one, and only one, component of either strain or
        stress, respectively:

        Volumetric strain.

        >>> obj.run_step('E', .1)

        Pressure:

        >>> obj.run_step('S', 1, scale=1e6)

        Notes
        -----
        Prescribed deformation gradient and displacement components are
        converted to strain. Internally, the driver works only with strain,
        stress, or their increments.

        For stress, strain (and their increments), or mixed steps, all
        components of deformation need not be prescribed (but
        `len(descriptors)` must be equal to `len(components)`). Any missing
        components are assumed to be 0 strain. Accordingly, the following two
        steps would be treated identically:

        >>> obj.run_step('ESSEEE', [1., 0., 0., 0., 0., 0.], scale=.1)
        >>> obj.run_step('ESS', [1., 0., 0.], scale=.1)

        Steps are run when they are added.

        """

        if self.material is None:
            raise RuntimeError("Material must be assigned before adding steps")

        if not self._initialized:
            self.initialize_data()

        descriptors, components = self._format_descriptors_and_components(
            descriptors, components
        )

        # Stress control must have kappa = 0
        if any([x in descriptors for x in ("S", "DS")]) and abs(kappa) > 1e-12:
            raise ValueError("Stress control requires kappa = 0")

        istep = len(self.steps)
        previous_step = self.steps[-1]

        if ufield is not None:
            if previous_step.ufield is None:
                s = (
                    "Invalid ufield specification on step {0}.\n"
                    "Was ufield passed to the MaterialPointSimulator's "
                    "constructor?"
                )
                raise ValueError(s.format(istep))
            if not is_listlike(ufield):
                ufield = [ufield]
            ufield = np.asarray(ufield)
            if ufield.shape != previous_step.ufield.shape:
                s = (
                    "Invalid ufield specification on step {0}.\n"
                    "ufield.shape is different than previous step"
                )
                raise ValueError(s.format(istep))
        if previous_step.ufield is not None and ufield is None:
            # Must define ufield for *all* steps
            ufield = np.asarray(previous_step.ufield)

        if not is_listlike(scale):
            # Scalar scale factor
            scale = np.ones(len(components)) * scale
        scale = np.asarray(scale)
        if len(scale) != len(components):
            raise ValueError("components and scale must have same length")

        # Apply scaling factors
        components = components * scale

        if "F" in descriptors:
            pass
            # Convert deformation gradient to strain
            # components, rotation = dfm.strain_from_defgrad(components, kappa)
            # if np.max(np.abs(rotation - np.eye(3))) > 1e-8:
            #    raise ValueError('QR decomposition of deformation gradient '
            #                     'gave unexpected rotations (rotations are '
            #                     'not yet supported)')
            # descriptors = ['E'] * 6

        elif "U" in descriptors:
            # Convert displacement to strain
            U = np.zeros((3, 3))
            DI3 = np.diag_indices(3)
            U[DI3] = components + 1.0
            components = dfm.strain_from_stretch(array_rep(U, (6,)), kappa)
            descriptors = ["E"] * 6

        elif "E" in descriptors and len(descriptors) == 1:
            # only one strain value given -> volumetric strain
            components = dfm.scalar_volume_strain_to_tensor(components[0], kappa)
            descriptors = ["E"] * 6

        elif "S" in descriptors and len(descriptors) == 1:
            # only one stress value given -> pressure
            Sij = -components[0]
            components = np.array([Sij, Sij, Sij, 0.0, 0.0, 0.0])
            descriptors = ["S"] * 6

        elif "DS" in descriptors and len(descriptors) == 1:
            # only one stress value given -> pressure
            ds = -components[0]
            components = np.array([ds, ds, ds, 0.0, 0.0, 0.0])
            descriptors = ["DS", "DS", "DS", "E", "E", "E"]

        if np.any(np.in1d(["E", "S", "DE", "DS"], descriptors)):
            # Stress/strain must have length == 6
            if len(descriptors) != 6:
                n = 6 - len(descriptors)
                descriptors.extend(["E"] * n)
                components = np.append(components, [0.0] * n)

        xc = "[{0}]".format(", ".join(["{0:g}".format(x) for x in components]))
        logger.debug(
            "Adding step {0:4d} with descriptors: {1}\n"
            "                   and components: {2}".format(
                istep, "".join(descriptors), xc
            )
        )

        begin = self.steps[-1].end

        if time_whole is not None:
            time_whole = float(time_whole)
            if time_whole < begin:
                i = len(self.steps) + 1
                raise ValueError(
                    "time_whole for step {0} " "< beginning time".format(i)
                )
            increment = time_whole - begin

        end = begin + increment
        step = Step(
            begin, end, frames, descriptors, components, temperature, kappa, ufield
        )

        # Add space for this step
        irow, icol = self.data.shape
        self.data = np.row_stack((self.data, np.zeros((frames, icol))))

        # Now run the thing - adding enough rows to the data array for this step
        logger.info("\rRunning step {0}... ".format(istep), extra=continued)
        assert step.begin == previous_step.end
        self.run_istep(
            istep,
            step.begin,
            step.end,
            step.frames,
            step.descriptors,
            step.components,
            step.temp,
            step.ufield,
            step.kappa,
            self.J0,
            self.data[irow - 1 :, :],
        )

        logger.info("done")
        self.steps.append(step)
        self.ran = True

        return step

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        self.assign_material(material)

    def assign_material(self, material):
        """Assign the material model to the `MaterialPointSimulator`

        Parameters
        ----------
        material : Material
            A material model

        Notes
        -----
        `material` is assumed to be subclassed from the `Material` class.
        Accordingly, the following members are assumed to exist:

        - `material.name`: The name of the material. Default is `None`
        - `material.num_sdv`: Number of state dependent variables. Default is
          `None`
        - `material.sdv_names`: Names of state dependent variables (in order
          expected by model). Default is `None`. If `material.num_sdv` is not
          `None` and `material.sdv_names` is `None`, state dependent variables
          are given the names `SDV.1`, `SDV.2`, ..., `SDV.num_sdv`

        The following methods are assumed to exist:

        - `material.sdvini`: Initialize state dependent variables. All state
          dependent variables are assumed to have an initial value of 0. The
          method `sdvini` is used to change this initial value.
        - `material.eval`: The material state update.

        See the documentation for the `Material` base class for more information

        """
        if not hasattr(material, "eval"):
            raise Exception("Material models must define the `eval` method")
        self.mat_is_Material_subclass = hasattr(material, "base_eval")
        optional_attrs = ("name", "num_sdv", "sdv_names", "sdvini")
        not_defined = []
        for attr in optional_attrs:
            try:
                getattr(material, attr)
            except AttributeError:
                not_defined.append(attr)
        name = getattr(material, "name", None)
        logger.info("Assigning material {0!r}".format(name))
        if not_defined:
            attrs = ", ".join(not_defined)
            logger.warning("Optional material members not defined: " + attrs)
        self._material = material
        material.assigned = True

    def undo_step(self):
        """Undo the last step, resetting the state variables"""

        # total_frames = sum(_.frames for _ in self.steps)

        if len(self.steps) <= 1:
            raise Exception("Cannot undo initialization step")

        # num_kept_frames = sum(_.frames for _ in self.steps[:-1])

        self.steps = self.steps[:-1]
        self.data = self.data[:-1]

    def initialize_data(self):
        """When the material is assigned, initialize the database"""

        if self.material is None:
            raise RuntimeError("Material not assigned")

        # Setup the array of simulation data
        columns = list(self.columns.keys())
        num_vars = len(columns)
        # num_incs = 1
        num_incs = sum(step.frames for step in self.steps)
        self.data = np.zeros((num_incs, num_vars))

        # Put the initial state in the output database
        step = self.steps[0]
        statev = self.initialize_statev()
        strain = np.where(step.descriptors == "E", step.components, 0.0)
        stress = np.where(step.descriptors == "S", step.components, 0.0)
        defgrad = dfm.defgrad_from_strain(strain, step.kappa)
        glo_var_vals = [step.increment, 1, 0]
        elem_var_vals = self.astack(
            strain,
            np.zeros(6),
            stress,
            stress - np.zeros(6),
            defgrad,
            step.temp,
            step.ufield,
            statev,
        )
        self.data[0, 0] = step.end
        self.data[0, 1:4] = glo_var_vals
        self.data[0, 4:] = elem_var_vals

        # Call the material with a zero state to get the initial Jacobian
        dtemp = 0.0
        dufield = np.zeros_like(step.ufield)
        self.J0 = numerical_jacobian(
            self.eval,
            1,
            1,
            step.temp,
            dtemp,
            defgrad,
            defgrad,
            np.zeros(6),
            np.zeros(6),
            np.zeros(6),
            step.ufield,
            dufield,
            copy(statev),
            range(6),
        )

        # This step is not actually ran - it's just the initial state
        step.ran = True
        self._initialized = True

    @property
    def df(self):
        """Return the DataFrame containing simulation data"""
        from pandas import DataFrame

        if self._df is None or self._df.shape[0] < self.data.shape[0]:
            columns = list(self.columns.keys())
            self._df = DataFrame(self.data, columns=columns)
        return self._df

    def get_from_df(self, key):
        """Get `key` from the database

        Parameters
        ----------
        key : str
            key is the name of the variable to get from the database.

        Returns
        -------
        df : DataFrame
            A Pandas DataFrame containing the values for `key` for all times

        Notes
        -----
        `key` can be either:

        - a single component like `F.XX`, in which case a `DataSeries` will be
          returned containing `F.XX` through all time of the simulation
        - a name like `F`, in which case a `DataFrame` will be returned
          containing all of the components of `F` through all time of the
          simulation

        """
        if key in self.df:
            return self.df[key]
        keys = self.expand_name_to_keys(key, self.df.columns)
        return self.df[keys]

    def get_from_a(self, key):
        """Get the value of key from the data array"""
        if key in self.columns:
            return self.data[:, self.columns[key]]
        columns = list(self.columns.keys())
        keys = self.expand_name_to_keys(key, columns)
        if keys is None:
            return None
        ix = [columns[key] for key in keys]
        return self.data[:, ix]

    def get(self, key, df=None):
        df = df or environ.notebook
        if df:
            return self.get_from_df(key)
        return self.get_from_a(key)

    def get2(self, *keys, **kwargs):
        df = kwargs.get("df", None) or environ.notebook
        if df:
            if is_listlike(keys):
                keys = list(keys)
            return self.df[keys]
        ix = [self.columns[key] for key in keys]
        return self.data[:, ix]

    def plot(self, *args, **kwargs):
        return self.df.plot(*args, **kwargs)

    def dump(self, filename=None):
        """Write results to output database"""

        logger.info("Opening the output database... ", extra=continued)
        if filename is None:
            filename = self.jobid + "." + self.db_fmt

        root, ext = os.path.splitext(filename)
        if not ext:
            ext = "." + self.db_fmt
            filename = filename + ext
        assert ext in (".npz", ".exo")

        if ext == ".exo":
            self._write_exodb(filename)

        elif ext == ".npz":
            self._write_npzdb(filename)

    def _write_exodb(self, filename):
        """Write the results to a exodus database"""
        self.db = DatabaseFile(filename, "w")
        logger.info("done")
        logger.info("Output database: {0!r}".format(self.db.filename))
        logger.info("Initializing the output database... ", extra=continued)
        self.db.initialize(self.glo_var_names, self.elem_var_names)
        logger.info("done")

        logger.info("Writing data to {0!r}".format(self.db.filename))
        num_glo_vars = len(self.glo_var_names)
        i, j = 1, 1 + num_glo_vars
        start = time.time()
        for row in self.data:
            end_time = row[0]
            glo_var_vals = row[i:j]
            elem_var_vals = row[j:]
            self.db.save(end_time, glo_var_vals, elem_var_vals)
        dt = time.time() - start
        logger.info("Done writing data {0:.2f}".format(dt))
        logger.info("Closing the output database... ", extra=continued)
        self.db.close()
        logger.info("done")

    def dumpz(self, filename=None):
        """Write results to output database"""
        if filename is None:
            filename = self.jobid
        if not filename.endswith(".npz"):
            filename += ".npz"
        self._write_npzdb(filename)

    def _write_npzdb(self, filename):
        logger.info("Writing data to {0!r}... ".format(filename), extra=continued)
        columns = list(self.columns.keys())
        with open(filename, "wb") as fh:
            np.savez(fh, columns=columns, data=self.data)
        logger.info("done")

    def expand_name_to_keys(self, key, columns):
        names_and_cols = groupby_names(columns)
        if key not in names_and_cols:
            return None
        sep = COMPONENT_SEP
        keys = ["{0}{1}{2}".format(key, sep, x) for x in names_and_cols[key]]
        return keys

    @property
    def columns(self):
        if self._columns is not None:
            return self._columns
        columns = ["Time"] + self.glo_var_names + self.elem_var_names
        self._columns = OrderedDict([(x, i) for (i, x) in enumerate(columns)])
        return self._columns

    @property
    def glo_var_names(self):
        return ["DTime", "Step", "Frame"]

    @property
    def elem_var_names(self):
        """Returns the list of element variable names"""
        if self.material is None:
            raise ValueError("Material must first be assigned")
        if self._elem_var_names is not None:
            return self._elem_var_names

        def expand_var_name(name, components):
            sep = COMPONENT_SEP
            return ["{0}{1}{2}".format(name, sep, x) for x in components]

        # xc1 = ["X", "Y", "Z"]
        xc2 = ["XX", "YY", "ZZ", "XY", "YZ", "XZ"]
        xc3 = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
        elem_var_names = []
        elem_var_names.extend(expand_var_name("E", xc2))
        elem_var_names.extend(expand_var_name("DE", xc2))
        elem_var_names.extend(expand_var_name("S", xc2))
        elem_var_names.extend(expand_var_name("DS", xc2))
        elem_var_names.extend(expand_var_name("F", xc3))
        elem_var_names.append("Temp")

        # User defined field
        if self.num_ufield:
            ufield_names = expand_var_name("UFIELD", range(1, self.num_ufield + 1))
            elem_var_names.extend(ufield_names)

        # Material state variables
        num_sdv = getattr(self.material, "num_sdv", Material.num_sdv)
        sdv_names = getattr(self.material, "sdv_names", Material.sdv_names)
        if num_sdv:
            if sdv_names:
                assert len(sdv_names) == num_sdv
            else:
                sdv_names = expand_var_name("SDV", range(1, num_sdv + 1))
            elem_var_names.extend(sdv_names)

        # Names for material aux models
        if hasattr(self.material, "aux_models"):
            for (name, aux_model) in self.material.aux_models.items():
                elem_var_names.extend(aux_model.sdv_names)

        self._elem_var_names = elem_var_names
        return elem_var_names

    def initialize_statev(self):
        """Initialize the state dependent variables - including aux models"""
        numx = getattr(self.material, "num_sdv", Material.num_sdv)
        statev = None if numx is None else np.zeros(numx)
        try:
            statev = self.material.sdvini(statev)
        except AttributeError:
            pass

        aux_sdv = []
        if hasattr(self.material, "aux_models"):
            for (name, aux_model) in self.material.aux_models.items():
                xv = np.zeros(aux_model.num_sdv)
                aux_sdv.extend(aux_model.sdvini(xv))

        if aux_sdv:
            if statev is not None:
                statev = np.append(statev, aux_sdv)
            else:
                statev = np.array(aux_sdv)

        return statev

    def astack(self, E, DE, S, DS, F, T, UF, XV):
        """Concatenates input arrays into a single flattened array"""
        a = [E, DE, S, DS, F, T]
        if len(UF):
            a.append(UF)
        if XV is not None:
            a.append(XV)
        return np.hstack(tuple([x for x in a if x is not None]))

    def run_istep(
        self,
        istep,
        begin,
        end,
        frames,
        descriptors,
        components,
        temp,
        ufield,
        kappa,
        J0,
        data,
    ):
        """Run this step, using the previous step as the initial state

        Parameters
        ----------
        istep : int
            The step number to run

        """
        if "F" in descriptors:
            self._run_istep_F(
                istep,
                begin,
                end,
                frames,
                descriptors,
                components,
                temp,
                ufield,
                kappa,
                J0,
                data,
            )
        else:
            self._run_istep(
                istep,
                begin,
                end,
                frames,
                descriptors,
                components,
                temp,
                ufield,
                kappa,
                J0,
                data,
            )

    def _run_istep(
        self,
        istep,
        begin,
        end,
        frames,
        descriptors,
        components,
        temp,
        ufield,
        kappa,
        J0,
        data,
    ):
        assert istep != 0
        increment = end - begin

        # ---------------------------------------------------------------------- #
        # The following variables have values at
        # [begining, end, current] of step
        #
        # The deformation gradient has values at
        # [begining of step, end of step, beginning of frame, current]
        # ---------------------------------------------------------------------- #
        # Time
        time = np.array([begin, end, begin])

        # Strain and stress states
        start_strain = data[0, 4:10]
        start_stress = data[0, 16:22]
        end_strain = np.where(descriptors == "E", components, 0.0)
        end_stress = np.where(descriptors == "S", components, 0.0)
        strain = np.vstack((start_strain, end_strain, start_strain))
        stress = np.vstack((start_stress, end_stress, start_stress))

        # Temperature
        temp = np.array((data[0, 37], temp, data[0, 37]))
        dtemp = (temp[1] - temp[0]) / float(frames)

        # User defined field
        start = 38
        end = start + len(ufield)
        ufield = np.array((data[0, start:end], ufield, data[0, start:end]))
        dufield = (ufield[1] - ufield[0]) / float(frames)
        start = end

        # ---------------------------------------------------------------------- #
        # The following variables have values at
        # [begining, current] of step
        # ---------------------------------------------------------------------- #
        start_statev = data[0, start:]
        if not len(start_statev):
            statev = [None, None]
        else:
            statev = np.vstack((start_statev, start_statev))

        start_defgrad = data[0, 28:37]
        F = np.vstack((start_defgrad, start_defgrad))

        dtime = 1.0 if increment < 1.0e-14 else (time[1] - time[0]) / float(frames)

        # v array is an array of integers that contains the rows and columns of
        # the slice needed in the jacobian subroutine.
        nv = 0
        v = np.zeros(6, dtype=np.int)
        for (i, cij) in enumerate(components):
            descriptor = descriptors[i]
            if descriptor == "DE":  # -- strain rate
                strain[1, i] = strain[0, i] + cij * increment
            elif descriptor == "E":  # -- strain
                strain[1, i] = cij
            elif descriptor == "DS":  # -- stress rate
                stress[1, i] = stress[0, i] + cij * increment
                v[nv] = i
                nv += 1
            elif descriptor == "S":  # -- stress
                stress[1, i] = cij
                v[nv] = i
                nv += 1
            else:
                raise ValueError("Invalid descriptor {0!r}".format(descriptor))

        v = v[:nv]
        vx = [x for x in range(6) if x not in v]
        if increment < 1.0e-14:
            dedt = np.zeros_like(strain[1])
        else:
            dedt = (strain[1] - strain[0]) / increment

        # --- find current value of d: sym(velocity gradient)
        if not nv:
            # strain or strain rate prescribed and the strain rate is constant
            # over the entire step
            if abs(kappa) > 1.0e-14:
                d = dfm.rate_of_strain_to_rate_of_deformation(dedt, strain[2], kappa)
            elif environ.SQA:
                d = dfm.rate_of_strain_to_rate_of_deformation(dedt, strain[2], kappa)
                if not np.allclose(d, dedt):
                    logger.warning("SQA: d != dedt")
            else:
                d = np.array(dedt)

        else:

            # Initial guess for d[v]
            dedt[v] = 0.0
            Jsub = J0[[[x] for x in v], v]
            work = (stress[1, v] - stress[0, v]) / increment
            try:
                dedt[v] = la.solve(Jsub, work)
            except Exception:
                try:
                    dedt[v] -= la.lstsq(Jsub, work, rcond=None)[0]
                except Exception:
                    dedt[v] = 0
            dedt[v] = dedt[v] / VOIGT[v]

        # Process each frame of the step
        for iframe in range(frames):
            a1 = float(frames - (iframe + 1)) / frames
            a2 = float(iframe + 1) / frames

            strain[2] = a1 * strain[0] + a2 * strain[1]
            pstress = a1 * stress[0] + a2 * stress[1]

            if nv:
                # One or more stresses prescribed
                d = d_from_prescribed_stress(
                    self.eval,
                    time[2],
                    dtime,
                    temp[2],
                    dtemp,
                    F[0],
                    F[1],
                    strain[2] * VOIGT,
                    dedt * VOIGT,
                    stress[2],
                    ufield[2],
                    dufield,
                    statev[0],
                    v,
                    pstress[v],
                )
                d = d / VOIGT

            # compute the current deformation gradient and strain from
            # previous values and the deformation rate
            F[1], e = dfm.update_deformation(F[0], d, dtime, kappa)
            strain[2, v] = e[v]

            if environ.SQA and not np.allclose(strain[2, vx], e[vx]):
                logger.warning("SQA: bad strain on  step {0}".format(istep))

            state = self.eval(
                kappa,
                time[2],
                dtime,
                temp[2],
                dtemp,
                F[0],
                F[1],
                strain[2] * VOIGT,
                d * VOIGT,
                np.array(stress[2]),
                ufield[2],
                dufield,
                statev[1],
            )
            s, x, ddsdde = state
            self.ddsdde = ddsdde

            F[0] = F[1]
            dstress = s - stress[2]
            time[2] = a1 * time[0] + a2 * time[1]
            temp[2] = a1 * temp[0] + a2 * temp[1]
            ufield[2] = a1 * ufield[0] + a2 * ufield[1]
            stress[2], statev[1] = s, x
            statev[0] = statev[1]

            glo_var_vals = [dtime, istep + 1, iframe + 1]
            elem_var_vals = self.astack(
                strain[2], dedt, stress[2], dstress, F[1], temp[2], ufield[2], x
            )
            data[iframe + 1, 0] = time[2]
            data[iframe + 1, 1:4] = glo_var_vals
            data[iframe + 1, 4:] = elem_var_vals

    def _run_istep_F(
        self,
        istep,
        begin,
        end,
        frames,
        descriptors,
        components,
        temp,
        ufield,
        kappa,
        J0,
        data,
    ):
        assert istep != 0
        increment = end - begin

        # ---------------------------------------------------------------------- #
        # The following variables have values at
        # [begining, end, current] of step
        #
        # The deformation gradient has values at
        # [begining of step, end of step, beginning of frame, current]
        # ---------------------------------------------------------------------- #

        # Time
        time = np.array([begin, end, begin])

        # Strain and stress states
        start_strain = data[0, 4:10]
        start_stress = data[0, 16:22]
        start_defgrad = data[0, 28:37]
        # Will be computed from defgrad
        end_strain = np.zeros(6)
        end_stress = np.zeros(6)
        end_defgrad = components

        strain = np.vstack((start_strain, end_strain, start_strain))
        stress = np.vstack((start_stress, end_stress, start_stress))
        F = np.vstack((start_defgrad, end_defgrad, start_defgrad, I9))

        # Temperature
        temp = np.array((data[0, 37], temp, data[0, 37]))
        dtemp = (temp[1] - temp[0]) / float(frames)

        # User defined field
        start = 38
        end = start + len(ufield)
        ufield = np.array((data[0, start:end], ufield, data[0, start:end]))
        dufield = (ufield[1] - ufield[0]) / float(frames)
        start = end

        # ---------------------------------------------------------------------- #
        # The following variables have values at
        # [begining, current] of step
        # ---------------------------------------------------------------------- #
        start_statev = data[0, start:]
        if not len(start_statev):
            statev = [None, None]
        else:
            statev = np.vstack((start_statev, start_statev))

        dtime = 1.0 if increment < 1.0e-14 else (time[1] - time[0]) / float(frames)

        # Process each frame of the step
        for iframe in range(frames):
            a1 = float(frames - (iframe + 1)) / frames
            a2 = float(iframe + 1) / frames

            F[3] = a1 * F[0] + a2 * F[1]
            _, R0 = dfm.strain_from_defgrad(F[2], kappa)
            strain[2], R = dfm.strain_from_defgrad(F[3], kappa)
            if dtime < 1.0e-14:
                d = np.zeros(6)
            else:
                d = dfm.rate_of_defomation_from_defgrad(F[2], F[3], dtime)
            dedt = d  # FIXME: this is for output only, should be fixed

            # d_b = unrotate(R0, d)
            # strain_b = unrotate(R0, strain[2])
            # stress_b = unrotate(R0, stress[2])
            state = self.eval(
                kappa,
                time[2],
                dtime,
                temp[2],
                dtemp,
                F[2],
                F[3],
                strain[2] * VOIGT,
                d * VOIGT,
                np.array(stress[2]),
                ufield[2],
                dufield,
                statev[1],
            )
            s, x, ddsdde = state
            self.ddsdde = ddsdde

            F[2] = F[3]
            dstress = s - stress[2]
            time[2] = a1 * time[0] + a2 * time[1]
            temp[2] = a1 * temp[0] + a2 * temp[1]
            ufield[2] = a1 * ufield[0] + a2 * ufield[1]
            statev[1] = x
            statev[0] = statev[1]
            stress[2] = s
            # stress[2] = rotate(R, s)

            glo_var_vals = [dtime, istep + 1, iframe + 1]
            elem_var_vals = self.astack(
                strain[2], dedt, stress[2], dstress, F[3], temp[2], ufield[2], x
            )
            data[iframe + 1, 0] = time[2]
            data[iframe + 1, 1:4] = glo_var_vals
            data[iframe + 1, 4:] = elem_var_vals

    def eval(
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
        **kwds
    ):
        """Wrapper method to material.eval. This is called by Matmodlab so that
        aux models can first be evaluated. See documentation for eval.

        """
        if self.mat_is_Material_subclass:
            return self.material.base_eval(
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
                self.initial_temp,
                **kwds
            )
        else:
            # Not a subclass of the Material class, call its eval method
            return self.material.eval(
                time,
                dtime,
                temp,
                dtemp,
                F0,
                F,
                strain,
                d,
                stress,
                statev,
                ufield=ufield,
                dufield=dufield,
                **kwds
            )


class Step(object):
    def __init__(
        self, begin, end, frames, descriptors, components, temp, kappa, ufield=None
    ):
        assert len(components) == len(descriptors)
        #        assert len(descriptors) == 6
        self.begin = float(begin)
        self.end = float(end)
        self.increment = self.end - self.begin
        if abs(self.increment) > 0.0:
            assert end > begin
        self.frames = frames
        self.components = np.asarray(components)
        self.descriptors = np.asarray(descriptors)
        self.temp = temp
        self.kappa = kappa
        self.ran = False
        if ufield is not None:
            if not is_listlike(ufield):
                ufield = [ufield]
        else:
            ufield = []
        ufield = np.asarray(ufield)
        if len(ufield.shape) != 1:
            raise ValueError("ufield must be a 1D array")
        self.ufield = ufield
