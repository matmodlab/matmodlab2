from __future__ import print_function
import time
import numpy as np
from collections import OrderedDict

from .misc import is_stringlike, is_listlike
from .environ import environ
from .logio import logger, add_filehandler, splash
from .database import DatabaseFile, COMPONENT_SEP, groupby_names
from .tensor import array_rep
from .stress_control import d_from_prescribed_stress
from .deformation import update_deformation, strain_from_stretch, \
    defgrad_from_strain, rate_of_strain_to_rate_of_deformation
from .tensor import VOIGT
import matmodlab.core.linalg as la
continued = {'continued': 1}

__all__ = ['MaterialPointSimulator']

class MaterialPointSimulator(object):
    """The material point simulator

    The material point simulator exercises a material model just as a finite
    element solver would. The steps to creating and running a simulation with
    the `MaterialPointSimulator` are:

    1. Instantiate a `MaterialPointSimulator` simulator object, giving it a
       string `jobid`
    2. Assign a material model to the simulator
    3. Add simulation (deformation) steps to the simulator
    4. Run the simulator

    Output is written to an ExodusII database file. ExodusII database files can
    be viewed using the open source `tsviewer` or `ParaView`.

    Examples
    --------

    In the following simulation, the `ElasticMaterial` is exercised through a
    step of uniaxial strain and then a stress controlled step to bring the
    material to a state of zero stress.

    >>> jobid = 'Job-1'
    >>> mps = MaterialPointSimulator(jobid)
    >>> material = ElasticMaterial(E=10, Nu=.1)
    >>> mps.assign_material(material)
    >>> mps.add_step('EEEEEE', [1., 0., 0., 0., 0., 0.])
    >>> mps.add_step('SSSEEE', [0., 0., 0., 0., 0., 0.])
    >>> mps.run()

    """
    valid_descriptors = ['DE', 'E', 'S', 'DS', 'U', 'F']
    def __init__(self, jobid, initial_temp=0., write_db=None,
                 db_fmt='npz'):

        logger.info('Initializing the simulation')
        self.jobid = jobid

        # File I/O
        if environ.notebook:
            if write_db:
                logger.warning('Database will not automatically be created in '
                               'the notebook environment.  Use the '
                               'MaterialPointSimulator.dump method to create '
                               'a database file, if desired.')
            self.write_db = False
        else:
            self.write_db = True if write_db is None else write_db
        if self.write_db:
            # No log file if not db
            add_filehandler(logger, self.jobid+'.log')
        splash(logger)
        if db_fmt not in ('npz', 'exo'):
            raise ValueError('db_fmt must by npz or exo')
        self.db_fmt = db_fmt

        logger.info('Matmodlab simulation: {0!r}'.format(self.jobid))

        # Create initial step
        self.initial_temp = initial_temp
        logger.info('Creating initial step... ', extra=continued)
        self.steps = self._initialize_steps(self.initial_temp)
        logger.info('done')

        # Set defaults
        self.ran = False
        self._df = None
        self._columns = None
        self._material = None
        self._elem_var_names = None
        self.db = None
        self.data = None
        logger.info('Done initializing the simulation')

    def _initialize_steps(self, temp):
        """Create the initial step"""
        begin, end = 0., 0.
        components = np.zeros(6, dtype=np.float64)
        descriptors = ['E'] * 6
        return [Step(0, 0, 1, descriptors, components, temp, 0)]

    def _validate_descriptors(self, descriptors):
        """Validate the user given descriptors"""
        descriptors = list(descriptors)
        for (i, descriptor) in enumerate(descriptors):
            if descriptor.upper() not in self.valid_descriptors:
                raise ValueError('Invalid descriptor {0!r}'.format(descriptor))
            descriptors[i] = descriptor.upper()

        unique_descriptors = list(set(descriptors))
        if 'F' in unique_descriptors:
            if len(unique_descriptors) != 1:
                raise ValueError('Cannot mix F with other descriptors')
            elif len(descriptors) != 9:
                raise ValueError('Must specify all 9 components of F')

        elif 'U' in unique_descriptors:
            if len(unique_descriptors) != 1:
                raise ValueError('Cannot mix U with other descriptors')
            elif len(descriptors) != 3:
                raise ValueError('Must specify all 3 components of U')

        elif np.any(np.in1d(['E', 'S', 'DE', 'DS'], descriptors)):
            if len(descriptors) > 6:
                raise ValueError('At most 6 components of stress/strain '
                                 'can be prescribed')

        return descriptors

    def add_step(self, descriptors, components, increment=1., frames=1,
                 scale=1., kappa=0., temperature=0., time_whole=None):
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

        Tensor Component Ordering
        -------------------------
        Component ordering for components is:

        1. Symmetric tensors: XX, YY, ZZ, XY, YZ, XZ
        2. Unsymmetric tensors: XX, XY, XZ YX, YY, YZ ZX, ZY, ZZ
        3. Vectors: X, Y, Z

        Examples
        --------
        To create a step of uniaxial strain with magnitude .1:

        >>> obj.add_step('EEEEEE', [1., 0., 0., 0., 0., 0.], scale=.1)

        To create a step of uniaxial stress with magnitude 1e6:

        >>> obj.add_step('SSSEEE', [1., 0., 0., 0., 0., 0.], scale=1e6)

        Stress and strain (and their increments) can be mixed.  To create a step of uniaxial stress by holding the lateral stress components at 0. and deforming along the axial direction:

        >>> obj.add_step('ESSEEE', [1., 0., 0., 0., 0., 0.], scale=.1)

        To create a step of uniaxial strain, controlled by deformation gradient:

        >>> obj.add_step('FFFFFFFFF', [1.05, 0., 0., 1., 0., 0.], scale=1e6)

        Note, all 9 components of the deformation gradient must be prescribed.

        Special deformation cases are volumetric strain and pressure. Each is
        defined by prescribing one, and only one, component of either strain or
        stress, respectively:

        Volumetric strain.

        >>> obj.add_step('E', .1)

        Pressure:

        >>> obj.add_step('S', 1, scale=1e6)

        Notes
        -----
        Prescribed deformation gradient and displacement components are
        converted to strain. Internally, the driver works only with strain,
        stress, or their increments.

        For stress, strain (and their increments), or mixed steps, all
        components of deformation need not be prescribed (but
        `len(descriptors)` must be equal to `len(components)`). Any missing
        components are assumed to be strain. Accordingly, the following two
        steps would be treated identically:

        >>> obj.add_step('ESSEEE', [1., 0., 0., 0., 0., 0.], scale=.1)
        >>> obj.add_step('ESS', [1., 0., 0.], scale=.1)

        Steps are accumulated by the `MaterialPointSimulator` object and not
        actually run until the `MaterialPointSimulator.run()` method is called.

        """
        if not is_listlike(components):
            components = [components]
        components = np.array(components, dtype=np.float64)

        if is_stringlike(descriptors):
            if len(descriptors) == 1:
                # broadcast descriptor to each component
                descriptors = descriptors * len(components)
            descriptors = list(descriptors)
        if not is_listlike(descriptors):
            raise TypeError('descriptors must be list_like or string_like')
        descriptors = self._validate_descriptors(descriptors)

        if not is_listlike(scale):
            # Scalar scale factor
            scale = np.ones(len(components)) * scale
        scale = np.asarray(scale)

        # Sanity checks
        if len(descriptors) != len(components):
            raise ValueError('components and descriptors must have same length')
        if len(scale) != len(components):
            raise ValueError('components and scale must have same length')

        # Apply scaling factor
        components = components * scale

        if 'F' in descriptors:
            # Convert deformation gradient to strain
            F = np.reshape(components, (3, 3))
            jac = la.det(F)
            if jac <= 0:
                raise ValueError('Negative or zero initial Jacobian')

            # convert deformation gradient to strain E with associated
            # rotation given by axis of rotation x and angle of rotation theta
            R, V = np.linalg.qr(F)
            if np.max(np.abs(R - np.eye(3))) > np.finfo(np.float).eps:
                raise ValueError('QR decomposition of deformation gradient '
                                 'gave unexpected rotations (rotations are '
                                 'not yet supported)')
            U = np.dot(R.T, np.dot(V, R))
            components = strain_from_stretch(array_rep(U,(6,)), kappa)
            descriptors = ['E'] * 6

        elif 'U' in descriptors:
            # Convert displacement to strain
            U = np.zeros((3, 3))
            DI3 = np.diag_indices(3)
            U[DI3] = components + 1.
            components = strain_from_stretch(array_rep(U,(6,)), kappa)
            descriptors = ['E'] * 6

        elif 'E' in descriptors and len(descriptors) == 1:
            # only one strain value given -> volumetric strain
            ev = components[0]
            if kappa * ev + 1. < 0.:
                raise ValueError('1 + kappa * ev must be positive')

            if abs(kappa) < np.finfo(np.float).eps:
                eij = ev / 3.
            else:
                eij = ((kappa * ev + 1.) ** (1. / 3.) - 1.)
                eij = eij / kappa
            components = np.array([eij, eij, eij, 0., 0., 0.], dtype=np.float64)
            descriptors = ['E'] * 6

        elif 'S' in descriptors and len(descriptors) == 1:
            # only one stress value given -> pressure
            Sij = -components[0]
            components = np.array([Sij, Sij, Sij, 0., 0., 0.], dtype=np.float64)
            descriptors = ['S'] * 6

        elif 'DS' in descriptors and len(descriptors) == 1:
            # only one stress value given -> pressure
            ds = -components[0]
            components = np.array([ds, ds, ds, 0., 0., 0.], dtype=np.float64)
            descriptors = ['DS', 'DS', 'DS', 'E', 'E', 'E']

        if np.any(np.in1d(['E', 'S', 'DE', 'DS'], descriptors)):
            # Stress/strain must have length == 6
            if len(descriptors) != 6:
                n = 6 - len(descriptors)
                descriptors.extend(['E'] * n)
                components = np.append(components, [0.] * n)

        n = len(self.steps)
        xc = '[{0}]'.format(', '.join(['{0:g}'.format(x) for x in components]))
        logger.debug('Adding step {0:4d} with descriptors: {1}\n'
                     '                   and components: {2}'.format(
                         n, ''.join(descriptors), xc))

        begin = self.steps[-1].end

        if time_whole is not None:
            time_whole = float(time_whole)
            if time_whole < begin:
                i = len(self.steps)+1
                raise ValueError('time_whole for step {0} '
                                 '< beginning time'.format(i))
            increment = time_whole - begin

        end = begin + increment
        step = Step(begin, end, frames, descriptors, components,
                    temperature, kappa)
        self.steps.append(step)

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
        `material` is assumed to be subclassed from the `Material` class.  Accordingly, the following members are assumed to exist:

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
        required_attrs = ('name', 'num_sdv', 'sdv_names', 'sdvini', 'eval')
        for attr in required_attrs:
            try:
                getattr(material, attr)
            except AttributeError:
                attrs = ', '.join(required_attrs)
                raise Exception('Material models must define all of the '
                                'following attributes: {0}'.format(attrs))
        logger.info('Assigning material {0!r}'.format(material.name))
        self._material = material

    def run(self):
        """Run the simulation

        Notes
        -----
        This method initializes and sets up the output database and runs each
        step of the simulation.

        """
        start_sim = time.time()
        logger.info('Running the simulation')

        if self.material is None:
            raise RuntimeError('Material not assigned')

        if self.ran:
            raise RuntimeError('Already run')

        # Setup the array of simulation data
        columns = list(self.columns.keys())
        num_vars = len(columns)
        num_incs = sum(step.frames for step in self.steps)
        self.data = np.zeros((num_incs, num_vars))

        # Put the initial state in the output database
        step = self.steps[0]
        numx = self.material.num_sdv
        statev = None if numx is None else np.zeros(numx)
        statev = self.material.sdvini(statev)
        strain = np.where(step.descriptors=='E', step.components, 0.) * VOIGT
        stress = np.where(step.descriptors=='S', step.components, 0.)
        defgrad = defgrad_from_strain(strain, step.kappa)
        glo_var_vals = [step.increment, 1, 0]
        elem_var_vals = self.astack(strain, np.zeros(6),
                                    stress, stress-np.zeros(6),
                                    defgrad, step.temp, statev)
        self.data[0, 0  ] = step.end
        self.data[0, 1:4] = glo_var_vals
        self.data[0, 4:] = elem_var_vals
        step.ran = True

        # Run each step, skipping the first
        num_steps = len(self.steps)
        m = len(str(num_steps))
        string = '\rRunning simulation steps [{{0:{0}d}}/{1}]'.format(
            m, num_steps-1)
        start_steps = time.time()
        for i in range(len(self.steps)-1):
            istep = i + 1
            irow = sum(step.frames for step in self.steps[:istep])-1
            previous_step = self.steps[i]
            step = self.steps[istep]
            assert step.begin == previous_step.end
            logger.info(string.format(istep), extra=continued)
            self.run_istep(istep, step.begin, step.end, step.frames,
                           step.descriptors, step.components,
                           step.temp, step.kappa, self.data[irow:, :])
            step.ran = True
        dt = time.time() - start_steps
        logger.info(' done ({0:.2f} sec.)'.format(dt))
        logger.info('All steps complete')

        self.ran = True

        if self.write_db:
            if self.db_fmt == 'npz':
                self.dumpz()
            else:
                self.dump()

        dt = time.time() - start_sim
        logger.info('Simulation complete ({0:.2f} sec.)'.format(dt))

    @property
    def df(self):
        """Return the DataFrame containing simulation data"""
        from pandas import DataFrame
        if not self.ran:
            raise RuntimeError('Must be run before accessing database')
        elif self._df is None:
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
        columns = list(self.columns.keys())
        if key in columns:
            return self.data[:, columns[key]]
        keys = self.expand_name_to_keys(key, columns)
        if keys is None:
            return None
        ix = [columns[key] for key in keys]
        return self.data[:, ix]

    def get(self, key, df=None):
        if not self.ran:
            raise RuntimeError('Simulation must first be run')
        df = df or environ.notebook
        if df:
            return self.get_from_df(key)
        return self.get_from_a(key)

    def get2(self, *keys, **kwargs):
        if not self.ran:
            raise RuntimeError('Simulation must first be run')
        df = kwargs.get('df', None) or environ.notebook
        if df:
            return self.df[keys]
        ix = [self.columns[key] for key in keys]
        return self.data[:, ix]

    def plot(self, *args, **kwargs):
        return self.df.plot(*args, **kwargs)

    def dump(self, filename=None):
        """Write results to output database"""
        if not self.ran:
            raise RuntimeError('Simulation must first be run')

        logger.info('Opening the output database... ', extra=continued)
        if filename is None:
            filename = self.jobid
        self.db = DatabaseFile(filename, 'w')
        logger.info('done')
        logger.info('Output database: {0!r}'.format(self.db.filename))
        logger.info('Initializing the output database... ', extra=continued)
        self.db.initialize(self.glo_var_names, self.elem_var_names)
        logger.info('done')

        logger.info('Writing data to {0!r}'.format(self.db.filename))
        num_glo_vars = len(self.glo_var_names)
        i, j = 1, 1  + num_glo_vars
        start = time.time()
        for row in self.data:
            end_time = row[0]
            glo_var_vals = row[i:j]
            elem_var_vals = row[j:]
            self.db.save(end_time, glo_var_vals, elem_var_vals)
        dt = time.time() - start
        logger.info('Done writing data {0:.2f}'.format(dt))
        logger.info('Closing the output database... ', extra=continued)
        self.db.close()
        logger.info('done')

    def dumpz(self, filename=None):
        """Write results to output database"""
        if not self.ran:
            raise RuntimeError('Simulation must first be run')
        if filename is None:
            filename = self.jobid
        if not filename.endswith('.npz'):
            filename += '.npz'
        logger.info('Writing data to {0!r}... '.format(filename),
                    extra=continued)
        columns = list(self.columns.keys())
        with open(filename, 'wb') as fh:
            np.savez(fh, columns=columns, data=self.data)
        logger.info('done')

    def expand_name_to_keys(self, key, columns):
        names_and_cols = groupby_names(columns)
        if key not in names_and_cols:
            return None
        sep = COMPONENT_SEP
        keys = ['{0}{1}{2}'.format(key, sep, x) for x in names_and_cols[key]]
        return keys

    @property
    def columns(self):
        if self._columns is not None:
            return self._columns
        columns = ['Time']+self.glo_var_names+self.elem_var_names
        self._columns = OrderedDict([(x,i) for (i,x) in enumerate(columns)])
        return self._columns

    @property
    def glo_var_names(self):
        return ['DTime', 'Step', 'Frame']

    @property
    def elem_var_names(self):
        """Returns the list of element variable names"""
        if self.material is None:
            raise ValueError('Material must first be assigned')
        if self._elem_var_names is not None:
            return self._elem_var_names
        def expand_var_name(name, components):
            sep = COMPONENT_SEP
            return ['{0}{1}{2}'.format(name, sep, x) for x in components]
        xc1 = ['X', 'Y', 'Z']
        xc2 = ['XX', 'YY', 'ZZ', 'XY', 'YZ', 'XZ']
        xc3 = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
        elem_var_names = []
        elem_var_names.extend(expand_var_name('E', xc2))
        elem_var_names.extend(expand_var_name('DE', xc2))
        elem_var_names.extend(expand_var_name('S', xc2))
        elem_var_names.extend(expand_var_name('DS', xc2))
        elem_var_names.extend(expand_var_name('F', xc3))
        elem_var_names.append('Temp')
        if self.material.num_sdv is not None:
            n = self.material.num_sdv
            if self.material.sdv_names is not None:
                assert len(self.material.sdv_names) == n
                elem_var_names.extend(self.material.sdv_names)
            else:
                elem_var_names.extend(expand_var_name('SDV', range(1, n+1)))
        self._elem_var_names = elem_var_names
        return elem_var_names

    def astack(self, E, DE, S, DS, F, T, XV):
        """Concatenates input arrays into a single flattened array"""
        a = [E, DE, S, DS, F, T, XV]
        return np.hstack(tuple([x for x in a if x is not None]))

    def run_istep(self, istep, begin, end, frames, descriptors, components,
                  temp, kappa, data):
        """Run this step, using the previous step as the initial state

        Parameters
        ----------
        istep : int
            The step number to run

        """
        assert istep != 0
        material = self.material
        increment = end - begin

        #---------------------------------------------------------------------- #
        # The following variables have values at
        # [begining, end, current] of step
        #---------------------------------------------------------------------- #
        # Time
        time = np.array([begin, end, begin])

        # Temperature
        temp = np.array((data[1, 37], temp, data[1, 37]))
        dtemp = (temp[1] - temp[0]) / float(frames)

        # Strain and stress states
        previous_strain = data[0,  4:10] * VOIGT
        previous_stress = data[0, 16:22]
        current_strain = np.where(descriptors=='E', components, 0.) * VOIGT
        current_stress = np.where(descriptors=='S', components, 0.)
        strain = np.vstack((previous_strain, current_strain, previous_strain))
        stress = np.vstack((previous_stress, current_stress, previous_stress))

        #---------------------------------------------------------------------- #
        # The following variables have values at
        # [begining, current] of step
        #---------------------------------------------------------------------- #
        previous_statev = data[0, 38:]
        if not len(previous_statev):
            statev = [None, None]
        else:
            statev = np.vstack((previous_statev, previous_statev))
        F0 = data[0, 28:37]
        F = np.vstack((F0, F0))

        # v array is an array of integers that contains the rows and columns of
        # the slice needed in the jacobian subroutine.
        nv = 0
        v = np.zeros(6, dtype=np.int)
        for (i, cij) in enumerate(components):
            descriptor = descriptors[i]
            if descriptor == 'DE':         # -- strain rate
                strain[1, i] = strain[0, i] + cij * VOIGT[i] * increment
            elif descriptor == 'E':        # -- strain
                strain[1, i] = cij * VOIGT[i]
            elif descriptor == 'DS':       # -- stress rate
                stress[1, i] = stress[0, i] + cij * increment
                v[nv] = i
                nv += 1
            elif descriptor == 'S':        # -- stress
                stress[1, i] = cij
                v[nv] = i
                nv += 1
            else:
                raise ValueError('Invalid descriptor {0!r}'.format(descriptor))

        v = v[:nv]
        vx = [x for x in range(6) if x not in v]
        if increment < 1.e-14:
            dedt = np.zeros_like(strain[1])
            dtime = 1.
        else:
            dedt = (strain[1] - strain[0]) / increment
            dtime = (time[1] - time[0]) / float(frames)

        # --- find current value of d: sym(velocity gradient)
        if not nv:
            # strain or strain rate prescribed and the strain rate is constant
            # over the entire step
            if abs(kappa) > 1.e-14:
                d = rate_of_strain_to_rate_of_deformation(dedt, strain[2], kappa)
            elif environ.SQA:
                d = rate_of_strain_to_rate_of_deformation(dedt, strain[2], kappa)
                if not np.allclose(d, dedt):
                    logger.warn('SQA: d != dedt')
            else:
                d = np.array(dedt)

        else:
            # Initial guess for d[v]
            dedt[v] = 0.
            #Jsub = J0[[[x] for x in v], v]
            #work = (stress[1,v] - stress[0,v]) / increment
            #try:
            #    dedt[v] = solve(Jsub,  work)
            #except:
            #    dedt[v] -= lstsq(Jsub, work)[0]

        # Process each frame of the step
        for iframe in range(frames):
            a1 = float(frames - (iframe + 1)) / frames
            a2 = float(iframe + 1) / frames
            strain[2] = a1 * strain[0] + a2 * strain[1]
            pstress = a1 * stress[0] + a2 * stress[1]

            if nv:
                # One or more stresses prescribed
                d = d_from_prescribed_stress(
                    self.material, time[2], dtime, temp[2], dtemp,
                    F[0], F[1], strain[2], dedt, stress[2],
                    statev[0], v, pstress[v])

            # compute the current deformation gradient and strain from
            # previous values and the deformation rate
            F[1], e = update_deformation(F[0], d, dtime, kappa)
            strain[2,v] = e[v]
            if environ.SQA and not np.allclose(strain[2,vx], e[vx]):
                logger.warn('SQA: bad strain on  step {0}'.format(istep))

            state = material.eval(time[2], dtime, temp[2], dtemp,
                                  F[0], F[1],
                                  np.array(strain[2]), d,
                                  np.array(stress[2]), statev[1])
            s, x, ddsdde = state

            dstress = s - stress[2]
            F[0] = F[1]
            time[2] = a1 * time[0] + a2 * time[1]
            temp[2] = a1 * temp[0] + a2 * temp[1]
            stress[2], statev[1] = s, x
            statev[0] = statev[1]

            glo_var_vals = [dtime, istep+1, iframe+1]
            elem_var_vals = self.astack(strain[2]/VOIGT, dedt/VOIGT,
                                        stress[2], dstress, F[1], temp[2], x)
            data[iframe+1, 0] = time[2]
            data[iframe+1, 1:4] = glo_var_vals
            data[iframe+1, 4:] = elem_var_vals

class Step(object):
    def __init__(self, begin, end, frames,
                 descriptors, components, temp, kappa):
        assert len(components) == 6
        assert len(descriptors) == 6
        self.begin = float(begin)
        self.end = float(end)
        self.increment = self.end - self.begin
        if abs(self.increment) > 0.:
            assert end > begin
        self.frames = frames
        self.components = np.asarray(components)
        self.descriptors = np.asarray(descriptors)
        self.temp = temp
        self.kappa = kappa
        self.ran = False
