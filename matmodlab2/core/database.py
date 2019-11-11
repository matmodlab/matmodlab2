import os
import re
import datetime
import numpy as np
from scipy.io.netcdf import NetCDFFile

COMPONENT_SEP = '.'

def read_exodb(filename):
    db = DatabaseFileReader(filename)
    return db.df

def read_npzdb(filename):
    from pandas import DataFrame
    f = np.load(filename, allow_pickle=True)
    return DataFrame(f['data'], columns=f['columns'])

def read_db(filename):
    if filename.endswith('npz'):
        return read_npzdb(filename)
    elif filename.endswith('exo'):
        db = DatabaseFileReader(filename)
        return db.df
    elif filename.endswith(('dat', 'txt')):
        from pandas import read_table
        return read_table(filename, sep='\s+')
    else:
        raise ValueError('Unknown file extension')

def cat(*args):
    return ''.join(str(a).strip() for a in args)

def adjstr(string):
    return '{0:33s}'.format(string)[:33]

def stringify2(a):
    stringified = []
    for row in a:
        try:
            item = ''.join(row)
        except TypeError:
            item = b''.join(row).decode('utf-8')
        stringified.append(item.strip())
    return stringified

def DatabaseFile(jobid, mode='r'):
    """The database file factory method"""
    if mode not in 'rw':
        raise ValueError('Mode must be r or w')

    exts = ('.exo', '.gen', '.base_exo', '.e', '.g', '.npz')
    if jobid.endswith(exts):
        filename = jobid
        jobid = os.path.splitext(os.path.basename(jobid))[0]
    else:
        filename = jobid + exts[0]

    if not filename.endswith(exts):
        raise ValueError('Not a valid ExodusII file extension')

    if mode == 'w':
        if not filename.endswith(exts[:-1]):
            raise ValueError('DatabaseWriter is only for exodus files')
        return DatabaseFileWriter(jobid, filename)
    if filename.endswith('.npz'):
        from pandas import DataFrame
        f = np.load(filename, allow_pickle=True)
        df = DataFrame(f['data'], columns=f['columns'])
    else:
        db = DatabaseFileReader(filename)
        db.fh.close()
        df = db.df
    return df

def groupby_names(names, sep=None, disp=0):
    return _DatabaseFile.groupby_names(names, sep=sep, disp=disp)

class _DatabaseFile(object):
    coordx = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5]) + .5
    coordy = np.array([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]) + .5
    coordz = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]) + .5

    def get_elem_var_names(self):
        return stringify2(self.fh.variables['name_elem_var'].data)

    def get_glo_var_names(self):
        return stringify2(self.fh.variables['name_glo_var'].data)

    def get_qa_records(self):
        return stringify2(self.fh.variables['qa_records'].data[0])

    def get_jobid(self):
        qa_records = self.get_qa_records()
        return qa_records[1]

    @staticmethod
    def groupby_names(names, sep=None, disp=0):
        """Group variables by name

        Returns
        -------
        names_and_cols : dict
            Dictionary with name:column number[s] pairs

        """

        # Mappings from component name to number
        stc = dict(zip(('XX', 'YY', 'ZZ', 'XY', 'YZ', 'XZ'), range(6)))
        tc = dict(zip(('XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ'), range(9)))
        vc = dict(zip(('X', 'Y', 'Z'), range(3)))

        sep = sep or COMPONENT_SEP
        names_and_components = {}
        for (i, name) in enumerate(names):
            try:
                name, x = name.rsplit(sep, 1)
                names_and_components.setdefault(name, []).append((i, x))
            except ValueError:
                names_and_components.setdefault(name, []).append((i, None))

        names_and_cols = {}
        for (name, components) in names_and_components.items():
            if len(components) == 1 and components[0][1] is None:
                names_and_cols[name] = components[0][0]
                continue

            if len(components) == 3:
                components = sorted(components, key=lambda x: vc[x[1]])
            elif len(components) == 6:
                components = sorted(components, key=lambda x: stc[x[1]])
            elif len(components) == 9:
                components = sorted(components, key=lambda x: tc[x[1]])
            else:
                raise TypeError('Unknown components for {0!r}'.format(name))

            if disp:
                names_and_cols[name] = [x[0] for x in components]
            else:
                names_and_cols[name] = [x[1] for x in components]

        return names_and_cols

class DatabaseFileWriter(_DatabaseFile):
    mode = 'w'
    def __init__(self, jobid, filename):
        '''
        Notes
        -----
        The EXOFile class is an interface to the Exodus II api. Its methods
        are named after the analogous method from the Exodus II C bindings,
        minus the prefix 'ex_'.

        '''
        self.jobid = jobid
        self.filename = filename
        self.fh = NetCDFFile(filename, mode='w')

    def initialize(self, glo_var_names, elem_var_names):
        """Initialize the output database

        Parameters
        ----------
        glo_var_names : list of string
        elem_var_names : list of string

        """
        # ------------------------------------------------------------------- #
        # -------------------------------- standard ExodusII dimensioning --- #
        # ------------------------------------------------------------------- #
        self.fh.floating_point_word_size = 4
        self.fh.version = 5.0300002
        self.fh.file_size = 1
        self.fh.api_version = 5.0300002
        self.fh.title = 'Matmodlab material point simulation'

        self.fh.filename = os.path.basename(self.filename)
        self.fh.jobid = self.jobid

        self.fh.createDimension('time_step', None)

        self.fh.createDimension('len_string', 33)
        self.fh.createDimension('len_line', 81)
        self.fh.createDimension('four', 4)

        self.fh.createDimension('num_dim', 3)
        self.fh.createDimension('num_nodes', 8)
        self.fh.createDimension('num_elem', 1)

        # ------------------------------------------------------------------- #
        # ---------------------------------------------------- QA records --- #
        # ------------------------------------------------------------------- #
        now = datetime.datetime.now()
        day = now.strftime("%m/%d/%y")
        hour = now.strftime("%H:%M:%S")
        self.fh.createDimension('num_qa_rec', 1)
        self.fh.createVariable('qa_records', 'c',
                               ('num_qa_rec', 'four', 'len_string'))
        self.fh.variables['qa_records'][0, 0, :] = adjstr('Matmodlab')
        self.fh.variables['qa_records'][0, 1, :] = adjstr(self.jobid)
        self.fh.variables['qa_records'][0, 2, :] = adjstr(day)
        self.fh.variables['qa_records'][0, 3, :] = adjstr(hour)

        # ------------------------------------------------------------------- #
        # ------------------------------------------------- record arrays --- #
        # ------------------------------------------------------------------- #
        self.fh.createVariable('time_whole', 'f', ('time_step',))

        # ------------------------------------------------------------------- #
        # --------------------------------------- element block meta data --- #
        # ------------------------------------------------------------------- #
        # block IDs - standard map
        self.fh.createDimension('num_el_blk', 1)
        self.fh.createVariable('eb_prop1', 'i', ('num_el_blk',))
        self.fh.variables['eb_prop1'][:] = np.arange(1, dtype=np.int32)+1
        self.fh.variables['eb_prop1'].name = 'ID'

        self.fh.createVariable('eb_status', 'i', ('num_el_blk',))
        self.fh.variables['eb_status'][:] = np.ones(1, dtype=int)

        self.fh.createVariable('eb_names', 'c', ('num_el_blk', 'len_string'))
        self.fh.variables['eb_names'][0][:] = adjstr('ElementBlock1')

        # element map
        self.fh.createDimension('num_el_in_blk1', 1)
        self.fh.createDimension('num_nod_per_el1', 8)
        self.fh.createVariable('elem_map1', 'i', ('num_elem',))
        self.fh.variables['elem_map1'][:] = np.arange(1, dtype=np.int32)+1

        # set up the element block connectivity
        dim = ('num_el_in_blk1', 'num_nod_per_el1')
        self.fh.createVariable('connect1', 'i', dim)
        self.fh.variables['connect1'][:] = np.arange(8, dtype=np.int32)+1
        self.fh.variables['connect1'].elem_type = 'HEX'

        # ------------------------------------------------------------------- #
        # -------------------------------------------------- Element data --- #
        # ------------------------------------------------------------------- #
        num_elem_var = len(elem_var_names)
        self.fh.createDimension('num_elem_var', num_elem_var)
        dim = ('num_elem_var', 'len_string')
        self.fh.createVariable('name_elem_var', 'c', dim)
        for (i, name) in enumerate(elem_var_names):
            key = adjstr(name)
            self.fh.variables['name_elem_var'][i, :] = key
            self.fh.createVariable('vals_elem_var{0}eb1'.format(i+1),
                                   'f', ('time_step', 'num_el_in_blk1'))
        self.fh.createVariable('elem_var_tab', 'i', ('num_elem_var',))
        elem_var_tab = np.ones(num_elem_var, dtype=np.int32)
        self.fh.variables['elem_var_tab'][:] = elem_var_tab


        # ------------------------------------------------------------------- #
        # ----------------------------------------------------- Node data --- #
        # ------------------------------------------------------------------- #
        vertices = [self.coordx, self.coordy, self.coordz]
        self.fh.createVariable('coor_names', 'c', ('num_dim', 'len_string'))
        for i in range(3):
            key = 'coord' + 'xyz'[i]
            self.fh.variables['coor_names'][i][:] = adjstr(key)
            self.fh.createVariable(key, 'f', ('num_nodes',))
            self.fh.variables[key][:] = vertices[i]

        self.fh.createDimension('num_nod_var', 3)
        dim = ('num_nod_var', 'len_string')
        self.fh.createVariable('name_nod_var', 'c', dim)
        for i in range(3):
            key = 'displ' + 'xyz'[i]
            self.fh.variables['name_nod_var'][i, :] = adjstr(key)
            self.fh.createVariable('vals_nod_var{0}'.format(i+1), 'f',
                                   ('time_step', 'num_nodes'))

        # ------------------------------------------------------------------- #
        # ---------------------------------------------- Global variables --- #
        # ------------------------------------------------------------------- #
        self.fh.createDimension('num_glo_var', len(glo_var_names))
        dim = ('num_glo_var', 'len_string')
        self.fh.createVariable('name_glo_var', 'c', dim)
        for (i, key) in enumerate(glo_var_names):
            self.fh.variables['name_glo_var'][i, :] = adjstr(key)
        self.fh.createVariable('vals_glo_var', 'f', ('time_step', 'num_glo_var'))

        self.initialized = True
        return

    def update_displ(self, elem_var_vals):
        """Update the node positions based on the deformation gradient"""
        elem_var_names = self.get_elem_var_names()
        names_and_cols = self.groupby_names(elem_var_names, disp=1)
        cols = names_and_cols['F']
        F = np.array(elem_var_vals[cols]).reshape((3,3))
        displ = []
        for i in range(8):
            X = np.array([self.coordx[i], self.coordy[i], self.coordz[i]])
            x = np.dot(F, X)
            displ.append(x)
        displ = np.array(displ).T
        return displ

    def save(self, time, glo_var_vals, elem_var_vals):
        """Save the step information to the database file

        Parameters
        ----------
        time : float
            Time at end of increment
        glo_var_vals : ndarray
            Global variable values, in same order put in to the database
        elem_var_vals : ndarray
            Element variable values, in same order put in to the database

        """

        # write time value
        count = len(self.fh.variables['time_whole'].data)
        self.fh.variables['time_whole'][count] = time
        self.fh.variables['vals_glo_var'][count] = glo_var_vals

        # get node and element fields
        elem_var_names = self.get_elem_var_names()
        if len(elem_var_names) != len(elem_var_vals):
            l1, l2 = len(elem_var_names), len(elem_var_vals)
            message = 'Expected {0} sdv got {1}'.format(l1, l2)
            raise RuntimeError(message)

        for (i, elem_var_val) in enumerate(elem_var_vals):
            key = 'vals_elem_var{0}eb1'.format(i+1)
            self.fh.variables[key][count] = elem_var_val

        nod_var_vals = self.update_displ(elem_var_vals)
        assert nod_var_vals.shape == (3, 8)
        for (i, nod_var_val) in enumerate(nod_var_vals):
            key = 'vals_nod_var{0}'.format(i+1)
            self.fh.variables[key][count] = nod_var_val

        return

    def close(self):
        """Close the database file"""
        self.fh.close()

class DatabaseFileReader(_DatabaseFile):
    def __init__(self, filename):
        self.filename = filename
        if not os.path.isfile(self.filename):
            raise OSError('{0}: no such file'.format(self.filename))
        self.fh = NetCDFFile(self.filename, 'r')
        self.df = self.read_db()
        self.jobid = self.get_jobid()
        self.fh.close()

    def read_db(self, blk_num=1, elem_num=1):
        """Read the ExodusII database file filename.

        Parameters
        ----------
        blk_num : int
            The element block number to read
        elem_num : int
            The element number to read

        Returns
        -------
        df : pandas.DataFrame
            A DataFrame with names of variables as columns

        Notes
        -----
        This is a single element reader

        """
        from pandas import DataFrame
        fh = self.fh

        # global/element vars and mapping
        num_glo_var = fh.dimensions.get('num_glo_var', 0)
        if num_glo_var:
            name_glo_var = self.get_glo_var_names()
            gmap = dict(zip(name_glo_var, range(len(name_glo_var))))

        name_elem_var = self.get_elem_var_names()
        emap = dict(zip(name_elem_var, range(len(name_elem_var))))

        # retrieve the data from the database
        head = ['Time']
        if num_glo_var:
            head.extend([key for key in name_glo_var])
        head.extend([key for key in name_elem_var])

        data = []
        times = fh.variables['time_whole'].data.flatten()
        for (i, time) in enumerate(times):
            row = [time]
            if num_glo_var:
                vals_glo_var = fh.variables['vals_glo_var'].data[i]
                for var in name_glo_var:
                    var_num = gmap[var]
                    row.append(vals_glo_var[var_num])
            for var in name_elem_var:
                var_num = emap[var]+1
                name = 'vals_elem_var{0}eb1'.format(var_num, blk_num)
                row.append(fh.variables[name].data[i, elem_num-1])
            data.append(row)

        data = np.asarray(data)
        if len(head) != data.shape[1]:
            raise ValueError('inconsistent data')

        return DataFrame(data, columns=head)
