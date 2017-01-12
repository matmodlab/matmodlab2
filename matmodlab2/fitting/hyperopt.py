from numpy import *
import warnings
from itertools import permutations
from scipy.optimize import leastsq

UNIAXIAL_DATA = 'Uniaxial Data'
BIAXIAL_DATA = 'Biaxial Data'
SHEAR_DATA = 'Shear Data'

def IJ(N, i2dep=1):
    ij = []
    for n in range(N+2):
        ij.extend([(i,j) for i in range(n)[::-1] for j in range(n) if i+j==n-1])
    ij = ij[1:]
    if not i2dep:
        ij = [(i,j) for (i,j) in ij if not j]
    return ij

class OptimizeError(Exception):
    pass

class HyperelasticOptimizer:

    def __init__(self, dtype, strain, stress, order, i2dep):

        self.IJ = IJ(order, i2dep=i2dep)
        np = len(self.IJ)
        if np > len(strain):
            raise OptimizeError('Order of fit too high for data')

        self.order = order
        self.i2dep = bool(i2dep)
        self.dtype = dtype
        self.strain = strain
        self.stress = stress

        xdata, f = _data_type_helpers(dtype, strain)
        res = self._single_opt_p(xdata, f)
        (popt, pcov, infodict, errmsg, error) = res

        self.popt = popt
        self.pcov = pcov
        self.infodict = infodict
        self.errmsg = errmsg
        self.error = error

    def _single_opt_p(self, xdata, f):
        """Find the optimized parameters for xdata and ydata"""

        ydata = self.stress
        order = self.order
        i2dep = self.i2dep
        func = _general_function
        p0 = ones(len(self.IJ))
        kw = {'order': order, 'i2dep': i2dep}
        args = (kw, xdata, ydata, f)
        res = leastsq(func, p0, args=args, full_output=1)
        (popt, pcov, infodict, errmsg, ier) = res

        if ier not in [1, 2, 3, 4]:
            msg = "Optimal parameters not found: " + errmsg
            raise RuntimeError(msg)

        warn_cov = False
        if pcov is None:
            # indeterminate covariance
            pcov = zeros((len(popt), len(popt)), dtype=float)
            pcov.fill(inf)
            warn_cov = True
        else:
            if len(ydata) > len(p0):
                s_sq = (asarray(func(popt, *args))**2).sum()
                s_sq /= (len(ydata) - len(p0))
                pcov = pcov * s_sq
            else:
                pcov.fill(inf)
                warn_cov = True

        if warn_cov:
            warnings.warn('Covariance of the parameters could not be estimated')

        yp = f(xdata, *popt, **kw)
        err = sqrt(mean((yp - ydata) ** 2)) / abs(average(ydata))

        # check if Drucker's stability criterion is satisfied
        #dy = diff(yp)
        #dx = diff(xdata)
        #if any(dy * dx < -1e-12):
        #    raise OptimizeError('Drucker stability criterion violated')

        return popt, pcov, infodict, errmsg, err

    def eval(self, **kw):
        overlay = kw.pop('overlay', None)
        if overlay is not None:
            dtype = overlay.dtype
            kw['order'] = overlay.order
            kw['i2dep'] = overlay.i2dep
            p = overlay.popt
        else:
            dtype = kw.pop('dtype', self.dtype)
            kw['order'] = kw.pop('order', self.order)
            kw['i2dep'] = kw.pop('i2dep', self.i2dep)
            p = kw.pop('p', self.popt)
        strain = kw.pop('strain', self.strain)
        xdata, f = _data_type_helpers(dtype, strain)
        return f(xdata, *p, **kw)

    def mp_plot(self, overlay=None, filename=None, show=True):
        import matplotlib.pyplot as plt
        plt.scatter(self.strain, self.stress, label='{0}, data'.format(self.dtype))
        ee = linspace(self.strain.min(), self.strain.max(), 100)
        ss = self.eval(strain=ee)
        plt.plot(ee, ss, label='{0}, fit'.format(self.dtype))
        if overlay is not None:
            try:
                overlay + []
            except (TypeError, ValueError):
                overlay = [overlay]
            for fit in overlay:
                ss = self.eval(strain=ee, p=fit.popt, order=fit.order,
                               dtype=fit.dtype, i2dep=fit.i2dep)
                plt.plot(ee, ss, label='{0}, fit'.format(fit.dtype))
        plt.legend(loc='best')
        if filename is not None:
            plt.savefigure(filename)
            show = False
        if show:
            plt.show()

    def bp_plot(self, strain=None, overlay=None, points=True, **kwargs):
        import bokeh.plotting as bp
        TOOLS = 'resize,pan,wheel_zoom,box_zoom,reset,save'
        plot = bp.figure(tools=TOOLS, **kwargs)

        if points:
            plot.circle(self.strain, self.stress,
                        legend='{0}, data'.format(self.dtype))
        if strain is None:
            strain = linspace(self.strain.min(), self.strain.max(), 100)
        ss = self.eval(strain=strain)
        plot.line(strain, ss, legend='{0}, fit'.format(self.dtype))
        if overlay is not None:
            try:
                overlay + []
            except TypeError:
                overlay = [overlay]
            for fit in overlay:
                ss = self.eval(strain=strain, p=fit.popt, order=fit.order,
                               dtype=fit.dtype, i2dep=fit.i2dep)
                plot.line(strain, ss, color='red',
                          legend='{0}, fit'.format(fit.dtype))
        return plot

    def todict(self):
        p = dict([('C{0}{1}'.format(i,j), self.popt[k])
                  for k, (i,j) in enumerate(self.IJ)])
        return p

    def summary(self):
        p = ['C{0}{1}={2:.3f}'.format(i,j,self.popt[k])
             for k, (i,j) in enumerate(self.IJ)]
        s = """\
            Data type: {0}
Number of data points: {1}
     Polynomial order: {2}
        I2 dependence: {3}
           Parameters: {4}
                Error: {5}
        """.format(self.dtype.split()[0], self.strain.shape[0], self.order,
                   self.i2dep, ', '.join(p), self.error)
        return s

def _hyperelastic(xdata, *p, **kw):
    """Evaluate the hyper elastic model

    Parameters
    ----------
    xdata : array_like (3,)
        The principal stretches
    p : tuple of real
        The hyperelastic coefficients
    kw : dict
        Optional keyword arguments

    Returns
    -------
    nominal_stress : ndarray
        The nominal stress

    """
    order = kw.get('order', 2)
    i2dep = kw.get('i2dep', 1)

    ij = IJ(order, i2dep=i2dep)
    if len(ij) != len(p):
        raise ValueError('inconsistent parameter length')

    # helper quantities
    nominal_stress = zeros_like(xdata)
    xdata = asarray(xdata)
    I = ones(3)
    for (ix, x) in enumerate(xdata):
        I1 = sum(x)
        I2 = (I1 ** 2 - sum(x * x)) / 2.
        xi = 1. / x

        A = zeros(2)
        for k in range(len(ij)):
            i, j = ij[k]
            if i - 1 >= 0:
                A[0] += p[k] * i * (I1 - 3) ** (i - 1) * (I2 - 3) ** (j)
            if j - 1 >= 0:
                A[1] += p[k] * j * (I1 - 3) ** (i) * (I2 - 3) ** (j - 1)

        B = zeros((2,3))
        B[0] = I - I1 * xi / 3.
        B[1] = I1 * I - xi - 2. * I2 * xi / 3.

        pk2_stress = sum(A[j] * B[j] for j in [0, 1])

        # Nominal stress
        nominal_stress[ix] = sqrt(x) * pk2_stress

    return nominal_stress

def _uniaxial_func(xdata, *p, **kw):
    """Uniaxial stress"""
    s = _hyperelastic(xdata, *p, **kw)
    return s[:,0] - s[:,-1]

def _biaxial_func(xdata, *p, **kw):
    """Biaxial stress"""
    s = _hyperelastic(xdata, *p, **kw)
    return s[:,0]

def _shear_func(xdata, *p, **kw):
    """Shear stress"""
    s = _hyperelastic(xdata, *p, **kw)
    return (s[:,0] - s[:,-1]) / 2.

def _data_type_helpers(dtype, strain):
    """Returns the deformation and associated stress function for the data type"""
    stretch = asarray(strain) + 1
    if dtype == UNIAXIAL_DATA:
        C = array([[lam, 1./sqrt(lam), 1./sqrt(lam)] for lam in stretch])
        return C, _uniaxial_func
    elif dtype == BIAXIAL_DATA:
        C = array([[lam, lam, 1./lam**2] for lam in stretch])
        return C, _biaxial_func
    elif dtype == SHEAR_DATA:
        C = array([[lam, 1./lam, 1.] for lam in stretch])
        return C, _shear_func
    raise ValueError('unrecogized data type')

def _general_function(params, options, xdata, ydata, function):
    return function(xdata, *params, **options) - ydata

def hyperopt(dtype, strain, stress, order=None, i2dep=None):
    strain = asarray(strain)
    stress = asarray(stress)
    if i2dep is None:
        opt = {}
        for i2dep in (0, 1):
            try:
                p = hyperopt(dtype, strain, stress, order, i2dep)
            except OptimizeError:
                continue
            opt[i2dep] = p
        if not opt:
            raise OptimizeError('unable to determine optimal parameters')
        i2dep = sorted(opt, key=lambda x: opt[x].error)[0]
        opt = opt[i2dep]

    elif order is None:
        # Find the order that gives the smallest error
        opt = {}
        for i in range(1, 6):
            order = i
            try:
                p = HyperelasticOptimizer(dtype, strain, stress, order, i2dep)
            except OptimizeError:
                break
            opt[order] = p
        if not opt:
            raise OptimizeError('unable to determine optimal parameters')
        order = sorted(opt, key=lambda x: opt[x].error)[0]
        opt = opt[order]

    else:
        np = len(IJ(order, i2dep=i2dep))
        if np > strain.shape[0]:
            raise OptimizeError('Order of fit too high for data')
        opt = HyperelasticOptimizer(dtype, strain, stress, order, i2dep)

    return opt

def hyperopt2(*args, **kwargs):
    nargs = len(args)
    if nargs % 3:
        raise OptimizeError('input data required to be triplets')
    maxn = kwargs.get('maxn', 5)

    # gather many fits
    d = []
    for i in range(nargs)[::3]:
        dtype, e, s = args[i:i+3]
        for i2 in [True, False]:
            for o in range(1, maxn):
                try:
                    p = hyperopt(dtype, e, s, order=o, i2dep=i2)
                except OptimizeError:
                    continue
                d.append(p)

    def err(f1, f2):
        y1 = f1.eval()
        y2 = f1.eval(overlay=f2)
        return sqrt(mean((y1-y2)**2))

    # get the relative error between fits
    fopt = None
    error = 1e45
    for (f1, f2) in permutations(d, r=2):
        if f1.dtype == f2.dtype:
            continue
        e = err(f1, f2)
        if e < error:
            error = e
            fopt = f2

    fopt.error2 = error / average(abs(fopt.stress))
    fopt.dtype2 = 'Multi'
    return fopt

if __name__ == '__main__':
    from pandas import read_excel
    f = '../examples/Treloar_hyperelastic_data.xlsx'
    O = 2
    I2dep = 1

    df1 = read_excel(f, sheetname='Pure Shear')
    s1 = df1['Engineering Stress (MPa)']
    e1 = df1['Engineering Strain']
    p1 = hyperopt(SHEAR_DATA, e1, s1)
    p1.mp_plot()

    df2 = read_excel(f, sheetname='Uniaxial')
    s2 = df2['Engineering Stress (MPa)']
    e2 = df2['Engineering Strain']
    p2 = hyperopt(UNIAXIAL_DATA, e2, s2)
    p2.mp_plot()

    p1.mp_plot(overlay=p2)
