from __future__ import print_function
import os
import re
import sys
import logging
import numpy as np
from itertools import cycle

try:
    import pandas
except ImportError:
    pandas = None
try:
    import scipy.optimize as sciopt
except ImportError:
    sciopt = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import bokeh.plotting as bp
except ImportError:
    bp = None

__all__ = [
    "MasterCurve",
    "CurveFitter",
    "mc_init_notebook",
    "MODIFIED_POWER",
    "POWER",
    "PRONY",
    "POLYNOMIAL",
    "COBYLA",
    "POWELL",
    "FMIN",
]

MODIFIED_POWER = "Modified Power"
POWER = "Power"
PRONY = "Prony"
POLYNOMIAL = "Polynomial"

COBYLA = "Cobyla"
POWELL = "Powell"
FMIN = "Fmin"

PHONY = 123456.789
EPS = np.finfo(float).eps
BASE = 10.0


class Environment:
    pass


environ = Environment()
environ.notebook = 0


def _loadcsv(filename):
    """Load the csv file written out by MasterCurve.dump"""

    class CSVData:
        data = {}

        def get(self, temp):
            return self.data[temp]

        def __iter__(self):
            return iter(self.data.items())

    dtype = np.float64
    array = np.array
    cstack = np.column_stack

    assert os.path.isfile(filename)
    lines = open(filename).readlines()

    for (i, line) in enumerate(lines):
        if re.search("Version", line):
            version = float(lines[i + 1])
            assert version > 1.0
        if re.search("Curve Fitter", line):
            fitter = lines[i + 1]  # noqa
        if re.search("Fit Error", line):
            fiterr = float(lines[i + 1])  # noqa
        if re.search("WLF", line):
            wlf_coeffs = [float(x) for x in lines[i + 1].split(",")]
            assert len(wlf_coeffs) == 2

        if re.search("Data", line):
            j = i + 1

        if re.search("Master Curve", line):
            k = i + 1

    d = CSVData()

    desc = lines[j].split(",")
    temps = array([float(x) for x in desc[2:]])
    data = array(
        [
            [None if not x.split() else float(x) for x in y.split(",")]
            for y in lines[j + 1 : k - 1]
        ]
    )
    d.master = array([[float(x) for x in y.split(",")] for y in lines[k + 1 :]])

    d.toat = data[:, 0]
    d.logtoat = data[:, 1]

    d.raw_master = cstack((d.toat, d.logtoat, array(data[:, 2], dtype=dtype)))
    d.ref_temp = temps[0]

    d.temps = temps[1:]
    for (i, temp) in enumerate(temps[1:]):
        td = data[:, i + 3]
        idx = [j for (j, x) in enumerate(td) if x is not None]
        a = array(td[idx], dtype=dtype)
        if len(a) == 0:
            continue
        a = cstack((d.toat[idx], d.logtoat[idx], a))
        d.data[temp] = a

    return d


def log(x, base=BASE):
    e = 2.718281828459045
    if abs(base - 10.0) < EPS:
        return np.log10(x)
    elif abs(base - 2.0) < EPS:
        return np.log2(x)
    elif abs(base - e) < EPS:
        return np.log(x)
    else:
        return np.log(x) / np.log(BASE)


def interp1d(xy, x, findx=False, clip=False):
    """Wrapper around numpy's interp"""
    xp = xy[:, 0]
    yp = xy[:, 1]
    if findx:
        xp, yp = yp, xp
    # xd = np.diff(xp)
    if np.allclose(-1, np.sign(np.diff(xp))):
        # descending curve, reverse it
        xp, yp = xp[::-1], yp[::-1]
    if not clip:
        return np.interp(x, xp, yp)

    yval = np.interp(x, xp, yp, left=PHONY, right=PHONY)
    if abs(yval - PHONY) < 1.0e-12:
        return None
    return yval


def islist(a):
    return isinstance(a, (list, tuple, np.ndarray))


def multidim(a):
    try:
        if islist(a) and islist(a[0]):
            return True
    except IndexError:
        return False


def joinn(l, sep=",", num=False, end="\n", ffmt=".18f"):
    if num:
        realfmt = lambda r: "{0:{1}}".format(float(r), ffmt)
        l = [realfmt(x) if x is not None else "" for x in l]
    if not multidim(l):
        line = sep.join(l)
    else:
        line = "\n".join(sep.join(s) for s in l)
    return line + end


def bounding_box(curve):
    """Determine the box that bounds curve

    Parameters
    ----------
    curve : ndarray
        curve[i, 0] is the x coordinate of the ith data point
        curve[i, 1] is the y coordinate of the ith data point

    Returns
    -------
    box : ndarray
        box[0, i] is xm[in,ax]
        box[1, i] is ym[in,ax]

    """
    if curve[0, 1] > curve[-1, 1]:
        # y values are decreasing from left to right
        xmin, ymin = curve[-1, 0], curve[-1, 1]
        xmax, ymax = curve[0, 0], curve[0, 1]
    else:
        xmin, ymin = curve[0, 0], curve[0, 1]
        xmax, ymax = curve[-1, 0], curve[-1, 1]
    return np.array([[xmin, ymin], [xmax, ymax]])


def area(x, y, yaxis=False):
    if not yaxis:
        return np.trapz(y, x)
    return np.trapz(x, y)


COLORS = [
    "Blue",
    "Red",
    "Purple",
    "Green",
    "Orange",
    "HotPink",
    "Cyan",
    "Magenta",
    "Chocolate",
    "Yellow",
    "Black",
    "DodgerBlue",
    "DarkRed",
    "DarkViolet",
    "DarkGreen",
    "OrangeRed",
    "Teal",
    "DarkSlateGray",
    "RoyalBlue",
    "Crimson",
    "SeaGreen",
    "Plum",
    "DarkGoldenRod",
    "MidnightBlue",
    "DarkOliveGreen",
    "DarkMagenta",
    "DarkOrchid",
    "DarkTurquoise",
    "Lime",
    "Turquoise",
    "DarkCyan",
    "Maroon",
]


def gen_colors(keys):
    colors = {}
    c = cycle(COLORS)
    for key in keys:
        colors[key] = next(c).lower()
    return colors


def aslist(item):
    if item is None:
        return []
    if not isinstance(item, (list, tuple, np.ndarray)):
        item = [item]
    return [x for x in item]


class MasterCurve(object):
    fiterr = None

    def __init__(
        self,
        txy,
        ref_temp=75.0,
        apply_log=False,
        xfac=1.0,
        yfac=1.0,
        skip_temps=None,
        wlf_coeffs=None,
        xvar="Time",
        xunits="min",
        yvar="Er",
        yunits="psi",
        optimizer=FMIN,
        fitter=PRONY,
        optwlf=sciopt is not None,
        **kwargs
    ):
        """Initialize the master curve object

        Parameters
        ----------
        txy : array_like (n, 3)
            txy[i] is the ith [Temp, X, Y]
        ref_temp : real
            Reference temperature
        optimizer : str [FMIN]
            The scipy.optimize optimizing procedure
        optwlf : bool [True]
            Optimize the WLF parameters
        fitter : str [prony]
            Name of CurveFitter
        skip_temps : list [None]
            Temperatures to skip
        wlf_coeffs : list [None]
            Initial guesses for C1 and C2

        kwargs : dict
            keywords [optional] to pass to fitter

        """
        if pandas is None:
            raise RuntimeError("master curve fitting requires pandas")

        columns = ("Temp", "X", "Log[X]", "Y")
        txy = np.asarray(txy)
        txy[:, -1] *= yfac
        if apply_log:
            txy[:, 1] *= xfac
            logx = log(txy[:, 1])
            txy = np.insert(txy, 2, logx, axis=1)
        else:
            x = (BASE ** txy[:, 1]) * xfac
            logx = log(x)
            txy = np.insert(txy, 1, x, axis=1)
            txy = np.insert(txy, 2, logx, axis=1)

        self.ref_temp = ref_temp
        self.skip_temps = aslist(skip_temps)
        self.df = pandas.DataFrame(txy, columns=columns)

        self.wlf_coeffs = wlf_coeffs
        self.optwlf = optwlf
        self.optimizer = optimizer
        self.kwds = dict(**kwargs)
        cf = CurveFitter(fitter)
        self._cf = {0: cf(**kwargs)}

        self.xvar = xvar
        self.yvar = yvar
        self.xunits = xvar
        self.yunits = yunits

        self._fit = 0

    @property
    def cf(self):
        return self._cf.get(1, self._cf[0])

    @cf.setter
    def cf(self, item):
        if item is not None:
            self._cf[1] = item

    def fit(
        self,
        wlf_coeffs=None,
        skip_temps=None,
        ref_temp=None,
        optimize=None,
        fitter=None,
        **kwargs
    ):

        skip_temps = aslist(skip_temps)
        skip_temps.extend(self.skip_temps)

        if fitter is not None:
            fitter = CurveFitter(fitter)(**kwargs)
        self.cf = fitter

        # make skip temps a list, if not already
        df = self.df.copy()
        for temp in skip_temps:
            df = df[~(np.abs(df["Temp"] - temp) < EPS)]

        ref_temp = self.ref_temp if ref_temp is None else ref_temp
        wlf_coeffs = self.wlf_coeffs if wlf_coeffs is None else wlf_coeffs

        if not any(np.abs(df["Temp"] - ref_temp) < EPS):
            raise ValueError(
                "Reference temperature {0} not " "found in data".format(ref_temp)
            )

        wlf_opt = self.opt_wlf_coeffs(df, ref_temp, wlf_coeffs, optimize)
        self.dfm = self.shift_data(df, ref_temp, wlf_opt)
        self.mc_fit = self.fit_shifted_data(self.dfm)
        self.wlf_opt = wlf_opt
        self._fit = 1

    def opt_wlf_coeffs(self, df, ref_temp, wlf_coeffs, opt):
        """Generate the optimized master curve"""

        temps = np.unique(df["Temp"])
        if len(temps) == 1:
            # only one data set
            return np.zeros(2)

        if wlf_coeffs is None:
            wlf_coeffs = self.get_wlf_coeffs(df, ref_temp)

        if opt is None:
            opt = self.optwlf

        if not opt:
            return wlf_coeffs

        def func(xopt, *args):
            """Objective function returning the area between the fitted curve
            and shifted data

            """
            if np.any(np.abs(xopt[1] + temps - ref_temp) < EPS):
                self.fiterr = 1000.0
                return self.fiterr

            df1 = self.shift_data(df.copy(), ref_temp, xopt)
            fit = self.fit_shifted_data(df1)

            # determine error between fitted curve and master curve
            yvals = []
            for logx in df1["Log[X/aT]"]:
                yvals.append(self.cf.eval(fit, logx))
            yvals = np.asarray(yvals)
            error = np.sqrt(np.mean((yvals - df1["Y"]) ** 2))
            self.fiterr = error  # / area(data[:,0],data[:,1])
            return self.fiterr

        if self.optimizer == COBYLA:
            cons = [
                lambda x: 1 if abs(x[1] + temp - ref_temp) > EPS else -1
                for temp in temps
            ]
            wlf_coeffs = sciopt.fmin_cobyla(func, wlf_coeffs, cons, disp=0)
        elif self.optimizer == POWELL:
            wlf_coeffs = sciopt.fmin_powell(func, wlf_coeffs, disp=0)
        else:
            wlf_coeffs = sciopt.fmin(func, wlf_coeffs, disp=0)

        return wlf_coeffs

    def shift_data(self, df, ref_temp, wlf):
        """Compute the master curve for data series"""
        # reference temperature curve
        def f(x):
            temp = np.asarray(x["Temp"])[0]
            shift = -wlf[0] * (temp - ref_temp) / (wlf[1] + temp - ref_temp)
            x["Log[X/aT]"] = np.asarray(x["Log[X]"]) - shift
            return x

        df = df.groupby("Temp").apply(f)
        return df

    def fit_shifted_data(self, df):
        """Fit the master curve"""
        t = np.asarray(df["Log[X/aT]"])
        d = np.asarray(df["Y"])
        return self.cf.fit_points(t, d)

    @staticmethod
    def x_shift(df1, df2):
        """

        Parameters
        ----------
        df1 : ndarray
            Base curve to shift to consisting of list of x,y pairs
        df2 : ndarray
            Curve to shift consisting of list of x,y pairs

         Returns
         -------
         shift : real
             A scalar shift value

        Notes
        -----
        Single curves must be monotonically increasing or decreasing. Composite
        curve can be more complex and x values should generally be increasing
        (noise is okay)

        shift value returned is such that x points should be shifted by
        subtracting the shift value

        """
        ref_curve = np.asarray(df1[["Log[X]", "Y"]])
        curve = np.asarray(df2[["Log[X]", "Y"]])

        ref_bnds = bounding_box(ref_curve)
        crv_bnds = bounding_box(curve)

        if crv_bnds[1, 1] > ref_bnds[1, 1]:
            # y values for curve larger than ref_curve, check for overlap
            if crv_bnds[0, 1] < ref_bnds[1, 1]:
                ypt = (ref_bnds[1, 1] + crv_bnds[0, 1]) / 2.0
                x_crv = interp1d(curve, ypt, findx=True)
                x_ref = interp1d(ref_curve, ypt, findx=True)
            else:
                # no overlap
                x_ref, x_crv = ref_curve[0, 0], curve[-1, 0]
        else:
            # y values for ref_curve larger than curve, check for overlap
            if ref_bnds[0, 1] < crv_bnds[1, 1]:
                ypt = (ref_bnds[0, 1] + crv_bnds[1, 1]) / 2.0
                x_crv = interp1d(curve, ypt, findx=True)
                x_ref = interp1d(ref_curve, ypt, findx=True)
            else:
                # no overlap
                x_ref, x_crv = ref_curve[-1, 0], curve[0, 0]

        return -(x_ref - x_crv)

    def get_wlf_coeffs(self, df, T0):
        """Defines the WLF shift

        Notes
        -----
        returns a tuple containing best fits for C1 and C2:

         log(aT) = -C1(T-ref_temp)/(C2+T-ref_temp)

         The coefficients are determined by the least squares fit

              x = (A^TA)^-1 A^Tb  (T is matrix transpose)

         where A is nx2 matrix (n is number of T and logaT pairs) of rows or
         [(T-Tr) logaT] for each T,logaT pair b is a nx1 vector where each row is
         -logaT*(T-Tr) for each pair x is a 2x1 vector of C1 and C2

        """
        # shift the data to the reference curve
        dg = df.groupby("Temp")
        rc = dg.get_group(T0)

        # Shift each data set.  xs is [T, logaT]
        xs = np.array([[g, self.x_shift(rc, df1)] for (g, df1) in dg])
        if all([abs(x) < 1.0e-6 for x in xs[:, 1]]):
            logging.warn(
                "No shift found, consider specifying " "initial WLF coefficients"
            )

        # Setup A Matrix
        A = np.zeros((xs.shape[0], 2))
        A[:, 0] = xs[:, 0] - T0
        A[:, 1] = xs[:, 1]

        # Setup b Vector
        b = -A[:, 0] * A[:, 1]

        # Calculate WLF Coefficients
        ATA = np.dot(A.T, A)
        ATb = np.dot(A.T, b)
        try:
            wlf_coeffs = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            logging.warn("Using least squares to find wlf coefficients")
            wlf_coeffs = np.linalg.lstsq(ATA, ATb)[0]

        return wlf_coeffs

    def plot(self, **kwargs):
        if environ.notebook == 2:
            p = self._bp_plot(**kwargs)
            bp.show(p)
            return p
        return self._mp_plot(**kwargs)

    def _bp_plot(self, raw=False, show_fit=False):
        if bp is None:
            raise ImportError("bokeh")

        if raw:
            x_label = "Log[{0}] ({1})".format(self.xvar, self.xunits)
        else:
            x_label = "Log[{0}/aT] ({1})".format(self.xvar, self.xunits)
        y_label = "{0} ({1})".format(self.yvar, self.yunits)
        plot = bp.figure(x_axis_label=x_label, y_axis_label=y_label)
        if raw:
            plot.title = "Raw, unshifted data"
            dg = self.df.groupby("Temp")
            colors = gen_colors([str(temp) for temp in dg.groups.keys()])
            for (temp, df) in dg:
                plot.scatter(
                    df["Log[X]"],
                    df["Y"],
                    legend="{0}".format(temp),
                    color=colors[str(temp)],
                )
            return plot

        if not self._fit:
            self.fit()

        dg = self.dfm.groupby("Temp")
        colors = gen_colors([str(temp) for temp in dg.groups.keys()])
        plot.title = "Data shift: {0}".format(self.cf.name)
        for (temp, df) in dg:
            plot.scatter(
                df["Log[X/aT]"],
                df["Y"],
                legend="{0}".format(temp),
                color=colors[str(temp)],
            )
        if show_fit:
            xp, yp = self._mc_points()
            plot.line(xp, yp, color="black", line_width=1.5, legend="Master Curve Fit")
        return plot

    def _mp_plot(
        self, raw=False, show_fit=False, filename=None, legend_loc="best", legend_ncol=1
    ):
        """Plot the given data or shifted data and/or the fit"""
        if plt is None:
            raise ImportError("matplotlib")

        plt.clf()
        plt.cla()
        xl, xu = self.xvar, self.xunits
        yl, yu = self.yvar, self.yunits
        if raw:
            plt.xlabel(r"$\log\left(%s\right)$ (%s)" % (xl, xu))
        else:
            plt.xlabel(r"$\log\left(\frac{%s}{aT}\right)$ (%s)" % (xl, xu))
        ylabel = r"$%s$ (%s)" % (yl, yu)
        plt.ylabel(ylabel)

        if raw:
            dg = self.df.groupby("Temp")
            colors = gen_colors([str(temp) for temp in dg.groups.keys()])
            for (temp, df) in dg:
                plt.scatter(
                    df["Log[X]"],
                    df["Y"],
                    label="{0}".format(temp),
                    color=colors[str(temp)],
                )

        else:
            if not self._fit:
                self.fit()
            dg = self.dfm.groupby("Temp")
            colors = gen_colors([str(temp) for temp in dg.groups.keys()])
            for (temp, df) in dg:
                plt.scatter(
                    df["Log[X/aT]"],
                    df["Y"],
                    label="{0}".format(temp),
                    color=colors[str(temp)],
                )

            if show_fit:
                xp, yp = self._mc_points()
                plt.plot(xp, yp, "k-", lw=1.5)

        if legend_loc is not None:
            kwds = {"loc": legend_loc, "ncol": legend_ncol, "scatterpoints": 1}
            if show_fit:
                kwds["title"] = "${0}$ = {1}".format(self.yvar, self.cf.plot_label)
                kwds["fancybox"] = True
            plt.legend(**kwds)

        if environ.notebook:
            plt.show()
        elif filename is None:
            plt.show()
        if filename is not None:
            plt.savefig(filename, transparent=True)

        return

    def to_csv(self, filename):
        """Dump data to a file

        Parameters
        ----------
        filename : str
            file path

        Notes
        -----
        Writes to each row of filename

        """
        if not self._fit:
            self.fit()

        fh = open(filename, "w")
        fh.write("mcgen\nVersion\n1.2\n")
        fh.write("Curve Fitter\n{0}\n".format(self.cf.name))
        fh.write("Curve Fitting Information\n")
        # write out the fit info
        line = self.cf.dump_info(self.mc_fit, delimiter=",")
        fh.write(line)
        try:
            fh.write("Fit Error\n{0:.18f}\n".format(self.fiterr))
        except ValueError:
            pass
        fh.write(joinn(["WLF C1", "WLF C2"], sep=","))
        fh.write(joinn([self.wlf_opt[0], self.wlf_opt[1]], sep=",", num=True))

        fh.write("Data\n")
        self.dfm.to_csv(fh, float_format="%.18f", index=False)
        fh.close()

    def to_excel(self, filename):
        if not self._fit:
            self.fit()

        def cell(i, j):
            return "{0}{1}".format(chr(j + ord("A")), i + 1)

        writer = pandas.ExcelWriter(filename)
        worksheet = writer.book.create_sheet()
        worksheet.title = "mcgen Meta"
        worksheet[cell(0, 0)] = "mcgen Version"
        worksheet[cell(0, 1)] = "1.2"

        worksheet[cell(1, 0)] = "Curve Fitter"
        worksheet[cell(1, 0)] = self.cf.name

        worksheet[cell(2, 0)] = "Curve Fitting Information"
        lines = self.cf.dump_info(self.mc_fit, delimiter=",")
        for (i, line) in enumerate(lines.split("\n")):
            for (j, item) in enumerate(line.split(",")):
                worksheet[cell(3 + i, j)] = "{0}".format(item.strip())
        n = 3 + i
        try:
            worksheet[cell(n, 0)] = "Fit Error"
            worksheet[cell(n, 1)] = "{0:.18f}\n".format(self.fiterr)
            n += 1
        except ValueError:
            pass
        worksheet[cell(n, 0)] = "WLF C1"
        worksheet[cell(n, 1)] = "WLF C1"
        worksheet[cell(n + 1, 0)] = "{0}".format(self.wlf_opt[0])
        worksheet[cell(n + 1, 1)] = "{0}".format(self.wlf_opt[1])

        self.dfm.to_excel(writer, sheet_name="mcgen Data", index=False)
        writer.save()
        return

    def _mc_points(self, n=50):
        xmin = np.amin(self.dfm["Log[X/aT]"])
        xmax = np.amax(self.dfm["Log[X/aT]"])
        xvals = np.linspace(xmin, xmax, n)
        yvals = np.array([self.cf.eval(self.mc_fit, x) for x in xvals])
        return xvals, yvals

    @classmethod
    def Import(cls, filename, **kwargs):
        root, ext = os.path.splitext(filename)
        if ext.lower() == ".csv":
            return ReadCSV(filename, **kwargs)
        raise TypeError("Unexpected file extension {0!r}".format(ext))

    def Export(self, filename):
        root, ext = os.path.splitext(filename)
        if ext.lower() == ".csv":
            return self.to_csv(filename)
        elif ext.lower() == ".xlsx":
            return self.to_excel(filename)
        raise TypeError("Unexpected file extension {0!r}".format(ext))

    @property
    def description(self):
        if not self._fit:
            self.fit()

        fh = sys.stdout
        fh.write("mcgen Version 1.2\n")
        fh.write("Curve Fitter: {0}\n".format(self.cf.name))
        fh.write("Curve Fitting Information\n")

        # write out the fit info
        line = self.cf.dump_info(self.mc_fit, delimiter=",")
        fh.write(line)
        try:
            fh.write("Fit Error\n{0:.18f}\n".format(self.fiterr))
        except ValueError:
            pass
        fh.write(joinn(["WLF C1", "WLF C2"], sep=","))
        fh.write(joinn([self.wlf_opt[0], self.wlf_opt[1]], sep=",", num=True))

        # fh.write('Data\n')
        # self.dfm.to_csv(fh, float_format='%.18f', index=False)


class _CurveFitter(object):
    """CurveFitter base class"""

    name = None
    key = None
    plot_label = "Curve fit"
    requires_opt = False

    def fit_points(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def dump_info(self, *args, **kwargs):
        raise NotImplementedError


class PronyFit(_CurveFitter):
    name = "Prony"
    key = PRONY
    plot_label = r"$\sum_{i=1}^{n} y_i e^{\frac{t/a_T}{\tau_i}}$"

    def __init__(self, *args, **kwargs):
        optprony = kwargs.pop("optprony", False)
        self.optprony = optprony and sciopt is not None

    def fit_points(self, xp, yp):
        """Retuns the best fits for a Prony series

        Parameters
        ----------
        xp : ndarray
            x points (log(t/aT))
        yp : ndarray
            y points

        Returns
        -------
        fit : ndarray
            (tau_i, Y_i) for Prony series fit (last point is ( ,Y_inf))

        Notes
        -----
        Fits
                       ---
                       \        -(t/aT) / tau_i
            Y = Y_0 +  /   Y_i e
                       ---

        with a least squares fit.

        xp should be given in ascending order.

        """
        n = len(xp)
        mn = np.amin(xp)
        mx = np.amax(xp)

        # number of decade points
        ndp = int(mx - mn + 1)

        # decades
        nn = ndp + 1
        tau = np.empty(nn)
        tau[:-1] = [BASE ** (round(mn) + i) for i in range(ndp)]

        d = np.zeros((n, nn))
        d[:, -1] = 1.0
        for i in range(n):
            d[i, :ndp] = np.exp(-(BASE ** xp[i]) / tau[:ndp])
        try:
            dtdi = np.linalg.inv(np.dot(d.T, d))
        except np.linalg.LinAlgError:
            raise ValueError("adjust initial WLF coefficients")
        ddd = np.dot(dtdi, d.T)
        coeffs = np.dot(ddd, yp)
        if self.optprony:
            # finish off the optimization.  The following optimizes both
            # tau and coeffs
            def func(xarg, *args):
                ni, xp, yp = args
                ci = xarg[:ni]
                ti = xarg[ni:]
                y = [self._eval(ti, ci, x) for x in xp]
                err = np.sqrt(np.mean((yp - y) ** 2))
                return err

            xarg = np.append(coeffs, tau)
            xopt = sciopt.fmin(func, xarg, args=(nn, xp, yp), disp=0)
            coeffs = xopt[:nn]
            tau = xopt[nn:]

        return np.column_stack((tau, coeffs))

    def _eval(self, ti, ci, z):
        s = np.sum(ci[:-1] * np.exp(-z / ti[:-1]))
        return ci[-1] + s

    def eval(self, fit, z):
        """Determine the value on the curve defined by a Prony series at z = t / aT

        Parameters
        ----------
        fit : ndarray
            Array returned by fit_points

        Returns
        -------
        val : real
            The value of the Prony series at z

        """
        ti, ci = fit[:, 0], fit[:, 1]
        return self._eval(ti, ci, BASE**z)

    def dump_info(self, fit, ffmt=".18f", delimiter=","):
        line = []
        ti, ci = fit[:, 0], fit[:, 1]
        line.append(["tau_{0:d}".format(i + 1) for i in range(len(ti) - 1)])
        line.append(["{0:{1}}".format(r, ffmt) for r in ti[:-1]])
        line.append(["y_{0:d}".format(i + 1) for i in range(len(ci) - 1)])
        line[-1].append("y_inf")
        line.append(["{0:{1}}".format(r, ffmt) for r in ci])
        line = joinn(line, sep=delimiter)
        return line


class ModifiedPowerFit(_CurveFitter):
    name = "Modified Power"
    key = MODIFIED_POWER
    requires_opt = True
    plot_label = r"$y_0 + y_1 \left(\frac{t}{a_T}\right) ^ a$"

    def __init__(self, *args, **kwargs):
        pass

    def fit_points(self, xp, yp):
        """Retuns the best fits for a modified power curve

        Parameters
        ----------
        xp : ndarray
            x points (log(t/aT))
        yp : ndarray
            y points

        Returns
        -------
        fit : ndarray
            The fit parameters [Ee, E1, a]

        Notes
        -----
        Fits

                                  a
               Er = Ee + E1 (t/aT)

        """

        def func(p, x, y):
            return y - self._eval(p[0], p[1], p[2], x)

        out, success = sciopt.leastsq(func, [200.0, 100.0, -0.1], args=(xp, yp))
        return out[:3]

    def eval(self, fit, x):
        Ee, E1, a = fit
        return self._eval(Ee, E1, a, x)

    @staticmethod
    def _eval(Ee, E1, a, x):
        return Ee + E1 * (BASE**x) ** a

    def dump_info(self, fit, ffmt=".18f", delimiter=","):
        line = []
        line.append(["Ee", "E1", "a"])
        line.append(["{0:{1}}".format(r, ffmt) for r in fit])
        line = joinn(line, sep=delimiter)
        return line


class PowerFit(_CurveFitter):
    name = "Power"
    key = POWER
    requires_opt = True
    plot_label = r"$y_0\left(\frac{t}{a_T}\right) ^ a$"

    def __init__(self, *args, **kwargs):
        pass

    def fit_points(self, xp, yp):
        """Retuns the best fits for a power curve

        Parameters
        ----------
        xp : ndarray
            x points (log(t/aT))
        yp : ndarray
            y points

        Returns
        -------
        fit : ndarray
            The fit parameters [E0, a]

        Notes
        -----
        Fits

                                  a
               Er = E0 (t/aT)

        """

        def func(p, x, y):
            return y - self._eval(p[0], p[1], x)

        out, success = sciopt.leastsq(func, [100.0, -0.1], args=(xp, yp))
        return out[:2]

    def eval(self, fit, x):
        E0, a = fit
        return self._eval(E0, a, x)

    @staticmethod
    def _eval(E0, a, x):
        return E0 * (BASE**x) ** a

    def dump_info(self, fit, ffmt=".18f", delimiter=","):
        line = []
        line.append(["E0", "a"])
        line.append(["{0:{1}}".format(r, ffmt) for r in fit])
        line = joinn(line, sep=delimiter)
        return line


class PolyFit(_CurveFitter):
    name = "Polynomial"
    key = POLYNOMIAL

    def __init__(self, *args, **kwargs):
        self.order = kwargs.get("order", 2)
        self.p = None
        pass

    @property
    def plot_label(self):
        f = r"\log{\left(\frac{t}{a_T}\right)}"
        if self.p is None:
            return r"$c_0 + c_1 %(f)s + ... + c_n %(f)s^n$" % {"f": f}
        np = self.p.shape[0]
        l = ["c_0"]
        if np > 1:
            l.append("c_1 %(f)s" % {"f": f})
        if np > 2:
            l.extend([r"\cdots", r"c_%(n)d %(f)s^%(n)d" % {"f": f, "n": np - 1}])
        return r"${0}$".format(" + ".join(l))

    def fit_points(self, xp, yp):
        """Retuns the best fits for a polynomial curve

        Parameters
        ----------
        xp : ndarray
            x points (log(t/aT))
        yp : ndarray
            y points

        """
        self.p = np.polyfit(xp, yp, self.order)
        return self.p

    def eval(self, fit, x):
        return np.poly1d(fit)(x)

    def dump_info(self, fit, ffmt=".18f", delimiter=","):
        line = []
        line.append(["p_{0}".format(i) for i in range(fit.shape[0])])
        line.append(["{0:{1}}".format(r, ffmt) for r in fit[::-1]])
        line = joinn(line, sep=delimiter)
        return line


def CurveFitter(key):
    """Curve Fitter factory method"""
    fitters = _CurveFitter.__subclasses__()
    for f in fitters:
        if f.key == key:
            break
    else:
        raise ValueError("{0}: unrecognized fitter".format(key))
    if f.requires_opt and not sciopt:
        raise ValueError("{0}: requires scipy.optimize".format(key))
    return f


RE = re.compile("[ \,]")


def _split(string, comments, i=0):
    return [x for x in RE.split(string.strip().split(comments, 1)[i]) if x.split()]


def ReadCSV(
    filename,
    apply_log=True,
    ref_temp=75.0,
    cols=[0, 1, 2],
    xvar="Time",
    xunits="min",
    yvar="Er",
    yunits="psi",
    skip_temps=None,
    xfac=1.0,
    yfac=1.0,
    comments="#",
    **kwargs
):
    """MasterCurve factory method

    Parameters
    ----------
    filename : str
        File in which data is found

    Returns
    -------
    mc : MasterCurve

    Notes
    -----
    Each line of filename must be formatted as

      Temperature, x, y

    """
    fown = False
    try:
        if isinstance(filename, str):
            fown = True
            fh = iter(open(filename))
        else:
            fh = iter(filename)
    except (TypeError):
        message = "filename must be a string, file handle, or generator"
        raise ValueError(message)

    data = []
    try:
        for (i, line) in enumerate(fh.readlines()):
            line = _split(line, comments)
            if not line:
                continue
            try:
                line = [float(x) for x in line]
            except ValueError:
                raise ValueError(
                    "expected floates in line{0} " "got {1}".format(i + 1, line)
                )
            data.append(line)
    finally:
        if fown:
            fh.close()

    data = np.array(data)[:, cols]
    data = np.asarray(sorted(data, key=lambda x: (x[0], x[1])))
    return MasterCurve(
        data,
        ref_temp=ref_temp,
        apply_log=apply_log,
        skip_temps=skip_temps,
        xfac=xfac,
        yfac=yfac,
        xvar=xvar,
        xunits=xunits,
        yvar=yvar,
        yunits=yunits,
        **kwargs
    )


def mc_init_notebook(plot_lib="bokeh", i=1):
    lib = plot_lib.lower()
    if lib == "bokeh":
        if bp is None:
            raise ImportError("bokeh")
        if i:
            from bokeh.io import output_notebook

            output_notebook()
        environ.notebook = 2
    elif lib == "matplotlib":
        if plt is None:
            raise ImportError("matplotlib")
        plt.rcParams["figure.figsize"] = (15, 12)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 20
        plt.rcParams["font.serif"] = "Times New Roman"
        plt.rcParams["legend.scatterpoints"] = 1
        plt.rcParams["legend.handlelength"] = 0
        environ.notebook = 1
    else:
        raise ValueError("expected bokeh or matplotlib, got {0!r}".format(plot_lib))


def mcgen_test_data():
    from io import StringIO

    return StringIO(
        """\
# Each line shall be as
# Temperature, Time, Value
0.0000, 0.0100, 4857.0000
0.0000, 0.0316, 3444.0000
0.0000, 0.1000, 2489.0000
0.0000, 0.3162, 1815.0000
0.0000, 1.0000, 1375.0000
0.0000, 3.1623, 1067.0000
0.0000, 10.0000, 852.0000
0.0000, 16.5959, 774.0000
20.0000, 0.0100, 3292.0000
20.0000, 0.0316, 2353.0000
20.0000, 0.1000, 1730.0000
20.0000, 0.3162, 1284.0000
20.0000, 1.0000, 970.0000
20.0000, 3.1623, 746.0000
20.0000, 10.0000, 577.0000
20.0000, 16.5959, 505.0000
40.0000, 0.0100, 2159.0000
40.0000, 0.0316, 1592.0000
40.0000, 0.1000, 1179.0000
40.0000, 0.3162, 920.0000
40.0000, 1.0000, 733.0000
40.0000, 3.1623, 585.0000
40.0000, 10.0000, 484.0000
40.0000, 16.5959, 449.0000
75.0000, 0.0100, 1287.0000
75.0000, 0.0316, 985.0000
75.0000, 0.1000, 767.0000
75.0000, 0.3162, 616.0000
75.0000, 1.0000, 498.0000
75.0000, 3.1623, 410.0000
75.0000, 10.0000, 333.0000
75.0000, 16.5959, 311.0000
100.0000, 0.0100, 1123.0000
100.0000, 0.0316, 881.0000
100.0000, 0.1000, 708.0000
100.0000, 0.3162, 573.0000
100.0000, 1.0000, 471.0000
100.0000, 3.1623, 399.0000
100.0000, 10.0000, 341.0000
100.0000, 16.5959, 316.0000
130.0000, 0.0100, 810.0000
130.0000, 0.0316, 646.0000
130.0000, 0.1000, 523.0000
130.0000, 0.3162, 432.0000
130.0000, 1.0000, 364.0000
130.0000, 3.1623, 313.0000
130.0000, 10.0000, 271.0000
130.0000, 16.5959, 254.0000"""
    )


if __name__ == "__main__":
    # Baseline solution
    c = np.array([3.292, 181.82])
    p = np.array(
        [
            [0.0001, 2489],
            [0.001, 1482],
            [0.01, 803],
            [0.1, 402],
            [1, 207],
            [10, 124],
            [100, 101],
            [0, 222],
        ],
        dtype=np.float64,
    )
    mc = ReadCSV(
        mcgen_test_data(),
        ref_temp=75.0,
        apply_log=True,
        fitter=PRONY,
        optimizer=FMIN,
        optwlf=False,
    )
    s1 = "WLF coefficients not within tolerance"
    mc.fit()
    assert np.allclose(mc.wlf_opt, c, rtol=1.0e-3, atol=1.0e-3), s1
    s2 = "Prony series not within tolerance"
    assert np.allclose(mc.mc_fit[:, 1], p[:, 1], rtol=1.0e-2, atol=1.0e-2), s2
    mc.fit(optimize=True)
    print("Success")
