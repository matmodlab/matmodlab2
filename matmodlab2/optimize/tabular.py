from __future__ import print_function
import sys
import time
import argparse
import numpy as np
import xml.dom.minidom as xdom
from os.path import realpath, join, isdir, isfile, dirname, splitext
from ..core.environ import environ

U_ROOT = "SimulationData"
U_JOB = "job"
U_DATE = "date"
U_EVAL = "Evaluation"
U_EVAL_N = "n"
U_EVAL_D = "d"
U_EVAL_S = "status"
U_PARAMS = "Parameters"
U_RESP = "Responses"
IND = "  "


class TabularWriter(object):
    def __init__(self, filename, job):
        """Set up a logger object, which takes evaluation events and outputs
        an XML log file

        """
        self.stack = []

        self.filename = realpath(filename)
        if not self.filename.endswith(".edb"):
            self.filename += ".edb"
        self.evald = dirname(self.filename)
        if not isdir(self.evald):
            raise OSError("no such directory {0!r}".format(self.evald))
        self.start_document(job)
        pass

    def create_element(self, name, attrs):
        sp = IND * len(self.stack)
        a = " ".join('{0}="{1}"'.format(k, v) for (k, v) in attrs)
        with open(self.filename, "a") as stream:
            stream.write("{0}<{1} {2}/>\n".format(sp, name, a))
            stream.flush()
        return

    def start_element(self, name, attrs, end=False):
        sp = IND * len(self.stack)
        a = " ".join('{0}="{1}"'.format(k, v) for (k, v) in attrs)
        with open(self.filename, "a") as stream:
            stream.write("{0}<{1} {2}>\n".format(sp, name, a))
            stream.flush()
        self.stack.append(name)
        return

    def end_element(self, name):
        _name = self.stack.pop(-1)
        assert _name == name
        sp = IND * len(self.stack)
        with open(self.filename, "a") as stream:
            stream.write("{0}</{1}>\n".format(sp, name))
            stream.flush()
        return

    def start_document(self, job):
        with open(self.filename, "w") as stream:
            stream.write("""<?xml version="1.0"?>\n""")
            stream.flush()
        now = time.asctime(time.localtime())
        self.start_element(U_ROOT, ((U_JOB, job), (U_DATE, now)))
        return

    def end_document(self):
        _name = self.stack.pop(-1)
        assert _name == U_ROOT
        with open(self.filename, "a") as stream:
            stream.write("</{0}>\n".format(U_ROOT))
            stream.flush()
            stream.close()
        return

    def write_eval_info(self, n, s, d, parameters, responses=None):
        """Write information for this evaluation

        Parameters
        ----------
        n : int
            Evaluation number
        s : int
            Evaluation status
        d : int
            Evaluation directory
        parameters : list of tuple
            (name, value) pairs for each parameter
        respones : list of tuple (optional)
            (name, value) pairs for each response

        """
        d = d.replace(self.evald, ".")
        self.start_element(U_EVAL, ((U_EVAL_N, n), (U_EVAL_S, s), (U_EVAL_D, d)))
        self.create_element(U_PARAMS, parameters)
        if responses:
            self.create_element(U_RESP, responses)
        self.end_element(U_EVAL)
        return

    def close(self):
        """
        Clean up the logger object
        """
        self.end_document()
        return


def read_mml_evaldb(filepath):
    """Read the Material Model Laboratory tabular file

    Parameters
    ----------
    filepath : str
        Path to index file to read

    Returns
    -------
    sources : list of str
        Individual filepaths for each evaluation
    parameters : tuple of tuple
        (name, value) pairs for parameters for each evaluation

    """
    D = realpath(dirname(filepath))
    doc = xdom.parse(filepath)
    root = doc.getElementsByTagName(U_ROOT)[0]
    job = root.getAttribute(U_JOB)

    sources = []
    parameters = {}
    responses = {}
    for evaluation in root.getElementsByTagName(U_EVAL):
        n = evaluation.getAttribute(U_EVAL_N) # noqa
        d = realpath(join(D, evaluation.getAttribute(U_EVAL_D)))
        f = join(d, "{0}.exo".format(job))
        if isfile(f):
            sources.append(f)

        # get parameters
        nparams = evaluation.getElementsByTagName(U_PARAMS)[0]
        evars, enames = [], []
        for (name, value) in nparams.attributes.items():
            enames.append(name)
            evars.append(float(value))
        parameters[f] = zip(enames, evars)

        # get responses
        nresponses = evaluation.getElementsByTagName(U_RESP)
        if nresponses:
            rvars, rnames = [], []
            for (name, value) in nresponses[0].attributes.items():
                rnames.append(name)
                rvars.append(float(value))
            responses[f] = zip(rnames, rvars)

    return sources, parameters, responses


def read_mml_evaldb_nd(filepath, nonan=1):
    sources, parameters, responses = read_mml_evaldb(filepath)
    head = [x[0] for x in parameters[sources[0]]]
    resp = responses.get(sources[0])
    if resp:
        head.extend([x[0] for x in resp])
    data = []
    for source in sources:
        r = responses.get(source)
        if r is None:
            continue
        p = parameters[source]
        line = [x[1] for x in p]
        line.extend([x[1] for x in r])
        data.append(line)
    data = np.array(data)
    if nonan:
        # remove nan's
        rows = np.where(np.isnan(data))[0]
        data = np.delete(data, rows, 0)
    return head, data, len(responses[sources[0]])


def correlations(filepath, nonan=1):
    title = "CORRELATIONS AMONG INPUT AND OUTPUT VARIABLES CREATED BY MATMODLAB"
    head, data, nresp = read_mml_evaldb_nd(filepath, nonan=nonan)
    H = " " * 13 + " ".join("{0:>12s}".format(x) for x in head)
    with open(splitext(filepath)[0] + ".corr", "w") as fobj:
        fobj.write("{0}\n".format(title))
        # get correlation matrix
        corrcoef = np.corrcoef(data, rowvar=0)
        i = 1
        fobj.write("{0}\n".format(H))
        for row in corrcoef:
            fobj.write(
                "{0:>12} {1}\n".format(
                    head[i - 1], " ".join("{0: 12.2f}".format(x) for x in row[:i])
                )
            )
            i += 1
    return


def plot_correlations(filepath, nonan=1, pdf=0):
    if environ.notebook == 2 and not pdf:
        return plot_bokeh_correlations(filepath, nonan)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter  # noqa
    except ImportError:
        print("unable to import matplotlib")
        return
    head, data, nresp = read_mml_evaldb_nd(filepath, nonan=nonan)

    # create xy scatter plots
    y = data[:, -nresp]
    sort = np.argsort(y)
    y = y[sort]

    keys = head[:-nresp]
    colors = "bgrcmykw"

    pdf = "{0}.pdf".format(splitext(filepath)[0])
    plt.clf()

    # set up subplots
    fig, axs = plt.subplots(1, len(keys), sharey=True)
    if len(keys) == 1:
        axs = [axs]

    ylabel = r"{0}".format(head[-1])
    axs[0].set_ylabel(ylabel)
    for i, key in enumerate(keys):
        x = data[:, i][sort]
        m2, m, b = np.polyfit(x, y, 2)
        m2, (m, b) = 0, np.polyfit(x, y, 1)
        axs[i].plot(x, y, "{0}.".format(colors[i]), x, m2 * x * x + m * x + b, "-k")
        axs[i].set_xlabel(r"{0}".format(key))
        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45, fontsize="small")
        continue
    plt.savefig(pdf, transparent=True)

    return


def plot_bokeh_correlations(filepath, nonan=1):
    from bokeh.plotting import figure, gridplot

    head, data, nresp = read_mml_evaldb_nd(filepath, nonan=nonan)

    # create xy scatter plots
    y = data[:, -nresp]
    sort = np.argsort(y)
    y = y[sort]

    keys = head[:-nresp]
    colors = ("blue", "green", "red", "cyan", "maroon", "yellow", "black", "white")

    ylabel = r"{0}".format(head[-1])
    plots = []
    for i, key in enumerate(keys):
        x = data[:, i][sort]
        m2, m, b = np.polyfit(x, y, 2)
        m2, (m, b) = 0, np.polyfit(x, y, 1)
        TOOLS = "pan,wheel_zoom,box_zoom,reset,save,resize"
        y_axis_label = ylabel if not i else None
        p = figure(
            tools=TOOLS, x_axis_label=r"{0}".format(key), y_axis_label=y_axis_label
        )
        p.scatter(x, y, color=colors[i])
        p.line(x, m2 * x * x + m * x + b, color="black")
        plots.append(p)
    return gridplot([plots])


def is_evaldb(filename):
    if not isfile(filename) or not filename.endswith(".edb"):
        return False
    with open(filename, "r") as fh:
        for i in range(4):
            if U_ROOT in fh.readline():
                return True
    return False


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plot", "table"))
    parser.add_argument("filepath")
    args = parser.parse_args(argv)
    if args.action == "plot":
        sys.exit(plot_correlations(args.filepath))
    sys.exit(correlations(args.filepath))


if __name__ == "__main__":
    main(sys.argv[1:])
