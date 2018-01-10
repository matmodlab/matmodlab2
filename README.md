# The Material Model Laboratory

## What is it?

The material model laboratory (*Matmodlab2*) is a material point simulator developed as a tool for developing and analyzing material models for use in larger finite element codes.

## System Requirements

*Matmodlab2* has been built and tested extensively on several versions of linux
and the Apple Mac OSX 10 operating systems.

### Required Software

*Matmodlab2* requires the following software installed for your platform:

- [Python (3.5 or newer)](http://www.python.org) or newer

- [Numpy](http://www.numpy.org)

- [Scipy](http://www.scipy.org)


*Matmodlab2* is developed and tested using the [Anaconda](http://continuum.io) Python distributions.

### Optional Software

- [Pandas](http://www.pandas.pydata.org)

- [py.test](http://doc.pytest.org/en/latest)

## Installation

The easiest way to get started with *Matmodlab2* is to
clone or download *Matmodlab2* from the
[repository](https://www.github.com/matmodlab/matmodlab2), navigate to the
`matmodlab2` directory, and execute

```
python setup.py install
```

which will install *Matmodlab2* to your Python package's `site-packages`
directory.  Optionally, execute

```
python setup.py develop
```

and source files files are *linked* to the Python interpreter’s site-packages,
rather than copied. This way, changes made to source files are applied
immediately and do not require you to re-install *Matmodlab2*.

Another method to "install" *Matmodlab2* is to simply set the `PYTHONPATH`
environment variable to the `matmodlab2` directory.

## Documentation

The documentation consists of Jupyter notebooks contained in the `matmodlab2/notebooks` directory and vieweable at [Introduction to Matmodlab](https://github.com/tjfulle/matmodlab2/blob/master/notebooks/Introduction.ipynb)

## Troubleshooting

If you experience problems when building/installing/testing *Matmodlab2*, you
can ask help from Tim Fuller <timothy.fuller@utah.edu>
