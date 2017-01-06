import os
import numpy as np
from pandas import read_table

def amag(a):
    return np.sqrt(np.sum(a * a))

def rms_error(t1, d1, t2, d2, disp=1):
    """Compute the RMS and normalized RMS error

    """
    t1 = np.asarray(t1)
    d1 = np.asarray(d1)
    t2 = np.asarray(t2)
    d2 = np.asarray(d2)

    if t1.shape[0] == t2.shape[0]:
        rms = np.sqrt(np.mean((d1 - d2) ** 2))
    else:
        rms = interp_rms_error(t1, d1, t2, d2)
    dnom = np.amax(np.abs(d1))
    if dnom < 1.e-12: dnom = 1.
    if disp:
        return rms, rms / dnom
    return rms / dnom


def interp_rms_error(t1, d1, t2, d2):
    """Compute RMS error by interpolation

    """
    ti = max(np.amin(t1), np.amin(t2))
    tf = min(np.amax(t1), np.amax(t2))
    n = t1.shape[0]
    f1 = lambda x: np.interp(x, t1, d1)
    f2 = lambda x: np.interp(x, t2, d2)
    rms = np.sqrt(np.mean([(f1(t) - f2(t)) ** 2
                           for t in np.linspace(ti, tf, n)]))
    return rms

def same_as_baseline(mps, baseline=None, interp=0):

    dtol = 1.5E-06
    ftol = 1.E-04
    floor = 1.E-12

    df1 = mps.df
    if baseline is None:
        baseline = os.path.join('data', mps.jobid+'.base_dat')
    df2 = read_table(baseline, sep='\s+')

    t1 = df1['Time']
    t2 = df2['Time']

    if not interp:
        # interpolation will not be used when comparing values, so the
        # timesteps must be equal
        if t1.shape[0] != t2.shape[0]:
            raise Exception('Number of timesteps differ')

        if not np.allclose(t1, t2, atol=ftol, rtol=ftol):
            raise Exception('Timestep size in File1 and File2 differ')

    vars_to_compare = [x for x in df1.columns
                       if x in df2.columns and x != 'Time']

    passed, diffed, failed = [], [], []
    for key in vars_to_compare:
        d1 = np.where(df1[key] <= floor, 0., df1[key])
        d2 = np.where(df2[key] <= floor, 0., df2[key])
        if not interp:
            if np.allclose(d1, d2, atol=ftol, rtol=ftol):
                passed.append(key)
                continue

        if amag(d1) < 1.e-10 and amag(d2) < 1.e-10:
            passed.append(key)
            continue

        rms, nrms = rms_error(t1, d1, t2, d2)
        if nrms < dtol:
            passed.append(key)

        elif nrms < ftol:
            diffed.append(key)

        else:
            failed.append(key)

        continue

    if len(passed) == len(vars_to_compare):
        return True

    return False

def responses_are_same(a, b, vars):
    failtol = 1.E-02
    difftol = 5.E-03
    T = a[:, 0]
    t = b[:, 0]
    passed, diffed, failed = 0, 0, 0
    for col in range(1, len(vars)):
        X = a[:, col]
        x = b[:, col]
        nrms = rms_error(T, X, t, x, disp=0)
        if nrms < difftol:
            passed += 1
        elif nrms < failtol:
            diffed += 1
        else:
            failed += 1
    return passed == len(vars[1:])
